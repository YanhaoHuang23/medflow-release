import json
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from dataset import dataset_VQ
import models.ms_flow_matching as ms_flow
import models.ms_vqvae as vqvae
import options.option_ms_flow as option_flow
import utils.eval_msdformer as eval_trans
import utils.utils_model as utils_model

warnings.filterwarnings('ignore')

MIMIC_FEATURES = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'Temp', 'SpO2']


def feature_names_from_args(args):
    raw = getattr(args, 'feature_names', None)
    if raw:
        names = [part.strip() for part in str(raw).split(',') if part.strip()]
        if not names:
            raise ValueError('--feature-names was provided but no non-empty names were parsed')
        return names
    return MIMIC_FEATURES


def is_rank_0() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(labels: np.ndarray, mode: str, num_classes: int):
    if mode == 'none':
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f'Cannot build class weights with empty class counts: {counts.tolist()}')
    max_count = counts.max()
    if mode == 'auto_sqrt':
        weights = np.sqrt(max_count / counts)
    elif mode == 'auto_inverse':
        weights = max_count / counts
    else:
        raise ValueError(f'Unsupported class weight mode: {mode}')
    return torch.tensor(weights, dtype=torch.float32)


def write_positive_mixup_report(report_dir: str, report: dict):
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'positive_mixup_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)


def build_positive_mixup_samples(
    positive_data: np.ndarray,
    multiplier: float,
    alpha: float,
    seed: int,
    report_dir: str = None,
):
    if multiplier <= 1.0:
        return None
    if alpha <= 0:
        raise ValueError(f'--positive-mixup-alpha must be > 0, got {alpha}')
    positive_data = np.asarray(positive_data, dtype=np.float32)
    if positive_data.ndim != 3:
        raise ValueError(f'Expected positive_data with shape (N,T,F), got {positive_data.shape}')
    num_positive = int(positive_data.shape[0])
    if num_positive < 2:
        raise ValueError(f'Need at least two positive samples for mixup, got {num_positive}')

    num_extra = int(round(num_positive * (float(multiplier) - 1.0)))
    if num_extra <= 0:
        return None

    rng = np.random.default_rng(int(seed))
    idx_a = rng.integers(0, num_positive, size=num_extra)
    idx_b = rng.integers(0, num_positive, size=num_extra)
    same = idx_a == idx_b
    while np.any(same):
        idx_b[same] = rng.integers(0, num_positive, size=int(np.sum(same)))
        same = idx_a == idx_b

    lam = rng.beta(float(alpha), float(alpha), size=(num_extra, 1, 1)).astype(np.float32)
    mixed = lam * positive_data[idx_a] + (1.0 - lam) * positive_data[idx_b]
    if report_dir is not None:
        write_positive_mixup_report(report_dir, {
            'mode': 'random',
            'num_positive': num_positive,
            'requested_num_extra': int(num_extra),
            'num_extra': int(num_extra),
            'alpha': float(alpha),
        })
    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def build_knn_positive_mixup_samples(
    positive_data: np.ndarray,
    multiplier: float,
    alpha: float,
    seed: int,
    knn_k: int,
    report_dir: str,
):
    if multiplier <= 1.0:
        return None
    if alpha <= 0:
        raise ValueError(f'--positive-mixup-alpha must be > 0, got {alpha}')
    positive_data = np.asarray(positive_data, dtype=np.float32)
    if positive_data.ndim != 3:
        raise ValueError(f'Expected positive_data with shape (N,T,F), got {positive_data.shape}')
    num_positive = int(positive_data.shape[0])
    if num_positive < 2:
        raise ValueError(f'Need at least two positive samples for knn mixup, got {num_positive}')
    num_extra = int(round(num_positive * (float(multiplier) - 1.0)))
    if num_extra <= 0:
        return None
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            '--positive-mixup-mode knn requires scikit-learn to run nearest-neighbor search.'
        ) from exc

    flat = positive_data.reshape(num_positive, -1)
    flat_mean = flat.mean(axis=0, keepdims=True)
    flat_std = flat.std(axis=0, keepdims=True) + 1e-6
    flat_std_data = (flat - flat_mean) / flat_std
    k = int(min(max(knn_k, 1), num_positive - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(flat_std_data)
    neighbor_idx = nbrs.kneighbors(flat_std_data, return_distance=False)[:, 1:]

    rng = np.random.default_rng(int(seed))
    idx_a = rng.integers(0, num_positive, size=num_extra)
    picked_neighbor = rng.integers(0, k, size=num_extra)
    idx_b = neighbor_idx[idx_a, picked_neighbor]
    lam = rng.beta(float(alpha), float(alpha), size=(num_extra, 1, 1)).astype(np.float32)
    mixed = lam * positive_data[idx_a] + (1.0 - lam) * positive_data[idx_b]
    write_positive_mixup_report(report_dir, {
        'mode': 'knn',
        'num_positive': num_positive,
        'requested_num_extra': int(num_extra),
        'num_extra': int(num_extra),
        'alpha': float(alpha),
        'knn_k': int(k),
    })
    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def build_safe_random_positive_mixup_samples(
    positive_data: np.ndarray,
    multiplier: float,
    alpha: float,
    seed: int,
    safe_quantile: float,
    max_violation_frac: float,
    candidate_multiplier: float,
    report_dir: str,
):
    if multiplier <= 1.0:
        return None
    if alpha <= 0:
        raise ValueError(f'--positive-mixup-alpha must be > 0, got {alpha}')
    positive_data = np.asarray(positive_data, dtype=np.float32)
    if positive_data.ndim != 3:
        raise ValueError(f'Expected positive_data with shape (N,T,F), got {positive_data.shape}')
    num_positive = int(positive_data.shape[0])
    if num_positive < 2:
        raise ValueError(f'Need at least two positive samples for safe random mixup, got {num_positive}')
    requested_num_extra = int(round(num_positive * (float(multiplier) - 1.0)))
    if requested_num_extra <= 0:
        return None

    q = min(max(float(safe_quantile), 0.0), 0.2)
    max_violation_frac = min(max(float(max_violation_frac), 0.0), 1.0)
    candidate_multiplier = max(float(candidate_multiplier), 1.0)
    low = np.quantile(positive_data, q, axis=(0, 1)).astype(np.float32)
    high = np.quantile(positive_data, 1.0 - q, axis=(0, 1)).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    kept = []
    total_candidates = 0
    attempts = 0
    max_attempts = 8
    while sum(len(chunk) for chunk in kept) < requested_num_extra and attempts < max_attempts:
        remaining = requested_num_extra - sum(len(chunk) for chunk in kept)
        candidate_count = int(np.ceil(remaining * candidate_multiplier))
        idx_a = rng.integers(0, num_positive, size=candidate_count)
        idx_b = rng.integers(0, num_positive, size=candidate_count)
        same = idx_a == idx_b
        while np.any(same):
            idx_b[same] = rng.integers(0, num_positive, size=int(np.sum(same)))
            same = idx_a == idx_b
        lam = rng.beta(float(alpha), float(alpha), size=(candidate_count, 1, 1)).astype(np.float32)
        candidates = lam * positive_data[idx_a] + (1.0 - lam) * positive_data[idx_b]
        violation = ((candidates < low.reshape(1, 1, -1)) | (candidates > high.reshape(1, 1, -1))).mean(axis=(1, 2))
        accepted = candidates[violation <= max_violation_frac]
        if len(accepted) > 0:
            kept.append(accepted.astype(np.float32))
        total_candidates += int(candidate_count)
        attempts += 1

    if kept:
        mixed = np.concatenate(kept, axis=0)[:requested_num_extra]
    else:
        mixed = np.empty((0,) + tuple(positive_data.shape[1:]), dtype=np.float32)
    if len(mixed) == 0:
        raise ValueError(
            'safe_random positive mixup rejected all candidates. '
            'Increase --positive-mixup-safe-max-violation-frac or lower --positive-mixup-safe-quantile.'
        )
    write_positive_mixup_report(report_dir, {
        'mode': 'safe_random',
        'num_positive': num_positive,
        'requested_num_extra': int(requested_num_extra),
        'num_extra': int(len(mixed)),
        'alpha': float(alpha),
        'safe_quantile': float(q),
        'safe_max_violation_frac': float(max_violation_frac),
        'safe_candidate_multiplier': float(candidate_multiplier),
        'total_candidates': int(total_candidates),
        'attempts': int(attempts),
        'feature_low': low.astype(float).tolist(),
        'feature_high': high.astype(float).tolist(),
    })
    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def parse_danger_feature_spec(spec: str, feature_names=None):
    parsed = []
    feature_names = list(feature_names or MIMIC_FEATURES)
    feature_to_idx = {name.lower(): idx for idx, name in enumerate(feature_names)}
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f'Invalid --positive-mixup-danger-spec item "{item}". '
                'Use comma-separated feature:low|high entries, e.g. SpO2:low,RR:high.'
            )
        feature_name, direction = [part.strip() for part in item.split(':', 1)]
        key = feature_name.lower()
        if key not in feature_to_idx:
            raise ValueError(f'Unknown danger feature "{feature_name}". Valid features: {feature_names}')
        direction = direction.lower()
        if direction not in {'low', 'high'}:
            raise ValueError(f'Unsupported danger direction "{direction}" for {feature_name}; use low or high.')
        parsed.append((feature_names[feature_to_idx[key]], feature_to_idx[key], direction))
    if not parsed:
        raise ValueError('--positive-mixup-danger-spec must include at least one feature:direction item.')
    return parsed


def compute_positive_danger_scores(
    positive_data: np.ndarray,
    danger_spec: str,
    danger_quantile: float,
    feature_names=None,
):
    positive_data = np.asarray(positive_data, dtype=np.float32)
    q = min(max(float(danger_quantile), 1e-4), 0.49)
    components = {}
    total = np.zeros(positive_data.shape[0], dtype=np.float64)
    for feature_name, feature_idx, direction in parse_danger_feature_spec(danger_spec, feature_names=feature_names):
        values = positive_data[:, :, feature_idx]
        if direction == 'low':
            sample_stat = np.quantile(values, q, axis=1)
            threshold = float(np.quantile(sample_stat, q))
            scale = float(np.std(sample_stat) + 1e-6)
            score = np.maximum((threshold - sample_stat) / scale, 0.0)
        else:
            sample_stat = np.quantile(values, 1.0 - q, axis=1)
            threshold = float(np.quantile(sample_stat, 1.0 - q))
            scale = float(np.std(sample_stat) + 1e-6)
            score = np.maximum((sample_stat - threshold) / scale, 0.0)
        components[f'{feature_name}:{direction}'] = {
            'threshold': threshold,
            'sample_stat_mean': float(np.mean(sample_stat)),
            'sample_stat_std': float(np.std(sample_stat)),
            'score_mean': float(np.mean(score)),
            'score_q50': float(np.quantile(score, 0.50)),
            'score_q90': float(np.quantile(score, 0.90)),
            'score_q99': float(np.quantile(score, 0.99)),
        }
        total += score
    if np.all(total <= 0):
        total = np.ones_like(total, dtype=np.float64)
    total = total / (float(np.mean(total)) + 1e-6)
    return total.astype(np.float64), components


def build_danger_positive_mixup_samples(
    positive_data: np.ndarray,
    multiplier: float,
    alpha: float,
    seed: int,
    mode: str,
    danger_spec: str,
    danger_quantile: float,
    danger_power: float,
    lambda_min: float,
    knn_k: int,
    report_dir: str,
    feature_names=None,
):
    if multiplier <= 1.0:
        return None
    if alpha <= 0:
        raise ValueError(f'--positive-mixup-alpha must be > 0, got {alpha}')
    positive_data = np.asarray(positive_data, dtype=np.float32)
    if positive_data.ndim != 3:
        raise ValueError(f'Expected positive_data with shape (N,T,F), got {positive_data.shape}')
    num_positive = int(positive_data.shape[0])
    if num_positive < 2:
        raise ValueError(f'Need at least two positive samples for danger mixup, got {num_positive}')
    requested_num_extra = int(round(num_positive * (float(multiplier) - 1.0)))
    if requested_num_extra <= 0:
        return None

    danger_scores, components = compute_positive_danger_scores(
        positive_data,
        danger_spec=danger_spec,
        danger_quantile=danger_quantile,
        feature_names=feature_names,
    )
    power = max(float(danger_power), 0.0)
    anchor_weights = np.power(danger_scores + 1e-6, power)
    if not np.isfinite(anchor_weights).all() or float(anchor_weights.sum()) <= 0.0:
        anchor_weights = np.ones(num_positive, dtype=np.float64)
    anchor_probs = anchor_weights / anchor_weights.sum()

    rng = np.random.default_rng(int(seed))
    idx_a = rng.choice(num_positive, size=requested_num_extra, replace=True, p=anchor_probs)
    if mode == 'danger_knn':
        try:
            from sklearn.neighbors import NearestNeighbors
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                '--positive-mixup-mode danger_knn requires scikit-learn to run nearest-neighbor search.'
            ) from exc
        flat = positive_data.reshape(num_positive, -1)
        flat_mean = flat.mean(axis=0, keepdims=True)
        flat_std = flat.std(axis=0, keepdims=True) + 1e-6
        flat_std_data = (flat - flat_mean) / flat_std
        k = int(min(max(knn_k, 1), num_positive - 1))
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(flat_std_data)
        neighbor_idx = nbrs.kneighbors(flat_std_data, return_distance=False)[:, 1:]
        picked_neighbor = rng.integers(0, k, size=requested_num_extra)
        idx_b = neighbor_idx[idx_a, picked_neighbor]
    elif mode == 'danger_random':
        k = None
        idx_b = rng.integers(0, num_positive, size=requested_num_extra)
        same = idx_a == idx_b
        while np.any(same):
            idx_b[same] = rng.integers(0, num_positive, size=int(np.sum(same)))
            same = idx_a == idx_b
    else:
        raise ValueError(f'Unsupported danger mixup mode: {mode}')

    lam = rng.beta(float(alpha), float(alpha), size=(requested_num_extra, 1, 1)).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)
    if float(lambda_min) > 0:
        lam = np.maximum(lam, min(max(float(lambda_min), 0.0), 1.0))
    mixed = lam * positive_data[idx_a] + (1.0 - lam) * positive_data[idx_b]

    sampled_anchor_counts = np.bincount(idx_a, minlength=num_positive)
    report = {
        'mode': mode,
        'num_positive': num_positive,
        'requested_num_extra': int(requested_num_extra),
        'num_extra': int(requested_num_extra),
        'alpha': float(alpha),
        'lambda_min': float(lambda_min),
        'lambda_mean': float(lam.mean()),
        'lambda_q05': float(np.quantile(lam.reshape(-1), 0.05)),
        'lambda_q50': float(np.quantile(lam.reshape(-1), 0.50)),
        'lambda_q95': float(np.quantile(lam.reshape(-1), 0.95)),
        'danger_spec': str(danger_spec),
        'danger_quantile': float(danger_quantile),
        'danger_power': float(danger_power),
        'danger_score_mean': float(np.mean(danger_scores)),
        'danger_score_q50': float(np.quantile(danger_scores, 0.50)),
        'danger_score_q90': float(np.quantile(danger_scores, 0.90)),
        'danger_score_q99': float(np.quantile(danger_scores, 0.99)),
        'anchor_danger_mean': float(np.mean(danger_scores[idx_a])),
        'anchor_danger_q50': float(np.quantile(danger_scores[idx_a], 0.50)),
        'anchor_unique_frac': float(np.count_nonzero(sampled_anchor_counts) / max(num_positive, 1)),
        'anchor_max_count': int(sampled_anchor_counts.max()),
        'components': components,
    }
    if k is not None:
        report['knn_k'] = int(k)
    write_positive_mixup_report(report_dir, report)
    with open(os.path.join(report_dir, 'positive_mixup_danger_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def build_cluster_positive_mixup_samples(
    positive_data: np.ndarray,
    multiplier: float,
    alpha: float,
    seed: int,
    n_clusters: int,
    cluster_balance: str,
    cluster_min_count: int,
    cluster_cap_multiplier: float,
    report_dir: str,
):
    if multiplier <= 1.0:
        return None
    if alpha <= 0:
        raise ValueError(f'--positive-mixup-alpha must be > 0, got {alpha}')
    positive_data = np.asarray(positive_data, dtype=np.float32)
    if positive_data.ndim != 3:
        raise ValueError(f'Expected positive_data with shape (N,T,F), got {positive_data.shape}')
    num_positive = int(positive_data.shape[0])
    if num_positive < 2:
        raise ValueError(f'Need at least two positive samples for cluster mixup, got {num_positive}')
    n_clusters = int(min(max(n_clusters, 1), num_positive))

    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            '--positive-mixup-mode cluster requires scikit-learn to run KMeans.'
        ) from exc

    flat = positive_data.reshape(num_positive, -1)
    feature_mean = flat.mean(axis=0, keepdims=True)
    feature_std = flat.std(axis=0, keepdims=True) + 1e-6
    flat_std = (flat - feature_mean) / feature_std
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
    cluster_ids = kmeans.fit_predict(flat_std)
    counts = np.bincount(cluster_ids, minlength=n_clusters).astype(np.int64)
    min_count = max(int(cluster_min_count), 2)
    valid_clusters = np.where(counts >= min_count)[0]
    if len(valid_clusters) == 0:
        raise ValueError(
            f'Cluster-aware mixup needs at least one cluster with >= {min_count} '
            'positive samples. Lower --positive-mixup-cluster-min-count.'
        )

    if cluster_balance == 'none':
        probs = counts[valid_clusters].astype(np.float64)
    elif cluster_balance == 'inverse_sqrt':
        probs = 1.0 / np.sqrt(counts[valid_clusters].astype(np.float64))
    elif cluster_balance == 'inverse':
        probs = 1.0 / counts[valid_clusters].astype(np.float64)
    else:
        raise ValueError(f'Unsupported cluster balance: {cluster_balance}')
    probs = probs / probs.sum()

    requested_num_extra = int(round(num_positive * (float(multiplier) - 1.0)))
    if requested_num_extra <= 0:
        return None
    rng = np.random.default_rng(int(seed))

    cap_multiplier = float(cluster_cap_multiplier)
    if cap_multiplier > 0:
        capacities = np.floor(counts[valid_clusters].astype(np.float64) * cap_multiplier).astype(np.int64)
        capacities = np.maximum(capacities, 0)
    else:
        capacities = np.full(len(valid_clusters), requested_num_extra, dtype=np.int64)
    total_capacity = int(capacities.sum())
    num_extra = min(requested_num_extra, total_capacity)
    if num_extra <= 0:
        raise ValueError(
            'Cluster-aware mixup has zero sampling capacity after min-count filtering '
            'and per-cluster capping. Lower filtering/capping constraints.'
        )

    target_counts = np.zeros(len(valid_clusters), dtype=np.int64)
    remaining = int(num_extra)
    active = capacities > 0
    while remaining > 0 and np.any(active):
        active_idx = np.where(active)[0]
        active_probs = probs[active_idx].astype(np.float64)
        active_probs = active_probs / active_probs.sum()
        draw = rng.multinomial(remaining, active_probs)
        room = capacities[active_idx] - target_counts[active_idx]
        add = np.minimum(draw, room)
        target_counts[active_idx] += add
        new_remaining = int(num_extra - target_counts.sum())
        if new_remaining == remaining:
            addable = active_idx[room > 0]
            if len(addable) == 0:
                break
            picked = rng.choice(addable)
            target_counts[picked] += 1
            new_remaining -= 1
        remaining = new_remaining
        active = target_counts < capacities

    chosen_clusters = np.repeat(valid_clusters, target_counts)
    num_extra = int(len(chosen_clusters))
    if num_extra <= 0:
        raise ValueError(
            'Cluster-aware mixup produced zero sampled clusters after applying caps.'
        )
    rng.shuffle(chosen_clusters)
    idx_a = np.empty(num_extra, dtype=np.int64)
    idx_b = np.empty(num_extra, dtype=np.int64)
    for i, cluster_id in enumerate(chosen_clusters):
        members = np.where(cluster_ids == cluster_id)[0]
        pair = rng.choice(members, size=2, replace=False)
        idx_a[i], idx_b[i] = pair[0], pair[1]

    lam = rng.beta(float(alpha), float(alpha), size=(num_extra, 1, 1)).astype(np.float32)
    mixed = lam * positive_data[idx_a] + (1.0 - lam) * positive_data[idx_b]

    cluster_report = {
        'mode': 'cluster',
        'num_positive': num_positive,
        'requested_num_extra': requested_num_extra,
        'num_extra': num_extra,
        'n_clusters': int(n_clusters),
        'cluster_balance': cluster_balance,
        'cluster_min_count': int(cluster_min_count),
        'cluster_cap_multiplier': float(cluster_cap_multiplier),
        'counts': counts.tolist(),
        'filtered_clusters': np.setdiff1d(np.arange(n_clusters), valid_clusters).astype(int).tolist(),
        'valid_clusters': valid_clusters.astype(int).tolist(),
        'cluster_capacities': dict(zip(valid_clusters.astype(int).tolist(), capacities.astype(int).tolist())),
        'target_counts': dict(zip(valid_clusters.astype(int).tolist(), target_counts.astype(int).tolist())),
        'sampled_counts': np.bincount(chosen_clusters, minlength=n_clusters).astype(int).tolist(),
    }
    os.makedirs(report_dir, exist_ok=True)
    write_positive_mixup_report(report_dir, cluster_report)
    with open(os.path.join(report_dir, 'positive_mixup_cluster_report.json'), 'w', encoding='utf-8') as f:
        json.dump(cluster_report, f, indent=2, sort_keys=True)

    summary_rows = []
    for cluster_id in range(n_clusters):
        members = positive_data[cluster_ids == cluster_id]
        if len(members) == 0:
            continue
        summary_rows.append({
            'cluster': int(cluster_id),
            'count': int(len(members)),
            'feature_mean': members.mean(axis=(0, 1)).astype(float).tolist(),
            'feature_std': members.std(axis=(0, 1)).astype(float).tolist(),
        })
    with open(os.path.join(report_dir, 'positive_mixup_cluster_feature_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2, sort_keys=True)

    return np.clip(mixed, 0.0, 1.0).astype(np.float32)


def encode_numpy_samples(net, samples: np.ndarray, batch_size: int):
    chunks = []
    for start in tqdm(range(0, len(samples), batch_size), desc='Encoding positive mixup samples'):
        batch = torch.from_numpy(samples[start:start + batch_size]).cuda().float()
        with torch.no_grad():
            chunks.append(net.encode(batch).cpu().numpy())
    return np.concatenate(chunks, 0)


def train():
    args = option_flow.get_args_parser()
    seed_everything(args.seed)
    if args.senior_sampler is None:
        raise ValueError('--senior-sampler is required for CE-FM senior mainline')
    if args.class_specific_output_head and args.class_specific_adapter:
        raise ValueError('--class-specific-output-head and --class-specific-adapter are mutually exclusive.')

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('gpu', args.gpu, 'visible', os.environ.get('CUDA_VISIBLE_DEVICES'))

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- stage-1 dataloader used for encoding & evaluation -----
    train_loader_token = dataset_VQ.DATALoader(
        args.dataname,
        args.encode_batch_size,
        window_size=args.window_size,
        unit_length=2 ** args.down_t,
        dataset_type='train',
        data_path=args.data_path,
        class_label=args.class_label,
        return_label=args.conditional_flow,
        input_dim=args.input_dim,
    )

    # ----- load stage-1 frozen VQ -----
    net = vqvae.VQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )

    if args.resume_pth is None:
        raise ValueError('--resume-pth is required for stage-2 flow matching training')
    print('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net = net.float().cuda()
    for p in net.parameters():
        p.requires_grad = False

    # ----- pre-encode token dataset -----
    train_dataset = []
    train_labels = []
    positive_mixup_source = []
    nb_used = set({})
    for batch in tqdm(train_loader_token, desc='Encoding dataset into stage-1 tokens'):
        labels = None
        if args.conditional_flow:
            batch, labels = batch
            positive_mask = labels == int(args.positive_mixup_label)
            if args.positive_mixup_multiplier > 1.0 and positive_mask.any():
                positive_mixup_source.append(batch[positive_mask].cpu().numpy())
        batch = batch.cuda().float()
        with torch.no_grad():
            target = net.encode(batch).cpu().numpy()
        nb_used = nb_used.union(set(target.reshape(-1).tolist()))
        train_dataset.append(target)
        if labels is not None:
            train_labels.append(labels.cpu().numpy())

    if args.conditional_flow and args.positive_mixup_multiplier > 1.0:
        if not positive_mixup_source:
            raise ValueError(
                f'No samples found for --positive-mixup-label {args.positive_mixup_label}; '
                'cannot build positive mixup augmentation.'
            )
        positive_mixup_source = np.concatenate(positive_mixup_source, 0)
        if args.positive_mixup_mode == 'cluster':
            mixed_samples = build_cluster_positive_mixup_samples(
                positive_mixup_source,
                args.positive_mixup_multiplier,
                args.positive_mixup_alpha,
                args.seed,
                args.positive_mixup_clusters,
                args.positive_mixup_cluster_balance,
                args.positive_mixup_cluster_min_count,
                args.positive_mixup_cluster_cap_multiplier,
                args.out_dir,
            )
        elif args.positive_mixup_mode == 'knn':
            mixed_samples = build_knn_positive_mixup_samples(
                positive_mixup_source,
                args.positive_mixup_multiplier,
                args.positive_mixup_alpha,
                args.seed,
                args.positive_mixup_knn_k,
                args.out_dir,
            )
        elif args.positive_mixup_mode == 'safe_random':
            mixed_samples = build_safe_random_positive_mixup_samples(
                positive_mixup_source,
                args.positive_mixup_multiplier,
                args.positive_mixup_alpha,
                args.seed,
                args.positive_mixup_safe_quantile,
                args.positive_mixup_safe_max_violation_frac,
                args.positive_mixup_safe_candidate_multiplier,
                args.out_dir,
            )
        elif args.positive_mixup_mode in {'danger_random', 'danger_knn'}:
            mixed_samples = build_danger_positive_mixup_samples(
                positive_mixup_source,
                args.positive_mixup_multiplier,
                args.positive_mixup_alpha,
                args.seed,
                args.positive_mixup_mode,
                args.positive_mixup_danger_spec,
                args.positive_mixup_danger_quantile,
                args.positive_mixup_danger_power,
                args.positive_mixup_lambda_min,
                args.positive_mixup_knn_k,
                args.out_dir,
                feature_names_from_args(args),
            )
        else:
            mixed_samples = build_positive_mixup_samples(
                positive_mixup_source,
                args.positive_mixup_multiplier,
                args.positive_mixup_alpha,
                args.seed,
                args.out_dir,
            )
        if mixed_samples is not None:
            mixed_tokens = encode_numpy_samples(net, mixed_samples, args.encode_batch_size)
            mixed_labels = np.full(
                mixed_tokens.shape[0],
                int(args.positive_mixup_label),
                dtype=np.int64,
            )
            nb_used = nb_used.union(set(mixed_tokens.reshape(-1).tolist()))
            train_dataset.append(mixed_tokens)
            train_labels.append(mixed_labels)
            print(
                'Positive data-level mixup enabled: '
                f'label={args.positive_mixup_label}, original_positive={len(positive_mixup_source)}, '
                f'added={len(mixed_samples)}, multiplier={args.positive_mixup_multiplier}, '
                f'alpha={args.positive_mixup_alpha}, mode={args.positive_mixup_mode}'
            )

    train_dataset = np.concatenate(train_dataset, 0)
    class_weights = None
    if args.conditional_flow:
        train_labels = np.concatenate(train_labels, 0).astype(np.int64)
        class_counts = np.bincount(train_labels, minlength=args.num_classes)
        class_weights = build_class_weights(train_labels, args.class_loss_weight, args.num_classes)
        print('Class counts:', class_counts.tolist())
        if class_weights is not None:
            print('Class loss weights:', class_weights.tolist())
    print('#####', train_dataset.shape, '######')
    print('The number of used code:', len(nb_used))
    num_train_samples = int(train_dataset.shape[0])
    dynamic_loss_components = [
        item.strip()
        for item in str(args.dynamic_loss_components).split(',')
        if item.strip()
    ]

    # ----- logger -----
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ----- stage-2 flow model -----
    flow_model = ms_flow.MultiScaleFlowMatching(
        nb_code=args.nb_code,
        patch_num=args.patch_num,
        code_dim=args.code_dim,
        fm_backbone=args.fm_backbone,
        flow_path=args.flow_path,
        num_train_samples=num_train_samples,
        latent_rank=args.latent_rank,
        latent_noise_std=args.latent_noise_std,
        t_scheduler=args.t_scheduler,
        hidden_dim=args.fm_hidden_dim,
        depth=args.fm_depth,
        n_heads=args.fm_heads,
        dropout=args.fm_dropout,
        train_mixup_prob=args.train_mixup_prob,
        train_mixup_alpha=args.train_mixup_alpha,
        structure_loss_weight=args.structure_loss_weight,
        senior_mean_reg_weight=args.senior_mean_reg_weight,
        senior_std_reg_weight=args.senior_std_reg_weight,
        senior_sample_noise_std=args.senior_sample_noise_std,
        source_prior_mode=args.source_prior_mode,
        num_classes=args.num_classes if args.conditional_flow else 0,
        cross_scale_conditioning=args.cross_scale_conditioning,
        class_specific_output_head=args.class_specific_output_head,
        class_specific_adapter=args.class_specific_adapter,
        label_conditioning_mode=args.label_conditioning_mode,
        token_type_conditioning=args.token_type_conditioning,
    )

    if args.resume_flow is not None:
        print('loading flow checkpoint from {}'.format(args.resume_flow))
        ckpt = torch.load(args.resume_flow, map_location='cpu')
        flow_model.load_state_dict(ckpt['flow'], strict=True)

    if args.sampling_mode == 'ctf_nearest':
        flow_model.set_training_code_indices(torch.from_numpy(train_dataset).long())
    if args.conditional_flow:
        flow_model.set_training_labels(torch.from_numpy(train_labels).long())

    flow_model = flow_model.cuda()
    flow_model.train()

    optimizer = optim.AdamW(
        flow_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    token_tensor = torch.from_numpy(train_dataset).long()
    sample_ids = torch.arange(num_train_samples, dtype=torch.long)
    if args.conditional_flow:
        label_tensor = torch.from_numpy(train_labels).long()
        base_dataset = TensorDataset(token_tensor, sample_ids, label_tensor)
        if args.class_balanced_sampler:
            class_counts_t = torch.bincount(label_tensor, minlength=args.num_classes).float()
            sample_weights = 1.0 / class_counts_t[label_tensor].clamp_min(1.0)
            sampler = WeightedRandomSampler(
                weights=sample_weights.double(),
                num_samples=len(base_dataset) * args.repeat_times,
                replacement=True,
            )
            train_dataset = base_dataset
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                sampler=sampler,
                num_workers=8,
                drop_last=False,
            )
        else:
            train_dataset = ConcatDataset([base_dataset for _ in range(args.repeat_times)])
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                args.batch_size,
                shuffle=True,
                num_workers=8,
                drop_last=False,
            )
    else:
        base_dataset = TensorDataset(token_tensor, sample_ids)
        train_dataset = ConcatDataset([base_dataset for _ in range(args.repeat_times)])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )
    print('train_loader steps:', len(train_loader))
    train_loader_iter = dataset_VQ.cycle(train_loader)

    # ----- evaluation only mode -----
    if args.if_test:
        eval_trans.evaluation_flow_matching(
            args,
            args.out_dir,
            train_loader_token,
            flow_model,
            net,
            logger,
            writer,
            nb_iter=0,
            best_iter=0,
            best_ds=99999,
            save=True,
        )
        print('Evaluation-only mode completed.')
        return

    # ----- initial eval -----
    start_time = time.time()
    best_mse_iter_test = 0
    best_mse = 99999
    best_iter_test, best_ds, best_mse_iter_test, best_mse, writer, logger = eval_trans.evaluation_flow_matching(
        args,
        args.out_dir,
        train_loader_token,
        flow_model,
        net,
        logger,
        writer,
        0,
        best_iter=0,
        best_ds=99999,
        best_mse=best_mse,
        best_mse_iter=best_mse_iter_test,
    )
    print(f'First evaluation time: {time.time() - start_time:.2f}s')

    # ----- training -----
    nb_iter = 0
    avg_loss_total = 0.0
    avg_metrics = defaultdict(float)

    while nb_iter <= args.total_iter:
        next_batch = next(train_loader_iter)
        batch_labels = None
        if args.conditional_flow:
            batch, batch_sample_ids, batch_labels = next_batch
            batch_labels = batch_labels.cuda().long()
        else:
            batch, batch_sample_ids = next_batch
        batch = batch.cuda().long()
        batch_sample_ids = batch_sample_ids.cuda().long()

        loss_fm, loss_dict = flow_model.compute_loss(
            code_indices=batch,
            quantizers=net.vqvae.quantizer,
            sample_ids=batch_sample_ids,
            labels=batch_labels,
            class_weights=class_weights.cuda() if class_weights is not None else None,
            label_dropout_prob=args.label_dropout_prob,
            class_aware_mixup=args.class_aware_mixup,
            dynamic_loss_weight=args.dynamic_loss_weight,
            dynamic_loss_components=dynamic_loss_components,
            contrastive_flow_weight=args.contrastive_flow_weight,
            contrastive_representation=args.contrastive_representation,
            contrastive_negative_mode=args.contrastive_negative_mode,
            contrastive_objective=args.contrastive_objective,
            contrastive_margin=args.contrastive_margin,
            contrastive_temperature=args.contrastive_temperature,
            balanced_fm_loss=args.balanced_fm_loss,
        )

        optimizer.zero_grad()
        loss_fm.backward()
        optimizer.step()

        nb_iter += 1
        avg_loss_total += loss_dict['loss_total']
        for metric_key, metric_value in loss_dict.items():
            if metric_key == 'loss_total':
                continue
            avg_metrics[metric_key] += metric_value

        if nb_iter % args.print_iter == 0:
            avg_loss_total /= args.print_iter
            writer.add_scalar('./LossFM/train_total', avg_loss_total, nb_iter)

            msg_parts = [f'Train. Iter {nb_iter} : LossFM. {avg_loss_total:.6f}']
            prefix_order = [
                'loss_s',
                'ce_s',
                'structure_s',
                'dynamic_s',
                'contrastive_s',
                'contrastive_valid_frac_s',
                'balanced_class_count_s',
                'mean_s',
                'std_s',
                'reg_s',
                'acc_s',
            ]

            def metric_sort_key(metric_name: str):
                for idx, prefix in enumerate(prefix_order):
                    if metric_name.startswith(prefix):
                        try:
                            scale_idx = int(metric_name.split('s')[-1])
                        except ValueError:
                            scale_idx = 0
                        return (idx, scale_idx, metric_name)
                return (len(prefix_order), 0, metric_name)

            for metric_key in sorted(avg_metrics.keys(), key=metric_sort_key):
                metric_val = avg_metrics[metric_key] / args.print_iter
                writer.add_scalar(f'./LossFM/{metric_key}', metric_val, nb_iter)
                msg_parts.append(f'{metric_key}={metric_val:.6f}')
            avg_metrics = defaultdict(float)

            logger.info(', '.join(msg_parts))
            avg_loss_total = 0.0

        if nb_iter % args.eval_iter == 0 and is_rank_0():
            start_time = time.time()
            best_iter_test, best_ds, best_mse_iter_test, best_mse, writer, logger = eval_trans.evaluation_flow_matching(
                args,
                args.out_dir,
                train_loader_token,
                flow_model,
                net,
                logger,
                writer,
                nb_iter=nb_iter,
                best_iter=best_iter_test,
                best_ds=best_ds,
                best_mse=best_mse,
                best_mse_iter=best_mse_iter_test,
            )
            print(f'evaluation time: {time.time() - start_time:.2f}s')

        if nb_iter == args.total_iter and is_rank_0():
            torch.save({'flow': flow_model.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
            msg_final = (
                f'Train completed. Best DS iter {best_iter_test} : DS. {best_ds:.6f}; '
                f'Best MSE iter {best_mse_iter_test} : MSE. {best_mse:.6f}'
            )
            logger.info(msg_final)
            return


if __name__ == '__main__':
    train()
