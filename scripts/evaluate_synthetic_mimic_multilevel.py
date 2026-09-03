#!/usr/bin/env python
"""Multi-level evaluation for MIMIC tuple synthetic time-series.

This script complements downstream TSTR/TSRTR with realism, temporal dynamics,
and privacy-oriented diagnostics. Lower is better for every score emitted here
except the reported label rates and nearest-neighbor distances.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURES = ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"]


def ensure_ntf(x, input_dim=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D time-series tensor, got {x.shape}")
    if input_dim is not None:
        input_dim = int(input_dim)
        if x.shape[1] == input_dim:
            return x.transpose(0, 2, 1)
        if x.shape[2] == input_dim:
            return x
        raise ValueError(
            f"Expected {input_dim}-feature tensor in (N,{input_dim},T) or (N,T,{input_dim}), got {x.shape}"
        )
    if x.shape[1] <= x.shape[2]:
        return x.transpose(0, 2, 1)
    if x.shape[2] < x.shape[1]:
        return x
    raise ValueError(f"Cannot infer tuple orientation; pass --input-dim for shape {x.shape}")


def load_tuple(path, input_dim=None):
    x, y = pd.read_pickle(path)
    x = ensure_ntf(x, input_dim=input_dim)
    y = np.asarray(y, dtype=np.int64)
    if len(x) != len(y):
        raise ValueError(f"{path}: len(X)={len(x)} != len(y)={len(y)}")
    return x, y


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stats_from(x):
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(x.std(axis=(0, 1), keepdims=True), 1e-6).astype(np.float32)
    return mean, std


def normalize(x, stats):
    mean, std = stats
    x = (x.astype(np.float32) - mean) / std
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def subsample(x, y=None, max_samples=4096, seed=42, stratify=True):
    if max_samples <= 0 or len(x) <= max_samples:
        return (x, y) if y is not None else x
    rng = np.random.default_rng(seed)
    if y is not None and stratify:
        parts = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            n_cls = max(1, int(round(max_samples * len(idx) / len(y))))
            parts.append(rng.choice(idx, size=min(n_cls, len(idx)), replace=False))
        idx = np.concatenate(parts)
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        elif len(idx) < max_samples:
            rest = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
            if len(rest):
                extra = rng.choice(rest, size=min(max_samples - len(idx), len(rest)), replace=False)
                idx = np.concatenate([idx, extra])
    else:
        idx = rng.choice(len(x), size=max_samples, replace=False)
    rng.shuffle(idx)
    return (x[idx], y[idx]) if y is not None else x[idx]


class ArrayDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_embedding=False):
        _, h = self.gru(x)
        emb = h[-1]
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits


class GRUPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)


def train_classifier(x_train, y_train, x_val, y_val, args, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = GRUClassifier(input_dim=x_train.shape[-1], hidden_dim=args.hidden_dim, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    best_state = None
    best_acc = -1.0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        acc = classifier_accuracy(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def classifier_accuracy(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(yb.numpy())
    return float(accuracy_score(np.concatenate(labels), np.concatenate(preds)))


def extract_embeddings(model, x, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = DataLoader(ArrayDataset(x), batch_size=args.batch_size, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for xb in loader:
            _, emb = model(xb.to(device), return_embedding=True)
            outs.append(emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_predictor(x_train, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = GRUPredictor(input_dim=x_train.shape[-1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    loader = DataLoader(ArrayDataset(x_train), batch_size=args.batch_size, shuffle=True)
    for _ in range(args.epochs):
        model.train()
        for xb in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb[:, :-1])
            loss = loss_fn(pred, xb[:, 1:])
            loss.backward()
            opt.step()
    return model


def predictive_mae(model, x_eval, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = DataLoader(ArrayDataset(x_eval), batch_size=args.batch_size, shuffle=False)
    losses = []
    with torch.no_grad():
        model.eval()
        for xb in loader:
            xb = xb.to(device)
            pred = model(xb[:, :-1])
            losses.append(torch.abs(pred - xb[:, 1:]).mean().item() * len(xb))
    return float(np.sum(losses) / len(x_eval))


def stable_sqrtm(mat):
    try:
        from scipy.linalg import sqrtm

        out = sqrtm(mat)
        if np.iscomplexobj(out):
            out = out.real
        return out
    except Exception:
        vals, vecs = np.linalg.eigh((mat + mat.T) / 2.0)
        vals = np.maximum(vals, 0.0)
        return (vecs * np.sqrt(vals)) @ vecs.T


def fid_score(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    covmean = stable_sqrtm(cov_a @ cov_b)
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * covmean))


def discriminative_score(real_x, syn_x, stats, args, seed):
    n = min(len(real_x), len(syn_x), args.max_samples)
    real_x = subsample(real_x, max_samples=n, seed=seed)
    syn_x = subsample(syn_x, max_samples=n, seed=seed + 1)
    x = np.concatenate([normalize(real_x, stats), normalize(syn_x, stats)], axis=0)
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    model = train_classifier(x_tr, y_tr, x_va, y_va, args, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    acc = classifier_accuracy(model, DataLoader(ArrayDataset(x_te, y_te), batch_size=args.batch_size), device)
    return float(abs(acc - 0.5)), float(acc)


def feature_metrics(real_x, real_y, syn_x, syn_y, ranges, features):
    rows = {}
    masks = [("all", np.ones(len(real_y), dtype=bool), np.ones(len(syn_y), dtype=bool))]
    for cls in sorted(set(np.unique(real_y).astype(int).tolist()) | set(np.unique(syn_y).astype(int).tolist())):
        masks.append((f"class{cls}", real_y == cls, syn_y == cls))
    for cls_name, mask_real, mask_syn in masks:
        if not np.any(mask_real) or not np.any(mask_syn):
            continue
        r = real_x[mask_real]
        s = syn_x[mask_syn]
        prefix = f"{cls_name}_"
        rows[prefix + "mean_abs_diff"] = float(np.mean(np.abs(s.mean(axis=(0, 1)) - r.mean(axis=(0, 1)))))
        rows[prefix + "std_abs_diff"] = float(np.mean(np.abs(s.std(axis=(0, 1)) - r.std(axis=(0, 1)))))
        for q in (0.01, 0.05, 0.50, 0.95, 0.99):
            rows[prefix + f"q{int(q * 100):02d}_abs_diff"] = float(
                np.mean(np.abs(np.quantile(s, q, axis=(0, 1)) - np.quantile(r, q, axis=(0, 1))))
            )
        ac_diffs = []
        for f in range(r.shape[-1]):
            if r.shape[1] < 2:
                continue
            r_ac = np.mean((r[:, :-1, f] - r[:, :-1, f].mean()) * (r[:, 1:, f] - r[:, 1:, f].mean()))
            s_ac = np.mean((s[:, :-1, f] - s[:, :-1, f].mean()) * (s[:, 1:, f] - s[:, 1:, f].mean()))
            ac_diffs.append(abs(float(s_ac - r_ac)))
        rows[prefix + "lag1_autocov_abs_diff"] = float(np.mean(ac_diffs)) if ac_diffs else float("nan")

    if ranges:
        total = 0
        bad = 0
        for i, feat in enumerate(features):
            if feat not in ranges:
                continue
            lo, hi = ranges[feat]
            vals = syn_x[:, :, i]
            total += vals.size
            bad += int(np.sum((vals < lo) | (vals > hi)))
        rows["range_violation_rate"] = float(bad / max(total, 1))
    return rows


def nn_metrics(real_train_x, syn_x, args, seed):
    real_flat = subsample(real_train_x.reshape(len(real_train_x), -1), max_samples=args.max_samples, seed=seed)
    syn_flat = subsample(syn_x.reshape(len(syn_x), -1), max_samples=args.max_samples, seed=seed + 7)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(real_flat)
    dist, _ = nn.kneighbors(syn_flat)
    dist = dist[:, 0]
    duplicate_eps = float(args.duplicate_eps)
    return {
        "privacy_train_nn_q01": float(np.quantile(dist, 0.01)),
        "privacy_train_nn_q05": float(np.quantile(dist, 0.05)),
        "privacy_train_nn_median": float(np.median(dist)),
        "privacy_near_duplicate_rate": float(np.mean(dist <= duplicate_eps)),
    }


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_synthetic(items):
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError("--synthetic must be method_name=path")
        name, path = item.split("=", 1)
        out.append((name, Path(path)))
    return out


def parse_feature_names(raw, meta, input_dim):
    if raw:
        names = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    else:
        names = [str(x).strip().lower() for x in meta.get("feature_names", []) if str(x).strip()]
    if not names:
        names = DEFAULT_FEATURES[: int(input_dim)]
    if len(names) != int(input_dim):
        names = [f"f{i:02d}" for i in range(int(input_dim))]
    if len(names) != int(input_dim):
        raise ValueError(f"Expected {input_dim} feature names, got {len(names)}: {names}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Realism/dynamics/privacy metrics for tuple synthetic time-series data.")
    parser.add_argument("--data-dir", default="data/processed/mimic_mortality_reasonable_7of7")
    parser.add_argument("--input-dim", type=int, default=None, help="optional tuple feature dimension; inferred when omitted")
    parser.add_argument("--feature-names", default=None, help="optional comma-separated feature names")
    parser.add_argument("--synthetic", action="append", default=[], help="method_name=path/to/synthetic.pkl")
    parser.add_argument("--out", default="results/mimic_mortality_reasonable_multilevel_metrics.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-eps", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    x_train, y_train = load_tuple(data_dir / "train_tuple.pkl", input_dim=args.input_dim)
    if args.input_dim is None:
        args.input_dim = int(x_train.shape[-1])
    x_val, y_val = load_tuple(data_dir / "val_tuple.pkl", input_dim=args.input_dim)
    x_test, y_test = load_tuple(data_dir / "test_tuple.pkl", input_dim=args.input_dim)
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    real_ref_x = np.concatenate([x_train, x_val], axis=0)
    real_ref_y = np.concatenate([y_train, y_val], axis=0)
    stats = stats_from(x_train)

    ranges = {}
    meta = {}
    meta_path = data_dir / "preprocess_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            ranges = meta.get("feature_ranges", {})
    features = parse_feature_names(args.feature_names, meta, args.input_dim)

    x_tr_n, y_tr_s = subsample(normalize(x_train, stats), y_train, args.max_samples, args.seed)
    x_va_n, y_va_s = subsample(normalize(x_val, stats), y_val, args.max_samples, args.seed)
    x_te_n, y_te_s = subsample(normalize(x_test, stats), y_test, args.max_samples, args.seed)
    context_model = train_classifier(x_tr_n, y_tr_s, x_va_n, y_va_s, args, num_classes=num_classes)
    real_context = extract_embeddings(context_model, x_te_n, args)

    real_predictor = train_predictor(x_tr_n, args)
    real_predictive_mae = predictive_mae(real_predictor, x_te_n, args)

    rows = []
    for method, path in parse_synthetic(args.synthetic):
        x_syn, y_syn = load_tuple(path, input_dim=args.input_dim)
        row = {
            "method": method,
            "n_synthetic": int(len(y_syn)),
            "synthetic_y1_rate": float(np.mean(y_syn == 1)),
            "real_train_y1_rate": float(np.mean(y_train == 1)),
            "real_predictive_mae": real_predictive_mae,
        }
        for cls in range(num_classes):
            row[f"synthetic_class{cls}_rate"] = float(np.mean(y_syn == cls))
            row[f"real_train_class{cls}_rate"] = float(np.mean(y_train == cls))
        disc, disc_acc = discriminative_score(real_ref_x, x_syn, stats, args, args.seed)
        row["discriminative_score"] = disc
        row["discriminator_accuracy"] = disc_acc

        x_syn_norm, y_syn_sub = subsample(normalize(x_syn, stats), y_syn, args.max_samples, args.seed)
        syn_context = extract_embeddings(context_model, x_syn_norm, args)
        row["context_fid"] = fid_score(real_context, syn_context)

        pred_model = train_predictor(x_syn_norm, args)
        row["predictive_score_mae"] = predictive_mae(pred_model, x_te_n, args)
        row["predictive_score_excess_mae"] = row["predictive_score_mae"] - real_predictive_mae

        row.update(feature_metrics(x_test, y_test, x_syn, y_syn, ranges, features))
        row.update(nn_metrics(x_train, x_syn, args, args.seed))
        rows.append(row)
        print(
            f"{method}: DS={row['discriminative_score']:.4f}, "
            f"PS={row['predictive_score_mae']:.4f}, Context-FID={row['context_fid']:.2f}"
        )

    write_rows(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
