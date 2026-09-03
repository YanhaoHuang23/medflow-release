#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.ms_flow_matching as ms_flow  # noqa: E402
import models.ms_vqvae as vqvae  # noqa: E402
from medflow_release.classifier.classifier_train import TimeSeriesDataset  # noqa: E402
from medflow_release.classifier.model import RNNClassifier  # noqa: E402


def ensure_ntc(x, input_dim=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tuple tensor, got {x.shape}")
    if input_dim is not None:
        input_dim = int(input_dim)
        if x.shape[1] == input_dim:
            return x.transpose(0, 2, 1)
        if x.shape[2] == input_dim:
            return x
        raise ValueError(
            f"Expected {input_dim}-feature data in (N,{input_dim},T) or (N,T,{input_dim}), got {x.shape}"
        )
    if x.shape[1] <= x.shape[2]:
        return x.transpose(0, 2, 1)
    if x.shape[2] < x.shape[1]:
        return x
    raise ValueError(f"Cannot infer tuple orientation; pass --input-dim for shape {x.shape}")


def load_tuple(path, input_dim=None):
    x, y = pd.read_pickle(path)
    x = ensure_ntc(x, input_dim=input_dim)
    y = np.asarray(y, dtype=np.int64)
    return x, y


def build_arch_args(args):
    return SimpleNamespace(
        dataname="mimic_icustay",
        input_dim=args.input_dim,
        quantizer=args.quantizer,
        nb_code=args.nb_code,
        patch_num=args.patch_num,
        window_size=args.window_size,
        mu=args.mu,
        beta=args.beta,
    )


def infer_num_train_samples(flow_ckpt):
    state = flow_ckpt["flow"] if "flow" in flow_ckpt else flow_ckpt
    key = "scale_models.0.sample_latents"
    if key not in state:
        raise KeyError(f"Cannot infer num_train_samples; missing {key} in flow checkpoint")
    return int(state[key].shape[0])


def load_vq(args, path, device):
    arch_args = build_arch_args(args)
    model = vqvae.VQVAE(
        arch_args,
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
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_flow(args, path, device, conditional, cross_scale_conditioning=None):
    ckpt = torch.load(path, map_location="cpu")
    if cross_scale_conditioning is None:
        cross_scale_conditioning = args.cross_scale_conditioning
    model = ms_flow.MultiScaleFlowMatching(
        nb_code=args.nb_code,
        patch_num=args.patch_num,
        code_dim=args.code_dim,
        fm_backbone=args.fm_backbone,
        flow_path="ot",
        num_train_samples=infer_num_train_samples(ckpt),
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
        source_prior_mode=getattr(args, "source_prior_mode", "learned"),
        num_classes=args.num_classes if conditional else 0,
        cross_scale_conditioning=cross_scale_conditioning,
        class_specific_output_head=getattr(args, "class_specific_output_head", False),
        class_specific_adapter=getattr(args, "class_specific_adapter", False),
        label_conditioning_mode=getattr(args, "label_conditioning_mode", "add"),
        token_type_conditioning=getattr(args, "token_type_conditioning", "none"),
    )
    model.load_state_dict(ckpt["flow"], strict=True)
    model.eval().to(device)
    return model


def inverse_minmax(x_norm, x_train):
    mins = np.nanmin(x_train, axis=(0, 1), keepdims=True)
    maxs = np.nanmax(x_train, axis=(0, 1), keepdims=True)
    return x_norm * np.maximum(maxs - mins, 1e-6) + mins


def normalize_like_train(x, x_train):
    mins = np.nanmin(x_train, axis=(0, 1), keepdims=True)
    maxs = np.nanmax(x_train, axis=(0, 1), keepdims=True)
    x_norm = (x - mins) / np.maximum(maxs - mins, 1e-6)
    return np.nan_to_num(x_norm, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def cache_training_tokens_for_ctf(args, vq_model, flow_model, x_source, x_train, device):
    if args.sampling_mode != "ctf_nearest":
        return
    x_norm = normalize_like_train(x_source, x_train)
    chunks = []
    for start in range(0, len(x_norm), args.batch_size):
        batch = torch.from_numpy(x_norm[start:start + args.batch_size]).float().to(device)
        chunks.append(vq_model.encode(batch).detach().cpu().long())
    flow_model.set_training_code_indices(torch.cat(chunks, dim=0))


def cache_training_labels_for_conditional_prior(args, flow_model, y_source):
    if not getattr(args, "label_conditioned_prior", False):
        return
    labels = np.asarray(y_source, dtype=np.int64)
    expected = int(flow_model.scale_models[0].sample_latents.shape[0])
    if expected != len(labels):
        if expected < len(labels):
            raise ValueError(
                f"Flow prior expects {expected} labels, but source labels have {len(labels)}. "
                "The flow checkpoint and data split do not match."
            )
        extra = expected - len(labels)
        mixup_multiplier = float(getattr(args, "positive_mixup_multiplier", 1.0))
        mixup_label = int(getattr(args, "positive_mixup_label", 1))
        if mixup_multiplier <= 1.0:
            raise ValueError(
                f"Flow prior expects {expected} labels, but source labels have {len(labels)}. "
                "If this checkpoint was trained with positive data-level mixup, pass "
                "--positive-mixup-multiplier and --positive-mixup-label during generation."
            )
        n_label = int(np.sum(labels == mixup_label))
        declared_extra = None
        flow_path = getattr(args, "flow", None)
        if flow_path:
            flow_dir = Path(flow_path).resolve().parent
            report_path = flow_dir / "positive_mixup_report.json"
            legacy_cluster_report_path = flow_dir / "positive_mixup_cluster_report.json"
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                declared_extra = int(report.get("num_extra", 0))
                print(f"Loaded actual positive mixup extra labels from {report_path}: {declared_extra}")
            elif legacy_cluster_report_path.exists():
                with open(legacy_cluster_report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                declared_extra = int(report.get("num_extra", 0))
                print(f"Loaded actual positive mixup extra labels from {legacy_cluster_report_path}: {declared_extra}")
        if declared_extra is None:
            declared_extra = int(round(n_label * (mixup_multiplier - 1.0)))
        if declared_extra != extra:
            raise ValueError(
                f"Positive mixup label count mismatch: flow expects {extra} extra labels, "
                f"but checkpoint/report implies {declared_extra} extra label={mixup_label} samples "
                f"(multiplier={mixup_multiplier}, base_count={n_label})."
            )
        labels = np.concatenate([labels, np.full(extra, mixup_label, dtype=np.int64)], axis=0)
        print(
            f"Extended conditional-prior labels for positive mixup: "
            f"base={len(y_source)}, extra={extra}, label={mixup_label}, total={len(labels)}"
        )
    flow_model.set_training_labels(torch.from_numpy(labels).long())


@torch.no_grad()
def cache_token_manifold_guidance(args, vq_model, flow_model, x_train, y_train, device):
    weight = float(getattr(args, "token_manifold_guidance_weight", 0.0))
    transition_weight = float(getattr(args, "token_manifold_transition_weight", 0.0))
    cross_scale_weight = float(getattr(args, "token_manifold_cross_scale_weight", 0.0))
    if weight == 0.0 and transition_weight == 0.0 and cross_scale_weight == 0.0:
        flow_model.set_token_manifold_biases(None)
        if hasattr(flow_model, "set_token_transition_biases"):
            flow_model.set_token_transition_biases(None)
        if hasattr(flow_model, "set_token_cross_scale_biases"):
            flow_model.set_token_cross_scale_biases(None)
        return
    target_label = int(getattr(args, "token_manifold_guidance_target_label", 1))
    mode = getattr(args, "token_manifold_guidance_mode", "log_odds")
    background = getattr(args, "token_manifold_guidance_background", "all")
    smoothing = float(getattr(args, "token_manifold_guidance_smoothing", 1.0))
    clamp = float(getattr(args, "token_manifold_guidance_clamp", 3.0))
    labels_mode = getattr(args, "token_manifold_guidance_labels", "target")
    if smoothing <= 0:
        raise ValueError("--token-manifold-guidance-smoothing must be > 0")

    y_train = np.asarray(y_train, dtype=np.int64)
    if labels_mode == "target":
        label_values = [target_label]
    elif labels_mode == "all":
        label_values = list(range(int(getattr(args, "num_classes", 2))))
    else:
        raise ValueError(f"Unsupported token manifold guidance labels mode: {labels_mode}")
    if not np.any(y_train == target_label):
        raise ValueError(f"No training samples found for token manifold target label={target_label}")

    def label_masks(label_value):
        target_mask = y_train == int(label_value)
        if not np.any(target_mask):
            return target_mask, None
        if background == "all":
            background_mask = np.ones_like(target_mask, dtype=bool)
        elif background == "not_target":
            background_mask = ~target_mask
            if not np.any(background_mask):
                raise ValueError("No non-target samples found for token manifold background")
        else:
            raise ValueError(f"Unsupported token manifold guidance background: {background}")
        return target_mask, background_mask

    def center_clip_unigram(bias):
        bias = bias - bias.mean()
        if clamp > 0:
            bias = np.clip(bias, -clamp, clamp)
        return bias.astype(np.float32)

    def center_clip_rows(bias):
        bias = bias - bias.mean(axis=-1, keepdims=True)
        if clamp > 0:
            bias = np.clip(bias, -clamp, clamp)
        return bias.astype(np.float32)

    def unigram_bias(local_idx, mask, background_mask, num_codes):
        flat_target = local_idx[mask].reshape(-1).numpy()
        target_counts = np.bincount(flat_target, minlength=num_codes).astype(np.float64) + smoothing
        target_probs = target_counts / target_counts.sum()
        if mode == "target_prior":
            bias = np.log(target_probs)
        elif mode == "log_odds":
            flat_background = local_idx[background_mask].reshape(-1).numpy()
            background_counts = np.bincount(flat_background, minlength=num_codes).astype(np.float64) + smoothing
            background_probs = background_counts / background_counts.sum()
            bias = np.log(target_probs) - np.log(background_probs)
        else:
            raise ValueError(f"Unsupported token manifold guidance mode: {mode}")
        return center_clip_unigram(bias)

    def transition_bias(local_idx, mask, background_mask, num_codes):
        if local_idx.shape[1] < 2:
            return np.zeros((num_codes, num_codes), dtype=np.float32)
        target_pairs = local_idx[mask].numpy()
        target_prev = target_pairs[:, :-1].reshape(-1)
        target_next = target_pairs[:, 1:].reshape(-1)
        target_counts = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
        np.add.at(target_counts, (target_prev, target_next), 1.0)
        target_probs = target_counts / target_counts.sum(axis=1, keepdims=True)
        if mode == "target_prior":
            bias = np.log(target_probs)
        elif mode == "log_odds":
            bg_pairs = local_idx[background_mask].numpy()
            bg_prev = bg_pairs[:, :-1].reshape(-1)
            bg_next = bg_pairs[:, 1:].reshape(-1)
            bg_counts = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
            np.add.at(bg_counts, (bg_prev, bg_next), 1.0)
            bg_probs = bg_counts / bg_counts.sum(axis=1, keepdims=True)
            bias = np.log(target_probs) - np.log(bg_probs)
        else:
            raise ValueError(f"Unsupported token manifold guidance mode: {mode}")
        return center_clip_rows(bias)

    def cross_bias(parent_idx, child_idx, mask, background_mask, parent_codes, child_codes):
        parent_len = parent_idx.shape[1]
        child_len = child_idx.shape[1]

        def accumulate(sample_mask):
            counts = np.full((parent_codes, child_codes), smoothing, dtype=np.float64)
            parent_np = parent_idx[sample_mask].numpy()
            child_np = child_idx[sample_mask].numpy()
            for child_pos in range(child_len):
                parent_pos = min((child_pos * parent_len) // child_len, parent_len - 1)
                np.add.at(counts, (parent_np[:, parent_pos], child_np[:, child_pos]), 1.0)
            return counts

        target_counts = accumulate(mask)
        target_probs = target_counts / target_counts.sum(axis=1, keepdims=True)
        if mode == "target_prior":
            bias = np.log(target_probs)
        elif mode == "log_odds":
            bg_counts = accumulate(background_mask)
            bg_probs = bg_counts / bg_counts.sum(axis=1, keepdims=True)
            bias = np.log(target_probs) - np.log(bg_probs)
        else:
            raise ValueError(f"Unsupported token manifold guidance mode: {mode}")
        return center_clip_rows(bias)

    x_norm = normalize_like_train(x_train, x_train)
    chunks = []
    for start in range(0, len(x_norm), args.batch_size):
        batch = torch.from_numpy(x_norm[start:start + args.batch_size]).float().to(device)
        chunks.append(vq_model.encode(batch).detach().cpu().long())
    global_tokens = torch.cat(chunks, dim=0)
    local_tokens = flow_model.split_global_indices(global_tokens)

    biases = []
    transition_biases = []
    cross_scale_biases = [None]
    for scale_idx, local_idx in enumerate(local_tokens):
        num_codes = int(flow_model.nb_code[scale_idx])
        if labels_mode == "target":
            target_mask, background_mask = label_masks(target_label)
            bias = unigram_bias(local_idx, target_mask, background_mask, num_codes)
            trans = transition_bias(local_idx, target_mask, background_mask, num_codes)
        else:
            class_biases = []
            class_transitions = []
            for label_value in label_values:
                target_mask, background_mask = label_masks(label_value)
                if background_mask is None:
                    class_biases.append(np.zeros((num_codes,), dtype=np.float32))
                    class_transitions.append(np.zeros((num_codes, num_codes), dtype=np.float32))
                else:
                    class_biases.append(unigram_bias(local_idx, target_mask, background_mask, num_codes))
                    class_transitions.append(transition_bias(local_idx, target_mask, background_mask, num_codes))
            bias = np.stack(class_biases, axis=0)
            trans = np.stack(class_transitions, axis=0)
        biases.append(torch.from_numpy(bias.astype(np.float32)))
        transition_biases.append(torch.from_numpy(trans.astype(np.float32)))

    for scale_idx in range(1, len(local_tokens)):
        parent_idx = local_tokens[scale_idx - 1]
        child_idx = local_tokens[scale_idx]
        parent_codes = int(flow_model.nb_code[scale_idx - 1])
        child_codes = int(flow_model.nb_code[scale_idx])
        if labels_mode == "target":
            target_mask, background_mask = label_masks(target_label)
            cross = cross_bias(parent_idx, child_idx, target_mask, background_mask, parent_codes, child_codes)
        else:
            class_cross = []
            for label_value in label_values:
                target_mask, background_mask = label_masks(label_value)
                if background_mask is None:
                    class_cross.append(np.zeros((parent_codes, child_codes), dtype=np.float32))
                else:
                    class_cross.append(cross_bias(parent_idx, child_idx, target_mask, background_mask, parent_codes, child_codes))
            cross = np.stack(class_cross, axis=0)
        cross_scale_biases.append(torch.from_numpy(cross.astype(np.float32)))

    flow_model.set_token_manifold_biases(biases)
    flow_model.set_token_transition_biases(transition_biases if transition_weight != 0.0 else None)
    flow_model.set_token_cross_scale_biases(cross_scale_biases if cross_scale_weight != 0.0 else None)
    print(
        "Token manifold guidance enabled: "
        f"weight={weight}, transition_weight={transition_weight}, cross_scale_weight={cross_scale_weight}, "
        f"target_label={target_label}, labels={labels_mode}, mode={mode}, "
        f"background={background}, smoothing={smoothing}, clamp={clamp}, "
        f"schedule={getattr(args, 'token_manifold_guidance_schedule', 'late_linear')}, "
        f"warmup_frac={getattr(args, 'token_manifold_guidance_warmup_frac', 0.35)}"
    )


@torch.no_grad()
def _encode_tokens_np(args, vq_model, x_norm, device):
    chunks = []
    for start in range(0, len(x_norm), args.batch_size):
        batch = torch.from_numpy(x_norm[start:start + args.batch_size]).float().to(device)
        chunks.append(vq_model.encode(batch).detach().cpu().long())
    return torch.cat(chunks, dim=0)


def cache_utility_token_guidance(args, vq_model, flow_model, x_train, y_train, x_val, y_val, device):
    weight = float(getattr(args, "utility_token_guidance_weight", 0.0))
    transition_weight = float(getattr(args, "utility_token_transition_weight", 0.0))
    cross_scale_weight = float(getattr(args, "utility_token_cross_scale_weight", 0.0))
    guidance_mode = getattr(args, "utility_token_guidance_mode", "unigram")
    if guidance_mode not in {"unigram", "transition", "cross_scale", "all"}:
        raise ValueError(f"Unsupported utility token guidance mode: {guidance_mode}")
    if weight == 0.0 and transition_weight == 0.0 and cross_scale_weight == 0.0:
        if hasattr(flow_model, "set_utility_token_biases"):
            flow_model.set_utility_token_biases(None)
        if hasattr(flow_model, "set_utility_token_transition_biases"):
            flow_model.set_utility_token_transition_biases(None)
        if hasattr(flow_model, "set_utility_token_cross_scale_biases"):
            flow_model.set_utility_token_cross_scale_biases(None)
        return
    if not hasattr(flow_model, "set_utility_token_biases"):
        raise AttributeError("flow_model does not support utility token biases")
    if transition_weight != 0.0 and not hasattr(flow_model, "set_utility_token_transition_biases"):
        raise AttributeError("flow_model does not support utility token transition biases")
    if cross_scale_weight != 0.0 and not hasattr(flow_model, "set_utility_token_cross_scale_biases"):
        raise AttributeError("flow_model does not support utility token cross-scale biases")

    target_label = int(getattr(args, "utility_token_guidance_target_label", 1))
    background_label = int(getattr(args, "utility_token_guidance_background_label", 0))
    top_frac = float(getattr(args, "utility_token_guidance_top_frac", 0.5))
    smoothing = float(getattr(args, "utility_token_guidance_smoothing", 1.0))
    clamp = float(getattr(args, "utility_token_guidance_clamp", 3.0))
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("--utility-token-guidance-top-frac must be in (0, 1]")
    if smoothing <= 0:
        raise ValueError("--utility-token-guidance-smoothing must be > 0")

    y_train = np.asarray(y_train, dtype=np.int64)
    target_mask = y_train == target_label
    background_mask = y_train == background_label
    if not np.any(target_mask):
        raise ValueError(f"No training samples found for utility target label={target_label}")
    if not np.any(background_mask):
        raise ValueError(f"No training samples found for utility background label={background_label}")

    util_args = SimpleNamespace(**vars(args))
    util_args.posthoc_influence_weight = float(getattr(args, "utility_token_guidance_influence_weight", 0.2))
    util_args.posthoc_influence_target_label = target_label
    util_args.posthoc_influence_background_label = background_label
    util_args.posthoc_influence_background_weight = float(getattr(args, "utility_token_guidance_background_weight", 0.25))
    util_args.posthoc_influence_target_split = "val"
    util_args.posthoc_influence_normalize = "cosine"
    util_args.posthoc_influence_max_target = int(getattr(args, "utility_token_guidance_max_target", 512))
    util_args.posthoc_influence_candidate_batch_size = int(getattr(args, "utility_token_guidance_candidate_batch_size", 64))
    util_args.posthoc_classifier_ckpt = getattr(args, "posthoc_classifier_ckpt", getattr(args, "classifier_ckpt", "runs/selector/best_model.pt"))
    util_args.posthoc_classifier_hidden_dim = int(getattr(args, "posthoc_classifier_hidden_dim", getattr(args, "classifier_hidden_dim", 256)))
    util_args.posthoc_classifier_num_layers = int(getattr(args, "posthoc_classifier_num_layers", getattr(args, "classifier_num_layers", 2)))
    util_args.posthoc_classifier_rnn_type = getattr(args, "posthoc_classifier_rnn_type", getattr(args, "classifier_rnn_type", "gru"))
    util_args.posthoc_classifier_dropout = float(getattr(args, "posthoc_classifier_dropout", getattr(args, "classifier_dropout", 0.2)))

    target_real = x_train[target_mask].astype(np.float32)
    target_labels = np.full(len(target_real), target_label, dtype=np.int64)
    utility_scores, target_scores, background_scores = _posthoc_influence_utility_scores(
        util_args,
        target_real,
        target_labels,
        x_train,
        y_train,
        x_val,
        y_val,
        target_label,
        device,
    )
    keep_n = max(1, int(round(len(target_real) * top_frac)))
    keep_idx = np.argsort(-utility_scores)[:keep_n]

    x_norm = normalize_like_train(x_train, x_train)
    global_tokens = _encode_tokens_np(args, vq_model, x_norm, device)
    local_tokens = flow_model.split_global_indices(global_tokens)
    target_global_idx = np.nonzero(target_mask)[0][keep_idx]
    background_global_idx = np.nonzero(background_mask)[0]

    def center_clip_vector(bias):
        bias = bias - bias.mean()
        if clamp > 0:
            bias = np.clip(bias, -clamp, clamp)
        return bias.astype(np.float32)

    def center_clip_rows(bias):
        bias = bias - bias.mean(axis=-1, keepdims=True)
        if clamp > 0:
            bias = np.clip(bias, -clamp, clamp)
        return bias.astype(np.float32)

    def transition_bias(local_np, num_codes):
        if local_np.shape[1] < 2:
            return np.zeros((num_codes, num_codes), dtype=np.float32)
        target_pairs = local_np[target_global_idx]
        background_pairs = local_np[background_global_idx]
        target_counts = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
        background_counts = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
        np.add.at(target_counts, (target_pairs[:, :-1].reshape(-1), target_pairs[:, 1:].reshape(-1)), 1.0)
        np.add.at(background_counts, (background_pairs[:, :-1].reshape(-1), background_pairs[:, 1:].reshape(-1)), 1.0)
        target_probs = target_counts / target_counts.sum(axis=1, keepdims=True)
        background_probs = background_counts / background_counts.sum(axis=1, keepdims=True)
        return center_clip_rows(np.log(target_probs) - np.log(background_probs))

    def cross_scale_bias(parent_np, child_np, parent_codes, child_codes):
        parent_len = parent_np.shape[1]
        child_len = child_np.shape[1]

        def accumulate(sample_idx):
            counts = np.full((parent_codes, child_codes), smoothing, dtype=np.float64)
            parent_sel = parent_np[sample_idx]
            child_sel = child_np[sample_idx]
            for child_pos in range(child_len):
                parent_pos = min((child_pos * parent_len) // child_len, parent_len - 1)
                np.add.at(counts, (parent_sel[:, parent_pos], child_sel[:, child_pos]), 1.0)
            return counts

        target_counts = accumulate(target_global_idx)
        background_counts = accumulate(background_global_idx)
        target_probs = target_counts / target_counts.sum(axis=1, keepdims=True)
        background_probs = background_counts / background_counts.sum(axis=1, keepdims=True)
        return center_clip_rows(np.log(target_probs) - np.log(background_probs))

    biases = []
    transition_biases = []
    for scale_idx, local_idx in enumerate(local_tokens):
        num_codes = int(flow_model.nb_code[scale_idx])
        local_np = local_idx.numpy()
        target_flat = local_np[target_global_idx].reshape(-1)
        background_flat = local_np[background_global_idx].reshape(-1)
        target_counts = np.bincount(target_flat, minlength=num_codes).astype(np.float64) + smoothing
        background_counts = np.bincount(background_flat, minlength=num_codes).astype(np.float64) + smoothing
        bias = np.log(target_counts / target_counts.sum()) - np.log(background_counts / background_counts.sum())
        biases.append(torch.from_numpy(center_clip_vector(bias)))
        transition_biases.append(torch.from_numpy(transition_bias(local_np, num_codes)))

    cross_scale_biases = [None]
    for scale_idx in range(1, len(local_tokens)):
        parent_np = local_tokens[scale_idx - 1].numpy()
        child_np = local_tokens[scale_idx].numpy()
        cross_scale_biases.append(
            torch.from_numpy(
                cross_scale_bias(
                    parent_np,
                    child_np,
                    int(flow_model.nb_code[scale_idx - 1]),
                    int(flow_model.nb_code[scale_idx]),
                )
            )
        )

    enable_unigram = guidance_mode in {"unigram", "all"} and weight != 0.0
    enable_transition = guidance_mode in {"transition", "all"} and transition_weight != 0.0
    enable_cross_scale = guidance_mode in {"cross_scale", "all"} and cross_scale_weight != 0.0
    flow_model.set_utility_token_biases(biases if enable_unigram else None)
    if hasattr(flow_model, "set_utility_token_transition_biases"):
        flow_model.set_utility_token_transition_biases(transition_biases if enable_transition else None)
    if hasattr(flow_model, "set_utility_token_cross_scale_biases"):
        flow_model.set_utility_token_cross_scale_biases(cross_scale_biases if enable_cross_scale else None)
    print(
        "Utility token guidance enabled: "
        f"mode={guidance_mode}, weight={weight}, transition_weight={transition_weight}, "
        f"cross_scale_weight={cross_scale_weight}, target_label={target_label}, background_label={background_label}, "
        f"top_frac={top_frac}, selected={keep_n}/{len(target_real)}, "
        f"influence_weight={util_args.posthoc_influence_weight}, "
        f"background_weight={util_args.posthoc_influence_background_weight}, "
        f"utility_mean={float(np.mean(utility_scores)):.4f}, "
        f"utility_selected_mean={float(np.mean(utility_scores[keep_idx])):.4f}, "
        f"target_selected_mean={float(np.mean(target_scores[keep_idx])):.4f}, "
        f"background_selected_mean={float(np.mean(background_scores[keep_idx])):.4f}"
    )


class InfluenceRefiner:
    def __init__(self, args, x_train, y_train, x_guide, y_guide, device):
        self.device = device
        self.weight = float(args.guidance_weight)
        self.steps = int(args.guidance_steps)
        self.refine_batch_size = int(args.guidance_batch_size)
        self.guidance_class = args.guidance_class
        self.guidance_refine_class = args.guidance_refine_class
        self.model = RNNClassifier(
            input_dim=int(args.input_dim),
            hidden_dim=args.guidance_hidden_dim,
            num_layers=args.guidance_num_layers,
            rnn_type=args.guidance_rnn_type,
            num_classes=1,
            dropout=args.guidance_dropout,
        ).to(device)
        ckpt = torch.load(args.guidance_ckpt, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()
        self.criterion = nn.BCEWithLogitsLoss()
        train_set = TimeSeriesDataset(x_train, y_train)
        if self.guidance_class == "positive":
            keep = y_guide == 1
            x_guide = x_guide[keep]
            y_guide = y_guide[keep]
        elif self.guidance_class == "negative":
            keep = y_guide == 0
            x_guide = x_guide[keep]
            y_guide = y_guide[keep]
        if len(y_guide) == 0:
            raise ValueError(f"No guidance samples left after --guidance-class {self.guidance_class}")
        guide_set = TimeSeriesDataset(x_guide, y_guide, stats=train_set.stats)
        self.mean, self.std = [t.to(device) for t in train_set.stats]
        guide_loader = DataLoader(guide_set, batch_size=args.guidance_batch_size, shuffle=False)
        self.cached_grads = self._compute_guidance_grads(guide_loader)

    def _normalize_grads(self, grads):
        total = torch.sqrt(sum((g ** 2).sum() for g in grads))
        return [g / (total + 1e-6) for g in grads]

    def _compute_guidance_grads(self, loader):
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_loss = torch.tensor(0.0, device=self.device)
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device).float()
            with torch.backends.cudnn.flags(enabled=False):
                logits = self.model(x)
            total_loss = total_loss + self.criterion(logits, y)
        grads = torch.autograd.grad(total_loss, params, allow_unused=True)
        grads = [g for g in grads if g is not None]
        return self._normalize_grads(grads)

    def _refine_batch(self, x_ntc, y):
        if self.weight == 0 or self.steps <= 0:
            return x_ntc
        x = torch.from_numpy(x_ntc).float().to(self.device)
        labels = torch.from_numpy(y).float().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(self.steps):
            x = x.detach().requires_grad_(True)
            x_norm = (x - self.mean) / self.std
            with torch.backends.cudnn.flags(enabled=False):
                logits = self.model(x_norm)
            loss = self.criterion(logits, labels)
            test_grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            test_grads = [g for g in test_grads if g is not None]
            test_grads = self._normalize_grads(test_grads)
            dot = sum((a * b).sum() for a, b in zip(test_grads, self.cached_grads))
            grad_x = torch.autograd.grad(dot, x)[0]
            x = x + self.weight * grad_x
        return x.detach().cpu().numpy().astype(np.float32)

    def refine(self, x_ntc, y):
        if self.weight == 0 or self.steps <= 0:
            return x_ntc
        if self.guidance_refine_class == "positive":
            mask = y == 1
        elif self.guidance_refine_class == "negative":
            mask = y == 0
        else:
            mask = np.ones_like(y, dtype=bool)
        if not mask.any():
            return x_ntc
        refined = x_ntc.copy()
        x_target = x_ntc[mask]
        y_target = y[mask]
        outputs = []
        for start in range(0, len(y_target), self.refine_batch_size):
            end = min(start + self.refine_batch_size, len(y_target))
            outputs.append(self._refine_batch(x_target[start:end], y_target[start:end]))
        refined[mask] = np.concatenate(outputs, axis=0)
        return refined


def _classifier_model_from_args(args, device):
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.posthoc_classifier_hidden_dim,
        num_layers=args.posthoc_classifier_num_layers,
        rnn_type=args.posthoc_classifier_rnn_type,
        num_classes=1,
        dropout=args.posthoc_classifier_dropout,
    ).to(device)
    ckpt = torch.load(args.posthoc_classifier_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def _feature_quantile_penalty_torch(x, low, high):
    width = (high - low).clamp_min(1e-6)
    low_violation = torch.relu(low.view(1, 1, -1) - x) / width.view(1, 1, -1)
    high_violation = torch.relu(x - high.view(1, 1, -1)) / width.view(1, 1, -1)
    return (low_violation + high_violation).mean()


def _continuous_utility_refine_x_norm(
    args,
    x_norm,
    label,
    x_train,
    y_train,
    device,
    *,
    enabled,
    labels_mode,
    steps,
    weight,
    anchor_weight,
    feature_penalty_weight,
    clamp,
    report_suffix,
):
    if not enabled or int(steps) <= 0 or float(weight) == 0.0:
        return x_norm
    if labels_mode == "target" and int(label) != int(args.posthoc_selector_target_label):
        return x_norm
    if labels_mode not in {"target", "all"}:
        raise ValueError(f"Unsupported utility refine labels mode: {labels_mode}")

    model = _classifier_model_from_args(args, device)
    train_set = TimeSeriesDataset(x_train, y_train)
    mean, std = [t.to(device) for t in train_set.stats]
    mins = torch.from_numpy(np.nanmin(x_train, axis=(0, 1), keepdims=True).astype(np.float32)).to(device)
    maxs = torch.from_numpy(np.nanmax(x_train, axis=(0, 1), keepdims=True).astype(np.float32)).to(device)
    scale = (maxs - mins).clamp_min(1e-6)
    target_real = x_train[y_train == int(label)].astype(np.float32)
    q = min(max(float(args.posthoc_feature_quantile), 0.0), 0.2)
    low = torch.from_numpy(np.quantile(target_real, q, axis=(0, 1)).astype(np.float32)).to(device)
    high = torch.from_numpy(np.quantile(target_real, 1.0 - q, axis=(0, 1)).astype(np.float32)).to(device)

    labels = torch.full((len(x_norm),), int(label), device=device, dtype=torch.float32)
    before_probs = []
    after_probs = []
    anchor_dist = []
    feature_penalties = []
    outputs = []
    batch_size = int(getattr(args, "latent_utility_refine_batch_size", args.batch_size))
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    for start in range(0, len(x_norm), batch_size):
        end = min(start + batch_size, len(x_norm))
        x0_norm = torch.from_numpy(x_norm[start:end]).float().to(device)
        x0 = x0_norm * scale + mins
        x = x0.detach().clone()
        y = labels[start:end]
        with torch.no_grad():
            logits0 = model((x - mean) / std).squeeze(-1)
            before_probs.append(torch.sigmoid(logits0).detach().cpu())
        for _ in range(int(steps)):
            x = x.detach().requires_grad_(True)
            # cuDNN RNN backward requires train mode, but the utility classifier
            # should stay frozen/eval. Disable cuDNN only for this gradient path.
            with torch.backends.cudnn.flags(enabled=False):
                logits = model((x - mean) / std).squeeze(-1)
            cls_loss = criterion(logits, y)
            anchor = ((x - x0) / scale).pow(2).mean()
            feature_penalty = _feature_quantile_penalty_torch(x, low, high)
            loss = cls_loss + float(anchor_weight) * anchor + float(feature_penalty_weight) * feature_penalty
            grad = torch.autograd.grad(loss, x)[0]
            grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
            update = -float(weight) * grad / grad_norm
            if clamp > 0:
                update = update.clamp(min=-clamp, max=clamp)
            x = x + update * scale
            x = torch.maximum(torch.minimum(x, maxs), mins)
        with torch.no_grad():
            logits1 = model((x - mean) / std).squeeze(-1)
            after_probs.append(torch.sigmoid(logits1).detach().cpu())
            anchor_dist.append((((x - x0) / scale).pow(2).mean(dim=(1, 2)).sqrt()).detach().cpu())
            feature_penalties.append(_feature_quantile_penalty_torch(x, low, high).detach().cpu().view(1))
            outputs.append(((x - mins) / scale).detach().cpu().numpy().astype(np.float32))

    refined = np.concatenate(outputs, axis=0)
    before = torch.cat(before_probs).numpy()
    after = torch.cat(after_probs).numpy()
    anchor = torch.cat(anchor_dist).numpy()
    fpen = torch.cat(feature_penalties).numpy()
    report_path = getattr(args, "_posthoc_report_path", None)
    if report_path is not None:
        path = Path(report_path)
        path = path.parent / f"{path.stem}_{report_suffix}_label{int(label)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": int(label),
                    "steps": int(steps),
                    "weight": float(weight),
                    "prob_before_mean": float(np.mean(before)),
                    "prob_after_mean": float(np.mean(after)),
                    "anchor_distance_mean": float(np.mean(anchor)),
                    "feature_penalty_mean": float(np.mean(fpen)),
                },
                f,
                indent=2,
            )
    print(
        f"Continuous utility refine label={label}: n={len(x_norm)}, steps={int(steps)}, "
        f"weight={float(weight):.4f}, prob_before={float(np.mean(before)):.4f}, "
        f"prob_after={float(np.mean(after)):.4f}, anchor={float(np.mean(anchor)):.4f}"
    )
    return refined


@torch.no_grad()
def sample_model(args, vq_model, flow_model, n, label, device):
    outputs = []
    left = int(n)
    while left > 0:
        bs = min(args.batch_size, left)
        labels = None
        if flow_model.num_classes > 0:
            labels = torch.full((bs,), int(label), device=device, dtype=torch.long)
        token_idx = flow_model.sample_token_indices(
            batch_size=bs,
            quantizers=vq_model.vqvae.quantizer,
            flow_steps=args.flow_steps,
            solver=args.solver,
            sample_temperature=args.sample_temperature,
            noise_scale=args.noise_scale,
            senior_sampler=args.senior_sampler,
            kde_bandwidth_factor=args.kde_bandwidth_factor,
            kde_max_centers=args.kde_max_centers,
            device=device,
            dtype=torch.float32,
            labels=labels,
            label_guidance_scale=args.label_guidance_scale,
            token_manifold_guidance_weight=args.token_manifold_guidance_weight,
            token_manifold_guidance_target_label=args.token_manifold_guidance_target_label,
            token_manifold_transition_weight=args.token_manifold_transition_weight,
            token_manifold_cross_scale_weight=args.token_manifold_cross_scale_weight,
            utility_token_guidance_weight=args.utility_token_guidance_weight,
            utility_token_guidance_target_label=args.utility_token_guidance_target_label,
            utility_token_transition_weight=args.utility_token_transition_weight,
            utility_token_cross_scale_weight=args.utility_token_cross_scale_weight,
            token_manifold_guidance_schedule=args.token_manifold_guidance_schedule,
            token_manifold_guidance_warmup_frac=args.token_manifold_guidance_warmup_frac,
            sampling_mode=args.sampling_mode,
            label_conditioned_prior=args.label_conditioned_prior,
        )
        x = vq_model.forward_decoder(token_idx).detach().cpu().numpy()
        outputs.append(x)
        left -= bs
    return np.concatenate(outputs, axis=0)


def target_counts(args, y_source):
    counts = Counter(y_source.tolist())
    num_classes = int(getattr(args, "num_classes", 2))
    labels = list(range(num_classes)) if num_classes > 0 else sorted(counts)
    if num_classes > 2:
        # Multi-class datasets preserve every real class count. The binary
        # positive_multiplier is deliberately a MIMIC-style opt-in mechanism.
        if args.num_samples is None:
            return {label: int(counts.get(label, 0)) for label in labels}
        total = int(args.num_samples)
        total_source = max(sum(counts.get(label, 0) for label in labels), 1)
        raw = np.asarray([total * counts.get(label, 0) / total_source for label in labels], dtype=np.float64)
        allocated = np.floor(raw).astype(np.int64)
        remainder = total - int(allocated.sum())
        if remainder > 0:
            order = np.argsort(-(raw - allocated), kind="stable")
            for idx in order[:remainder]:
                allocated[idx] += 1
        return {label: int(allocated[idx]) for idx, label in enumerate(labels)}
    if args.num_samples is None:
        return {
            0: counts.get(0, 0),
            1: int(round(counts.get(1, 0) * float(args.positive_multiplier))),
        }
    total = int(args.num_samples)
    frac1 = counts.get(1, 0) / max(sum(counts.values()), 1)
    n1 = int(round(total * frac1))
    return {0: total - n1, 1: n1}


@torch.no_grad()
def _posthoc_classifier_probs(args, x_real, labels, x_train, y_train, device):
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.posthoc_classifier_hidden_dim,
        num_layers=args.posthoc_classifier_num_layers,
        rnn_type=args.posthoc_classifier_rnn_type,
        num_classes=1,
        dropout=args.posthoc_classifier_dropout,
    ).to(device)
    ckpt = torch.load(args.posthoc_classifier_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    train_set = TimeSeriesDataset(x_train, y_train)
    dataset = TimeSeriesDataset(x_real, labels, stats=train_set.stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    probs = []
    for batch_x, _ in loader:
        logits = model(batch_x.to(device))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def _posthoc_classifier_head_grad_embeddings(args, x_real, labels, x_train, y_train, device, batch_size):
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.posthoc_classifier_hidden_dim,
        num_layers=args.posthoc_classifier_num_layers,
        rnn_type=args.posthoc_classifier_rnn_type,
        num_classes=1,
        dropout=args.posthoc_classifier_dropout,
    ).to(device)
    ckpt = torch.load(args.posthoc_classifier_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    train_set = TimeSeriesDataset(x_train, y_train)
    dataset = TimeSeriesDataset(x_real, labels, stats=train_set.stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    parts = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)
        rnn_out, _ = model.rnn(batch_x)
        hidden = rnn_out[:, -1, :]
        logits = model.fc(hidden).squeeze(-1)
        residual = torch.sigmoid(logits) - batch_y
        weight_grad = residual.unsqueeze(1) * hidden
        grad_emb = torch.cat([weight_grad, residual.unsqueeze(1)], dim=1)
        parts.append(grad_emb.detach().cpu())
    return torch.cat(parts, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def _multiclass_posthoc_classifier_probs(args, x_real, labels, x_train, y_train, device):
    """Return p(class | x) for the frozen multi-class utility classifier."""
    num_classes = int(args.num_classes)
    if num_classes <= 2:
        raise ValueError("multiclass selector requires --num-classes > 2")
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.posthoc_classifier_hidden_dim,
        num_layers=args.posthoc_classifier_num_layers,
        rnn_type=args.posthoc_classifier_rnn_type,
        num_classes=num_classes,
        dropout=args.posthoc_classifier_dropout,
    ).to(device)
    ckpt = torch.load(args.posthoc_classifier_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    train_set = TimeSeriesDataset(x_train, y_train)
    dataset = TimeSeriesDataset(x_real, labels, stats=train_set.stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    probs = []
    for batch_x, _ in loader:
        logits = model(batch_x.to(device))
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


@torch.no_grad()
def _multiclass_classifier_head_grad_embeddings(args, x_real, labels, x_train, y_train, device, batch_size):
    """Analytic CE gradients of the classifier head for every candidate.

    For CE, dL/dlogits = softmax(logits) - one_hot(label). Concatenating the
    linear-head weight and bias gradients gives a compact, class-aware utility
    embedding without retaining a full recurrent computation graph.
    """
    num_classes = int(args.num_classes)
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.posthoc_classifier_hidden_dim,
        num_layers=args.posthoc_classifier_num_layers,
        rnn_type=args.posthoc_classifier_rnn_type,
        num_classes=num_classes,
        dropout=args.posthoc_classifier_dropout,
    ).to(device)
    ckpt = torch.load(args.posthoc_classifier_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    train_set = TimeSeriesDataset(x_train, y_train)
    dataset = TimeSeriesDataset(x_real, labels, stats=train_set.stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    parts = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)
        rnn_out, _ = model.rnn(batch_x)
        hidden = rnn_out[:, -1, :]
        logits = model.fc(hidden)
        residual = torch.softmax(logits, dim=1) - F.one_hot(batch_y, num_classes=num_classes).float()
        weight_grad = residual.unsqueeze(2) * hidden.unsqueeze(1)
        grad_emb = torch.cat([weight_grad.flatten(start_dim=1), residual], dim=1)
        parts.append(grad_emb.detach().cpu())
    return torch.cat(parts, dim=0).numpy().astype(np.float32)


def _multiclass_influence_scores(args, x_real, labels, x_train, y_train, x_val, y_val, label, device):
    if x_val is None or y_val is None:
        raise ValueError("multiclass_influence_utility requires validation data")
    target_real, target_labels = _select_influence_target_samples(
        args, x_train, y_train, x_val, y_val, int(label)
    )
    target_grads = _multiclass_classifier_head_grad_embeddings(
        args,
        target_real,
        target_labels,
        x_train,
        y_train,
        device,
        int(args.posthoc_influence_candidate_batch_size),
    )
    candidate_grads = _multiclass_classifier_head_grad_embeddings(
        args,
        x_real,
        labels,
        x_train,
        y_train,
        device,
        int(args.posthoc_influence_candidate_batch_size),
    )
    target_grad = target_grads.mean(axis=0, keepdims=True)
    return _gradient_similarity(candidate_grads, target_grad, args.posthoc_influence_normalize)


def _select_influence_target_samples(args, x_train, y_train, x_val, y_val, target_label):
    split = args.posthoc_influence_target_split
    if split == "val":
        target_x, target_y = x_val, y_val
    elif split == "train":
        target_x, target_y = x_train, y_train
    else:
        raise ValueError(f"Unsupported --posthoc-influence-target-split: {split}")

    target_label = int(target_label)
    target_real = target_x[target_y == target_label]
    if len(target_real) == 0:
        raise ValueError(f"No target samples found for influence target label={target_label} in {split}")
    max_target = int(args.posthoc_influence_max_target)
    if max_target > 0 and len(target_real) > max_target:
        rng = np.random.default_rng(42 + target_label)
        idx = rng.choice(len(target_real), size=max_target, replace=False)
        target_real = target_real[idx]
    target_labels = np.full(len(target_real), target_label, dtype=np.int64)
    return target_real.astype(np.float32), target_labels


def _gradient_similarity(cand_grads, target_grad, normalize):
    raw = (cand_grads * target_grad).sum(axis=1)
    if normalize == "cosine":
        denom = (
            np.linalg.norm(cand_grads, axis=1)
            * max(float(np.linalg.norm(target_grad)), 1e-12)
        )
        raw = raw / np.maximum(denom, 1e-12)
    return raw.astype(np.float32)


def _posthoc_influence_utility_scores(args, x_real, labels, x_train, y_train, x_val, y_val, label, device):
    if float(args.posthoc_influence_weight) == 0.0:
        zeros = np.zeros(len(x_real), dtype=np.float32)
        return zeros, zeros, zeros
    if x_val is None or y_val is None:
        raise ValueError("influence_utility selector requires validation data for target gradients")

    target_label = int(args.posthoc_influence_target_label)
    batch_size = int(args.posthoc_influence_candidate_batch_size)
    target_real, target_labels = _select_influence_target_samples(
        args, x_train, y_train, x_val, y_val, target_label
    )
    target_grads = _posthoc_classifier_head_grad_embeddings(
        args,
        target_real,
        target_labels,
        x_train,
        y_train,
        device,
        batch_size,
    )
    target_grad = target_grads.mean(axis=0, keepdims=True)
    cand_grads = _posthoc_classifier_head_grad_embeddings(
        args,
        x_real.astype(np.float32),
        labels,
        x_train,
        y_train,
        device,
        batch_size,
    )
    target_score = _gradient_similarity(cand_grads, target_grad, args.posthoc_influence_normalize)

    background_score = np.zeros_like(target_score, dtype=np.float32)
    background_weight = float(getattr(args, "posthoc_influence_background_weight", 0.0))
    if background_weight != 0.0:
        background_label = int(getattr(args, "posthoc_influence_background_label", 0))
        background_real, background_labels = _select_influence_target_samples(
            args, x_train, y_train, x_val, y_val, background_label
        )
        background_grads = _posthoc_classifier_head_grad_embeddings(
            args,
            background_real,
            background_labels,
            x_train,
            y_train,
            device,
            batch_size,
        )
        background_grad = background_grads.mean(axis=0, keepdims=True)
        background_score = _gradient_similarity(cand_grads, background_grad, args.posthoc_influence_normalize)

    combined = target_score - background_weight * background_score
    return combined.astype(np.float32), target_score.astype(np.float32), background_score.astype(np.float32)


def _class_symmetric_influence_scores(args, x_real, labels, probs, x_train, y_train, x_val, y_val, label, device):
    label = int(label)
    influence_args = SimpleNamespace(**vars(args))
    influence_args.posthoc_influence_weight = 1.0
    influence_args.posthoc_influence_target_label = label
    influence_args.posthoc_influence_background_label = 1 - label
    if label == 0:
        influence_weight = float(args.posthoc_y0_influence_weight)
        influence_args.posthoc_influence_background_weight = float(args.posthoc_y0_background_weight)
    else:
        influence_weight = float(args.posthoc_y1_influence_weight)
        influence_args.posthoc_influence_background_weight = float(args.posthoc_y1_background_weight)
    influence_score, target_score, background_score = _posthoc_influence_utility_scores(
        influence_args,
        x_real,
        labels,
        x_train,
        y_train,
        x_val,
        y_val,
        label,
        device,
    )
    hard_negative_score = np.zeros_like(probs, dtype=np.float32)
    if label == 0 and float(args.posthoc_y0_hard_negative_weight) != 0.0:
        real_y0 = x_train[y_train == 0].astype(np.float32)
        if len(real_y0) > 0:
            real_y0_labels = np.zeros(len(real_y0), dtype=np.int64)
            real_y0_probs = _posthoc_classifier_probs(args, real_y0, real_y0_labels, x_train, y_train, device)
            target = float(np.quantile(real_y0_probs, 0.95))
            threshold = float(args.posthoc_positive_threshold)
            target = min(target, threshold - 1e-3)
            width = max(threshold - target, 1e-3)
            hard_negative_score = -np.abs(probs - target) / width
            hard_negative_score = hard_negative_score - 4.0 * np.maximum(probs - threshold, 0.0) / width
            hard_negative_score = hard_negative_score.astype(np.float32)
    return influence_weight, influence_score, target_score, background_score, hard_negative_score


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (values - np.nanmean(values)) / scale


def _sample_feature_embedding(x_real):
    mean = x_real.mean(axis=1)
    std = x_real.std(axis=1)
    low = np.quantile(x_real, 0.10, axis=1)
    high = np.quantile(x_real, 0.90, axis=1)
    return np.concatenate([mean, std, low, high], axis=1).astype(np.float32)


def _candidate_density_penalty(x_real, seed=42, max_refs=1024, batch_size=2048):
    emb = _sample_feature_embedding(x_real)
    if len(emb) <= 1:
        return np.zeros(len(emb), dtype=np.float32)
    rng = np.random.default_rng(seed)
    ref_n = min(int(max_refs), len(emb))
    ref_idx = rng.choice(len(emb), size=ref_n, replace=False)
    refs = emb[ref_idx]
    center = emb.mean(axis=0, keepdims=True)
    scale = np.maximum(emb.std(axis=0, keepdims=True), 1e-6)
    emb = (emb - center) / scale
    refs = (refs - center) / scale
    densities = []
    for start in range(0, len(emb), batch_size):
        chunk = emb[start:start + batch_size]
        sqdist = ((chunk[:, None, :] - refs[None, :, :]) ** 2).mean(axis=2)
        kth = min(8, sqdist.shape[1] - 1)
        local = np.partition(sqdist, kth, axis=1)[:, kth]
        densities.append(1.0 / np.sqrt(local + 1e-6))
    return np.concatenate(densities, axis=0).astype(np.float32)


@torch.no_grad()
def _posthoc_token_manifold_scores(args, vq_model, flow_model, x_norm, label, x_train, y_train, device):
    if vq_model is None or flow_model is None or float(args.posthoc_token_weight) == 0.0:
        return np.zeros(len(x_norm), dtype=np.float32)
    target_real = x_train[y_train == int(label)]
    if len(target_real) == 0:
        return np.zeros(len(x_norm), dtype=np.float32)

    def encode_np(x):
        chunks = []
        for start in range(0, len(x), args.batch_size):
            batch = torch.from_numpy(x[start:start + args.batch_size]).float().to(device)
            chunks.append(vq_model.encode(batch).detach().cpu().long())
        return torch.cat(chunks, dim=0)

    target_norm = normalize_like_train(target_real, x_train)
    real_tokens = flow_model.split_global_indices(encode_np(target_norm))
    cand_tokens = flow_model.split_global_indices(encode_np(x_norm))
    scores = np.zeros(len(x_norm), dtype=np.float32)
    smoothing = max(float(args.posthoc_token_smoothing), 1e-6)
    for real_idx, cand_idx, num_codes in zip(real_tokens, cand_tokens, flow_model.nb_code):
        real_np = real_idx.numpy()
        cand_np = cand_idx.numpy()
        counts = np.bincount(real_np.reshape(-1), minlength=int(num_codes)).astype(np.float64) + smoothing
        logp = np.log(counts / counts.sum())
        scores += logp[cand_np].mean(axis=1).astype(np.float32)
        if cand_np.shape[1] > 1:
            trans_counts = np.full((int(num_codes), int(num_codes)), smoothing, dtype=np.float64)
            for pos in range(real_np.shape[1] - 1):
                np.add.at(trans_counts, (real_np[:, pos], real_np[:, pos + 1]), 1.0)
            trans_logp = np.log(trans_counts / trans_counts.sum(axis=1, keepdims=True))
            trans = np.zeros(len(cand_np), dtype=np.float32)
            for pos in range(cand_np.shape[1] - 1):
                trans += trans_logp[cand_np[:, pos], cand_np[:, pos + 1]].astype(np.float32)
            scores += trans / max(cand_np.shape[1] - 1, 1)
    return scores / max(len(cand_tokens), 1)


def _quota_keep_indices(score, probs, keep_n, positive_rate, positive_threshold):
    order = np.argsort(-score)
    if positive_rate is None or positive_rate < 0:
        return order[:keep_n]
    target_pos = int(round(float(positive_rate) * int(keep_n)))
    target_pos = max(0, min(target_pos, int(keep_n)))
    positive = order[probs[order] >= float(positive_threshold)]
    negative = order[probs[order] < float(positive_threshold)]
    pos_keep = positive[:min(target_pos, len(positive))]
    remaining = int(keep_n) - len(pos_keep)
    neg_keep = negative[:remaining]
    if len(neg_keep) < remaining:
        used = set(pos_keep.tolist())
        extra = [idx for idx in order.tolist() if idx not in used and idx not in set(neg_keep.tolist())]
        neg_keep = np.concatenate([neg_keep, np.asarray(extra[:remaining - len(neg_keep)], dtype=np.int64)])
    keep_idx = np.concatenate([pos_keep, neg_keep]).astype(np.int64)
    if len(keep_idx) > keep_n:
        keep_idx = keep_idx[:keep_n]
    return keep_idx


def _distribution_quota_keep_indices(score, probs, target_probs, keep_n, bin_quantiles):
    quantiles = np.asarray(bin_quantiles, dtype=np.float64)
    if quantiles.ndim != 1 or len(quantiles) < 3:
        raise ValueError("--posthoc-prob-bin-quantiles must contain at least 3 values")
    if not np.all(np.diff(quantiles) > 0):
        raise ValueError("--posthoc-prob-bin-quantiles must be strictly increasing")
    if quantiles[0] != 0.0 or quantiles[-1] != 1.0:
        raise ValueError("--posthoc-prob-bin-quantiles must start with 0.0 and end with 1.0")

    edges = np.quantile(target_probs, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    n_bins = len(edges) - 1
    target_counts = np.diff(quantiles) * int(keep_n)
    floors = np.floor(target_counts).astype(np.int64)
    remainder = int(keep_n) - int(floors.sum())
    if remainder > 0:
        frac_order = np.argsort(-(target_counts - floors))
        floors[frac_order[:remainder]] += 1

    ranked_by_bin = []
    selected_parts = []
    used = set()
    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (probs >= edges[bin_idx]) & (probs <= edges[bin_idx + 1])
        else:
            mask = (probs >= edges[bin_idx]) & (probs < edges[bin_idx + 1])
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-score[idx])]
        ranked_by_bin.append(idx)
        take = min(int(floors[bin_idx]), len(idx))
        if take > 0:
            chosen = idx[:take]
            selected_parts.append(chosen)
            used.update(chosen.tolist())

    selected = np.concatenate(selected_parts).astype(np.int64) if selected_parts else np.empty(0, dtype=np.int64)
    deficit = int(keep_n) - len(selected)
    if deficit > 0:
        global_order = np.argsort(-score)
        extra = [idx for idx in global_order.tolist() if idx not in used]
        selected = np.concatenate([selected, np.asarray(extra[:deficit], dtype=np.int64)])
    if len(selected) > keep_n:
        selected = selected[np.argsort(-score[selected])[:keep_n]]

    selected_counts = []
    for bin_idx in range(n_bins):
        if len(selected) == 0:
            selected_counts.append(0)
            continue
        if bin_idx == n_bins - 1:
            mask = (probs[selected] >= edges[bin_idx]) & (probs[selected] <= edges[bin_idx + 1])
        else:
            mask = (probs[selected] >= edges[bin_idx]) & (probs[selected] < edges[bin_idx + 1])
        selected_counts.append(int(mask.sum()))
    info = {
        "prob_bin_quantiles": quantiles.tolist(),
        "prob_bin_edges": [None if not np.isfinite(x) else float(x) for x in edges.tolist()],
        "target_bin_counts": floors.astype(int).tolist(),
        "selected_bin_counts": selected_counts,
    }
    return selected.astype(np.int64), info


def _write_selector_report(
    args,
    label,
    keep_idx,
    probs,
    utility_score,
    feature_penalty,
    token_score,
    density_penalty,
    score,
    influence_score=None,
    influence_target_score=None,
    influence_background_score=None,
    target_positive_rate=None,
    positive_threshold=0.5,
    distribution_info=None,
):
    report_path = getattr(args, "_posthoc_report_path", None)
    if report_path is None:
        return
    selected = np.asarray(keep_idx, dtype=np.int64)
    payload = {
        "label": int(label),
        "selector": args.posthoc_selector,
        "num_candidates": int(len(probs)),
        "num_selected": int(len(selected)),
        "classifier_prob_before_mean": float(np.mean(probs)),
        "classifier_prob_after_mean": float(np.mean(probs[selected])),
        "classifier_prob_after_q50": float(np.quantile(probs[selected], 0.50)),
        "classifier_prob_after_q95": float(np.quantile(probs[selected], 0.95)),
        "candidate_positive_rate": float(np.mean(probs >= float(positive_threshold))),
        "selected_positive_rate": float(np.mean(probs[selected] >= float(positive_threshold))),
        "target_positive_rate": None if target_positive_rate is None else float(target_positive_rate),
        "positive_threshold": float(positive_threshold),
        "utility_score_after_mean": float(np.mean(utility_score[selected])),
        "feature_penalty_after_mean": float(np.mean(feature_penalty[selected])),
        "token_score_after_mean": float(np.mean(token_score[selected])),
        "density_penalty_after_mean": float(np.mean(density_penalty[selected])),
        "score_after_mean": float(np.mean(score[selected])),
        "weights": {
            "utility": float(args.posthoc_utility_weight),
            "influence": float(getattr(args, "posthoc_influence_weight", 0.0)),
            "feature_penalty": float(args.posthoc_feature_penalty_weight),
            "token": float(args.posthoc_token_weight),
            "diversity": float(args.posthoc_diversity_weight),
        },
    }
    if influence_score is not None:
        payload["influence_score_before_mean"] = float(np.mean(influence_score))
        payload["influence_score_after_mean"] = float(np.mean(influence_score[selected]))
        payload["influence_score_after_q50"] = float(np.quantile(influence_score[selected], 0.50))
        payload["influence_score_after_q95"] = float(np.quantile(influence_score[selected], 0.95))
    if influence_target_score is not None:
        payload["influence_target_score_before_mean"] = float(np.mean(influence_target_score))
        payload["influence_target_score_after_mean"] = float(np.mean(influence_target_score[selected]))
    if influence_background_score is not None:
        payload["influence_background_score_before_mean"] = float(np.mean(influence_background_score))
        payload["influence_background_score_after_mean"] = float(np.mean(influence_background_score[selected]))
        payload["influence_background_weight"] = float(getattr(args, "posthoc_influence_background_weight", 0.0))
        payload["influence_background_label"] = int(getattr(args, "posthoc_influence_background_label", 0))
    if distribution_info is not None:
        payload["distribution_quota"] = distribution_info
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_multiclass_selector_report(
    args,
    label,
    keep_idx,
    class_probs,
    utility_score,
    feature_penalty,
    token_score,
    density_penalty,
    influence_score,
    score,
    distribution_info,
):
    report_path = getattr(args, "_posthoc_report_path", None)
    if report_path is None:
        return
    selected = np.asarray(keep_idx, dtype=np.int64)
    class_names_raw = getattr(args, "class_names", None) or ""
    class_names = [name.strip() for name in str(class_names_raw).split(",") if name.strip()]
    class_name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
    payload = {
        "label": int(label),
        "class_name": class_name,
        "selector": args.posthoc_selector,
        "num_candidates": int(len(class_probs)),
        "num_selected": int(len(selected)),
        "target_class_probability_before_mean": float(np.mean(class_probs)),
        "target_class_probability_after_mean": float(np.mean(class_probs[selected])),
        "target_class_probability_after_q05": float(np.quantile(class_probs[selected], 0.05)),
        "target_class_probability_after_q50": float(np.quantile(class_probs[selected], 0.50)),
        "target_class_probability_after_q95": float(np.quantile(class_probs[selected], 0.95)),
        "utility_score_after_mean": float(np.mean(utility_score[selected])),
        "influence_score_after_mean": float(np.mean(influence_score[selected])),
        "feature_penalty_after_mean": float(np.mean(feature_penalty[selected])),
        "token_score_after_mean": float(np.mean(token_score[selected])),
        "density_penalty_after_mean": float(np.mean(density_penalty[selected])),
        "score_after_mean": float(np.mean(score[selected])),
        "weights": {
            "utility": float(args.posthoc_utility_weight),
            "influence": float(args.posthoc_influence_weight),
            "feature_penalty": float(args.posthoc_feature_penalty_weight),
            "token": float(args.posthoc_token_weight),
            "diversity": float(args.posthoc_diversity_weight),
        },
        "distribution_quota": distribution_info,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _multiclass_posthoc_select_candidates(args, x_norm, label, keep_n, x_train, y_train, x_val, y_val, device, vq_model, flow_model):
    x_real = inverse_minmax(x_norm, x_train).astype(np.float32)
    labels = np.full(len(x_real), int(label), dtype=np.int64)
    probs = _multiclass_posthoc_classifier_probs(args, x_real, labels, x_train, y_train, device)
    class_probs = probs[:, int(label)]

    target_real = x_train[y_train == int(label)]
    if len(target_real) == 0:
        raise ValueError(f"No real samples found for multi-class selector label={label}")
    target_labels = np.full(len(target_real), int(label), dtype=np.int64)
    target_probs = _multiclass_posthoc_classifier_probs(
        args, target_real.astype(np.float32), target_labels, x_train, y_train, device
    )[:, int(label)]

    q = min(max(float(args.posthoc_feature_quantile), 0.0), 0.2)
    low = np.quantile(target_real, q, axis=(0, 1)).astype(np.float32)
    high = np.quantile(target_real, 1.0 - q, axis=(0, 1)).astype(np.float32)
    width = np.maximum(high - low, 1e-6)
    low_violation = np.maximum(low.reshape(1, 1, -1) - x_real, 0.0) / width.reshape(1, 1, -1)
    high_violation = np.maximum(x_real - high.reshape(1, 1, -1), 0.0) / width.reshape(1, 1, -1)
    feature_penalty = (low_violation + high_violation).mean(axis=(1, 2))

    lower = float(np.quantile(target_probs, float(args.posthoc_prob_lower_quantile)))
    upper = float(np.quantile(target_probs, float(args.posthoc_prob_upper_quantile)))
    prob_width = max(upper - lower, 1e-3)
    below = np.maximum(lower - class_probs, 0.0) / prob_width
    above = np.maximum(class_probs - upper, 0.0) / prob_width
    utility_score = class_probs - float(args.posthoc_prob_tail_penalty_weight) * (below + above)
    influence_score = _multiclass_influence_scores(
        args, x_real, labels, x_train, y_train, x_val, y_val, int(label), device
    )
    token_score = _posthoc_token_manifold_scores(args, vq_model, flow_model, x_norm, label, x_train, y_train, device)
    density_penalty = _candidate_density_penalty(x_real, max_refs=args.posthoc_diversity_refs)
    score = (
        float(args.posthoc_influence_weight) * _zscore(influence_score)
        + float(args.posthoc_utility_weight) * _zscore(utility_score)
        + float(args.posthoc_token_weight) * _zscore(token_score)
        - float(args.posthoc_feature_penalty_weight) * _zscore(feature_penalty)
        - float(args.posthoc_diversity_weight) * _zscore(density_penalty)
    )
    keep_idx, distribution_info = _distribution_quota_keep_indices(
        score,
        class_probs,
        target_probs,
        keep_n,
        args.posthoc_prob_bin_quantiles,
    )
    _write_multiclass_selector_report(
        args,
        label,
        keep_idx,
        class_probs,
        utility_score,
        feature_penalty,
        token_score,
        density_penalty,
        influence_score,
        score,
        distribution_info,
    )
    print(
        f"Multi-class selected label={label}: candidates={len(x_norm)}, keep={keep_n}, "
        f"p_label_before={float(class_probs.mean()):.4f}, p_label_after={float(class_probs[keep_idx].mean()):.4f}, "
        f"influence_after={float(influence_score[keep_idx].mean()):.4f}, "
        f"feature_penalty={float(feature_penalty[keep_idx].mean()):.4f}"
    )
    return x_norm[keep_idx]


def posthoc_select_candidates(args, x_norm, label, keep_n, x_train, y_train, x_val, y_val, device, vq_model=None, flow_model=None):
    selector_labels = getattr(args, "posthoc_selector_labels", "target")
    selector_applies = selector_labels == "all" or int(label) == int(args.posthoc_selector_target_label)
    if args.posthoc_selector == "none" or not selector_applies:
        return x_norm[:keep_n]
    if len(x_norm) <= keep_n:
        return x_norm
    if args.posthoc_selector == "multiclass_influence_utility":
        return _multiclass_posthoc_select_candidates(
            args, x_norm, label, keep_n, x_train, y_train, x_val, y_val, device, vq_model, flow_model
        )
    x_real = inverse_minmax(x_norm, x_train).astype(np.float32)
    labels = np.full(len(x_real), int(label), dtype=np.int64)
    probs = _posthoc_classifier_probs(args, x_real, labels, x_train, y_train, device)

    target_real = x_train[y_train == int(label)]
    if len(target_real) == 0:
        raise ValueError(f"No real samples found for posthoc selector label={label}")
    target_labels = np.full(len(target_real), int(label), dtype=np.int64)
    target_probs = _posthoc_classifier_probs(args, target_real.astype(np.float32), target_labels, x_train, y_train, device)
    q = min(max(float(args.posthoc_feature_quantile), 0.0), 0.2)
    low = np.quantile(target_real, q, axis=(0, 1)).astype(np.float32)
    high = np.quantile(target_real, 1.0 - q, axis=(0, 1)).astype(np.float32)
    width = np.maximum(high - low, 1e-6)
    low_violation = np.maximum(low.reshape(1, 1, -1) - x_real, 0.0) / width.reshape(1, 1, -1)
    high_violation = np.maximum(x_real - high.reshape(1, 1, -1), 0.0) / width.reshape(1, 1, -1)
    feature_penalty = (low_violation + high_violation).mean(axis=(1, 2))
    distribution_info = None
    influence_score = np.zeros_like(probs, dtype=np.float32)
    influence_target_score = np.zeros_like(probs, dtype=np.float32)
    influence_background_score = np.zeros_like(probs, dtype=np.float32)
    if args.posthoc_selector == "classifier_quantile":
        utility_score = probs
        token_score = np.zeros_like(probs, dtype=np.float32)
        density_penalty = np.zeros_like(probs, dtype=np.float32)
        score = probs - float(args.posthoc_feature_penalty_weight) * feature_penalty
        target_positive_rate = None
    elif args.posthoc_selector in {"utility_manifold", "distribution_quota", "influence_utility", "class_symmetric_influence"}:
        target_class_probs = target_probs if int(label) == 1 else (1.0 - target_probs)
        class_probs = probs if int(label) == 1 else (1.0 - probs)
        lower = float(np.quantile(target_class_probs, float(args.posthoc_prob_lower_quantile)))
        upper = float(np.quantile(target_class_probs, float(args.posthoc_prob_upper_quantile)))
        width = max(upper - lower, 1e-3)
        below_penalty = np.maximum(lower - class_probs, 0.0) / width
        above_penalty = np.maximum(class_probs - upper, 0.0) / width
        utility_score = class_probs - float(args.posthoc_prob_tail_penalty_weight) * (below_penalty + above_penalty)
        token_score = _posthoc_token_manifold_scores(args, vq_model, flow_model, x_norm, label, x_train, y_train, device)
        density_penalty = _candidate_density_penalty(x_real, max_refs=args.posthoc_diversity_refs)
        influence_weight = float(args.posthoc_influence_weight)
        hard_negative_score = np.zeros_like(probs, dtype=np.float32)
        if args.posthoc_selector == "influence_utility":
            influence_score, influence_target_score, influence_background_score = _posthoc_influence_utility_scores(
                args,
                x_real,
                labels,
                x_train,
                y_train,
                x_val,
                y_val,
                label,
                device,
            )
        elif args.posthoc_selector == "class_symmetric_influence":
            (
                influence_weight,
                influence_score,
                influence_target_score,
                influence_background_score,
                hard_negative_score,
            ) = _class_symmetric_influence_scores(
                args,
                x_real,
                labels,
                probs,
                x_train,
                y_train,
                x_val,
                y_val,
                label,
                device,
            )
        score = (
            float(influence_weight) * _zscore(influence_score)
            + float(args.posthoc_utility_weight) * _zscore(utility_score)
            + float(args.posthoc_token_weight) * _zscore(token_score)
            + (float(args.posthoc_y0_hard_negative_weight) if int(label) == 0 else 0.0) * _zscore(hard_negative_score)
            - float(args.posthoc_feature_penalty_weight) * _zscore(feature_penalty)
            - float(args.posthoc_diversity_weight) * _zscore(density_penalty)
        )
        if int(label) == 0:
            target_positive_rate = float(np.mean(target_probs >= float(args.posthoc_positive_threshold)))
        elif args.posthoc_positive_rate_target == "auto":
            target_positive_rate = float(np.mean(target_probs >= float(args.posthoc_positive_threshold)))
        else:
            target_positive_rate = float(args.posthoc_positive_rate_target)
    else:
        raise ValueError(f"Unsupported posthoc selector: {args.posthoc_selector}")
    if args.posthoc_selector == "distribution_quota":
        keep_idx, distribution_info = _distribution_quota_keep_indices(
            score,
            probs,
            target_probs,
            keep_n,
            args.posthoc_prob_bin_quantiles,
        )
    else:
        keep_idx = _quota_keep_indices(
            score,
            probs,
            keep_n,
            target_positive_rate,
            float(args.posthoc_positive_threshold),
        )
    _write_selector_report(
        args,
        label,
        keep_idx,
        probs,
        utility_score,
        feature_penalty,
        token_score,
        density_penalty,
        score,
        influence_score=influence_score,
        influence_target_score=influence_target_score,
        influence_background_score=influence_background_score,
        target_positive_rate=target_positive_rate,
        positive_threshold=float(args.posthoc_positive_threshold),
        distribution_info=distribution_info,
    )
    print(
        f"Post-hoc selected label={label}: candidates={len(x_norm)}, keep={keep_n}, "
        f"prob_mean_before={float(probs.mean()):.4f}, prob_mean_after={float(probs[keep_idx].mean()):.4f}, "
        f"influence_after={float(influence_score[keep_idx].mean()):.4f}, "
        f"pos_rate_after={float(np.mean(probs[keep_idx] >= float(args.posthoc_positive_threshold))):.4f}, "
        f"penalty_mean_after={float(feature_penalty[keep_idx].mean()):.4f}, "
        f"token_score_after={float(token_score[keep_idx].mean()):.4f}, "
        f"density_after={float(density_penalty[keep_idx].mean()):.4f}"
    )
    return x_norm[keep_idx]


def main():
    parser = argparse.ArgumentParser(description="Generate MedFlow tuples with conditional sampling and optional TMG.")
    parser.add_argument("--mode", choices=["class_specific", "conditional", "influence"], required=True)
    parser.add_argument("--data-dir", default="data/processed/mimic_icustay")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed used for stochastic flow sampling")
    parser.add_argument("--input-dim", type=int, default=None, help="optional tuple feature dimension; inferred when omitted")
    parser.add_argument("--feature-names", default=None, help="optional comma-separated feature names for reports")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--positive-multiplier", type=float, default=1.0, help="multiply y=1 count when --num-samples is not set")
    parser.add_argument("--positive-mixup-multiplier", type=float, default=1.0, help="positive data-level mixup multiplier used when the flow checkpoint was trained")
    parser.add_argument("--positive-mixup-label", type=int, default=1, help="label appended by positive data-level mixup during flow training")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")

    parser.add_argument("--vq", default=None)
    parser.add_argument("--flow", default=None)
    parser.add_argument("--class0-vq", default=None)
    parser.add_argument("--class0-flow", default=None)
    parser.add_argument("--class1-vq", default=None)
    parser.add_argument("--class1-flow", default=None)

    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--nb-code", type=int, nargs="+", default=[128, 512, 512])
    parser.add_argument("--patch-num", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--vq-act", default="relu", choices=["relu", "silu", "gelu"])
    parser.add_argument("--vq-norm", default=None)
    parser.add_argument("--quantizer", default="ema_reset_sim")
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--fm-backbone", default="dit1d", choices=["dit1d", "transformer1d"])
    parser.add_argument("--fm-hidden-dim", type=int, default=512)
    parser.add_argument("--fm-depth", type=int, default=6)
    parser.add_argument("--fm-heads", type=int, default=8)
    parser.add_argument("--fm-dropout", type=float, default=0.1)
    parser.add_argument("--class-specific-output-head", action="store_true")
    parser.add_argument("--class-specific-adapter", action="store_true")
    parser.add_argument("--label-conditioning-mode", default="add", choices=["add", "film"])
    parser.add_argument("--token-type-conditioning", default="none", choices=["none", "class", "class_scale"])
    parser.add_argument("--latent-rank", type=int, default=128)
    parser.add_argument("--latent-noise-std", type=float, default=0.01)
    parser.add_argument("--t-scheduler", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--train-mixup-prob", type=float, default=0.5)
    parser.add_argument("--train-mixup-alpha", type=float, default=1.0)
    parser.add_argument("--structure-loss-weight", type=float, default=10.0)
    parser.add_argument("--senior-mean-reg-weight", type=float, default=0.1)
    parser.add_argument("--senior-std-reg-weight", type=float, default=0.1)
    parser.add_argument("--senior-sample-noise-std", type=float, default=0.01)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=30)
    parser.add_argument("--solver", default="euler", choices=["euler", "heun"])
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--label-guidance-scale", type=float, default=1.0)
    parser.add_argument("--token-manifold-guidance-weight", type=float, default=0.0)
    parser.add_argument("--token-manifold-guidance-target-label", type=int, default=1)
    parser.add_argument("--token-manifold-guidance-labels", default="target", choices=["target", "all"])
    parser.add_argument("--token-manifold-guidance-mode", default="log_odds", choices=["log_odds", "target_prior"])
    parser.add_argument("--token-manifold-guidance-background", default="all", choices=["all", "not_target"])
    parser.add_argument("--token-manifold-guidance-smoothing", type=float, default=1.0)
    parser.add_argument("--token-manifold-guidance-clamp", type=float, default=3.0)
    parser.add_argument("--token-manifold-transition-weight", type=float, default=0.0)
    parser.add_argument("--token-manifold-cross-scale-weight", type=float, default=0.0)
    parser.add_argument("--utility-token-guidance-weight", type=float, default=0.0)
    parser.add_argument("--utility-token-transition-weight", type=float, default=0.0)
    parser.add_argument("--utility-token-cross-scale-weight", type=float, default=0.0)
    parser.add_argument("--utility-token-guidance-mode", default="unigram", choices=["unigram", "transition", "cross_scale", "all"])
    parser.add_argument("--utility-token-guidance-target-label", type=int, default=1)
    parser.add_argument("--utility-token-guidance-background-label", type=int, default=0)
    parser.add_argument("--utility-token-guidance-top-frac", type=float, default=0.5)
    parser.add_argument("--utility-token-guidance-smoothing", type=float, default=1.0)
    parser.add_argument("--utility-token-guidance-clamp", type=float, default=3.0)
    parser.add_argument("--utility-token-guidance-influence-weight", type=float, default=0.2)
    parser.add_argument("--utility-token-guidance-background-weight", type=float, default=0.25)
    parser.add_argument("--utility-token-guidance-max-target", type=int, default=512)
    parser.add_argument("--utility-token-guidance-candidate-batch-size", type=int, default=64)
    parser.add_argument("--token-manifold-guidance-schedule", default="late_linear", choices=["constant", "late_linear", "late_cosine"])
    parser.add_argument("--token-manifold-guidance-warmup-frac", type=float, default=0.35)
    parser.add_argument("--label-conditioned-prior", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--senior-sampler", default="mixup", choices=["mixup", "kde", "gaussian"])
    parser.add_argument("--source-prior-mode", default="learned", choices=["learned", "gaussian"])
    parser.add_argument("--kde-bandwidth-factor", type=float, default=1.0)
    parser.add_argument("--kde-max-centers", type=int, default=2000)
    parser.add_argument("--sampling-mode", default="shared_context", choices=["shared_context", "ctf_nearest"])
    parser.add_argument("--cross-scale-conditioning", default="none", choices=["none", "ctf"])
    parser.add_argument("--class0-cross-scale-conditioning", default=None, choices=[None, "none", "ctf"])
    parser.add_argument("--class1-cross-scale-conditioning", default=None, choices=[None, "none", "ctf"])

    parser.add_argument("--guidance-ckpt", default=None)
    parser.add_argument("--guidance-weight", type=float, default=0.1)
    parser.add_argument("--guidance-steps", type=int, default=1)
    parser.add_argument("--guidance-class", choices=["all", "positive", "negative"], default="all")
    parser.add_argument("--guidance-refine-class", choices=["all", "positive", "negative"], default="all")
    parser.add_argument("--guidance-batch-size", type=int, default=128)
    parser.add_argument("--guidance-hidden-dim", type=int, default=256)
    parser.add_argument("--guidance-num-layers", type=int, default=2)
    parser.add_argument("--guidance-rnn-type", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--guidance-dropout", type=float, default=0.2)
    parser.add_argument("--posthoc-selector", default="none", choices=["none", "classifier_quantile", "utility_manifold", "distribution_quota", "influence_utility", "class_symmetric_influence", "multiclass_influence_utility"])
    parser.add_argument("--posthoc-candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--posthoc-selector-target-label", type=int, default=1)
    parser.add_argument("--posthoc-selector-labels", default="target", choices=["target", "all"])
    parser.add_argument("--posthoc-classifier-ckpt", default="runs/selector/best_model.pt")
    parser.add_argument("--posthoc-classifier-hidden-dim", type=int, default=256)
    parser.add_argument("--posthoc-classifier-num-layers", type=int, default=2)
    parser.add_argument("--posthoc-classifier-rnn-type", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--posthoc-classifier-dropout", type=float, default=0.2)
    parser.add_argument("--posthoc-feature-quantile", type=float, default=0.01)
    parser.add_argument("--posthoc-feature-penalty-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-utility-weight", type=float, default=1.0)
    parser.add_argument("--posthoc-influence-weight", type=float, default=0.0)
    parser.add_argument("--posthoc-influence-target-split", default="val", choices=["train", "val"])
    parser.add_argument("--posthoc-influence-target-label", type=int, default=1)
    parser.add_argument("--posthoc-influence-background-label", type=int, default=0)
    parser.add_argument("--posthoc-influence-background-weight", type=float, default=0.0)
    parser.add_argument("--posthoc-influence-normalize", default="cosine", choices=["cosine", "dot"])
    parser.add_argument("--posthoc-influence-max-target", type=int, default=512)
    parser.add_argument("--posthoc-influence-candidate-batch-size", type=int, default=64)
    parser.add_argument("--posthoc-y0-influence-weight", type=float, default=0.1)
    parser.add_argument("--posthoc-y1-influence-weight", type=float, default=0.2)
    parser.add_argument("--posthoc-y0-background-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-y1-background-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-y0-hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-token-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-token-smoothing", type=float, default=1.0)
    parser.add_argument("--posthoc-diversity-weight", type=float, default=0.10)
    parser.add_argument("--posthoc-diversity-refs", type=int, default=1024)
    parser.add_argument("--posthoc-prob-target-quantile", type=float, default=0.50)
    parser.add_argument("--posthoc-prob-distance-clip", type=float, default=4.0)
    parser.add_argument("--posthoc-positive-threshold", type=float, default=0.5)
    parser.add_argument("--posthoc-positive-rate-target", default="auto", help="'auto' uses real target-label positive rate, or pass a float such as 0.20")
    parser.add_argument("--posthoc-prob-lower-quantile", type=float, default=0.75)
    parser.add_argument("--posthoc-prob-upper-quantile", type=float, default=0.98)
    parser.add_argument("--posthoc-prob-tail-penalty-weight", type=float, default=0.25)
    parser.add_argument("--posthoc-prob-bin-quantiles", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 0.90, 0.97, 1.0])
    parser.add_argument("--posthoc-report", default=None)
    parser.add_argument("--class-names", default=None, help="optional comma-separated class names for multi-class selector reports")
    parser.add_argument("--latent-utility-refine", action="store_true")
    parser.add_argument("--latent-utility-refine-labels", default="target", choices=["target", "all"])
    parser.add_argument("--latent-utility-refine-steps", type=int, default=1)
    parser.add_argument("--latent-utility-refine-weight", type=float, default=0.0)
    parser.add_argument("--latent-utility-refine-space", default="embedding", choices=["logits", "embedding"])
    parser.add_argument("--latent-utility-refine-anchor-weight", type=float, default=1.0)
    parser.add_argument("--latent-utility-refine-token-prior-weight", type=float, default=0.1)
    parser.add_argument("--latent-utility-refine-feature-penalty-weight", type=float, default=0.5)
    parser.add_argument("--latent-utility-refine-clamp", type=float, default=0.05)
    parser.add_argument("--latent-utility-refine-batch-size", type=int, default=128)
    parser.add_argument("--inflow-utility-guidance-weight", type=float, default=0.0)
    parser.add_argument("--inflow-utility-guidance-start-frac", type=float, default=0.5)
    parser.add_argument("--inflow-utility-guidance-every", type=int, default=5)
    parser.add_argument("--inflow-utility-guidance-labels", default="target", choices=["target", "all"])
    parser.add_argument("--inflow-utility-guidance-space", default="embedding", choices=["logits", "embedding"])
    parser.add_argument("--inflow-utility-guidance-clamp", type=float, default=0.05)
    parser.add_argument("--inflow-utility-anchor-weight", type=float, default=1.0)
    args = parser.parse_args()

    # Sampling is otherwise stochastic but previously had no user-visible seed.
    # Keep the historical default (42) so existing commands remain reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    x_train, y_train = load_tuple(data_dir / "train_tuple.pkl", input_dim=args.input_dim)
    if args.input_dim is None:
        args.input_dim = int(x_train.shape[-1])
    x_val, y_val = load_tuple(data_dir / "val_tuple.pkl", input_dim=args.input_dim)
    counts = target_counts(args, y_train)

    x_norm_parts = []
    y_parts = []
    if args.mode == "class_specific":
        paths = {
            0: (args.class0_vq, args.class0_flow),
            1: (args.class1_vq, args.class1_flow),
        }
        for label, n in counts.items():
            if n <= 0:
                continue
            vq_model = load_vq(args, paths[label][0], device)
            if label == 0:
                csc = args.class0_cross_scale_conditioning
            else:
                csc = args.class1_cross_scale_conditioning
            if csc is None:
                csc = args.cross_scale_conditioning
            flow_model = load_flow(args, paths[label][1], device, conditional=False, cross_scale_conditioning=csc)
            cache_training_tokens_for_ctf(args, vq_model, flow_model, x_train[y_train == label], x_train, device)
            x_norm_parts.append(sample_model(args, vq_model, flow_model, n, label, device))
            y_parts.append(np.full(n, label, dtype=np.int64))
    else:
        vq_model = load_vq(args, args.vq, device)
        flow_model = load_flow(args, args.flow, device, conditional=True)
        cache_training_tokens_for_ctf(args, vq_model, flow_model, x_train, x_train, device)
        cache_training_labels_for_conditional_prior(args, flow_model, y_train)
        cache_token_manifold_guidance(args, vq_model, flow_model, x_train, y_train, device)
        cache_utility_token_guidance(args, vq_model, flow_model, x_train, y_train, x_val, y_val, device)
        for label, n in counts.items():
            if n <= 0:
                continue
            sample_n = int(n)
            selector_applies = args.posthoc_selector_labels == "all" or int(label) == int(args.posthoc_selector_target_label)
            if args.posthoc_selector != "none" and selector_applies:
                sample_n = max(sample_n, int(np.ceil(sample_n * float(args.posthoc_candidate_multiplier))))
            x_norm = sample_model(args, vq_model, flow_model, sample_n, label, device)
            if args.posthoc_report is not None:
                report_path = Path(args.posthoc_report)
            else:
                report_path = Path(args.out).with_suffix("")
                report_path = report_path.parent / f"{report_path.name}_selector_label{label}.json"
            args._posthoc_report_path = str(report_path)
            x_norm = _continuous_utility_refine_x_norm(
                args,
                x_norm,
                label,
                x_train,
                y_train,
                device,
                enabled=bool(args.latent_utility_refine),
                labels_mode=args.latent_utility_refine_labels,
                steps=args.latent_utility_refine_steps,
                weight=args.latent_utility_refine_weight,
                anchor_weight=args.latent_utility_refine_anchor_weight,
                feature_penalty_weight=args.latent_utility_refine_feature_penalty_weight,
                clamp=float(args.latent_utility_refine_clamp),
                report_suffix="latent_refine",
            )
            x_norm = _continuous_utility_refine_x_norm(
                args,
                x_norm,
                label,
                x_train,
                y_train,
                device,
                enabled=float(args.inflow_utility_guidance_weight) != 0.0,
                labels_mode=args.inflow_utility_guidance_labels,
                steps=max(1, int(np.ceil((1.0 - float(args.inflow_utility_guidance_start_frac)) * int(args.flow_steps) / max(int(args.inflow_utility_guidance_every), 1)))),
                weight=args.inflow_utility_guidance_weight,
                anchor_weight=args.inflow_utility_anchor_weight,
                feature_penalty_weight=args.latent_utility_refine_feature_penalty_weight,
                clamp=float(args.inflow_utility_guidance_clamp),
                report_suffix="late_inflow_lite",
            )
            x_norm = posthoc_select_candidates(args, x_norm, label, int(n), x_train, y_train, x_val, y_val, device, vq_model, flow_model)
            x_norm_parts.append(x_norm)
            y_parts.append(np.full(n, label, dtype=np.int64))

    x_norm = np.concatenate(x_norm_parts, axis=0)
    y_syn = np.concatenate(y_parts, axis=0)
    x_syn = inverse_minmax(x_norm, x_train)

    if args.mode == "influence":
        if args.guidance_ckpt is None:
            raise ValueError("--guidance-ckpt is required for influence mode")
        refiner = InfluenceRefiner(args, x_train, y_train, x_val, y_val, device)
        x_syn = refiner.refine(x_syn, y_syn)

    order = np.random.default_rng(args.seed).permutation(len(y_syn))
    x_out = x_syn[order].transpose(0, 2, 1).astype(np.float32)
    y_out = y_syn[order].astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((x_out, y_out), args.out)
    print(f"Saved {args.mode} MSDFlow samples to {args.out}: X={x_out.shape}, labels={dict(Counter(y_out.tolist()))}")


if __name__ == "__main__":
    main()
