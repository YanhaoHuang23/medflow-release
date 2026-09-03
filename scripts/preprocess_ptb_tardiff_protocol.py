#!/usr/bin/env python
"""Build auditable PTB tuples for the two conflicting TarDiff source protocols.

The local PTB directory is the processed artifact released with Medformer:
one ``(n_heartbeats, 300, 15)`` file per patient and a ``(label, subject_id)``
table. TarDiff Table 7 reports 288 steps, while the exact TarDiff dataloader
uses a leading-window slice. We therefore standardize a complete 300-step
heartbeat as Medformer does, then retain its first 288 steps.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROTOCOLS = {
    "tardiff_literal_801010": {
        "description": "TarDiff paper text: subject-independent stratified 80/10/10, seed=42.",
        "split": "stratified_80_10_10",
    },
    "medformer_code_551530": {
        "description": "Pinned Medformer PTBLoader code: label-order subject split a=0.55, b=0.70.",
        "split": "medformer_code_55_15_30",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Medformer PTB for TarDiff protocol probes.")
    parser.add_argument("--raw-root", required=True, help="Directory containing PTB Feature/ and Label/ files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=288)
    return parser.parse_args()


def split_subjects(subject_ids, labels, protocol, seed):
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if protocol == "tardiff_literal_801010":
        train_ids, heldout_ids, train_labels, heldout_labels = train_test_split(
            subject_ids,
            labels,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
        )
        val_ids, test_ids = train_test_split(
            heldout_ids,
            test_size=0.50,
            random_state=seed,
            stratify=heldout_labels,
        )
        return {"train": sorted(train_ids.tolist()), "val": sorted(val_ids.tolist()), "test": sorted(test_ids.tolist())}

    if protocol == "medformer_code_551530":
        out = {"train": [], "val": [], "test": []}
        # Deliberately preserve the label.npy order: this reproduces PTBLoader.
        for label in sorted(np.unique(labels).tolist()):
            ids = subject_ids[labels == label].tolist()
            a, b = int(0.55 * len(ids)), int(0.70 * len(ids))
            out["train"].extend(ids[:a])
            out["val"].extend(ids[a:b])
            out["test"].extend(ids[b:])
        return {name: sorted(ids) for name, ids in out.items()}

    raise ValueError(f"Unsupported protocol: {protocol}")


def standardize_heartbeat_batch(x):
    """Medformer normalize_batch_ts: StandardScaler over time per heartbeat/channel."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return (x - mean) / std


def label_counts(values):
    return {str(int(label)): int(count) for label, count in sorted(Counter(map(int, values)).items())}


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    feature_root = raw_root / "Feature"
    label_path = raw_root / "Label" / "label.npy"
    if not feature_root.is_dir() or not label_path.is_file():
        raise FileNotFoundError("Expected PTB Feature/ and Label/label.npy under --raw-root")

    label_rows = np.load(label_path).astype(np.int64)
    if label_rows.ndim != 2 or label_rows.shape[1] != 2:
        raise ValueError(f"Expected label.npy with shape (N,2), got {label_rows.shape}")
    labels, subject_ids = label_rows[:, 0], label_rows[:, 1]
    subject_to_label = {int(subject_id): int(label) for label, subject_id in label_rows}
    feature_paths = {int(path.stem.split("_")[-1]): path for path in feature_root.glob("feature_*.npy")}
    missing = sorted(set(subject_to_label) - set(feature_paths))
    if missing:
        raise FileNotFoundError(f"Missing feature files for PTB subject IDs: {missing[:10]}")

    splits = split_subjects(subject_ids, labels, args.protocol, args.seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_meta = {}
    for split_name, ids in splits.items():
        x_parts, y_parts, id_rows = [], [], []
        for subject_id in ids:
            x = np.load(feature_paths[subject_id], mmap_mode="r")
            if x.ndim != 3 or x.shape[1] < args.window_size or x.shape[2] != 15:
                raise ValueError(f"Unexpected PTB feature shape for subject {subject_id}: {x.shape}")
            # Match Medformer normalization on the full heartbeat before TarDiff's leading-window crop.
            x = standardize_heartbeat_batch(x)[:, : args.window_size, :]
            label = subject_to_label[subject_id]
            x_parts.append(np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False))
            y_parts.append(np.full(len(x), label, dtype=np.int64))
            id_rows.extend(
                {"subject_id": subject_id, "label": label, "class_name": "HC" if label == 0 else "MI", "split": split_name}
                for _ in range(len(x))
            )
        x_split = np.concatenate(x_parts, axis=0)
        y_split = np.concatenate(y_parts, axis=0)
        pd.to_pickle((x_split, y_split), output / f"{split_name}_tuple.pkl")
        pd.DataFrame(id_rows).to_csv(output / f"ids_{split_name}.csv", index=False)
        split_meta[split_name] = {
            "subjects": len(ids),
            "segments": int(len(y_split)),
            "subject_label_counts": label_counts([subject_to_label[i] for i in ids]),
            "segment_label_counts": label_counts(y_split),
            "tuple_shape": list(x_split.shape),
        }
        print(f"Wrote {split_name}: X={x_split.shape}, labels={label_counts(y_split)}")

    pd.concat([pd.read_csv(output / f"ids_{name}.csv") for name in ("train", "val", "test")]).to_csv(output / "ids_all.csv", index=False)
    meta = {
        "dataset": "PTB Diagnostic ECG Database / Medformer processed subset",
        "task": "binary myocardial infarction vs healthy control classification",
        "label_mapping": {"0": "HC", "1": "MI"},
        "source_root": str(raw_root),
        "protocol": args.protocol,
        "protocol_description": PROTOCOLS[args.protocol]["description"],
        "split_unit": "subject_id",
        "seed": args.seed,
        "total_subjects": int(len(subject_ids)),
        "total_segments": int(sum(item["segments"] for item in split_meta.values())),
        "raw_processed_shape": "(N, 300, 15)",
        "tuple_orientation": "(N, C, T)",
        "channels": 15,
        "sequence_length": args.window_size,
        "window_policy": "standardize full 300-step heartbeat, then retain leading 288 steps to match TarDiff dataloader slicing",
        "normalization": "per-heartbeat, per-channel StandardScaler over full 300-step heartbeat (Medformer normalize_batch_ts)",
        "source_evidence": {
            "tardiff_table7": "PTB: 64,356 samples, 15 channels, length 288",
            "medformer_processed_artifact": "198 subjects, 64,356 heartbeats, (N,300,15)",
            "medformer_ptb_loader": "subject-independent a=0.55, b=0.70 in pinned reproduction code",
        },
        "splits": split_meta,
    }
    with open(output / "preprocess_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Saved protocol metadata: {output / 'preprocess_meta.json'}")


if __name__ == "__main__":
    main()
