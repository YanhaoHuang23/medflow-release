#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "HC",
    1: "AD",
}

MEDFORMER_VAL_SUBJECTS = {15, 16, 19, 20}
MEDFORMER_TEST_SUBJECTS = {1, 2, 17, 18}


def counts(values):
    values = np.asarray(values, dtype=np.int64)
    return {str(int(k)): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def split_subjects_801010(label_rows: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for cls in sorted(np.unique(label_rows[:, 0]).astype(int).tolist()):
        subjects = label_rows[label_rows[:, 0] == cls, 1].astype(int)
        subjects = subjects[rng.permutation(len(subjects))]
        n = len(subjects)
        n_val = max(1, int(round(0.1 * n)))
        n_test = max(1, int(round(0.1 * n)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Class {cls} has too few subjects for 80/10/10 split: {n}")
        split["train"].extend(subjects[:n_train].tolist())
        split["val"].extend(subjects[n_train:n_train + n_val].tolist())
        split["test"].extend(subjects[n_train + n_val:].tolist())
    return {name: sorted(map(int, subjects)) for name, subjects in split.items()}


def split_subjects_medformer_fixed(label_rows: np.ndarray):
    subjects = set(label_rows[:, 1].astype(int).tolist())
    train = subjects - MEDFORMER_VAL_SUBJECTS - MEDFORMER_TEST_SUBJECTS
    if not train or not MEDFORMER_VAL_SUBJECTS <= subjects or not MEDFORMER_TEST_SUBJECTS <= subjects:
        raise ValueError("APAVA fixed split subject IDs do not match the available subjects")
    return {
        "train": sorted(train),
        "val": sorted(MEDFORMER_VAL_SUBJECTS),
        "test": sorted(MEDFORMER_TEST_SUBJECTS),
    }


def load_subject_features(raw_root: Path, subject_id: int) -> np.ndarray:
    path = raw_root / "Feature" / f"feature_{subject_id:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing APAVA feature file: {path}")
    x = np.load(path)
    if x.ndim != 3:
        raise ValueError(f"{path} must be (segments, time, channels), got {x.shape}")
    if x.shape[1:] != (256, 16):
        raise ValueError(f"{path} expected shape (*,256,16), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path} contains NaN or inf")
    return x.astype(np.float32)


def build_split(raw_root: Path, label_map: dict[int, int], subject_ids: list[int], split_name: str):
    xs = []
    ys = []
    rows = []
    for subject_id in subject_ids:
        label = int(label_map[subject_id])
        x_st = load_subject_features(raw_root, subject_id)
        n_segments = x_st.shape[0]
        xs.append(x_st.transpose(0, 2, 1))
        ys.append(np.full(n_segments, label, dtype=np.int64))
        for segment_id in range(n_segments):
            rows.append({
                "subject_id": int(subject_id),
                "segment_id": int(segment_id),
                "label": label,
                "label_name": LABEL_NAMES.get(label, str(label)),
                "split": split_name,
            })
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess APAVA EEG into TarDiff/MSDFlow tuple protocol.")
    parser.add_argument("--raw-root", required=True, help="Directory containing APAVA Feature/ and Label/ files")
    parser.add_argument("--out-dir", default="data/processed/apava_medformer_fixed_16ch_256")
    parser.add_argument("--protocol", choices=["medformer_fixed", "subject_801010"], default="medformer_fixed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    label_path = raw_root / "Label" / "label.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing APAVA label file: {label_path}")
    label_rows = np.load(label_path)[:, :2].astype(int)
    label_map = {int(subject_id): int(label) for label, subject_id in label_rows}
    if args.protocol == "medformer_fixed":
        split_subjects = split_subjects_medformer_fixed(label_rows)
        split_note = (
            "Wang et al. / Medformer fixed APAVA subject-independent split: "
            "validation subjects {15,16,19,20}, test subjects {1,2,17,18}, all remaining subjects for training."
        )
    else:
        split_subjects = split_subjects_801010(label_rows, seed=args.seed)
        split_note = (
            "TarDiff-stated 80/10/10 protocol interpreted as stratified subject-level split; "
            "this is used only to test the paper-text split statement."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = []
    split_summaries = {}
    for split_name, subject_ids in split_subjects.items():
        x, y, ids = build_split(raw_root, label_map, subject_ids, split_name)
        pd.to_pickle((x, y), out_dir / f"{split_name}_tuple.pkl")
        ids.to_csv(out_dir / f"ids_{split_name}.csv", index=False)
        all_ids.append(ids)
        split_summaries[split_name] = {
            "subjects": len(subject_ids),
            "subject_ids": [int(sid) for sid in subject_ids],
            "segments": int(len(y)),
            "class_subject_counts": counts([label_map[sid] for sid in subject_ids]),
            "class_segment_counts": counts(y),
            "tuple_shape": list(x.shape),
        }

    ids_all = pd.concat(all_ids, ignore_index=True)
    ids_all.to_csv(out_dir / "ids_all.csv", index=False)
    meta = {
        "dataset": "APAVA",
        "task": "binary EEG classification",
        "label_mapping": {str(k): v for k, v in LABEL_NAMES.items()},
        "tuple_orientation": "(N, C, T)",
        "channels": 16,
        "sequence_length": 256,
        "protocol": args.protocol,
        "split": split_note,
        "seed": int(args.seed),
        "source_root": str(raw_root),
        "total_subjects": int(len(label_rows)),
        "total_segments": int(len(ids_all)),
        "class_subject_counts": counts(label_rows[:, 0]),
        "class_segment_counts": counts(ids_all["label"].to_numpy()),
        "paper_reference": {
            "TarDiff_Table2_Real_AUPRC": 0.76692,
            "TarDiff_Table2_Real_AUROC": 0.72063,
            "TarDiff_Table2_TarDiff_AUPRC": 0.76519,
            "TarDiff_Table2_TarDiff_AUROC": 0.77097,
        },
        "splits": split_summaries,
    }
    with open(out_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote APAVA protocol to {out_dir}")
    print(json.dumps({
        "protocol": args.protocol,
        "total_segments": meta["total_segments"],
        "class_segment_counts": meta["class_segment_counts"],
        "splits": split_summaries,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
