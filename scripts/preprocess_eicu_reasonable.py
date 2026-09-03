#!/usr/bin/env python
"""Build documented eICU 24h vital-sign tuple protocols.

The output tuple format follows the existing TarDiff/MSDFlow convention:
``(X, y)`` where ``X`` is ``(N, C, T)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_SPECS = {
    "HR": ("heartrate", 20.0, 250.0),
    "RR": ("respiration", 1.0, 80.0),
    "SpO2": ("sao2", 50.0, 100.0),
}


def load_first_stays(root: Path) -> pd.DataFrame:
    cols = [
        "patientunitstayid",
        "uniquepid",
        "unitvisitnumber",
        "hospitalid",
        "hospitaldischargestatus",
        "unitdischargestatus",
        "unitdischargeoffset",
    ]
    patient = pd.read_csv(root / "patient.csv.gz", usecols=cols)
    patient = patient.dropna(subset=["patientunitstayid", "uniquepid"])
    patient["unitvisitnumber_sort"] = patient["unitvisitnumber"].fillna(10**9)
    patient = (
        patient.sort_values(["uniquepid", "unitvisitnumber_sort", "patientunitstayid"])
        .groupby("uniquepid", as_index=False)
        .first()
        .drop(columns=["unitvisitnumber_sort"])
        .reset_index(drop=True)
    )
    patient["patientunitstayid"] = patient["patientunitstayid"].astype(np.int64)
    patient["icu_los_days"] = patient["unitdischargeoffset"].astype(float) / 1440.0
    return patient


def build_grid(root: Path, cohort: pd.DataFrame, args) -> Tuple[np.ndarray, dict]:
    stay_ids = cohort["patientunitstayid"].to_numpy(dtype=np.int64)
    stay_to_row = {int(stay_id): i for i, stay_id in enumerate(stay_ids)}
    stay_set = set(stay_to_row.keys())
    n = len(stay_ids)
    t = int(args.steps)
    c = len(FEATURE_SPECS)
    grid = np.full((n, t, c), np.nan, dtype=np.float32)
    observed = np.zeros((n, c), dtype=np.int16)

    usecols = ["patientunitstayid", "observationoffset"] + [spec[0] for spec in FEATURE_SPECS.values()]
    chunks = 0
    kept_rows = 0
    for chunk in pd.read_csv(root / "vitalPeriodic.csv.gz", usecols=usecols, chunksize=args.chunksize):
        chunks += 1
        chunk = chunk[chunk["patientunitstayid"].isin(stay_set)]
        chunk = chunk[(chunk["observationoffset"] >= 0) & (chunk["observationoffset"] < args.window_minutes)]
        if chunk.empty:
            continue
        chunk["slot"] = (chunk["observationoffset"] // args.grid_minutes).astype(np.int16)
        chunk = chunk[(chunk["slot"] >= 0) & (chunk["slot"] < t)]
        if chunk.empty:
            continue
        chunk["row"] = chunk["patientunitstayid"].map(stay_to_row).astype(np.int32)
        kept_rows += len(chunk)

        for f_idx, (_feature, (col, lo, hi)) in enumerate(FEATURE_SPECS.items()):
            values = pd.to_numeric(chunk[col], errors="coerce")
            valid = chunk.loc[values.between(lo, hi, inclusive="both"), ["row", "slot"]].copy()
            if valid.empty:
                continue
            valid["value"] = values.loc[valid.index].astype(float)
            # Median aggregation is exact within each chunk. eICU rows for the same stay/offset
            # are normally local, so this avoids materializing all vital rows at once.
            grouped = valid.groupby(["row", "slot"], sort=False)["value"].median().reset_index()
            rows = grouped["row"].to_numpy(dtype=np.int64)
            slots = grouped["slot"].to_numpy(dtype=np.int64)
            grid[rows, slots, f_idx] = grouped["value"].to_numpy(dtype=np.float32)

    for f_idx in range(c):
        observed[:, f_idx] = np.isfinite(grid[:, :, f_idx]).sum(axis=1).astype(np.int16)
    meta = {
        "vital_periodic_chunks": int(chunks),
        "vital_periodic_rows_in_first_24h_for_first_stays": int(kept_rows),
        "duplicate_aggregation": "median per patientunitstayid/5-minute-slot within chunk",
    }
    return grid, observed, meta


def stratified_split(ids: np.ndarray, y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=y[temp_idx])
    return train_idx, val_idx, test_idx


def impute_with_train_median(x_ntc: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_ntc[train_idx], axis=(0, 1)).astype(np.float32)
    out = x_ntc.copy()
    for i in range(out.shape[0]):
        df = pd.DataFrame(out[i])
        df = df.ffill().bfill()
        arr = df.to_numpy(dtype=np.float32)
        missing = ~np.isfinite(arr)
        if missing.any():
            arr[missing] = np.take(med, np.where(missing)[1])
        out[i] = arr
    return out.astype(np.float32), med


def write_task(task: str, out_dir: Path, cohort: pd.DataFrame, x_ntc: np.ndarray, observed: np.ndarray, args, grid_meta: dict):
    if task == "mortality":
        task_cohort = cohort[cohort["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy()
        y_all = (task_cohort["hospitaldischargestatus"].to_numpy() == "Expired").astype(np.int64)
        positive_definition = 'patient.hospitaldischargestatus == "Expired"'
        paper_counts = {"negative": 27892, "positive": 3173}
    elif task == "icu_los":
        task_cohort = cohort[np.isfinite(cohort["icu_los_days"]) & (cohort["icu_los_days"] > 0)].copy()
        y_all = (task_cohort["icu_los_days"].to_numpy() > float(args.los_threshold_days)).astype(np.int64)
        positive_definition = f"patient.unitdischargeoffset / 1440 > {args.los_threshold_days:g} days"
        paper_counts = {"negative": 17859, "positive": 13206}
    else:
        raise ValueError(task)

    original_rows = task_cohort.index.to_numpy(dtype=np.int64)
    x_task = x_ntc[original_rows]
    observed_task = observed[original_rows]
    keep = (observed_task >= int(args.min_observed_slots)).all(axis=1)
    task_cohort = task_cohort.iloc[np.where(keep)[0]].reset_index(drop=True)
    x_task = x_task[keep]
    observed_task = observed_task[keep]
    y_all = y_all[keep]

    train_idx, val_idx, test_idx = stratified_split(task_cohort["uniquepid"].to_numpy(), y_all, args.seed)
    x_task, train_median = impute_with_train_median(x_task, train_idx)

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, idx in split_map.items():
        x_out = np.transpose(x_task[idx], (0, 2, 1)).astype(np.float32)
        y_out = y_all[idx].astype(np.int64)
        pd.to_pickle((x_out, y_out), out_dir / f"{split}_tuple.pkl")
        ids = task_cohort.iloc[idx].copy()
        ids["label"] = y_out
        ids.to_csv(out_dir / f"ids_{split}.csv", index=False)

    ids_all = task_cohort.copy()
    ids_all["label"] = y_all
    for i, feature in enumerate(FEATURE_SPECS):
        ids_all[f"observed_slots_{feature}"] = observed_task[:, i]
    ids_all.to_csv(out_dir / "ids_all.csv", index=False)

    def stats(idx):
        labels = y_all[idx]
        counts = np.bincount(labels, minlength=2)
        return {
            "n": int(len(idx)),
            "negative": int(counts[0]),
            "positive": int(counts[1]),
            "positive_rate": float(counts[1] / max(counts.sum(), 1)),
        }

    meta = {
        "protocol_name": out_dir.name,
        "dataset": "eICU Collaborative Research Database v2.0",
        "task": task,
        "positive_definition": positive_definition,
        "raw_root": str(args.raw_root),
        "unit": "first ICU stay per uniquepid",
        "window_minutes": int(args.window_minutes),
        "grid_minutes": int(args.grid_minutes),
        "steps": int(args.steps),
        "feature_names": list(FEATURE_SPECS.keys()),
        "feature_columns": {name: spec[0] for name, spec in FEATURE_SPECS.items()},
        "feature_ranges": {name.lower(): [float(spec[1]), float(spec[2])] for name, spec in FEATURE_SPECS.items()},
        "coverage_filter": {
            "min_observed_slots_per_feature": int(args.min_observed_slots),
            "total_slots": int(args.steps),
        },
        "imputation": "per-stay forward-fill/back-fill, then train-only feature median",
        "train_feature_median": {name: float(train_median[i]) for i, name in enumerate(FEATURE_SPECS)},
        "split": {"type": "uniquepid-level stratified 80/10/10", "seed": int(args.seed)},
        "split_stats": {split: stats(idx) for split, idx in split_map.items()},
        "all_stats": stats(np.arange(len(y_all))),
        "paper_reference_counts": paper_counts,
        "grid_meta": grid_meta,
        "tuple_format": "(X, y), X shape is (N, 3, 288)",
    }
    with open(out_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Wrote {task}: {out_dir}  N={len(y_all):,}  y1={int(y_all.sum()):,} ({float(y_all.mean()):.4f})")


def main():
    parser = argparse.ArgumentParser(description="Build transparent eICU 3-feature 288-step tuple protocols.")
    parser.add_argument("--raw-root", required=True, help="Directory containing the credentialed eICU files")
    parser.add_argument("--mortality-out-dir", default="data/processed/eicu_mortality_reasonable_3feat_288_cov97")
    parser.add_argument("--icu-los-out-dir", default="data/processed/eicu_icu_los_reasonable_3feat_288_cov97")
    parser.add_argument("--tasks", nargs="+", default=["mortality", "icu_los"], choices=["mortality", "icu_los"])
    parser.add_argument("--window-minutes", type=int, default=1440)
    parser.add_argument("--grid-minutes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--min-observed-slots", type=int, default=280)
    parser.add_argument("--los-threshold-days", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    root = Path(args.raw_root)
    cohort = load_first_stays(root)
    grid, observed, grid_meta = build_grid(root, cohort, args)
    for task in args.tasks:
        out_dir = Path(args.mortality_out_dir if task == "mortality" else args.icu_los_out_dir)
        write_task(task, out_dir, cohort, grid, observed, args, grid_meta)


if __name__ == "__main__":
    main()
