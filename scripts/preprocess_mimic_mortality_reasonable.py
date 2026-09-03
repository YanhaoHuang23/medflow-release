#!/usr/bin/env python
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"]

ITEM_TO_FEATURE = {
    211: "hr",
    220045: "hr",
    6: "sbp",
    51: "sbp",
    442: "sbp",
    455: "sbp",
    3313: "sbp",
    6701: "sbp",
    220050: "sbp",
    220179: "sbp",
    225309: "sbp",
    8364: "dbp",
    8368: "dbp",
    8440: "dbp",
    8441: "dbp",
    8555: "dbp",
    220051: "dbp",
    220180: "dbp",
    225310: "dbp",
    52: "map",
    443: "map",
    456: "map",
    3312: "map",
    6702: "map",
    220052: "map",
    220181: "map",
    225312: "map",
    618: "rr",
    220210: "rr",
    224689: "rr",
    224690: "rr",
    676: "temp",
    677: "temp",
    678: "temp",
    679: "temp",
    223761: "temp",
    223762: "temp",
    646: "spo2",
    220277: "spo2",
}

FAHRENHEIT_ITEMS = {678, 679, 223761}

RANGES = {
    "hr": (20.0, 250.0),
    "sbp": (30.0, 300.0),
    "dbp": (10.0, 200.0),
    "map": (20.0, 250.0),
    "rr": (1.0, 80.0),
    "temp": (25.0, 45.0),
    "spo2": (1.0, 100.0),
}


def _path(root, name):
    return str(Path(root) / name)


def _load_cohort(root, cohort_mode, adult_only, los_min_hours):
    stays = pd.read_csv(
        _path(root, "ICUSTAYS.csv.gz"),
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"],
        parse_dates=["INTIME", "OUTTIME"],
    )
    admissions = pd.read_csv(
        _path(root, "ADMISSIONS.csv.gz"),
        usecols=["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"],
    )
    patients = pd.read_csv(
        _path(root, "PATIENTS.csv.gz"),
        usecols=["SUBJECT_ID", "DOB"],
        parse_dates=["DOB"],
    )
    cohort = stays.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    cohort = cohort.merge(patients, on="SUBJECT_ID", how="left")
    cohort = cohort.dropna(subset=["ICUSTAY_ID", "INTIME"])
    cohort = cohort.sort_values(["SUBJECT_ID", "INTIME", "ICUSTAY_ID"]).reset_index(drop=True)
    raw_n = len(cohort)

    cohort["AGE_YEAR_APPROX"] = cohort["INTIME"].dt.year - cohort["DOB"].dt.year
    if adult_only:
        cohort = cohort[cohort["AGE_YEAR_APPROX"] >= 18].copy()
    after_adult_n = len(cohort)

    if los_min_hours > 0:
        los_hours = (cohort["OUTTIME"] - cohort["INTIME"]).dt.total_seconds() / 3600.0
        cohort = cohort[los_hours >= float(los_min_hours)].copy()
    after_los_n = len(cohort)

    if cohort_mode == "first_icu":
        cohort = (
            cohort.sort_values(["SUBJECT_ID", "INTIME", "ICUSTAY_ID"])
            .groupby("SUBJECT_ID", sort=False)
            .head(1)
            .copy()
        )
    elif cohort_mode != "all_icu":
        raise ValueError(f"Unsupported cohort_mode={cohort_mode}")

    cohort = cohort.sort_values("ICUSTAY_ID").reset_index(drop=True)
    cohort["ICUSTAY_ID"] = cohort["ICUSTAY_ID"].astype(np.int64)
    cohort["HOSPITAL_EXPIRE_FLAG"] = cohort["HOSPITAL_EXPIRE_FLAG"].astype(np.int64)
    return cohort, {
        "raw_icu_stays": int(raw_n),
        "after_adult_filter": int(after_adult_n),
        "after_los_filter": int(after_los_n),
        "after_cohort_mode": int(len(cohort)),
    }


def _clean_values(frame):
    vals = frame["VALUENUM"].astype(np.float32).to_numpy()
    itemids = frame["ITEMID"].astype(np.int64).to_numpy()
    temp_f = np.isin(itemids, list(FAHRENHEIT_ITEMS))
    vals[temp_f] = (vals[temp_f] - 32.0) * 5.0 / 9.0
    frame = frame.copy()
    frame["VALUE_CLEAN"] = vals

    keep = np.ones(len(frame), dtype=bool)
    features = frame["FEATURE"].to_numpy()
    for feature, (lo, hi) in RANGES.items():
        feature_keep = features == feature
        keep[feature_keep & ((vals < lo) | (vals > hi))] = False
    return frame.loc[keep]


def _coverage_mask(counts, coverage):
    present = counts.sum(axis=2) > 0
    feature_count = present.sum(axis=1)
    if coverage == "any":
        return feature_count >= 1
    if coverage == "atleast5":
        return feature_count >= 5
    if coverage == "atleast6":
        return feature_count >= 6
    if coverage == "all7":
        return feature_count == 7
    raise ValueError(f"Unsupported coverage={coverage}")


def _stratified_group_split(subject_ids, y, seed):
    rng = np.random.default_rng(seed)
    subject_ids = np.asarray(subject_ids)
    y = np.asarray(y)
    subject_label = {}
    for sid, label in zip(subject_ids, y):
        label = int(label)
        if sid in subject_label and subject_label[sid] != label:
            # HOSPITAL_EXPIRE_FLAG can differ across admissions; keep the subject in the
            # positive stratum if any selected stay is positive.
            subject_label[sid] = max(subject_label[sid], label)
        else:
            subject_label[sid] = label

    train_subjects = []
    val_subjects = []
    test_subjects = []
    for cls in sorted(set(subject_label.values())):
        sids = np.array([sid for sid, label in subject_label.items() if label == cls])
        rng.shuffle(sids)
        n = len(sids)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        train_subjects.append(sids[:n_train])
        val_subjects.append(sids[n_train:n_train + n_val])
        test_subjects.append(sids[n_train + n_val:])

    out = []
    for parts in (train_subjects, val_subjects, test_subjects):
        sids = set(np.concatenate(parts).tolist())
        idx = np.where(np.isin(subject_ids, list(sids)))[0]
        rng.shuffle(idx)
        out.append(idx)
    return out


def _fill_missing_train_median(x, split_indices):
    x = x.copy()
    train_idx = split_indices[0]
    feature_medians = np.nanmedian(x[train_idx], axis=(0, 2))
    feature_medians = np.nan_to_num(feature_medians, nan=0.0).astype(np.float32)
    for i in range(x.shape[0]):
        for c in range(x.shape[1]):
            s = pd.Series(x[i, c])
            s = s.ffill().bfill().fillna(float(feature_medians[c]))
            x[i, c] = s.to_numpy(dtype=np.float32)
    return x.astype(np.float32), feature_medians


def _write_ids(path, cohort, idx):
    cols = [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "INTIME",
        "OUTTIME",
        "LOS",
        "AGE_YEAR_APPROX",
        "HOSPITAL_EXPIRE_FLAG",
    ]
    optional_cols = ["ICU_LOS_GT_THRESHOLD"]
    cols = [c for c in cols + optional_cols if c in cohort.columns]
    frame = cohort.iloc[idx][cols].copy()
    frame.to_csv(path, index=False)


def _label_counts(y):
    counts = Counter(np.asarray(y).astype(int).tolist())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _build_labels(selected, args):
    if args.label_task == "mortality":
        y = selected["HOSPITAL_EXPIRE_FLAG"].to_numpy(dtype=np.int64)
        return y, {
            "label_task": "mortality",
            "label_column": "HOSPITAL_EXPIRE_FLAG",
            "positive_definition": "ADMISSIONS.HOSPITAL_EXPIRE_FLAG == 1",
            "paper_reference_counts": {
                "TarDiff_MIMIC_mortality_positive": 1680,
                "TarDiff_MIMIC_mortality_negative": 19240,
            },
        }
    if args.label_task == "icu_los":
        threshold = float(args.los_threshold_days)
        y = (selected["LOS"].astype(float).to_numpy() > threshold).astype(np.int64)
        selected["ICU_LOS_GT_THRESHOLD"] = y
        return y, {
            "label_task": "icu_los",
            "label_column": "ICUSTAYS.LOS",
            "los_threshold_days": threshold,
            "positive_definition": f"ICUSTAYS.LOS > {threshold:g} days",
            "paper_reference_counts": {
                "TarDiff_MIMIC_ICUStay_positive": 2869,
                "TarDiff_MIMIC_ICUStay_negative": 18051,
            },
        }
    raise ValueError(f"Unsupported label_task={args.label_task}")


def preprocess(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, cohort_counts = _load_cohort(
        args.mimic_root,
        cohort_mode=args.cohort,
        adult_only=args.adult_only,
        los_min_hours=args.los_min_hours,
    )
    stay_to_row = {stay_id: i for i, stay_id in enumerate(cohort["ICUSTAY_ID"].to_numpy())}
    intime_ns = cohort["INTIME"].astype("int64").to_numpy()
    n = len(cohort)

    sums = np.zeros((n, len(FEATURES), args.window), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.float32)

    itemids = set(ITEM_TO_FEATURE)
    feature_to_idx = {name: i for i, name in enumerate(FEATURES)}
    usecols = ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
    start_time = time.time()
    chunks_seen = 0
    rows_seen = 0
    matched_rows = 0
    kept_rows = 0
    print(
        f"Loaded cohort candidates: {n:,} ICU stays after cohort filters. "
        f"Scanning CHARTEVENTS with chunksize={args.chunksize:,}...",
        flush=True,
    )
    for chunk in pd.read_csv(
        _path(args.mimic_root, "CHARTEVENTS.csv.gz"),
        usecols=usecols,
        chunksize=args.chunksize,
        parse_dates=["CHARTTIME"],
        low_memory=False,
    ):
        chunks_seen += 1
        rows_seen += len(chunk)
        chunk = chunk.dropna(subset=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
        if chunk.empty:
            continue
        chunk["ITEMID"] = chunk["ITEMID"].astype(np.int64)
        chunk = chunk[chunk["ITEMID"].isin(itemids)]
        matched_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype(np.int64)
        row_idx = chunk["ICUSTAY_ID"].map(stay_to_row)
        chunk = chunk[row_idx.notna()].copy()
        if chunk.empty:
            continue
        row_idx = row_idx[row_idx.notna()].astype(np.int64).to_numpy()
        chart_ns = chunk["CHARTTIME"].astype("int64").to_numpy()
        hours = ((chart_ns - intime_ns[row_idx]) // (3600 * 10**9)).astype(np.int64)
        keep = (hours >= 0) & (hours < args.window)
        if not np.any(keep):
            continue
        chunk = chunk.loc[keep].copy()
        row_idx = row_idx[keep]
        hours = hours[keep]
        chunk["ROW_IDX"] = row_idx
        chunk["HOUR"] = hours
        chunk["FEATURE"] = chunk["ITEMID"].map(ITEM_TO_FEATURE)
        chunk = _clean_values(chunk)
        if chunk.empty:
            continue
        kept_rows += len(chunk)
        feature_idx = chunk["FEATURE"].map(feature_to_idx).to_numpy()
        values = chunk["VALUE_CLEAN"].astype(np.float32).to_numpy()
        row_idx = chunk["ROW_IDX"].astype(np.int64).to_numpy()
        hours = chunk["HOUR"].astype(np.int64).to_numpy()
        np.add.at(sums, (row_idx, feature_idx, hours), values)
        np.add.at(counts, (row_idx, feature_idx, hours), 1.0)
        if chunks_seen % args.log_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[chunk {chunks_seen}] scanned={rows_seen:,}, matched_vitals={matched_rows:,}, "
                f"kept_window_rows={kept_rows:,}, elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    with np.errstate(invalid="ignore", divide="ignore"):
        x_raw = sums / counts
    valid = _coverage_mask(counts, args.coverage)
    x_raw = x_raw[valid].astype(np.float32)
    mask = (counts[valid] > 0).astype(np.uint8)
    selected = cohort.loc[valid].reset_index(drop=True)
    y, label_meta = _build_labels(selected, args)
    subject_ids = selected["SUBJECT_ID"].to_numpy(dtype=np.int64)

    train_idx, val_idx, test_idx = _stratified_group_split(subject_ids, y, args.seed)
    split_indices = (train_idx, val_idx, test_idx)
    x, feature_medians = _fill_missing_train_median(x_raw, split_indices)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split, idx in splits.items():
        pd.to_pickle((x[idx], y[idx]), out_dir / f"{split}_tuple.pkl")
        np.save(out_dir / f"{split}_mask.npy", mask[idx])
        _write_ids(out_dir / f"ids_{split}.csv", selected, idx)
    _write_ids(out_dir / "ids_all.csv", selected, np.arange(len(selected)))

    present = mask.sum(axis=2) > 0
    coverage_hist = Counter(present.sum(axis=1).astype(int).tolist())
    meta = {
        "protocol_name": args.protocol_name,
        "mimic_root": str(args.mimic_root),
        "features": FEATURES,
        "feature_ranges": RANGES,
        "shape_all": list(x.shape),
        "mask_shape_all": list(mask.shape),
        **label_meta,
        "label_counts_all": _label_counts(y),
        "positive_rate_all": float(np.mean(y)) if len(y) else 0.0,
        "cohort": args.cohort,
        "anchor": "ICUSTAYS.INTIME",
        "window_hours": int(args.window),
        "coverage": args.coverage,
        "coverage_hist_num_features_present": {str(k): int(v) for k, v in sorted(coverage_hist.items())},
        "adult_only": bool(args.adult_only),
        "los_min_hours": float(args.los_min_hours),
        "split": "subject-level stratified 80/10/10",
        "seed": int(args.seed),
        "cohort_counts": cohort_counts,
        "chartevents_rows_seen": int(rows_seen),
        "chartevents_matched_vital_rows": int(matched_rows),
        "chartevents_kept_window_rows_after_cleaning": int(kept_rows),
        "imputation": "hourly mean, then per-sample ffill/bfill, then train-only feature median",
        "train_feature_medians": {FEATURES[i]: float(feature_medians[i]) for i in range(len(FEATURES))},
        "splits": {
            split: {
                "n": int(len(idx)),
                "label_counts": _label_counts(y[idx]),
                "positive_rate": float(np.mean(y[idx])) if len(idx) else 0.0,
                "unique_subjects": int(len(np.unique(subject_ids[idx]))),
            }
            for split, idx in splits.items()
        },
    }
    with open(out_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(
        description="Build a documented MIMIC-III ICU first-24h vital-sign protocol."
    )
    parser.add_argument("--mimic-root", required=True, help="Directory containing the credentialed MIMIC-III files")
    parser.add_argument("--out-dir", default="data/processed/mimic_mortality_reasonable_7of7")
    parser.add_argument("--protocol-name", default="mimic_mortality_reasonable_7of7")
    parser.add_argument("--label-task", choices=["mortality", "icu_los"], default="mortality")
    parser.add_argument("--los-threshold-days", type=float, default=3.0)
    parser.add_argument("--cohort", choices=["first_icu", "all_icu"], default="first_icu")
    parser.add_argument("--coverage", choices=["any", "atleast5", "atleast6", "all7"], default="all7")
    parser.add_argument("--adult-only", action="store_true")
    parser.add_argument("--los-min-hours", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
