from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tuple_summary(path: Path, input_dim: int) -> Dict[str, Any]:
    x, y = pd.read_pickle(path)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 3:
        raise ValueError(f"{path} must contain a rank-3 feature tensor, got {x.shape}")
    if x.shape[1] != input_dim and x.shape[2] != input_dim:
        raise ValueError(f"{path} does not contain the configured {input_dim} features: {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"{path} has {len(x)} features but {len(y)} labels")
    labels, counts = np.unique(y.astype(int), return_counts=True)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "class_counts": {str(int(label)): int(count) for label, count in zip(labels, counts)},
    }


def environment() -> Dict[str, Any]:
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "git_revision": git_revision,
    }


def write_json(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)


def write_manifest(
    path: Path,
    config: Dict[str, Any],
    command: Iterable[str],
    data_summaries: List[Dict[str, Any]],
    dry_run: bool,
) -> None:
    write_json(
        path,
        {
            "config": config,
            "command": list(command),
            "data": data_summaries,
            "environment": environment(),
            "dry_run": bool(dry_run),
            "randomness_scope": {
                "generator": "One training and generation seed per configuration unless independently rerun.",
                "evaluator": "TimesNet seeds measure evaluator-training variation, not generator-training variation.",
            },
        },
    )
