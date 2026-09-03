from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


class ConfigError(ValueError):
    """Raised when a release configuration is incomplete or inconsistent."""


DEFAULTS: Dict[str, Any] = {
    "run": {"seed": 42},
    "dataset": {"num_classes": 2},
    "model": {
        "vq": {
            "code_dim": 512,
            "nb_code": [128, 512, 512],
            "patch_num": [3, 6, 12],
            "quantizer": "ema_reset_sim",
            "width": 512,
            "depth": 3,
            "down_t": 2,
            "stride_t": 2,
            "dilation_growth_rate": 3,
            "activation": "relu",
        },
        "flow": {
            "backbone": "dit1d",
            "hidden_dim": 512,
            "depth": 6,
            "heads": 8,
            "dropout": 0.1,
            "senior_sampler": "mixup",
            "sampling_mode": "shared_context",
            "cross_scale_conditioning": "none",
            "class_balanced_sampler": True,
            "class_loss_weight": "auto_sqrt",
            "class_specific_output_head": True,
            "label_conditioned_prior": True,
            "train_mixup_prob": 0.5,
            "train_mixup_alpha": 1.0,
            "class_aware_mixup": True,
            "positive_mixup": {"multiplier": 1.0, "alpha": 0.2, "label": 1, "mode": "random"},
        },
    },
    "training": {
        "vq": {"iterations": 60000, "eval_every": 5000, "batch_size": 256, "lr": 0.0002},
        "flow": {"iterations": 60000, "eval_every": 5000, "batch_size": 256, "lr": 0.0002},
    },
    "checkpoints": {"vq": "net_best_mse.pth", "flow": "net_best_ds.pth"},
    "sampling": {
        "batch_size": 256,
        "flow_steps": 30,
        "solver": "euler",
        "temperature": 0.9,
        "tmg": {
            "enabled": True,
            "policy": "minority_only",
            "target_label": 1,
            "weight": 0.2,
            "transition_weight": 0.0,
            "cross_scale_weight": 0.0,
            "mode": "log_odds",
            "background": "not_target",
            "schedule": "constant",
        },
    },
    "evaluation": {
        "timesnet": {
            "enabled": True,
            "seeds": [42, 43, 44],
            "epochs": 40,
            "alphas": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "multilevel": {"enabled": True, "epochs": 8, "max_samples": 4096},
    },
    "posthoc_selector": {
        "enabled": False,
        "type": "influence_utility",
        "labels": "target",
        "candidate_multiplier": 2.0,
        "influence_weight": 0.2,
        "background_weight": 0.25,
        "target_split": "val",
        "normalize": "cosine",
        "max_target": 512,
        "candidate_batch_size": 128,
        "feature_quantile": 0.01,
        "feature_penalty_weight": 0.25,
        "utility_weight": 1.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _require(mapping: Dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"Missing required configuration value: {path}")
        value = value[part]
    return value


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ConfigError("A MedFlow configuration must be a YAML mapping.")
    config = _deep_merge(DEFAULTS, supplied)
    config["_config_path"] = str(path)
    for key in ("run.name", "paths.data_dir", "dataset.window_size", "dataset.input_dim"):
        _require(config, key)
    if int(config["dataset"]["input_dim"]) <= 0 or int(config["dataset"]["window_size"]) <= 0:
        raise ConfigError("dataset.input_dim and dataset.window_size must be positive.")
    vq = config["model"]["vq"]
    if len(vq["nb_code"]) != len(vq["patch_num"]):
        raise ConfigError("model.vq.nb_code and model.vq.patch_num must have the same number of scales.")
    policy = config["sampling"]["tmg"]["policy"]
    if policy not in {"minority_only", "per_class"}:
        raise ConfigError("sampling.tmg.policy must be minority_only or per_class.")
    selector_type = config["posthoc_selector"]["type"]
    if selector_type not in {"influence_utility", "none"}:
        raise ConfigError("This release supports posthoc_selector.type influence_utility or none.")
    return config


def release_root(config: Dict[str, Any]) -> Path:
    # Configurations live in ``<release>/configs/<group>/<experiment>.yaml``.
    return Path(config["_config_path"]).parents[2]


def resolve_path(config: Dict[str, Any], raw_path: Union[str, Path]) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (release_root(config) / path).resolve()


def validate_data_contract(config: Dict[str, Any]) -> List[Path]:
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    expected = [data_dir / f"{split}_tuple.pkl" for split in ("train", "val", "test")]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise ConfigError(f"Missing required tuple split(s): {names}")
    return expected
