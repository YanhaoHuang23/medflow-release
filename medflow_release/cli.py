from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import ConfigError, load_config, resolve_path, validate_data_contract
from .provenance import tuple_summary, write_json, write_manifest


def _implementation_root(user_value: Optional[str]) -> Path:
    if user_value:
        root = Path(user_value).resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    required = [root / "train_msvq.py", root / "train_msflowdit.py", root / "scripts" / "generate_msdflow_mimic.py"]
    if not all(path.is_file() for path in required):
        raise ConfigError(
            "Cannot locate the bundled MedFlow implementation. Set --implementation-root to a directory containing train_msvq.py."
        )
    return root


def _paths(config: Dict[str, Any]) -> Tuple[Path, Path, Path, Path]:
    root = resolve_path(config, config["paths"].get("artifact_root", "runs"))
    vq = root / "vq" / config["checkpoints"]["vq"]
    flow = root / "flow" / config["checkpoints"]["flow"]
    synthetic = root / "synthetic" / "medflow_tmg.pkl"
    return root, vq, flow, synthetic


def _selector_checkpoint(config: Dict[str, Any]) -> Path:
    root, _, _, _ = _paths(config)
    return root / "selector" / "best_model.pt"


def _common_dataset_args(config: Dict[str, Any]) -> List[str]:
    data = config["dataset"]
    args = [
        "--dataname", "mimic_icustay",
        "--data-path", str(resolve_path(config, config["paths"]["data_dir"])),
        "--input-dim", str(data["input_dim"]),
        "--window-size", str(data["window_size"]),
    ]
    if data.get("feature_names"):
        args.extend(["--feature-names", ",".join(data["feature_names"])])
    return args


def _generator_dataset_args(config: Dict[str, Any]) -> List[str]:
    """The generator consumes tuple paths directly and has no --dataname option."""
    data = config["dataset"]
    args = [
        "--data-dir", str(resolve_path(config, config["paths"]["data_dir"])),
        "--input-dim", str(data["input_dim"]),
        "--window-size", str(data["window_size"]),
    ]
    if data.get("feature_names"):
        args.extend(["--feature-names", ",".join(data["feature_names"])])
    return args


def _model_args(config: Dict[str, Any]) -> List[str]:
    vq = config["model"]["vq"]
    return [
        "--code-dim", str(vq["code_dim"]),
        "--nb-code", *map(str, vq["nb_code"]),
        "--patch-num", *map(str, vq["patch_num"]),
        "--quantizer", str(vq["quantizer"]),
        "--width", str(vq["width"]),
        "--depth", str(vq["depth"]),
        "--down-t", str(vq["down_t"]),
        "--stride-t", str(vq["stride_t"]),
        "--dilation-growth-rate", str(vq["dilation_growth_rate"]),
        "--vq-act", str(vq["activation"]),
    ]


def _tmg_args(config: Dict[str, Any]) -> List[str]:
    tmg = config["sampling"]["tmg"]
    if not tmg["enabled"]:
        return []
    labels = "target" if tmg["policy"] == "minority_only" else "all"
    return [
        "--token-manifold-guidance-weight", str(tmg["weight"]),
        "--token-manifold-guidance-target-label", str(tmg["target_label"]),
        "--token-manifold-guidance-labels", labels,
        "--token-manifold-guidance-mode", str(tmg["mode"]),
        "--token-manifold-guidance-background", str(tmg["background"]),
        "--token-manifold-guidance-schedule", str(tmg["schedule"]),
        "--token-manifold-transition-weight", str(tmg["transition_weight"]),
        "--token-manifold-cross-scale-weight", str(tmg["cross_scale_weight"]),
    ]


def _selector_args(config: Dict[str, Any]) -> List[str]:
    selector = config["posthoc_selector"]
    if not selector["enabled"]:
        return []
    return [
        "--posthoc-selector", str(selector["type"]),
        "--posthoc-selector-labels", str(selector["labels"]),
        "--posthoc-classifier-ckpt", str(_selector_checkpoint(config)),
        "--posthoc-candidate-multiplier", str(selector["candidate_multiplier"]),
        "--posthoc-influence-weight", str(selector["influence_weight"]),
        "--posthoc-influence-background-weight", str(selector["background_weight"]),
        "--posthoc-influence-target-split", str(selector["target_split"]),
        "--posthoc-influence-normalize", str(selector["normalize"]),
        "--posthoc-influence-max-target", str(selector["max_target"]),
        "--posthoc-influence-candidate-batch-size", str(selector["candidate_batch_size"]),
        "--posthoc-feature-quantile", str(selector["feature_quantile"]),
        "--posthoc-feature-penalty-weight", str(selector["feature_penalty_weight"]),
        "--posthoc-utility-weight", str(selector["utility_weight"]),
    ]


def _run_dir(config: Dict[str, Any], stage: str) -> Path:
    artifact_root, _, _, _ = _paths(config)
    return artifact_root / "manifests" / stage


def _execute(command: List[str], *, implementation_root: Path, gpu: str, dry_run: bool) -> None:
    printable = " ".join(command)
    print(printable)
    if dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, cwd=str(implementation_root), env=env, check=True)


def _record(config: Dict[str, Any], command: List[str], stage: str, dry_run: bool) -> Path:
    splits = validate_data_contract(config)
    summaries = [tuple_summary(path, int(config["dataset"]["input_dim"])) for path in splits]
    manifest = _run_dir(config, stage) / "run_manifest.json"
    write_manifest(manifest, config, command, summaries, dry_run)
    return manifest


def command_prepare(config: Dict[str, Any], dry_run: bool) -> None:
    splits = validate_data_contract(config)
    report = {
        "dataset": config["run"]["name"],
        "splits": [tuple_summary(path, int(config["dataset"]["input_dim"])) for path in splits],
        "privacy_note": "The release does not copy protected raw data or tuple data. Data remain outside this package.",
    }
    root, _, _, _ = _paths(config)
    path = root / "manifests" / "data_audit.json"
    if not dry_run:
        write_json(path, report)
    print(json.dumps(report, indent=2))


def command_validate(config: Dict[str, Any]) -> None:
    splits = validate_data_contract(config)
    report = {
        "status": "valid",
        "name": config["run"]["name"],
        "tmg_policy": config["sampling"]["tmg"]["policy"],
        "tuple_splits": [str(path) for path in splits],
    }
    print(json.dumps(report, indent=2))


def command_train_vq(config: Dict[str, Any], implementation_root: Path, gpu: str, dry_run: bool) -> None:
    root, _, _, _ = _paths(config)
    train = config["training"]["vq"]
    command = [sys.executable, "train_msvq.py", *_common_dataset_args(config), *_model_args(config),
               "--batch-size", str(train["batch_size"]), "--total-iter", str(train["iterations"]),
               "--eval-iter", str(train["eval_every"]), "--lr", str(train["lr"]),
               "--seed", str(config["run"]["seed"]), "--out-dir", str(root), "--exp-name", "vq", "--gpu", "0"]
    _record(config, command, "train_vq", dry_run)
    _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)


def command_train_flow(config: Dict[str, Any], implementation_root: Path, gpu: str, dry_run: bool) -> None:
    root, vq_ckpt, _, _ = _paths(config)
    if not dry_run and not vq_ckpt.is_file():
        raise ConfigError(f"Missing VQ checkpoint: {vq_ckpt}. Run train-vq first or set paths.artifact_root correctly.")
    train, flow, sampling = config["training"]["flow"], config["model"]["flow"], config["sampling"]
    pos_mixup = flow["positive_mixup"]
    command = [sys.executable, "train_msflowdit.py", *_common_dataset_args(config), *_model_args(config),
               "--conditional-flow", "--num-classes", str(config["dataset"]["num_classes"]),
               "--resume-pth", str(vq_ckpt), "--fm-backbone", str(flow["backbone"]),
               "--fm-hidden-dim", str(flow["hidden_dim"]), "--fm-depth", str(flow["depth"]),
               "--fm-heads", str(flow["heads"]), "--fm-dropout", str(flow["dropout"]),
               "--senior-sampler", str(flow["senior_sampler"]), "--sampling-mode", str(flow["sampling_mode"]),
               "--cross-scale-conditioning", str(flow["cross_scale_conditioning"]),
               "--train-mixup-prob", str(flow["train_mixup_prob"]), "--train-mixup-alpha", str(flow["train_mixup_alpha"]),
               "--positive-mixup-multiplier", str(pos_mixup["multiplier"]),
               "--positive-mixup-alpha", str(pos_mixup["alpha"]),
               "--positive-mixup-label", str(pos_mixup["label"]),
               "--positive-mixup-mode", str(pos_mixup["mode"]),
               "--batch-size", str(train["batch_size"]), "--total-iter", str(train["iterations"]),
               "--eval-iter", str(train["eval_every"]), "--lr", str(train["lr"]),
               "--seed", str(config["run"]["seed"]), "--out-dir", str(root), "--exp-name", "flow", "--gpu", "0"]
    if flow["class_balanced_sampler"]:
        command.append("--class-balanced-sampler")
    if flow["class_aware_mixup"]:
        command.append("--class-aware-mixup")
    if flow["class_specific_output_head"]:
        command.append("--class-specific-output-head")
    command.extend(["--class-loss-weight", str(flow["class_loss_weight"])])
    _record(config, command, "train_flow", dry_run)
    _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)


def command_train_selector(config: Dict[str, Any], implementation_root: Path, gpu: str, dry_run: bool) -> None:
    selector = config["posthoc_selector"]
    if not selector["enabled"]:
        print("Post-hoc selector is disabled by this configuration.")
        return
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    command = [
        sys.executable, "-m", "medflow_release.classifier.classifier_train",
        "--train_data", str(data_dir / "train_tuple.pkl"),
        "--val_data", str(data_dir / "val_tuple.pkl"),
        "--input_dim", str(config["dataset"]["input_dim"]),
        "--num_classes", "1",
        "--batch_size", str(config["sampling"]["batch_size"]),
        "--epochs", "40",
        "--seed", str(config["run"]["seed"]),
        "--ckpt_dir", str(_selector_checkpoint(config).parent),
    ]
    _record(config, command, "train_selector", dry_run)
    _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)


def command_generate(config: Dict[str, Any], implementation_root: Path, gpu: str, dry_run: bool) -> None:
    _, vq_ckpt, flow_ckpt, synthetic = _paths(config)
    if not dry_run:
        for name, path in (("VQ", vq_ckpt), ("Flow", flow_ckpt)):
            if not path.is_file():
                raise ConfigError(f"Missing {name} checkpoint: {path}. Run the preceding stage first.")
        if config["posthoc_selector"]["enabled"] and not _selector_checkpoint(config).is_file():
            raise ConfigError("Missing selector checkpoint. Run train-selector before generate.")
    flow, sampling = config["model"]["flow"], config["sampling"]
    pos_mixup = flow["positive_mixup"]
    command = [sys.executable, "scripts/generate_msdflow_mimic.py", "--mode", "conditional",
               *_generator_dataset_args(config), *_model_args(config), "--vq", str(vq_ckpt), "--flow", str(flow_ckpt),
               "--num-classes", str(config["dataset"]["num_classes"]), "--class-specific-output-head",
               "--label-conditioned-prior", "--fm-backbone", str(flow["backbone"]),
               "--fm-hidden-dim", str(flow["hidden_dim"]), "--fm-depth", str(flow["depth"]),
               "--fm-heads", str(flow["heads"]), "--fm-dropout", str(flow["dropout"]),
               "--senior-sampler", str(flow["senior_sampler"]), "--sampling-mode", str(flow["sampling_mode"]),
               "--cross-scale-conditioning", str(flow["cross_scale_conditioning"]),
               "--positive-mixup-multiplier", str(pos_mixup["multiplier"]),
               "--positive-mixup-label", str(pos_mixup["label"]),
               "--flow-steps", str(sampling["flow_steps"]), "--solver", str(sampling["solver"]),
               "--sample-temperature", str(sampling["temperature"]), "--batch-size", str(sampling["batch_size"]),
               "--seed", str(config["run"]["seed"]), "--out", str(synthetic), "--gpu", "0",
               *_tmg_args(config), *_selector_args(config)]
    _record(config, command, "generate", dry_run)
    _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)


def command_evaluate(config: Dict[str, Any], implementation_root: Path, gpu: str, dry_run: bool) -> None:
    root, _, _, synthetic = _paths(config)
    if not dry_run and not synthetic.is_file():
        raise ConfigError(f"Missing synthetic cohort: {synthetic}. Run generate first.")
    evaluator = config["evaluation"]["timesnet"]
    if not evaluator["enabled"]:
        print("TimesNet evaluation is disabled by this configuration.")
        return
    command = [sys.executable, "scripts/evaluate_tstr_tsrtr_timesnet.py", "--data-dir",
               str(resolve_path(config, config["paths"]["data_dir"])), "--input-dim", str(config["dataset"]["input_dim"]),
               "--num-classes", str(config["dataset"]["num_classes"]), "--out", str(root / "metrics" / "timesnet.csv"),
               "--epochs", str(evaluator["epochs"]), "--seeds", *map(str, evaluator["seeds"]),
               "--alphas", *map(str, evaluator["alphas"]), "--synthetic", f"medflow={synthetic}"]
    _record(config, command, "evaluate", dry_run)
    _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)
    multilevel = config["evaluation"]["multilevel"]
    if multilevel["enabled"]:
        command = [
            sys.executable, "scripts/evaluate_synthetic_mimic_multilevel.py",
            "--data-dir", str(resolve_path(config, config["paths"]["data_dir"])),
            "--input-dim", str(config["dataset"]["input_dim"]),
            "--feature-names", ",".join(config["dataset"].get("feature_names", [])),
            "--out", str(root / "metrics" / "multilevel.csv"),
            "--epochs", str(multilevel["epochs"]),
            "--max-samples", str(multilevel["max_samples"]),
            "--seed", str(config["run"]["seed"]),
            "--synthetic", "medflow={}".format(synthetic),
        ]
        _record(config, command, "evaluate_multilevel", dry_run)
        _execute(command, implementation_root=implementation_root, gpu=gpu, dry_run=dry_run)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clean, configuration-driven MedFlow release runner.")
    parser.add_argument("--implementation-root", help="Optional override for the bundled MedFlow implementation directory.")
    parser.add_argument("--gpu", default="0", help="Physical GPU id exposed to one legacy process.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs, write a manifest, and print the command without training.")
    parser.add_argument("command", choices=["validate", "prepare", "train-vq", "train-flow", "train-selector", "generate", "evaluate", "all"])
    parser.add_argument("config", help="YAML experiment configuration.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
        implementation_root = _implementation_root(args.implementation_root)
        handlers = {
            "prepare": command_prepare,
            "validate": command_validate,
            "train-vq": command_train_vq,
            "train-flow": command_train_flow,
            "train-selector": command_train_selector,
            "generate": command_generate,
            "evaluate": command_evaluate,
        }
        if args.command == "all":
            command_prepare(config, args.dry_run)
            command_train_vq(config, implementation_root, args.gpu, args.dry_run)
            command_train_flow(config, implementation_root, args.gpu, args.dry_run)
            command_train_selector(config, implementation_root, args.gpu, args.dry_run)
            command_generate(config, implementation_root, args.gpu, args.dry_run)
            command_evaluate(config, implementation_root, args.gpu, args.dry_run)
        elif args.command == "prepare":
            command_prepare(config, args.dry_run)
        elif args.command == "validate":
            command_validate(config)
        else:
            handlers[args.command](config, implementation_root, args.gpu, args.dry_run)
    except ConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
