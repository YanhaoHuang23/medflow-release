# MedFlow reproducible release

This is a self-contained MedFlow reproduction repository. It includes the
tokenizer, multi-scale conditional flow model, Token Marginal Guidance (TMG),
optional selector classifier, generation code, downstream/fidelity evaluators,
and raw-data preprocessing recipes. It deliberately excludes patient-level
data, checkpoints, TensorBoard logs, historical sweep outputs, and baseline
repositories.

Before making this repository public, the authors must replace the placeholder
`LICENSE` with the licence they are authorized to grant for their own code.

Before every push, run the local privacy/identity audit:

```bash
python scripts/audit_release.py
```

The release has one configuration schema and five explicit stages:

```text
tuple protocol -> data audit -> Stage-1 R-VQ -> Stage-2 conditional flow
               -> TMG sampling -> TimesNet evaluation -> manifests/metrics
```

All entry points invoked by `medflow` are contained in this repository:

- `train_msvq.py`: Stage-1 multi-scale VQ tokenizer;
- `train_msflowdit.py`: Stage-2 class-conditional flow matching;
- `scripts/generate_msdflow_mimic.py`: sampling, TMG, and optional selector;
- `scripts/evaluate_tstr_tsrtr_timesnet.py`: TSTR/TSRTR evaluation;
- `scripts/evaluate_synthetic_mimic_multilevel.py`: DS, PS, C-FID, feature, and nearest-neighbour evaluation;
- `scripts/preprocess_*.py`: MIMIC-III, eICU, APAVA, and PTB tuple construction.

## Install

Clone this repository and install it from its root:

```bash
pip install -e .
```

GPU training requires a PyTorch build compatible with the local CUDA runtime.

## Data contract and privacy

Every dataset is an external directory containing exactly these files:

```text
train_tuple.pkl
val_tuple.pkl
test_tuple.pkl
```

Each pickle stores `(X, y)`, where `X` is `(N, C, T)` or `(N, T, C)` and `y` is a one-dimensional integer label vector. The configured `input_dim` disambiguates the orientation. Do not place protected data, generated cohorts, or checkpoints under version control; `runs/` is ignored by default.

Run a non-mutating audit before training:

```bash
medflow validate configs/paper/mimic_mortality.yaml
medflow prepare configs/paper/mimic_mortality.yaml
```

`prepare` writes `runs/<experiment>/manifests/data_audit.json`, recording split shapes, class counts, finite-value checks, and SHA-256 hashes. It never copies the underlying data.

## Reproduce one configuration

The paper configurations cover all reported tasks:

- `mimic_mortality.yaml` and `mimic_icu_los.yaml`;
- `eicu_mortality.yaml` and `eicu_icu_los.yaml`;
- `apava.yaml` and `ptb.yaml`.

Use the exact configuration for a task:

```bash
medflow train-vq configs/paper/mimic_mortality.yaml --gpu 0
medflow train-flow configs/paper/mimic_mortality.yaml --gpu 0
medflow train-selector configs/paper/mimic_mortality.yaml --gpu 0
medflow generate configs/paper/mimic_mortality.yaml --gpu 0
medflow evaluate configs/paper/mimic_mortality.yaml --gpu 0
```

For command inspection without GPU use, add `--dry-run` before the subcommand:

```bash
medflow --dry-run all configs/paper/apava.yaml
```

Each stage writes a resolved configuration, command, tuple hashes, Python/PyTorch/CUDA environment, and declared randomness scope to `runs/<experiment>/manifests/<stage>/run_manifest.json`. These manifests are for private reproducibility records and are ignored by Git because they contain local data paths; scrub paths before sharing one externally.

## TMG policy is explicit

`sampling.tmg.policy` is a methodological setting, not a class-balance heuristic.

| Dataset family | Configured policy | Generated command |
| --- | --- | --- |
| MIMIC-III and eICU EHR tasks | `minority_only` | `--token-manifold-guidance-labels target` |
| APAVA | `per_class` | `--token-manifold-guidance-labels all` |
| PTB | `per_class` | `--token-manifold-guidance-labels all` |

Thus PTB retains dual-class TMG although its class ratio is 5.72:1. The `target_label` only names the class used by legacy compatibility checks; `per_class` builds and uses a separate bias for every requested class.

## Randomness statement

The paper configurations use one MedFlow training-and-generation seed (`42`) per setting. The standard EHR/APAVA TimesNet results use evaluator seeds `42/43/44`; these repetitions quantify evaluator-training variation only, not independent generator retrainings or independently sampled synthetic cohorts. The manifest preserves this distinction so it cannot be lost in table aggregation.

## Preprocessing recipes

Raw-data preprocessing is not run automatically because MIMIC/eICU/APAVA/PTB access and locations are institution-specific. The complete source recipes are included here:

```bash
python scripts/preprocess_mimic_mortality_reasonable.py --help
python scripts/preprocess_eicu_reasonable.py --help
python scripts/preprocess_apava_reasonable.py --help
python scripts/preprocess_ptb_tardiff_protocol.py --help
```

After preprocessing, copy no data into this directory; point `paths.data_dir` in a private config at the resulting tuple directory and run `medflow validate`.

For the shipped configurations, write tuples under `data/processed/`, for example:

```bash
python scripts/preprocess_mimic_mortality_reasonable.py \
  --mimic-root /path/to/mimic-iii-1.4 \
  --out-dir data/processed/mimic_mortality_reasonable_7of7

python scripts/preprocess_eicu_reasonable.py \
  --raw-root /path/to/eicu-2.0 \
  --tasks mortality icu_los \
  --min-observed-slots 280

python scripts/preprocess_apava_reasonable.py \
  --raw-root /path/to/APAVA

python scripts/preprocess_ptb_tardiff_protocol.py \
  --raw-root /path/to/PTB \
  --out-dir data/processed/ptb_medformer_code_551530_15ch_288 \
  --protocol medformer_code_551530
```

## What this release intentionally excludes

- Historical `scripts/run_*` sweep scripts and abandoned variants.
- `results/`, `output_*`, logs, checkpoints, and PDF/manuscript assets.
- `TarDiff/`, `external_baselines/`, and other third-party baseline source trees.
- Restricted clinical data and raw EEG/ECG files.

These remain outside the repository so the public artifact has one auditable
path per experiment. See `third_party/NOTICE.md` before publishing.
