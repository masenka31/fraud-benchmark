# Repository layout

Three separable pieces. Only the first is the benchmark itself; the second holds the
studies that read its output and never write to it, and the third is how they are launched.

## 1. Dataset preparation — `src/fraud_benchmark/data/`

Driven by the `fraud-benchmark` CLI. Stages, in order: **fetch → canonicalize → validate →
split → campaign → delay → write**, orchestrated by `pipeline.py`.

| module | does |
|---|---|
| `cli.py` | `list`, `prepare`, `info` |
| `pipeline.py` | stage orchestration, and the atomic directory swap on write |
| `sources.py` | fetching from Kaggle or git, with errors a human can act on |
| `adapters/` | one adapter per dataset; `base.py` holds the interface and registry |
| `adapters/files.py` | locating a named file, or reassembling a multi-part zip |
| `schema.py` | the canonical schema every adapter must produce, and its validator |
| `splitting.py` | temporal splits, cut on timestamp values so ties cannot straddle |
| `campaigns.py` | grouping frauds into campaigns (one entity, gap-bounded) |
| `delay.py` | the synthetic `reported_at` timestamp, lognormal per campaign |
| `censoring.py` | reading `reported_at` back: the labels a model at a cutoff would have |
| `config.py` | `configs/default.yaml` plus per-dataset overrides |

`censoring.py` is the one module here that no pipeline stage calls. It lives beside
`delay.py` anyway, because it is the only correct way to consume what `delay.py` writes,
and an experiment that reimplemented it slightly differently would silently measure
something else.

**This package may not import from `experiments`.** Preparation knows nothing about
features or models; the dependency runs one way only.

**An adapter's only job** is turning one dataset's raw files into a canonical frame. It
knows nothing about splitting, label delay, or output formats. Adding a dataset means
writing `to_canonical` and `column_mapping`, and optionally overriding `custom_splits` or
`auxiliary_frames`. Per-dataset documentation lives in [`datasets/`](datasets/README.md).

**Output** per dataset, under `data/processed/<name>/`:

- `data.parquet` — the canonical frame: `event_time`, `entity_id`, `amount`, `is_fraud`,
  `split`, `reported_at`, `campaign_id`, then every source column passed through.
- `dataset_card.json` — provenance: source, licence, row and fraud counts, time range,
  split strategy and counts, the realised delay parameters, column mapping, caveats.
- occasionally an auxiliary frame, e.g. IEEE-CIS's `unlabelled_test.parquet`.

## 2. Experiments — `src/fraud_benchmark/experiments/`

Everything that happens after a dataset is prepared. Reads `data/processed/` and
`data/features/`, writes to neither. Imports from `data` freely.

The shared modeling stack:

| module | does |
|---|---|
| `splits.py` | `standard_split` and `italy_holdout_split` on an already-prepared frame |
| `build_features.py` | causal velocity features for one dataset, cached to `data/features/` |
| `features.py` | the velocity feature definitions themselves |
| `features_v2.py` | 26 history-relative features: novelty, decline velocity, burst ratios |
| `ibm_features.py` | the IBM CCF feature pipeline — grouped MCC, relative geography |
| `seq_window.py` | flattened sequence window: target transaction + its previous 9, same user |
| `mlp.py` | a 3-layer MLP over that window |
| `geo.py` | the transformations that break IBM CCF's geography oracle |
| `encoding.py` | categorical encoding and scaling, fitted on train only |
| `models.py` | logistic regression, XGBoost, and the trivial-rule floor |
| `metrics.py` | average precision, plus precision/recall/F1 at a validation-chosen threshold |

### The leakage ablation — `experiments/ablation/`

One study among the several this stack supports, and the only one with its own
subpackage, because it is a grid rather than a single run. Does a model's score survive
removing the columns the audit flagged as generation artifacts? A 14-cell grid of
(dataset × `leaky`/`clean` × `oracle`/`censored` labels), run in two stages so the
expensive feature build happens once per dataset.

| module | does |
|---|---|
| `grid.py` | which cells exist — one definition, read by the job generator and the summary |
| `columns.py` | which columns a model may see; the two exclusion sets |
| `cell.py` | stage 2: run one cell, appending a record per fit |
| `summarize.py` | `results/runs/*.jsonl` → `results/summary.md` |

Stage 1 is `experiments/build_features.py`, shared with the other experiments.

Read `results/summary.md` first, then
[`verification-notes.md`](verification-notes.md) §"Known leakage".

## 3. Runners — `scripts/`

Argparse and an output path, nothing more: every runner imports its features, splits,
models and metrics from `experiments/`, so no importable logic lives here. The six
experiment runners (`italy_holdout`, `geo_dilution`, `ibm_features_v2`, `seq_window`,
`seq_mlp`, `seq_mlp_v2`) ask why IBM CCF scores 0.041 average precision when a one-line
rule scores 0.764. Every result is tabled in [`experiments.md`](experiments.md).

`scripts/figures/` holds the two plotting scripts that generate the figures embedded in
the documentation — neither is feature engineering or model training, and neither is an
sbatch job.

`scripts/slurm/generate.py` emits **every** sbatch file, the ablation grid and the
experiments alike, so `scripts/slurm/jobs/` can stay gitignored. Add a job by declaring it
there, never by hand-writing a file into that directory.

## Results — `results/`

| path | holds |
|---|---|
| `summary.md` | the ablation table, regenerated by `summarize.py` |
| `runs/*.jsonl` | one file per ablation cell; a shared file would race |
| `runs/retired/` | records for cells no longer in the grid, excluded from the summary |
| `*.jsonl` | one file per experiment in `scripts/` |
| `figures/` | generated plots |
| `logs/` | SLURM stdout and stderr, gitignored |

## Conventions worth knowing before changing anything

- **Average precision, never ROC AUC.** Base rates run from 0.10% to 3.5%; a
  negative-class-dominated metric stays high for a model with no useful precision.
- **Nothing is fitted on val or test** — encoders, scalers and vocabularies included.
- **Every feature window is left-closed** and excludes the row it describes.
- **Splits are cut on timestamp values, not row positions**, so tied timestamps cannot
  straddle a boundary.
- **The pipeline's output is read-only** to everything in `experiments/`.
- **`data/` may not import from `experiments/`.** Tests mirror the split: `tests/data/`
  and `tests/experiments/`.
