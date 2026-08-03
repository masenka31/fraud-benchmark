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
  `split`, `reported_at`, `campaign_id`, then every source column passed through **except
  the raw label**. Each adapter declares the `source_label_column` its `is_fraud` came
  from and the pipeline drops it, so the frame carries exactly one binary label. A raw
  label survives only where it carries what `is_fraud` loses — amaretto's five-class
  `Anomaly`, saml_d's `Laundering_type` — declared as `label_descriptive_columns` and
  excluded from every model by `experiments/columns.py`.
- `dataset_card.json` — provenance: source, licence, row and fraud counts, time range,
  split strategy and counts, the realised delay parameters, column mapping,
  `dropped_source_label`, `label_descriptive_columns`, caveats.
- occasionally an auxiliary frame, e.g. IEEE-CIS's `unlabelled_test.parquet`.

## 2. Experiments — `src/fraud_benchmark/experiments/`

Everything that happens after a dataset is prepared. Reads `data/processed/`, writes
`data/features/`, and never writes to `data/processed/`. Imports from `data` freely.

Three of the seven registered datasets are run experimentally: **IBM CCF**, **SAML-D**
and **Sparkov**, the last under three label-delay regimes. The other four (paysim,
banksim, ieee_cis, amaretto) are prepared and documented but not modelled, so nothing
in this package mentions them.

### Feature extraction — `experiments/features/`

One module per experimental dataset, and each one names every feature it produces. That
is the point of the layout: reading `features/saml_d.py` is meant to be the whole answer
to what SAML-D's model sees, so no feature is declared in shared code.

| module | does |
|---|---|
| `util.py` | the primitives — `EntityHistory` for past-only aggregates, clock and amount shape, haversine, and `write_features`, which enforces the contract |
| `ibm_ccf.py` | → `data/features/ibm_ccf.parquet`. Grouped MCC, relative geography, decline velocity, card and cardholder attributes |
| `sparkov.py` | → `data/features/sparkov.parquet`. Haversine distance from home, and **both** delay regimes in one file |
| `saml_d.py` | → `data/features/saml_d.parquet`. Two histories — sender and receiver — so fan-out and fan-in are both expressible |

A feature parquet carries `entity_id`, `event_time`, `reported_at`, `is_fraud`, `split`
and nothing else raw: every other column was computed deliberately. Features are
**unfitted** — engineered, but with no vocabulary capped, no ordinal code assigned and
nothing scaled — so one parquet is valid under both the standard split and the Italy
holdout. Columns naming an absolute place or a specific counterparty carry an
`artifact_` prefix, which is how a downstream filter drops the group that lets a model
memorise "Italy = fraud" instead of learning fraud.

Sparkov's three label regimes — no delay, delay, slow delay — are a choice of *column*,
not of file: the parquet carries `reported_at` and `reported_at_slow`, and ignoring both
is the no-delay regime.

### The model side

Where the split is chosen and the fitting happens. Fitting inside a feature build would
tie the parquet to one split, and two are in use.

| module | does |
|---|---|
| `columns.py` | what no model may ever see, whatever the experiment — the label part read from the adapter registry |
| `splits.py` | `standard_split` and `italy_holdout_split` on an already-prepared frame |
| `encoding.py` | categorical encoding and scaling, fitted on train only |
| `models.py` | logistic regression, XGBoost, and the trivial-rule floor |
| `mlp.py` | a 3-layer MLP |
| `metrics.py` | average precision, plus precision/recall/F1 at a validation-chosen threshold |

## 3. Runners — `scripts/`

`scripts/figures/` holds the two plotting scripts that generate the figures embedded in
the documentation — neither is feature engineering or model training, and neither is an
sbatch job.

`scripts/slurm/generate.py` emits **every** sbatch file, so `scripts/slurm/jobs/` can
stay gitignored. It is three feature builds, one per experimental dataset, with their
memory and walltime sized by row count. Add a job by declaring it there, never by
hand-writing a file into that directory.

## Results — `results/`

| path | holds |
|---|---|
| `*.jsonl`, `runs/` | records from the experiments this cleanup retired, kept as a history of what was measured |
| `summary.md` | the leakage-ablation table those records produced |
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
