# Repository layout

Three separable pieces. Only the first is the benchmark itself; the second holds the
studies that read its output and never write to it, and the third is how they are launched.

## 1. Dataset preparation — `src/fraud_benchmark/data/`

Driven by the `fraud-benchmark` CLI. Stages, in order: **fetch → canonicalize → validate →
split → campaign → delay → write**, orchestrated by `pipeline.py`.

| module | does |
|---|---|
| `cli.py` | the `fraud-benchmark` console script: `list`, `download`, `prepare`, `info` |
| `pipeline.py` | stage orchestration — `download` alone, or `prepare` end to end — and the atomic directory swap on write |
| `selection.py` | what `--all` and `--exclude-noncommercial` mean, and that one failed dataset never stops the rest. Shared by both front doors so they cannot drift |
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

### Running the paper protocols — `experiment.py`, `splits.py`, `causal_encoding.py`

Where the split is chosen and everything is fitted. Fitting inside a feature build would
tie the parquet to one split, and two are in use.

| module | does |
|---|---|
| `experiment.py` | paper-protocol preparation: parquet → artifact exclusion → label regime → split → train-fitted design matrix |
| `splits.py` | temporal, transaction-IID, customer-IID, and shared pre-Italy split assignment on an already-prepared frame |
| `causal_encoding.py` | chronological label-rate features that use training labels only and score before updating |
| `encoding.py` | categorical vocabularies and scaling, fitted on train only |
| `models.py` | the fixed XGBoost parameters and fit primitive shared by both protocols |
| `metrics.py` | average precision, plus precision/recall/F1 at a validation-chosen threshold |
| `columns.py` | what no model may ever see, whatever the experiment — the label part read from the adapter registry |

Each runner receives the same prepared matrix, labels, and split indices for every model
seed. The protocol fixes every comparison axis except the one named by its table.

## 3. Runners — `scripts/`

**Argparse and an output path, nothing else.** Every script here is a CLI over a module
in the installed package, so nothing importable lives outside `src/` and a test never
has to reach into `scripts/` to get at a decision.

The preparation stages are the first three rows, in order. Each takes `--dataset` or
`--all`. The final two runners reproduce the public paper experiments.

| script | over | does |
|---|---|---|
| `download.py` | `data.pipeline.download` | **1.** raw files → `data/raw/<name>/`, and stops |
| `prepare.py` | `data.pipeline.prepare` | **2.** raw files → `data/processed/<name>/` on the shared schema |
| `features.py` | `experiments.features` | **3.** a prepared dataset → `data/features/<name>.parquet` |
| `ibm_split_protocol.py` | `experiments` | pre-Italy temporal/IID comparison → three JSON records and one readable table |
| `sparkov_delay_protocol.py` | `experiments` | synthetic training-label-delay comparison → JSONL records and one readable table |
| `figures/plot_monthly_fraud.py` | — | IBM CCF's regime shift, from its feature parquet |
| `figures/plot_dataset_caveats.py` | — | the three dataset caveats that are a shape rather than a number |

**Stages 1 and 2 are reachable two ways**, and that is deliberate: `fraud-benchmark
download|prepare` serves someone who wants prepared datasets, these scripts serve
someone working on the study. Neither owns the behaviour — both call the same function
in `data/pipeline.py` and take their `--all` semantics from `data/selection.py`, and
`tests/test_scripts.py` asserts the identity of what each door calls rather than trusting
the two to stay similar. Paper protocols have no console command: they need the `dev`
extras, so adding them would make `fraud-benchmark list` fail on a base install.

The public figure scripts share one palette, `fraud_benchmark.figures`, so the figures
read as one system and a colour is defined once. Outputs land in `results/figures/`.

## Results — `results/`

| path | holds |
|---|---|
| `paper.md` | index of the two public paper experiments and their records |
| `paper/*.json` | the three IBM pre-Italy split records |
| `paper/sparkov_delay_*.json` | the three Sparkov training-label-delay records |
| `ibm_split_protocol.md` | readable IBM paper table |
| `sparkov_label_delay.md` | readable Sparkov paper table |
| `figures/` | generated plots, light and dark per figure |

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
