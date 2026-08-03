# fraud-benchmark

Downloads public fraud and AML transaction datasets, normalizes them to a shared schema,
assigns temporal train/validation/test splits, and attaches a synthetic label-availability
("fraud reported at") timestamp to fraudulent transactions — so that an experiment can
model the labels a detector would actually have had at training time, rather than the ones
that exist in hindsight.

No data is redistributed. Everything is fetched at run time with your own credentials.

**Documentation index: [`docs/README.md`](docs/README.md).**

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Kaggle credentials are required before anything can be downloaded, and IEEE-CIS
additionally needs its competition rules accepted once in a browser —
**[`docs/kaggle-setup.md`](docs/kaggle-setup.md)**.

```bash
fraud-benchmark list                # registered datasets and their licences
fraud-benchmark prepare paysim      # download, normalize, split, delay, write
fraud-benchmark info paysim         # print the prepared dataset's card
```

Output lands in `data/processed/<name>/` as `data.parquet` plus a `dataset_card.json`
recording provenance, counts, split strategy, delay parameters and caveats.

## Datasets

Eight registered: seven distinct sources plus `sparkov_slow`, which is `sparkov` under a
second delay regime. 73 million transactions in total.

| dataset | rows | fraud rate | span | licence |
|---|---:|---:|---:|---|
| [`paysim`](docs/datasets/paysim.md) | 6,362,620 | 0.129% | 30d | CC BY-SA 4.0 |
| [`banksim`](docs/datasets/banksim.md) | 594,643 | 1.211% | 179d | CC BY-NC-SA 4.0 |
| [`sparkov`](docs/datasets/sparkov.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`sparkov_slow`](docs/datasets/sparkov_slow.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`saml_d`](docs/datasets/saml_d.md) | 9,504,852 | 0.104% | 320d | CC BY-NC-SA 4.0 |
| [`ibm_ccf`](docs/datasets/ibm_ccf.md) | 24,386,900 | 0.122% | 10,649d | Apache-2.0 ⚠ |
| [`ieee_cis`](docs/datasets/ieee_cis.md) | 590,540 | 3.499% | 181d | competition rules |
| [`amaretto`](docs/datasets/amaretto.md) | 29,704,090 | 0.274% | 83d | MIT |

⚠ **Read a dataset's page before using it.** Several carry artifacts sharp enough to
invalidate a result — `ibm_ccf` most of all, where validation frauds are 100% a single
merchant country and a one-line rule outscores every model. Each page gives the numbers,
the schema mapping, and the disclaimers: **[`docs/datasets/`](docs/datasets/README.md)**.

Six of the eight are fully synthetic; IEEE-CIS is the only real-world data.

## Time splits

Temporal, 80/10/10 by default, and **cut on timestamp values rather than row positions** —
datasets with coarse time resolution (PaySim's hours, BankSim's days) have heavy ties, and
slicing through a tied block would leak same-instant information across the boundary. A
consequence: the realised ratios drift slightly from 80/10/10 when ties are large.

`sparkov` and `sparkov_slow` are the exception. They keep their upstream test file as the
test split so results stay comparable with published work, with validation carved from the
train tail — roughly 63/7/30.

Every split boundary and count is recorded in the dataset card.

## Label delay

Real detectors learn from labels that arrive late. To model that, fraudulent transactions
are grouped into **campaigns** — one entity, each fraud within a configurable gap of the
last — and one delay is drawn per campaign from a lognormal in days, measured from the
campaign's **last** transaction. Every fraud in the campaign gets that same `reported_at`.

The number that matters is how many **train** frauds are still unreported at the train
cutoff. Those rows are not missing from training data; they sit in it looking legitimate.

| dataset | delay median | train labels censored |
|---|---:|---:|
| `saml_d` | 30d (AML) | 20.0% |
| `amaretto` | 7d | 17.5% |
| `paysim` | 1d | 10.1% |
| `sparkov_slow` | 15d, σ 1.665 | 8.9% |
| `banksim` | 7d | 8.1% |
| `ieee_cis` | 7d | 7.1% |
| `sparkov` | 7d | 2.2% |
| `ibm_ccf` | 7d | 0.0% |

`ibm_ccf` is effectively undelayed — 7 days against a 9,629-day train window censors 3
labels of 24,924 — so it cannot exercise a delay-aware method. `sparkov` and
`sparkov_slow` are the one paired contrast: identical rows, only `reported_at` differs.

The delay is calibrated per **domain** and whatever censoring results is reported rather
than tuned. Every `reported_at` in this repository is synthetic; no public dataset here
ships a real one. Full picture, including the two datasets whose spans cannot carry a
realistic clock: **[`docs/label-delay.md`](docs/label-delay.md)**.

## Beyond the pipeline

`src/fraud_benchmark/experiments/` holds everything that happens after a dataset is
prepared, and reads the pipeline's output without ever writing to it.

Three of the eight datasets are run experimentally — **IBM CCF**, **SAML-D** and
**Sparkov**, the last under three label-delay regimes. Each has one module in
`experiments/features/` that names every feature it produces and writes
`data/features/<dataset>.parquet`; reading that one file is meant to be the whole answer
to what its model sees. The encoding, model and metric stack sits alongside, and is where
a split is chosen and anything is fitted.

```bash
python -m fraud_benchmark.experiments.features.sparkov       # -> data/features/sparkov.parquet
python scripts/run_experiment.py --dataset sparkov \
    --model xgboost --history 10 --label-delay slow          # -> results/experiments/
python scripts/summarize.py --write                          # -> results/experiments.md
```

An experiment picks a dataset, a model (`xgboost`, `mlp`, `logistic`), how many previous
transactions to concatenate, and which labels *training* is allowed to see — `off`, `on`
or `slow`, the last being Sparkov's harsher reporting regime. Validation and test always
use true labels. `scripts/slurm/jobs/submit_all.sh` runs the whole 25-cell grid from
cold.

See [`docs/architecture.md`](docs/architecture.md) for the layout,
[`docs/experiments.md`](docs/experiments.md) for what each axis measures, and
[`docs/README.md`](docs/README.md) for everything else.

## Tests

```bash
pytest
```

Network tests are deselected by default. Run them with `pytest -m network` — they perform
real Kaggle downloads.

## Licence

The code in this repository is MIT licensed (see `LICENSE`).

**The datasets are not.** Each carries its own terms, which you accept directly with the
upstream provider when you download it. Two sources are **NonCommercial** (BankSim,
SAML-D), three are **ShareAlike** (PaySim, BankSim, SAML-D), and IEEE-CIS is governed by
Kaggle competition rules rather than an open licence. `prepare --all
--exclude-noncommercial` skips the restricted ones. Full table:
**[`docs/dataset-licenses.md`](docs/dataset-licenses.md)**.
