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
fraud-benchmark download paysim     # fetch the raw files and stop
fraud-benchmark prepare paysim      # normalize, split, delay, write (downloads if needed)
fraud-benchmark info paysim         # print the prepared dataset's card
```

Output lands in `data/processed/<name>/` as `data.parquet` plus a `dataset_card.json`
recording provenance, counts, split strategy, delay parameters and caveats.

`download` exists so a slow fetch can be done once and separately; `prepare` finds the
raw files already there and skips it. Both take `--all`, and one unavailable dataset
fails on its own without stopping the rest.

Every preparation stage is also a script under `scripts/`; the two paper experiments
have dedicated reproducible runners described below.

## Datasets

Eight registered: seven distinct sources plus `sparkov_slow`, which is `sparkov` under a
second delay regime. 73 million transactions in total. Six of the eight are fully synthetic;
IEEE-CIS is the only real-world data.

### Recommended

We recommend to use the following datasets for sequential fraud (and AML) detection.
The datasets contain entity features, have a realistic time-span, and the artifacts
of the data do not disqualify them from benchmark purposes.

| dataset | rows | fraud rate | span | licence |
|---|---:|---:|---:|---|
| [`sparkov`](docs/datasets/sparkov.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`sparkov_slow`](docs/datasets/sparkov_slow.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`saml_d`](docs/datasets/saml_d.md) | 9,504,852 | 0.104% | 320d | CC BY-NC-SA 4.0 |
| [`ibm_ccf`](docs/datasets/ibm_ccf.md) | 24,386,900 | 0.122% | 10,649d | Apache-2.0 ⚠ |

There is a known artifact for IBM CFF dataset where nearly all of fraudulent transactions
in validation and test periods come with `country = Italy`.

### Use with caution

The following datasets are available, but are recommended to use with caution.

| dataset | rows | fraud rate | span | licence |
|---|---:|---:|---:|---|
| [`paysim`](docs/datasets/paysim.md) | 6,362,620 | 0.129% | 30d | CC BY-SA 4.0 |
| [`banksim`](docs/datasets/banksim.md) | 594,643 | 1.211% | 179d | CC BY-NC-SA 4.0 |
| [`ieee_cis`](docs/datasets/ieee_cis.md) | 590,540 | 3.499% | 181d | competition rules |
| [`amaretto`](docs/datasets/amaretto.md) | 29,704,090 | 0.274% | 83d | MIT |

⚠ **Read a dataset's page before using it.** Each page gives the numbers,
the schema mapping, and the disclaimers: **[`docs/datasets/`](docs/datasets/README.md)**.

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

## Paper experiments

`src/fraud_benchmark/experiments/` holds everything that happens after a dataset is
prepared, and reads the pipeline's output without ever writing to it.

The public supplementary material contains two experiments: IBM CCF temporal versus IID
evaluation over a shared pre-Italy population, and Sparkov under three synthetic
training-label-delay regimes. Build the required inputs with the same download, prepare,
and feature stages used by the dataset pipeline:

```bash
.venv/bin/python scripts/download.py --dataset sparkov
.venv/bin/python scripts/prepare.py --dataset sparkov
.venv/bin/python scripts/features.py --dataset sparkov
```

Then reproduce the paper tables:

```bash
.venv/bin/python scripts/ibm_split_protocol.py
.venv/bin/python scripts/sparkov_delay_protocol.py
```

The runners write readable tables to `results/ibm_split_protocol.md` and
`results/sparkov_label_delay.md`; [`results/paper.md`](results/paper.md) indexes the
machine-readable records and states exactly which rows and regimes each experiment uses.

See [`docs/architecture.md`](docs/architecture.md) for the layout,
[`docs/experiments.md`](docs/experiments.md) for the complete protocols, and
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
