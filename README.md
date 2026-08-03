# fraud-benchmark

Downloads public fraud and AML transaction datasets, normalizes them to a shared schema,
assigns temporal train/validation/test splits, and attaches a synthetic label-availability
("fraud reported at") timestamp to fraudulent transactions.

See `docs/superpowers/specs/2026-08-01-fraud-benchmark-design.md` for the full design.

## What's in here

Three separable pieces. Only the first is the benchmark itself; the other two are
studies that read its output and never write to it.

**1. The pipeline** — `src/fraud_benchmark/`, driven by the `fraud-benchmark` CLI.

| module | does |
|---|---|
| `cli.py` | `list`, `prepare`, `info` |
| `pipeline.py` | stage orchestration: fetch → canonicalize → validate → split → delay → write |
| `sources.py` | fetching from Kaggle or git, with errors a human can act on |
| `datasets/` | one adapter per dataset; `base.py` holds the interface and registry — documented in [`docs/datasets/`](docs/datasets/README.md) |
| `schema.py` | the canonical schema every adapter must produce, and its validator |
| `splitting.py` | temporal splits, cut on timestamp values so ties cannot straddle |
| `campaigns.py` | grouping frauds into campaigns (one entity, gap-bounded) |
| `delay.py` | the synthetic `reported_at` timestamp, lognormal per campaign — see `docs/label-delay.md` |
| `config.py` | `configs/default.yaml` plus per-dataset overrides |

Output: `data/processed/<name>/data.parquet` + `dataset_card.json`.

**2. The leakage ablation** — `src/fraud_benchmark/ablation/`. Does a model's score
survive removing the columns the audit flagged as generation artifacts? A 14-cell
grid of (dataset × leaky/clean × oracle/censored labels).

| module | does |
|---|---|
| `grid.py` | which cells exist — the single definition, read by the generator and the summary |
| `build_features.py` | stage 1: causal velocity features, cached per dataset |
| `features.py` | the velocity feature definitions themselves |
| `columns.py` | which columns a model may see; the two exclusion sets |
| `cell.py` | stage 2: run one cell, append one record per fit |
| `encoding.py`, `models.py`, `metrics.py` | train-only encoding, the three models, average precision |
| `summarize.py` | `results/runs/*.jsonl` → `results/summary.md` |

Read `results/summary.md` first, then `docs/verification-notes.md` §"Known leakage".

**3. The IBM CCF experiments** — `scripts/`. Follow-ups asking why IBM CCF scores
0.041 when a one-line rule scores 0.764: feature engineering, geography ablation,
flattened sequence windows, MLPs. **Start at `docs/experiments.md`**, which tables
every result and names the rough edges. `scripts/slurm/generate.py` emits every
sbatch file, including these.

Results live in `results/`: `summary.md` and `runs/` for the ablation, one
`*.jsonl` per experiment, `figures/` for plots, `logs/` for SLURM output.
`runs/retired/` holds records for cells no longer in the grid and is deliberately
excluded from the summary.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Kaggle credentials

Required before any dataset can be downloaded. Pick **one** of the methods below —
`kagglehub` tries them in this order and uses the first it finds.

**1. Access token file (recommended)**

Copy your Kaggle access token into `~/.kaggle/access_token` as plain text — no
extension, no JSON, no quotes. Trailing whitespace is stripped, so a trailing newline
is fine.

```zsh
mkdir -p ~/.kaggle && read -rs "TOKEN?Paste token: " && \
  printf '%s' "$TOKEN" > ~/.kaggle/access_token && \
  chmod 600 ~/.kaggle/access_token && unset TOKEN
```

Using `read` keeps the token out of your shell history. In bash, use
`read -rsp "Paste token: " TOKEN` instead.

**2. Environment variable**

```bash
export KAGGLE_API_TOKEN=<token>
```

**3. Legacy `kaggle.json`**

Kaggle Settings → **API** → **Create New Token** downloads a `kaggle.json` containing
a username and key. Save it to `~/.kaggle/kaggle.json` and `chmod 600` it, or export
its two values as `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Note that the `key` inside `kaggle.json` is **not** the same credential as an access
token — do not paste it into `~/.kaggle/access_token`.

**Verify:**

```bash
.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"
```

The credentials must live in your **home** directory, not in this repository. `.gitignore`
guards against committing them, but keeping them outside the repo is safer.

IEEE-CIS additionally requires accepting its competition rules once in a browser — see
`docs/kaggle-setup.md`.

## Usage

    fraud-benchmark list
    fraud-benchmark prepare paysim
    fraud-benchmark info paysim

## Datasets and label delay

Eight registered: seven distinct sources, plus `sparkov_slow`. That variant is
row-identical to `sparkov` on every column but `reported_at` and reuses its raw
download, so a method can be compared across two label-delay regimes on identical
data — the only such pair in the suite.

`censored` is the share of **train** frauds a model training at the train cutoff
would have the wrong label for, measured at the shipped config. It is a consequence
of the delay distribution against each dataset's train window, not a tuned target.

| dataset | rows | fraud rate | span | delay median | censored |
|---|---:|---:|---:|---:|---:|
| `paysim` | 6,362,620 | 0.129% | 30d | 1d | 10.1% |
| `banksim` | 594,643 | 1.211% | 179d | 7d | 8.1% |
| `sparkov` | 1,852,394 | 0.521% | 730d | 7d | 2.2% |
| `sparkov_slow` | 1,852,394 | 0.521% | 730d | 15d, σ 1.665 | 8.9% |
| `saml_d` | 9,504,852 | 0.104% | 320d | 30d | 20.0% |
| `ibm_ccf` | 24,386,900 | 0.122% | 10,649d | 7d | 0.0% |
| `ieee_cis` | 590,540 | 3.499% | 181d | 7d | 7.1% |
| `amaretto` | 29,704,090 | 0.274% | 83d | 7d | 17.5% |

**One page per dataset in [`docs/datasets/`](docs/datasets/README.md)** — the numbers, the
schema mapping, and the artifacts and disclaimers each one carries. Read the page before
using a dataset; several have label-recoverable columns sharp enough to invalidate a
result, `ibm_ccf` most of all.

73 million transactions. IBM CCF is effectively undelayed — a 7-day delay against a
9,629-day train window censors 3 labels of 24,924 — so it cannot exercise a
delay-aware method; `sparkov` at 2.2% is nearly as weak, which is why
`sparkov_slow` exists. **`docs/label-delay.md`** has the full picture: how the delay
is constructed, which datasets it bites on, the two whose spans cannot carry a
realistic clock, and how to change it.

Two earlier variants, `ibm_ccf_subsample_fast` and `ibm_ccf_subsample_slow`, were
retired: their delay contrast was unmeasurable (average precision 0.975 against
0.975, at a 0.001 seed noise floor). See `docs/verification-notes.md`.

## Tests

    pytest

Network tests are deselected by default. Run them with `pytest -m network` — they perform
real Kaggle downloads.

## Licence

The code in this repository is MIT licensed (see `LICENSE`).

**The datasets are not.** Each carries its own terms, which you accept directly with the
upstream provider when you download it. This repository distributes no data — `data/` is
gitignored and everything is fetched at run time with your own credentials.

Two of the seven sources (BankSim, SAML-D) are **NonCommercial**, and three
(PaySim, BankSim, SAML-D) are **ShareAlike**. IEEE-CIS is governed by Kaggle competition
rules rather than an open licence. See **`docs/dataset-licenses.md`** for the full table,
and check `data_license` in any prepared dataset's `dataset_card.json`.
