# Fraud & AML Benchmark Suite — Design

Date: 2026-08-01

## Purpose

Download seven public fraud and AML transaction datasets, normalize them to a shared
schema, assign temporal train/validation/test splits, and attach a synthetic
label-availability ("fraud reported at") timestamp to every fraudulent transaction.

The result is a benchmark where a model can be evaluated under realistic label delay:
at prediction time, only labels that had actually been reported are available.

## Datasets in scope

| Name | Source | Access | Entity key |
|---|---|---|---|
| PaySim | `ealaxi/paysim1` | Kaggle dataset | `nameOrig` (customer) |
| BankSim | `ealaxi/banksim1` | Kaggle dataset | `customer` |
| IEEE-CIS / Vesta | `ieee-fraud-detection` | Kaggle **competition** | derived uid (see caveat) |
| Sparkov (Shenoy) | `kartik2112/fraud-detection` | Kaggle dataset | `cc_num` (card) |
| IBM CCF (Altman) | `ealtman2019/credit-card-transactions` | Kaggle dataset | `User`, switchable to card |
| SAML-D | `berkanoztas/synthetic-transaction-monitoring-dataset-aml` | Kaggle dataset | sender account |
| Amaretto | `github.com/necst/amaretto_dataset` | git | per-repo, resolved during implementation |

### Excluded

**ULB Credit Card Fraud** (`mlg-ulb/creditcardfraud`) is deliberately excluded. It has no
card or customer identifier, so fraud campaigns cannot be grouped and the label-delay step
would have to fabricate entities. Inclusion criterion: a dataset must carry a real
entity identifier.

### IEEE-CIS caveat

IEEE-CIS has no true card identifier. `card1` is a hashed card attribute, not an ID. The
adapter constructs an entity id from the card/address columns plus a `D1`-derived account
start offset — the well-known Kaggle community "uid" heuristic. This is an approximation,
not ground truth, and campaign grouping for this dataset inherits that approximation. The
caveat is recorded in the dataset card, not just here.

IEEE-CIS also has no absolute timestamps: `TransactionDT` is a seconds offset from an
unstated reference. It is anchored to a configured `start_date`.

## Canonical schema

Every processed dataset writes these six columns first, in this order, followed by all
remaining source columns unchanged.

| Column | Dtype | Notes |
|---|---|---|
| `event_time` | `datetime64[ns]` | Naive, treated as UTC. Anchored to `start_date` for IEEE-CIS. |
| `entity_id` | `string` | Non-null. The account/card that campaign grouping keys on. |
| `amount` | `float64` | Native currency, unconverted. |
| `is_fraud` | `bool` | |
| `split` | `category` | `train` / `val` / `test` |
| `reported_at` | `datetime64[ns]` | Added by the final stage; null where `is_fraud` is false. |

Two deliberate non-goals:

- **No FX conversion.** SAML-D carries separate payment and received currencies; IBM CCF
  stores `$`-prefixed strings. Amounts are parsed to float in native units and currency
  columns pass through. Cross-dataset amount comparison is not meaningful.
- **Entity semantics are not unified.** Customer, card, and sender-account are different
  concepts. Each is "the thing a fraud campaign runs against" for its dataset; the dataset
  card records which was used.

## Architecture

Adapter registry plus shared pipeline stages. Each dataset is one small module exposing a
source spec and a `to_canonical(raw) -> DataFrame` function. Everything downstream is
shared code that behaves identically for all datasets. Adapters know nothing about
splitting, delay, or IO.

```
fraud-benchmark/
├── pyproject.toml            # package, deps, `fraud-benchmark` CLI entry point
├── README.md
├── .gitignore                # data/, credentials, venv
├── configs/default.yaml
├── src/fraud_benchmark/
│   ├── schema.py             # canonical columns, dtypes, validation
│   ├── config.py             # YAML load/merge into dataclasses
│   ├── sources.py            # kaggle dataset | kaggle competition | git
│   ├── datasets/
│   │   ├── base.py           # DatasetAdapter protocol + registry
│   │   ├── paysim.py  banksim.py  ieee_cis.py  sparkov.py
│   │   └── ibm_ccf.py  saml_d.py  amaretto.py
│   ├── splitting.py
│   ├── delay.py              # built last
│   ├── pipeline.py
│   └── cli.py                # list / prepare / info
├── tests/
└── data/                     # gitignored
    ├── raw/<dataset>/        # untouched source files
    └── processed/<dataset>/  # data.parquet + dataset_card.json
```

## Pipeline stages

1. **fetch** — resolve source spec, download to `data/raw/<dataset>/`. Cached; re-fetched
   only with `--force`.
2. **canonicalize** — call the adapter.
3. **validate** — no nulls in `event_time` / `entity_id` / `amount`; `is_fraud` boolean;
   `event_time` within a plausible range. Fails before anything is written.
4. **split** — sort by `event_time`, cut at cumulative quantiles (default 80/10/10,
   configurable globally and per dataset). Cuts land on **timestamp values, not row
   positions**, so rows sharing a timestamp are never split across a boundary.
5. **delay** — adds `reported_at`. Built last; see below.
6. **write** — parquet plus `dataset_card.json`, written to a temp path and atomically
   renamed so an interrupted run leaves no half-written dataset.

`dataset_card.json` records source URL and resolved version, row and fraud counts, split
boundary timestamps, the source column behind each canonical field, and any derivation
caveat (IEEE-CIS uid and anchor date, IBM CCF entity key choice).

## Label delay (final stage)

Deferred until stages 1–4 and all adapters are working. Agreed shape:

- Sample a delay per reported fraud from a distribution over time — Poisson or lognormal;
  the exact choice and parameters are settled when this stage is designed.
- **Fraud campaigns share one timestamp.** When a single entity has several fraudulent
  transactions in a short consecutive burst, they are realistically discovered and reported
  together. All transactions in such a campaign receive the **same** `reported_at`.
- Campaign detection keys on `entity_id` plus a time gap threshold.

Non-fraud rows have null `reported_at`.

## Configuration

`configs/default.yaml`, global defaults with per-dataset overrides. Nothing is hardcoded
in adapters. `--config` selects an alternative file.

```yaml
paths:  {raw: data/raw, processed: data/processed}
split:  {ratios: [0.8, 0.1, 0.1]}
datasets:
  ieee_cis: {start_date: "2017-12-01"}
  ibm_ccf:  {entity_key: user}          # or: card
```

## Failure handling

Expected failures get specific, actionable messages rather than raw tracebacks:

- **Missing or invalid Kaggle credentials** — name the exact file or environment variables
  to set and point at the setup guide.
- **IEEE-CIS 403** — competition rules must be accepted once in a browser. The generic auth
  error is misleading here, so this case is detected and reported separately.
- **Schema validation failure** — name the failing column and rule; write nothing.

Hardware target is 128 GB RAM and 4 cores. Every dataset, including IBM CCF (~24M rows),
is read into memory directly with explicit dtypes. No chunking, no out-of-core machinery.

## Testing

Readability over coverage theatre. No test touches the network.

- Unit tests over small synthetic frames for splitting (including the tied-timestamp
  boundary case), schema validation, and later the campaign-grouping logic. This is where
  the real logic lives and where tests earn their keep.
- One fixture test per adapter against a ~20-row committed file mirroring that source's
  real column layout, so an upstream schema change fails a specific, obvious test.
- A single integration test that genuinely downloads, marked `network` and deselected by
  default.

## Environment

`pyproject.toml` with an installable `fraud_benchmark` package and a `fraud-benchmark` CLI
entry point. Deps: `kagglehub`, `pandas`, `pyarrow`, `pyyaml`. Plain pip and venv on the
system Python 3.13. No uv, no conda.

## Build order

1. `git init`, pyproject, package skeleton, `.gitignore`
2. Kaggle credential setup — a written guide the user follows, verified with a live call
3. Schema, config, sources, and PaySim end-to-end including splits — proves the spine
4. The remaining six adapters
5. The label-delay stage
