# Datasets

One page per registered dataset: the numbers, the schema mapping, and — the reason these
pages exist — the artifacts and disclaimers. Seven distinct sources plus `sparkov_slow`,
which is `sparkov` under a second delay regime.

Sorted by how much the labels are recoverable from a single column, worst first.

| dataset | rows | fraud rate | licence | in one line |
|---|---:|---:|---|---|
| [`ibm_ccf`](ibm_ccf.md) | 24,386,900 | 0.122% | Apache-2.0 ⚠ | ⚠⚠ val frauds are 100% one country; a one-line rule beats every model |
| [`paysim`](paysim.md) | 6,362,620 | 0.129% | CC BY-SA 4.0 | ⚠ 320 timesteps are pure fraud — 44% of frauds, from the clock alone |
| [`amaretto`](amaretto.md) | 29,704,090 | 0.274% | MIT | ⚠ capital markets, 5 classes; one whole class is `amount % 1000 == 0` |
| [`banksim`](banksim.md) | 594,643 | 1.211% | CC BY-NC-SA 4.0 | ⚠ fraud injected into named categories, up to 95% |
| [`ieee_cis`](ieee_cis.md) | 590,540 | 3.499% | competition rules | the only real data; flags are feature-set structure, not leakage |
| [`saml_d`](saml_d.md) | 9,504,852 | 0.104% | CC BY-NC-SA 4.0 | AML; nothing oracle-like, 13 unconverted currencies |
| [`sparkov`](sparkov.md) | 1,852,394 | 0.521% | CC0 1.0 | **clean** — zero flagged values; the negative control |
| [`sparkov_slow`](sparkov_slow.md) | 1,852,394 | 0.521% | CC0 1.0 | `sparkov` with a harsher delay; identical rows |

73 million transactions. Four are **NonCommercial or rules-governed**
(`banksim`, `saml_d`, `ieee_cis`) — see [`../dataset-licenses.md`](../dataset-licenses.md).

## Read these too

- [`../label-delay.md`](../label-delay.md) — the `reported_at` timestamp, per-dataset
  regimes, and which datasets the delay actually bites on.
- `dataset_card.json` in any prepared dataset's output directory — the same facts,
  machine-readable, as they were at preparation time.

## Two things that apply to every dataset

1. **The canonical frame carries exactly one binary label, `is_fraud`.** Each adapter
   declares the source column it derived that from — `isFraud`, `fraud`, `Is Fraud?`,
   `Is_laundering` — and the pipeline drops it, so there is no second label to remember
   to exclude. Two columns are kept, because `is_fraud` loses what they carry:
   amaretto's `Anomaly` (0 plus five FATF typology classes) and saml_d's
   `Laundering_type` (the typology for laundering rows). Both are declared
   label-descriptive by their adapters and are excluded from every model by
   `fraud_benchmark.experiments.columns.ALWAYS_EXCLUDED`, which builds that part of its
   list from the adapter registry rather than restating column names. Each dataset's
   card records what was removed in `dropped_source_label`.
2. **Every `reported_at` is synthetic.** No dataset here ships a real reporting timestamp.

Figures in `figures/` are emitted for light and dark by
`scripts/figures/plot_dataset_caveats.py`; the pages embed the light variant.
