# Archived results

Everything here was produced by code that no longer exists. It is kept because the
measurements are still the evidence behind several decisions in the current codebase,
and deleting them would keep the conclusions while losing the reasoning.

**Nothing here is comparable to `results/experiments/`.** The feature sets differ, the
splits differ, and the encoding was decided in a different place. Read a number here as
a historical fact about a retired pipeline, not as a baseline for a new run.

| path | produced by | what it measured |
|---|---|---|
| `summary.md`, `runs/` | `experiments/ablation/` | The 14-cell leakage ablation: dataset × `leaky`/`clean` columns × `oracle`/`censored` labels. `runs/retired/` holds cells dropped from the grid before it finished. |
| `italy_holdout.jsonl` | `scripts/italy_holdout.py` | IBM CCF on a split whose train half predates the first Italy fraud. |
| `geo_dilution.jsonl` | `scripts/geo_dilution.py` | Reassigning 95% of Italy rows to a real non-Italy merchant identity. Measured on a subsample that is itself retired. |
| `ibm_features_v2.jsonl`, `ibm_relative_geo.jsonl` | `scripts/ibm_features_v2.py` | 26 history-relative features against the curated `v1` set. They scored *worse* (0.0280 against 0.0333). |
| `seq_window_*.jsonl` | `scripts/seq_window.py` | XGBoost on a flattened window: target + its previous 9 rows. Lagged columns earned 13% of total gain. |
| `seq_mlp*.jsonl` | `scripts/seq_mlp.py`, `seq_mlp_v2.py` | The same window through a 3-layer MLP. Contains the largest effect anyone measured here: one-hot instead of ordinal codes, nothing else changed, took 0.0537 to 0.1804. |

Three of those findings are why the current code looks as it does. The artifact group in
`experiments/features/` exists because of the ablation and the Italy holdout; the narrow
`HISTORY_COLUMNS` default exists because lagged columns earned 13% of gain; and
`estimators/mlp.py` one-hot expands by default because of the 3.4x encoding gap.

Prose and the full tables: [`../../docs/experiments.md`](../../docs/experiments.md).
