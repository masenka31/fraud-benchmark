# Experiments on IBM CCF

Everything in `scripts/` other than `plot_monthly_fraud.py`. They all ask one
question, from different angles: **IBM CCF scores 0.041 average precision under the
leakage ablation while a one-line rule scores 0.764 — what, if anything, closes
that gap?**

The ablation itself is separate: `src/fraud_benchmark/ablation/`, results in
`results/runs/`, table in `results/summary.md`. This file covers the follow-ups.

Every number below is **test average precision**, mean ± population sd over seeds
0/1/2, read from the `results/*.jsonl` files named in each row. Never ROC AUC — at
a 0.122% base rate it stays high for a model with no useful precision.

## Two splits, and why numbers do not cross between them

| split | built by | train | note |
|---|---|---|---|
| `standard` | `standard_split` in the scripts | 80% by `event_time` quantile | Comparable across the scripts here |
| `italy_holdout` | `italy_holdout.build_split` | 13,350,884 rows ending 1s before the first Italy fraud | Exactly 80/10/10 by construction, zero Italy frauds in train |
| ablation | `splitting.assign_splits` | 80% cut on timestamp *values* | Tie-safe; **different** from `standard` |

Two traps worth stating plainly:

* **0.041 is not comparable to 0.0333.** The ablation's 0.041 comes from ~45 raw
  columns ordinally encoded; the scripts' `v1` = 0.0333 comes from 36 curated
  features (grouped MCC, relative geography, robust-scaled magnitudes) on a
  quantile split. Same dataset, different pipeline. Compare within a column of the
  table below, not across this paragraph.
* **The `italy_holdout` split changes almost nothing.** The standard 80% cut falls
  on 2017-05-14 and the first Italy fraud is 2017-11-19, so train contains zero
  Italy frauds *either way*. The holdout only makes the boundary exact, at the cost
  of 6.2M training rows. See docs/verification-notes.md.

## Results

### Feature engineering (trees, XGBoost, fixed hyperparameters)

| experiment | condition | standard | italy_holdout | source |
|---|---|---:|---:|---|
| ablation | leaky (45 raw cols) | 0.041 ± 0.006 | — | `results/runs/` |
| ablation | clean | 0.019 | — | `results/runs/` |
| `ibm_features_v2.py` | v1 (36 curated) | 0.0333 ± 0.0012 | — | `results/ibm_features_v2.jsonl` |
| `ibm_features_v2.py` | v1_plus (no location identity) | 0.0242 ± 0.0003 | — | same |
| `ibm_features_v2.py` | **v2 (v1 + 26 history-relative)** | **0.0280 ± 0.0012** | — | same |
| `italy_holdout.py` | v1 features, exact holdout | — | 0.0247 ± 0.0005 | `results/italy_holdout.jsonl` |
| `geo_dilution.py` | geo_diluted (95% of Italy reassigned) | 0.1947 ± 0.0057 | — | `results/geo_dilution.jsonl` ⚠ |
| `geo_dilution.py` | no_geo_keep_mcc | 0.0569 ± 0.0019 | — | same ⚠ |
| — | trivial rule (`Merchant State == Italy`) | 0.764 | — | `results/runs/` |

⚠ The two `geo_dilution` figures were measured on `ibm_ccf_subsample_fast`, since
retired. The script now defaults to the full `ibm_ccf`; a re-run will not reproduce
them.

**Findings.**

1. **The 26 hand-crafted history-relative features made trees worse**: v2 0.0280
   against v1 0.0333, a gap ~4x the seed spread. Context novelty, decline velocity,
   burst ratios and entity-baseline deviation — the features whose absence was the
   hypothesised reason a flattened window underperforms — do not recover anything
   here. That hypothesis is not supported.
2. **Location identity is worth about 0.009** (v1 0.0333 → v1_plus 0.0242) and
   `same_state`/`same_city` do not replace it. `ibm_relative_geo.py` asked exactly
   this and produced byte-identical numbers; it was deleted as a duplicate, and its
   record is kept at `results/ibm_relative_geo.jsonl`.
3. **Breaking the geography oracle leaves a real signal** (0.195), well above
   dropping location outright (0.057) — but on the retired subsample, so treat it as
   directional only.

### Architecture (same features, same splits, same metric)

| model | encoding | standard | italy_holdout | source |
|---|---|---:|---:|---|
| XGBoost, flattened window (target + 9 lags) | ordinal | 0.0516 ± 0.0008 | 0.0442 ± 0.0005 | `results/seq_window_*.jsonl` |
| 3-layer MLP, same window | ordinal | 0.0537 ± 0.0087 | 0.0482 ± 0.0047 | `results/seq_mlp_{standard,italy_holdout}.jsonl` |
| 3-layer MLP, same window | **one-hot** | **0.1804 ± 0.0274** | 0.0740 ± 0.0109 | `results/seq_mlp_onehot_*.jsonl` |
| 3-layer MLP, window + 26 v2 features | one-hot | *running* | *0.0499, seed 0 only* | `results/seq_mlp_v2_*.jsonl` |

**Findings.**

4. **Input encoding beat architecture by 3.4x.** One-hot instead of ordinal codes,
   nothing else changed, took the MLP from 0.0537 to 0.1804. Ordinal codes imply an
   ordering that does not exist (merchant state 5 is not "more" than 4); trees are
   indifferent, a dense layer is not. Learned embeddings are the untried next step.
5. **Flattening the history barely helps trees.** 0.0516 against 0.0333 for the
   same features unflattened, with lagged columns earning only 13% of total gain.
6. **Every configuration loses on `italy_holdout`**, one-hot MLP most of all
   (0.1804 → 0.0740). Whatever the models learn does not survive the regime shift.

### External reference points

Quoted in the script docstrings for orientation, **not measured in this
repository**: RNN 0.335, CAST fine-tuned 0.565. Verify before citing.

## Running them

All sbatch files are generated — including these:

```bash
.venv/bin/python scripts/slurm/generate.py        # writes scripts/slurm/jobs/
sbatch scripts/slurm/jobs/mlpoh_standard.sbatch   # one experiment at a time
```

Each expects `data/features/ibm_ccf.parquet` to exist already
(`python -m fraud_benchmark.ablation.build_features ibm_ccf`). Memory is the
binding constraint: the flattened window is a 15–17 GiB float32 block, so those
jobs request 250–280 GB and the `cpulong` partition.

## Known rough edges

`scripts/` grew as one-off experiments and has not been consolidated:

* `standard_split` is copy-pasted in two remaining scripts; the XGBoost parameter
  block is inlined four times although `ablation/models.py:fit_xgboost` already
  does exactly that; `italy_holdout.build_features` and `seq_window.base_features`
  are near-duplicates.
* `italy_holdout.py` and `features_v2.py` are imported *as libraries* by the other
  scripts (including private names such as `_money`), which works only because
  Python puts a script's own directory on `sys.path`. Three test files carry a
  `sys.path.insert` to reach them.
* `features_v2.py` (a library, no `main`) and `ibm_features_v2.py` (an experiment)
  are one character apart.

The fix is to promote the shared parts into `src/fraud_benchmark/experiments/` and
leave each file in `scripts/` as a thin entry point. Not yet done.
