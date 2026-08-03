# Experiments

Two halves. **"Results" is a closed record**: the code that produced it — six runners in
`scripts/`, the `experiments/ablation/` grid, and the feature modules they imported — was
removed when feature extraction was reorganised into one module per dataset. Those numbers
are why the new modules keep merchant geography behind an `artifact_` prefix, keep grouped
MCC out of it, and prefer relative geography to absolute, so they are kept rather than
deleted with the code. **"The current experimental surface" is what runs today.**

The retired work all asked one question, from different angles: **IBM CCF scores 0.041
average precision under the leakage ablation while a one-line rule scores 0.764 — what, if
anything, closes that gap?** The ablation itself was separate
(`experiments/ablation/`, results in `results/runs/`, table in `results/summary.md`); the
rest of this file covered the follow-ups.

Every number below is **test average precision**, mean ± population sd over seeds
0/1/2, read from the `results/*.jsonl` files named in each row. Never ROC AUC — at
a 0.122% base rate it stays high for a model with no useful precision.

## Two splits, and why numbers do not cross between them

| split | built by | train | note |
|---|---|---|---|
| `standard` | `experiments.splits.standard_split` | 80% by `event_time` quantile | Comparable across the runners here |
| `italy_holdout` | `experiments.splits.italy_holdout_split` | 13,350,884 rows ending 1s before the first Italy fraud | Exactly 80/10/10 by construction, zero Italy frauds in train |
| ablation *(retired)* | `data.splitting.assign_splits` | 80% cut on timestamp *values* | Tie-safe; **different** from `standard`. Still what `split` in a prepared frame means |

Two traps worth stating plainly:

* **0.041 is not comparable to 0.0333.** The ablation's 0.041 comes from ~45 raw
  columns ordinally encoded; the scripts' `v1` = 0.0333 comes from 36 curated
  features (grouped MCC, relative geography, robust-scaled magnitudes) on a
  quantile split. Same dataset, different pipeline. Compare within a column of the
  table below, not across this paragraph.
* **The `italy_holdout` split changes almost nothing.** The standard 80% cut falls
  on 2017-05-14 and the first Italy fraud is 2017-11-19, so train contains zero
  Italy frauds *either way*. The holdout only makes the boundary exact, at the cost
  of 6.2M training rows.

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

## The current experimental surface

Nothing above runs any more. What exists in its place is three feature parquets, built
one module per dataset:

```bash
python -m fraud_benchmark.experiments.features.ibm_ccf    # 24.4M rows, 82 features
python -m fraud_benchmark.experiments.features.saml_d     #  9.5M rows, 63 features
python -m fraud_benchmark.experiments.features.sparkov    # 1.85M rows, 50 features
```

Or via the generated sbatch files, which is how the big two are meant to run:

```bash
.venv/bin/python scripts/slurm/generate.py    # writes scripts/slurm/jobs/
scripts/slurm/jobs/submit_all.sh              # all three, no dependencies
```

Measured on the real data: sparkov 78s at 3.7 GB peak, saml_d 906s at 20.2 GB. The
sbatch requests (64 GB, 128 GB, 250 GB) are above those with room to spare.

Three axes are available to an experiment, and they are chosen when a model is fitted
rather than when a parquet is built:

| axis | values | how |
|---|---|---|
| feature set | with or without the generation artifacts | include or drop the `artifact_*` columns |
| label regime | no delay, delay, slow delay | ignore `reported_at`; use it; use `reported_at_slow` (sparkov only) |
| split | `standard`, `italy_holdout` | `experiments.splits`; the parquets are split-agnostic |

The leaky/clean axis of the retired ablation is the first of these, moved inside the
feature definitions: each module's docstring says which of its columns are in the
artifact group and what was measured about them. The three label regimes replace the
`oracle`/`censored` pair and the separate `sparkov_slow` dataset, since sparkov's parquet
carries both timestamps.

## Reproducing anything above

Every runner is gone, so a row in the tables above needs its model code written again
against the new parquets. Two things make that less than a rewrite from scratch: the
model side (`experiments/models.py`, `mlp.py`, `metrics.py`, `encoding.py`,
`splits.py`) is untouched and still holds the fixed hyperparameters every number above
was measured with, and the causal primitives the feature work needed are in
`experiments/features/util.py`.

Two caveats carry forward regardless:

* **The feature sets are not the same.** The old `v1` was 36 curated columns and the
  ablation's `leaky` was ~45 raw ones. IBM CCF's module now produces 82, unfitted. No
  number above transfers to it.
* **Encoding is no longer decided in the feature build.** Finding 4 — one-hot beating
  ordinal by 3.4x — is a fact about `experiments/encoding.py`, which still has both, and
  it is the strongest single result in this file. Anything rebuilt here should try
  one-hot or learned embeddings before ordinal codes.
