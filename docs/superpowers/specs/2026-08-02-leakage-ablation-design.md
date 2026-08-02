# Leakage Ablation Study — Design

**Date:** 2026-08-02
**Goal:** Measure how much of each dataset's apparent fraud-detection performance comes
from generation artifacts rather than learnable signal, by scoring real classifiers with
and without the leaky columns the audit identified.

**Motivating result:** `docs/verification-notes.md`, `## Known leakage`. A one-line rule
(`Merchant State == "Italy"`) scores F1 0.914 on IBM CCF validation. This study replaces
that anecdote with a measured ablation across models, datasets, and label regimes.

---

## Scope

Five dataset variants, chosen because they are the ones under consideration for the
benchmark's headline results:

| dataset | rows | frauds | base rate | entities |
|---|---:|---:|---:|---:|
| `ibm_ccf` | 24,386,900 | 29,757 | 0.122% | 2,000 |
| `ibm_ccf_subsample_fast` | 6,569,157 | 8,412 | 0.128% | 1,565 |
| `ibm_ccf_subsample_slow` | 6,569,157 | 8,412 | 0.128% | 1,565 |
| `saml_d` | 9,504,852 | 9,873 | 0.104% | 292,715 |
| `sparkov` | 1,852,394 | 9,651 | 0.521% | 999 |

`amaretto`, `banksim`, `ieee_cis` and `paysim` are out of scope. They are not candidates
for the headline recommendation.

## Not in scope

- Changing the pipeline, the adapters, or any processed data. This study reads
  `data/processed/` and writes only to `data/features/` and `results/`.
- Tuning models to win. The point is the *gap* between conditions, not the absolute
  ceiling.
- Repairing leakage. As with the audit, this measures and records.

---

## Experimental grid

Four axes.

**1. Dataset** — the five above.

**2. Feature set** — `leaky` (everything permitted) vs `clean` (leaky columns removed).

**3. Label regime** — `oracle` (all train labels) vs `censored` (only labels with
`reported_at <= train_end`, i.e. what a model training at the cutoff would really have).

**4. Model** — trivial rule, logistic regression, XGBoost × 3 seeds.

### Job grid

`ibm_ccf` runs `oracle` only: it has 3 censored train labels out of 24,924, so the
censored regime is arithmetically identical to oracle and would cost roughly eight hours
of the most expensive fits in the study to confirm. Documented here rather than measured.

| dataset | feature sets | label regimes | jobs |
|---|---|---|---:|
| `ibm_ccf` | leaky, clean | oracle | 2 |
| `ibm_ccf_subsample_fast` | leaky, clean | oracle, censored | 4 |
| `ibm_ccf_subsample_slow` | leaky, clean | oracle, censored | 4 |
| `saml_d` | leaky, clean | oracle, censored | 4 |
| `sparkov` | leaky, clean | oracle, censored | 4 |
| | | | **18** |

Each job runs every fitted model internally: 18 × (1 logistic + 3 XGBoost) = **72 fits**,
each scored on val and test. The trivial rule adds **3 further evaluations** — one each for
`ibm_ccf`, `ibm_ccf_subsample_fast` and `ibm_ccf_subsample_slow`, since it involves no
fitting and does not vary across cells.

### Sparkov is the negative control

The audit found **zero** flagged values in Sparkov, so its leaky set is empty and its
`leaky` and `clean` conditions are identical by construction. The observed gap between
them is therefore a direct measurement of this harness's own noise floor, and calibrates
how much of every other dataset's gap to believe. This is the single most important
control in the study and must not be dropped as redundant.

---

## Column handling

Two distinct exclusion sets. Conflating them is the most likely way to get a
meaningless result, so they are separate constants with separate tests.

### Always excluded, every condition

Leaving any of these in a feature matrix invalidates every number in this study.

| kind | columns |
|---|---|
| label and derivatives | `is_fraud`, `reported_at`, `campaign_id`, `split` |
| source raw labels | `Is Fraud?`, `Is_laundering`, `Laundering_type` |
| row identifiers | `trans_num` |
| provenance | `source_file` |

`source_file` is excluded because Sparkov's splits come from separate upstream files, so
it predicts the split perfectly. `Laundering_type` is a *description of the label* and is
non-null only for laundering rows.

The harness asserts that no always-excluded column reaches any model, rather than
trusting the list to be maintained correctly.

### Ablation set, removed only in `clean`

From `docs/verification-notes.md`, `## Known leakage`:

| dataset | leaky columns |
|---|---|
| `ibm_ccf`, `ibm_ccf_subsample_*` | `Merchant State`, `Merchant City`, `Zip`, `MCC`, `Errors?`, `Merchant Name` |
| `saml_d` | `Sender_account`, `Receiver_account`, `entity_id` |
| `sparkov` | *(none)* |

**`Merchant Name` is added beyond the tuple the audit suggested.** The audit's closing note
listed `Merchant State`, `Merchant City`, `Zip`, `MCC`, `Errors?` but not the merchant
identifier. Retaining it would defeat the clean condition: the audit measured whole
`merchant × MCC` groups running pure (56/56, 53/54, 33/35 inside Strasburg), and a merchant
id encodes its own location, so `Merchant Name` reintroduces the geography artifact under a
different name. Excluding it makes the clean condition genuinely clean at the cost of a
weaker feature set — which is the honest direction to err. This is a deliberate deviation
from the audit and is called out in the results.

`Sender_bank_location` and `Receiver_bank_location` are **kept** in SAML-D's clean
condition. The audit found their effects (5.8–6.3×) directional and plausible for money
laundering — the phenomenon, not an artifact.

Note that `entity_id` is leaky for SAML-D (its flagged values are recurring mule accounts)
but is *not* removed for the IBM CCF variants, where the audit measured entity
concentration and found it unremarkable: 977 of 1,565 users carry a fraud, worst user
1.82%.

---

## Features

### Causal velocity features

Computed per `entity_id` over the **full timeline**, not per split. A per-split
computation would give every split a cold-start artifact at its left edge, where entities
appear to have no history purely because the window was truncated.

| feature | definition |
|---|---|
| `txn_count_1h`, `_24h`, `_7d` | transactions by this entity in the preceding window |
| `amount_sum_24h`, `_7d` | summed amount over the preceding window |
| `amount_mean_7d` | mean amount over the preceding window |
| `amount_zscore_vs_7d` | `(amount - mean_7d) / std_7d`, 0 when std is 0 or undefined |
| `seconds_since_prev_txn` | gap to this entity's previous transaction; null on first |
| `merchant_novelty` | 1 if this entity has not transacted with this merchant before |

**Every window is left-closed and excludes the current row.** A feature that includes the
current transaction is lookahead within the row and would leak the amount into its own
z-score. This is the most likely place for a subtle bug, so it is tested against a
hand-computed fixture before any model runs — see the Testing section.

`merchant_novelty` requires a merchant-like column, which `ibm_ccf` (`Merchant Name`),
`sparkov` (`merchant`) and `saml_d` (`Receiver_account`) all have.

**It is kept in the `clean` condition even where its source column is leaky.** An earlier
draft dropped it, on the assumption that a feature derived from a leaky column inherits the
leak. Measured on `ibm_ccf_subsample_fast`, that assumption is false:

| rows considered | base rate | `novel=1` rate | lift |
|---|---:|---:|---:|
| all | 0.1281% | 1.7251% | 13.47× |
| Italy excluded | 0.0568% | 0.8463% | **14.89×** |

The lift *rises* when the geography artifact is removed, so novelty is not a proxy for it.
It compresses merchant identity to one bit — "seen before or not" — which does not transmit
*which* country. Dropping it would have discarded real signal to no purpose.

**A separate open question, recorded not resolved.** Novelty's lift is 13–15× on IBM CCF
and 8.4× on SAML-D, against only **1.60× on Sparkov**, the one dataset with no flagged
values. A plausible explanation is that these generators draw fraud merchants close to
uniformly at random, making nearly every fraud novel by construction — a generation
artifact of a different kind from the geography one, and not something this study is scoped
to settle. The results section reports novelty's standalone lift per dataset alongside the
ablation so the reader can see it.

### Anything fitted is fitted on train only

- `StandardScaler` for logistic regression: `fit` on train, `transform` on val and test.
- Categorical encoding: categories learned from train. Values appearing only in val or
  test map to a dedicated `__unseen__` bucket, never silently onto a real level.
- XGBoost needs no scaling but shares the encoder.

Velocity features involve no fitting — they are past-only aggregations — which is why
they are computed once over the whole timeline and cached.

---

## Models

| model | role | configuration |
|---|---|---|
| trivial rule | floor | `Merchant State == "Italy"` for `ibm_ccf*`. Not defined for `saml_d`/`sparkov`, which have no single-value oracle; recorded as null there. No fitting. **Evaluated once per dataset**, not per cell — see below. |
| logistic regression | linear reference | L2, `class_weight='balanced'`, on scaled features. Deterministic — one run, no seeds. |
| XGBoost | main model | `hist`, `nthread=4`, `scale_pos_weight` = train negative/positive ratio, 3 seeds (0, 1, 2). |

The floor matters: the earlier probe found a boosted model scoring *below* the one-line
rule on the subsample. Without the floor in the table that result is invisible.

**The trivial rule is fixed, so it is evaluated once per dataset and reused.** It performs
no fitting and reads no feature matrix, so its score cannot vary with feature set or label
regime — it is the same number in all four cells. Computing it per cell would imply a
variation that does not exist. It is emitted once, with `feature_set` and `label_regime`
recorded as `null`, and repeated across the summary rows for readability.

XGBoost hyperparameters are fixed across conditions, chosen once on `sparkov` validation.
Tuning per condition would confound the ablation with tuning effort.

## Metrics

**Average precision throughout.** ROC AUC is not computed. Base rates here are 0.10%–0.52%,
where ROC AUC is dominated by the negative class and stays high for a model with no useful
precision.

Secondary: precision, recall and F1 at the best-F1 threshold chosen **on validation** and
applied unchanged to test.

**Validation selects, test reports.** Model and threshold choices come off val. Test
numbers are computed once per configuration and are not iterated against.

**Evaluation sets are never downsampled.** Val and test are scored at full size and full
class balance; the harness asserts their row counts match the source parquet. Training uses
full data too — no negative downsampling in the primary results.

---

## Execution

### Two SLURM stages

**Stage 1 — feature build.** One job per dataset (5 jobs). Reads
`data/processed/<name>/data.parquet`, builds velocity features over the full timeline,
writes `data/features/<name>.parquet`. Run once; the 18 model jobs read the cache rather
than each rebuilding 24M rows of aggregates.

**Stage 2 — model runs.** 18 jobs, each `--dependency=afterok` on its dataset's stage-1
job. Each loads its cached features once and runs all models for its
(dataset, feature set, label regime) cell.

### Resources

Cluster facts: `cpu` partition is 48-core / 384 GB nodes, 1-day limit; `cpulong` allows 3
days; default account `smidlva1`; there is no `module` system, so jobs invoke
`/home/maskomic/projects/fraud-benchmark/.venv/bin/python` by absolute path.

| stage | partition | cpus | mem | time |
|---|---|---:|---:|---:|
| feature build, `ibm_ccf` | `cpulong` | 4 | 128G | 12:00:00 |
| feature build, others | `cpu` | 4 | 128G | 8:00:00 |
| model, `ibm_ccf` | `cpulong` | 4 | 128G | 24:00:00 |
| model, others | `cpu` | 4 | 128G | 12:00:00 |

Jobs are small and numerous so they run in parallel across the cluster. Wall-clock for the
whole grid is expected to be a few hours, bounded by the `ibm_ccf` jobs, not by the sum.

### Resumability

Every model evaluation appends one JSON object to `results/runs.jsonl` immediately after
it is scored. A job that dies loses at most the fit in flight, and re-running skips cells
already present. Each record carries the full configuration, both split scores, timings,
and the resolved feature list — so a result can be traced to exactly the columns that
produced it.

---

## Testing

Velocity features are where a silent lookahead bug would do the most damage and be hardest
to spot, so they carry the heaviest tests.

1. **Hand-computed fixture.** A small frame with two entities and known timestamps, where
   every window feature has been worked out by hand. Asserts exact equality.
2. **No-lookahead property test.** For a random row, recomputing its features from a frame
   truncated at that row's timestamp must give an identical result. This is the direct
   statement of causality and would catch an off-by-one on the window edge.
3. **Exclusion assertion.** Building a matrix that contains any always-excluded column
   raises, verified per dataset.
4. **Clean-condition assertion.** For each dataset, the `clean` matrix contains none of its
   leaky columns. `merchant_novelty` is explicitly expected to be present — see the
   Features section for why a derived feature is not automatically leaky.
5. **Encoder unseen-category handling.** A category present only in val maps to
   `__unseen__` and does not collide with a train level.
6. **Evaluation integrity.** Val and test row counts after feature building equal the
   source parquet's split counts.

Tests run on fixtures, not real data, so they stay in the fast suite.

---

## Outputs

- `results/runs.jsonl` — one record per model evaluation.
- `results/summary.md` — the ablation table: average precision per dataset × feature set ×
  label regime × model, for val and test, with the leakage gap as a derived column.
- A `## Leakage ablation` section appended to `docs/verification-notes.md`, in the style of
  the existing audit section: what was measured, what it means, what it changes.

## Success criteria

The study succeeds if it answers, with numbers:

1. How much average precision does each dataset lose when its leaky columns are removed?
2. Is that loss larger than the Sparkov control's noise floor?
3. Does a real model beat the trivial rule on the IBM CCF variants?
4. Is the label-delay effect measurable on the subsamples with a proper model, or was the
   earlier probe's "below noise" verdict correct?

A result of "the gap is within noise" is a successful outcome, not a failed study.
