# Paper experiments

This public repository records two supplementary experiments from the paper. Both use
full experimental populations, preserve temporal preprocessing semantics, select any
thresholds on validation data only, and report average precision as the primary metric.
The exact result locations are indexed in [`results/paper.md`](../results/paper.md).

Before running either experiment, build the corresponding feature parquet:

```bash
.venv/bin/python scripts/features.py --dataset ibm_ccf
.venv/bin/python scripts/features.py --dataset sparkov
```

## Model and input features

Both protocols fit an XGBoost `XGBClassifier` with 300 trees, maximum depth 6,
learning rate 0.1, row and column subsampling of 0.8, histogram tree building,
four fitting threads, and `aucpr` as the training evaluation metric. Each uses
seeds 0--4 and sets `scale_pos_weight` to the ratio of negative to positive
*available training labels*. See
[`models.py`](../src/fraud_benchmark/experiments/models.py) for the fixed settings.

The model receives the feature columns below, not the parquet's key columns
(`entity_id`, `event_time`, `reported_at`, `reported_at_slow` when present,
`is_fraud`, and `split`). Columns prefixed `artifact_` are excluded from the
model matrix. Categorical columns are ordinal-encoded with an encoder fitted on
the training split only. The feature builders are
[`ibm_ccf.py`](../src/fraud_benchmark/experiments/features/ibm_ccf.py) and
[`sparkov.py`](../src/fraud_benchmark/experiments/features/sparkov.py).

**Shared features (31), used by both protocols:**

```text
hour, minute, weekday, day, month, is_weekend, hour_sin, hour_cos
amount, amount_log1p, amount_is_refund, amount_cents,
amount_is_round_10, amount_is_round_100, amount_is_micro
txn_count_1h, txn_count_24h, txn_count_7d, txn_count_30d
amount_sum_24h, amount_sum_7d, amount_sum_30d,
amount_mean_7d, amount_zscore_vs_7d, seconds_since_prev_txn
burst_1h_over_24h, burst_24h_over_7d, amount_24h_over_7d
amount_over_entity_mean, amount_over_entity_max, entity_txn_ordinal
```

## IBM CCF: pre-Italy temporal versus IID splits

The IBM comparison uses every transaction strictly before the first Italy fraud at
`2017-11-19 12:06:00`, yielding one shared population of 20,403,999 rows. It compares:

- `pre_italy`: chronological 80/10/10;
- `pre_italy_iid_rows`: deterministic transaction-IID 80/10/10;
- `pre_italy_iid_customers`: deterministic customer-disjoint IID splitting, balanced
  against the 80/10/10 transaction-count targets.

All three regimes use the same rows, source features, causal engineered features,
XGBoost settings, and model seeds 0--4. Only split assignment changes. Label-derived
features use training labels only, are computed chronologically, and score a training row
before incorporating that row's label.

**IBM model inputs (82):** the 31 shared features above, these 43 IBM features,
and eight causal features added in
[`ibm_split_protocol.py`](../scripts/ibm_split_protocol.py):

```text
first_merchant_for_entity, first_mcc_for_entity,
first_state_for_entity, first_city_for_entity,
first_channel_for_entity, first_card_for_entity,
first_foreign_for_entity
secs_since_same_merchant, secs_since_same_mcc,
secs_since_same_state, secs_since_same_channel, secs_since_same_card
distinct_merchants_24h, prior_distinct_merchants
errors_1h, errors_24h, errors_7d
same_state, same_city, merchant_is_online,
merchant_state_missing, merchant_is_foreign
mcc_group, use_chip
card_brand, card_type, has_chip, card_on_dark_web,
cards_issued, days_since_acct_open, days_to_expiry,
days_since_pin_change
age, years_to_retirement, gender, fico, num_cards,
credit_limit, total_debt, income_person, income_zip,
debt_to_income, amount_over_credit_limit

state_prior_rarity, state_not_seen_previously, foreign_state_rarity
causal_target_rate_artifact_merchant_state
causal_target_rate_artifact_merchant_city
causal_target_rate_artifact_merchant_name
causal_target_rate_artifact_mcc
causal_target_rate_artifact_error_type
```

The five `causal_target_rate_*` columns summarize the named artifact fields
using chronological, training-only labels; the raw `artifact_` columns are not
model inputs.

```bash
.venv/bin/python scripts/ibm_split_protocol.py
```

The runner writes three JSON records under `results/paper/` and the readable table
[`results/ibm_split_protocol.md`](../results/ibm_split_protocol.md). Use `--render-only`
to regenerate the Markdown from the recorded JSON without refitting.

## Sparkov: synthetic training-label delay

The Sparkov comparison uses one `data/features/sparkov.parquet` population under three
training-label regimes:

- `off`: oracle/true training labels;
- `on`: synthetic default reporting timestamps from `reported_at`;
- `slow`: synthetic fat-tailed stress timestamps from `reported_at_slow`.

All regimes use the same rows, split, features, XGBoost settings, and model seeds 0--4.
A fraud reported after the training cutoff remains in training with provisional label
`0`; it is not dropped. Validation and test are always scored against true `is_fraud`
labels. The slow regime is a stress test, not an estimate of real reporting behavior.

**Sparkov model inputs (46):** the 31 shared features above plus:

```text
first_merchant_for_entity, first_category_for_entity
secs_since_same_merchant, secs_since_same_category
distinct_merchants_24h, distinct_categories_7d,
prior_distinct_merchants
distance_from_home_km, distance_over_entity_mean,
distance_over_entity_max
category, job, gender, age_at_txn, city_pop_log
```

```bash
.venv/bin/python scripts/sparkov_delay_protocol.py
```

The runner writes machine-readable records under `results/paper/` and the readable table
[`results/sparkov_label_delay.md`](../results/sparkov_label_delay.md).
Use `--render-only` to regenerate the Markdown without refitting.

## Shared experiment implementation

The public code under `src/fraud_benchmark/experiments/` contains the feature builders,
split functions, causal encoders, model preparation, and fixed XGBoost fitting required by the
two protocols. It fits preprocessing on training rows only and keeps evaluation labels
uncensored.

## LSTM sequence extension

The LSTM runner adds a sequence classifier to both studies without changing the
recorded XGBoost protocols above. Install the optional `lstm` dependencies with a
PyTorch build appropriate for the compute node's GPU. The runner uses a one-layer,
one-way LSTM with 64 hidden units; numeric features and missingness indicators are
projected with the categorical embeddings (8 dimensions each) to width 64. It uses
0.2 dropout, weighted binary cross entropy, AdamW with learning rate 0.001 and
weight decay 0.0001, gradient clipping at 1, batch size 1,024, and at most eight
epochs. The best checkpoint is selected by validation average precision, with
patience two. Seeds 0--4 are fixed. No transaction label, reporting timestamp,
or label-availability flag is an LSTM input.

### IBM endpoint experiment

The IBM extension sorts each entity's transactions chronologically and forms
disjoint, complete 30-transaction chunks. It predicts **only the final transaction
in each chunk** from that transaction and its 29 predecessors. Incomplete trailing
chunks have no target. Chunk membership and endpoint selection are fixed before
the split is assigned, so every IBM cell uses the same targets. This is a
reduced-target experiment, roughly one target per 30 source transactions. Its
average precision must not be compared with the full-row IBM table above. On
the shipped feature parquet, 20,403,999 source rows yield 679,372 endpoints
with 854 frauds. The temporal validation and test endpoint sets have 189 and
36 frauds respectively, so test AP may be sensitive to a few predictions.
Transactions from the same entity with identical minute timestamps retain their
source row order inside a chunk; that order is deterministic but does not prove
their true within-minute order.

The temporal cell uses each endpoint's original pre-Italy temporal assignment.
The customer-IID cell uses the original customer assignment. The new chunk-IID
cell assigns whole chunks with one fixed seed; it is not the original
transaction-IID cell. It still trains on some labels from dates later than
scored dates, so it measures an optimistic split rather than chronological
deployment. A new XGBoost comparator uses the **same endpoints, split assignment,
and 77 label-free feature names** as the LSTM. XGBoost receives only the final
transaction's feature values, while the LSTM receives those features for all
30 transactions in the chunk. Both use numeric values scaled from the training
split. The LSTM also receives missingness indicators and category embeddings;
XGBoost receives missing numeric values and ordinal-encoded categories. Those
77 feature names are the 74 non-artifact
IBM columns listed above plus `state_prior_rarity`,
`state_not_seen_previously`, and `foreign_state_rarity`. The five
`causal_target_rate_*` features are excluded because they summarize prior labels.

### Sparkov sequence experiment

Sparkov retains every transaction as a target under the existing temporal split.
For each target, the runner gathers that transaction and up to 29 strictly
earlier-time transactions from the same entity. It uses the 46 non-artifact
Sparkov features listed above. The `off`, `on`, and `slow` runs share those
windows and differ only in synthetic training-label availability. A delayed
fraud stays in training with provisional label 0; validation and test use the
true labels.

```bash
module purge
module load PyTorch/2.13.0-foss-2025b-CUDA-12.9.1
.venv/bin/python scripts/lstm_protocol.py --dataset ibm --device cuda
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --device cuda
```

The runner writes separate JSON records under `results/paper/` and, after all
cells are available, generates `results/ibm_lstm_protocol.md` and
`results/sparkov_lstm_delay.md`. Noncanonical seed runs require a separate
`--results-dir` and are diagnostics, not paper results. The default `auto`
device uses CUDA when the installed PyTorch build provides it; verify the
selected device in each result record. On this host, the
`PyTorch/2.13.0-foss-2025b-CUDA-12.9.1` module was verified on an A100 node;
its cuDNN version does not support the V100 nodes. Use `--regime` and `--resume`
to schedule or restart cells independently.

To repeat the protocols with a 10-transaction sequence, pass
`--window-length 10` to each runner command. The 10-transaction records have
`_len10` suffixes and generate `results/ibm_lstm_protocol_len10.md` and
`results/sparkov_lstm_delay_len10.md`, leaving the 30-transaction records
unchanged. This length comparison retains the original feature sets and all
other model settings. IBM gets more final-transaction targets because its
chunks are disjoint; those targets differ from the length-30 endpoints, so an
IBM AP difference between lengths is not a pure sequence-length effect.
Sparkov keeps every transaction as a target at both lengths.

### Recorded sequence results

The IBM endpoint LSTM has mean test average precision of 0.4370 (temporal),
0.6779 (chunk IID), and 0.7138 (customer IID) over seeds 0--4. The matched
XGBoost values are 0.3672, 0.5896, and 0.6499 respectively. See the
[`IBM endpoint table`](../results/ibm_lstm_protocol.md) for validation scores,
test-minus-validation AP gaps, variation across seeds, and target counts.
The temporal test set has only 36
fraud endpoints; its AP difference is descriptive and should not be treated as
precise evidence of a model advantage. The IID regimes also use different
assignments, so their AP difference combines split semantics with population
composition and should not be interpreted as a pure leakage effect.

On Sparkov, mean LSTM test AP is 0.9800 with no delay, 0.9774 with the default
synthetic delay, and 0.9722 with the slow synthetic delay. The matched
no-delay differences are -0.0026 and -0.0078. See the
[`Sparkov delay table`](../results/sparkov_lstm_delay.md) for validation scores
and test-minus-validation AP gaps. These synthetic regimes show sensitivity to the
specified reporting delays; they do not estimate real-world delay behavior.

The 10-transaction repeat has 2,039,680 IBM endpoints. Mean test AP is
0.5853/0.7272/0.7164 for the temporal/chunk-IID/customer-IID LSTM cells and
0.3939/0.6554/0.6279 for their endpoint-matched XGBoost comparators. The
[`length-10 IBM table`](../results/ibm_lstm_protocol_len10.md) gives validation
AP, seed variation, and target counts. Because the shorter chunks select
different endpoints, the IBM difference from length 30 cannot be attributed
to sequence length alone. The temporal test set has 95 fraud endpoints at
length 10, versus 36 at length 30.

Sparkov retains all target rows at length 10. Mean test AP is 0.9786 with no
delay, 0.9786 with the default synthetic delay, and 0.9739 with the slow
synthetic delay; the unrounded changes from no delay are -0.0001 and -0.0047.
The [`length-10 Sparkov table`](../results/sparkov_lstm_delay_len10.md)
contains validation AP and seed variation. These are five-seed descriptive
results with the same synthetic-delay caveats as the length-30 protocol.

### Raw-ish feature ablation

For the Sparkov feature ablation, retain only values from the current event,
its timestamp, and static cardholder/card attributes. Deterministic transforms
of those values are included; entity transaction-history summaries are excluded.
The selected names are already present in the feature parquet files, so this
ablation can select columns without rebuilding source data.

**Shared by IBM and Sparkov (15):**

```text
hour, minute, weekday, day, month, is_weekend, hour_sin, hour_cos,
amount, amount_log1p, amount_is_refund, amount_cents,
amount_is_round_10, amount_is_round_100, amount_is_micro
```

**IBM only (26; 41 total):**

```text
same_state, same_city, merchant_is_online, merchant_state_missing,
merchant_is_foreign, mcc_group, use_chip,
card_brand, card_type, has_chip, card_on_dark_web, cards_issued,
days_since_acct_open, days_to_expiry, days_since_pin_change,
age, years_to_retirement, gender, fico, num_cards, credit_limit,
total_debt, income_person, income_zip, debt_to_income,
amount_over_credit_limit
```

**Sparkov only (6; 21 total):**

```text
distance_from_home_km, category, job, gender, age_at_txn, city_pop_log
```

The Sparkov raw-ish protocol uses its 21-feature selection at both sequence
lengths while keeping the same rows, split, label-delay regimes, architecture,
and five seeds as the full-feature Sparkov runs:

```bash
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --feature-set rawish --window-length 10 --device cuda
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --feature-set rawish --window-length 30 --device cuda
```

Its separate records have `_rawish_len10` or `_rawish_len30` suffixes, and the
generated tables are `results/sparkov_lstm_delay_rawish_len10.md` and
`results/sparkov_lstm_delay_rawish_len30.md`. The 41-feature IBM selection above
is a documented proposal; it is not part of these Sparkov runs.

This removes rolling counts, sums, means, z-scores, recency gaps, entity
ordinals, first-occurrence and distinct-count flags, rarity scores, and every
label-derived field. No entity ID, absolute timestamp, report timestamp,
or label is an input. The earlier length-10 runs use their full current feature
sets; these Sparkov cells change the feature axis separately.
