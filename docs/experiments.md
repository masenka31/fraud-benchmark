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
