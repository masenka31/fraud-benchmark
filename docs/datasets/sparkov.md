# Sparkov (Shenoy) — simulated card transactions, real timestamps

[kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
· CC0 1.0 (public domain) · **simulated with Sparkov/Faker**

**The clean one.** The audit found **zero** flagged values — the only generated dataset
in the suite whose labels are not recoverable from a single column value. It is the
benchmark's negative control: in the leakage ablation its `leaky` and `clean` conditions
are identical by construction, so the gap between them measures the harness's own noise
floor (0.000–0.002).

| | |
|---|---|
| rows / frauds | 1,852,394 / 9,651 (**0.521%**) |
| entities | 999 cards, ~1,850 rows each |
| span | 730 days, second granularity |
| splits | **63 / 7 / 30** — not the project default, see below |
| columns | 29 |
| delay | median 7d, **2.2% of train labels censored**; 1,005 campaigns, median size 10 |

**Schema.** `event_time` ← `trans_date_trans_time` · `entity_id` ← `cc_num` ·
`amount` ← `amt` · `is_fraud` ← `is_fraud` (cast in place).
Passed through: `merchant`, `category`, customer demographics (`first`, `last`,
`gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`),
`merch_lat`, `merch_long`, `trans_num`, `unix_time`, `source_file`.

## Artifacts and disclaimers

⚠ **It does not use the project's global temporal split.** The upstream test file is
preserved as the test split so results stay comparable with published work, and
validation is the last 10% of the upstream train file (`datasets.sparkov.val_fraction`).
Hence 63/7/30. The two files are consecutive in time, so train → val → test is still
strictly ordered.

⚠ **`source_file` predicts the split perfectly** — it records which upstream file each
row came from. Drop it before fitting; it is in
`fraud_benchmark.experiments.columns.ALWAYS_EXCLUDED`.

⚠ **`unix_time` and `trans_date_trans_time` restate `event_time` as an absolute clock.**
Because the splits are temporal, a tree can isolate the split boundary as a threshold
(train max 1367492969 < val min 1367493022). A dtype check does not catch them —
`trans_date_trans_time` is stored as a string and `unix_time` as an int.

- `job`, `city`, `state` and `dob` are **per-customer constants** across 999 customers,
  so any apparent effect in them is a customer effect seen through a demographic column.
  The strongest anywhere is `job == "TEFL teacher"` at 8.1× over 760 rows.
- `category` tops out at `shopping_net` 3.1× and `grocery_pos` 2.4× — the categories
  real card fraud does favour, not an injected artifact.
- Customer names, addresses, jobs and dates of birth are Faker output, not real people.
- `trans_num` is a row identifier; the per-file index column is dropped as meaningless
  after concatenation.
- **2.2% censoring is weak** — see [`sparkov_slow`](sparkov_slow.md) for the harsher
  regime on identical rows.
