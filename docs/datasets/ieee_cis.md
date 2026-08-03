# IEEE-CIS / Vesta — real e-commerce transactions

[kaggle.com/c/ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection)
· **Kaggle competition rules**, not an open licence · **the only real-world data here**

Requires accepting the competition rules **in a browser** once before download —
credentials alone return 403. See [`../kaggle-setup.md`](../kaggle-setup.md).

| | |
|---|---|
| rows / frauds | 590,540 / 20,663 (**3.499%** — by far the highest base rate here) |
| entities | 290,492 **derived** pseudo-cards (see below) |
| span | 181 days, second granularity |
| splits | 80 / 10 / 10, temporal |
| columns | **441** — 394 transaction + identity columns, mostly anonymised |
| delay | median 7d, **7.1% of train labels censored**; 16,170 campaigns, 85.3% singletons |

**Schema.** `event_time` ← `TransactionDT` (seconds since a configured `start_date`) ·
`entity_id` ← derived uid · `amount` ← `TransactionAmt` · `is_fraud` ← `isFraud`.
Passed through: `ProductCD`, `card1-6`, `addr1-2`, `dist1-2`, the e-mail domains, and
Vesta's engineered blocks — `C1-14` (counts), `D1-15` (timedeltas), `M1-9` (matches),
`V1-339` (undocumented aggregates), `id_01-38`, `DeviceType`, `DeviceInfo`.

## Artifacts and disclaimers

⚠ **`entity_id` is not a real card id — it is derived.** IEEE-CIS ships no card
identifier, so the adapter uses the community heuristic: `card1` + `addr1` + a
`D1`-derived account start day. **Rows missing `addr1` or `D1` (~11%) get a per-row
unique id** rather than being fused into a shared bucket, which would fabricate campaigns
out of unrelated transactions. Consequences: the 290,492 entity count is inflated by
those singletons, and campaign grouping is meaningless for them.

⚠ **The 470 flagged values are feature-set structure, not generation leakage.** Every one
lands in an engineered column (`V*`: 114 columns, plus `C*`, `D*`, `id_*`) and none covers
as many as 20,000 rows — the largest is `V242 == 2.0` at 5,218 rows / 35.1% fraud. Vesta's
`V`/`C`/`D` blocks are undocumented aggregates built by a fraud vendor, so narrow
high-fraud slices are exactly what they should look like. The **raw** columns are
unremarkable at a 3.5% base rate: `ProductCD == C` 3.3×, `card6 == credit` 1.9×.

- **`isFraud` is dropped by the pipeline.** `is_fraud` is the only label in the output;
  the card records the removal in `dropped_source_label`.
- **The competition test set has no labels** and is excluded from the benchmark. It is
  written separately as `unlabelled_test.parquet` — 506,691 rows, spanning
  2018-07-02 → 2018-12-31, cleanly after the labelled period.
- **Identity data covers only ~25% of transactions**; the `id_*`, `DeviceType` and
  `DeviceInfo` columns are null for the rest. That is the data, not a join failure.
- `TransactionDT` is a seconds offset with **no stated origin**, so absolute dates carry
  no meaning — they are anchored to a configured `start_date`.
- Preparation emits a `PerformanceWarning: DataFrame is highly fragmented` from
  `df.insert` on a 394-column frame. Cosmetic.
- **Competition rules govern use** and generally prohibit redistribution. Treated as
  non-commercial: excluded by `prepare --all --exclude-noncommercial`.
