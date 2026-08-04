# SAML-D — synthetic AML transaction monitoring

[kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
· **CC BY-NC-SA 4.0 (NonCommercial + ShareAlike)** · **synthetic**

**Money laundering, not card fraud.** The label is `Is_laundering`, so "fraud" here means
a laundering typology. It carries the suite's most realistic delay regime, and the audit
found **nothing oracle-like** — no value anywhere reaches a 50% fraud rate.

| | |
|---|---|
| rows / frauds | 9,504,852 / 9,873 (**0.104%** — the lowest base rate here) |
| entities | 292,715 sender accounts |
| span | 320 days, second granularity |
| splits | 80 / 10 / 10, temporal |
| columns | 19 |
| delay | median **30d** (AML), **20.0% of train labels censored** — the harshest realistic regime; 7,887 campaigns |

**Schema.** `event_time` ← `Date`+`Time` · `entity_id` ← `Sender_account` ·
`amount` ← `Amount` · `is_fraud` ← `Is_laundering`.
Passed through: `Receiver_account`, `Payment_currency`, `Received_currency`,
`Sender_bank_location`, `Receiver_bank_location`, `Payment_type`, `Laundering_type`.

## Artifacts and disclaimers

⚠ **`Laundering_type` describes the label, and is kept on purpose.** It records the
typology for laundering rows (Smurfing, Fan-Out, …) and `Normal_*` values otherwise —
information `is_fraud` reduces to a bool, which is why the frame keeps it while
`Is_laundering` is dropped. The adapter declares it in `label_descriptive_columns`, and
`fraud_benchmark.experiments.columns.ALWAYS_EXCLUDED` reads that declaration, so it stays
in the data and out of every model. Never use it as an input.

⚠ **Amounts span 13 currencies and are NOT converted.** `Payment_currency` and
`Received_currency` differ per row, so **cross-row amount comparison is not meaningful**
and neither is any aggregate built on `amount` without first normalising.

⚠ **`entity_id` is the sender only.** Laundering is a multi-party phenomenon; the
receiving side lives in `Receiver_account` and is passed through but not keyed on.
`experiments/features/saml_d.py` builds a second history keyed on it, so fan-in is
available as a feature; campaign grouping — and therefore `reported_at` — is per sender.

⚠ **The engineered counterparty features roughly double the achievable score, and
the strongest one is a generator property.** XGBoost on the 63 features in
`experiments/features/saml_d.py` reaches **0.9919 ± 0.0003** test average precision,
against **0.500** for the retired ablation's raw columns on the same dataset. The gap is
not leakage — every feature is past-only and tested as such — it is one feature:
`first_receiver_for_entity`, "first time this sender has paid this receiver", earns
**35% of the tree's total gain**, with `payment_type` next at 23%.

That flag is predictive here mostly *by construction*. The laundering typologies create
fresh sender→receiver pairs (fan-out to mules) while normal traffic recurs against
established ones, so "new counterparty" separates the two almost by definition of how
the data was generated. A real AML system would also find new counterparties
informative, but nowhere near this cleanly. **Do not read 0.99 as evidence that
laundering detection is solved**; read it as this generator being separable.

⚠ **Fan-out separates the opposite way from the textbook pattern.** Measured over all
9.5M rows of the feature parquet, laundering rows average **2.90** distinct receivers in 7
days against **6.96** for normal rows, and **1.45** distinct senders per receiver against
**3.72**. Smurfing predicts the reverse. The cause is the generator: **61% of the normal
rows are explicit fan patterns** — `Normal_Small_Fan_Out` 3.48M, `Normal_Fan_Out` 2.30M,
`Normal_Fan_In` 2.10M — so the background traffic is fan-heavy by construction, while the
9,873 laundering rows spread across a dozen typologies of which only some are fan-shaped
(`Structuring` 1,870, `Smurfing` 932, `Layered_Fan_In` 656). The gap is large and a model
does not care about its sign, but **do not read a high fan-out here as evidence of
laundering**, and do not carry that reading to another dataset.

- **The strongest effects are the phenomenon, not artifacts.** `Receiver_bank_location`
  Nigeria 6.3×, Morocco 6.3×, Albania 5.8×; `Payment_type` Cash Deposit 6.0×, Cash
  Withdrawal 4.3× — directional in the way money laundering actually is. The audit kept
  the two `*_bank_location` columns in its `clean` condition for this reason; the feature
  module prefixes them `artifact_` because they name absolute places, not because they are
  suspect.
- The 73 flagged values are all account ids (25 receiver, 24 sender, plus their
  `entity_id` duplicates), none holding more than 0.4% of frauds — recurring mule
  accounts.
- 85.1% of campaigns are singletons; the largest is 10 rows.
- Its 30-day median is the AML regime, 4× the card-fraud default — an alert opens an
  investigation and the SAR clock starts only at detection. See
  [`../label-delay.md`](../label-delay.md).
- **NonCommercial**: excluded by `prepare --all --exclude-noncommercial`.
