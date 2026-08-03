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

⚠ **`Laundering_type` describes the label.** It records the typology for laundering rows
(Smurfing, Fan-Out, …) and `Normal_*` values otherwise, so it is label-adjacent, not an
input. It is in the ablation's `ALWAYS_EXCLUDED` set. Drop it, together with
`Is_laundering`.

⚠ **Amounts span 13 currencies and are NOT converted.** `Payment_currency` and
`Received_currency` differ per row, so **cross-row amount comparison is not meaningful**
and neither is any aggregate built on `amount` without first normalising.

⚠ **`entity_id` is the sender only.** Laundering is a multi-party phenomenon; the
receiving side lives in `Receiver_account` and is passed through but not keyed on.
Campaign grouping — and therefore `reported_at` — is per sender.

- **The strongest effects are the phenomenon, not artifacts.** `Receiver_bank_location`
  Nigeria 6.3×, Morocco 6.3×, Albania 5.8×; `Payment_type` Cash Deposit 6.0×, Cash
  Withdrawal 4.3× — directional in the way money laundering actually is. The audit kept
  the two `*_bank_location` columns in the `clean` condition for this reason.
- The 73 flagged values are all account ids (25 receiver, 24 sender, plus their
  `entity_id` duplicates), none holding more than 0.4% of frauds — recurring mule
  accounts.
- 85.1% of campaigns are singletons; the largest is 10 rows.
- Its 30-day median is the AML regime, 4× the card-fraud default — an alert opens an
  investigation and the SAR clock starts only at detection. See
  [`../label-delay.md`](../label-delay.md).
- **NonCommercial**: excluded by `prepare --all --exclude-noncommercial`.
