# Amaretto — synthetic capital-market AML

[github.com/necst/amaretto_dataset](https://github.com/necst/amaretto_dataset)
· MIT · **fully synthetic**, built from aggregate real market parameters

**Not payments.** Rows are securities buy/sell orders, so `amount` is a normalised trade
value rather than a transfer.

| | |
|---|---|
| rows / anomalies | 29,704,090 / 81,268 (**0.274%**) |
| entities | 400 clients — but only **21** carry any anomaly |
| span | 83 days, second granularity |
| splits | 80 / 10 / 10, temporal |
| columns | 19 |
| delay | median 7d (card default, **too fast for AML** — see below), **17.5% censored**; campaign gap **1h**, 1,852 campaigns |

**Schema.** `event_time` ← `EntryDate` · `entity_id` ← `Originator` ·
`amount` ← `Normalized Amount` · `is_fraud` ← `Anomaly > 0`.
Passed through: `Transaction ID`, `InputOutput`, `Market`, `Product ISIN`,
`Product Type`, `Product Class`, `Currency`, `Anomaly`.

## Artifacts and disclaimers

⚠ **`Anomaly` is NOT binary.** It is 0 plus **five classes** matching the FATF typologies
described upstream. `is_fraud` is `Anomaly > 0` and the class is retained, so any
per-class claim must read `Anomaly`, not `is_fraud`. It is also the source label column —
drop it before fitting.

⚠ **One whole typology is a one-liner.** `amount` an exact multiple of 1,000 is 2,817
rows, **91.4% anomalous**, and those rows are **all 2,576 members of `Anomaly` class 2 —
the entire class, with no members outside the rule.** Values are $8,000–$13,000; $13,000
is 463 rows at 100%. On test the rule scores 0.948 precision.

⚠ **Three clients hold 83% of all anomalies** — Client_066 45.1%, Client_212 22.9%,
Client_126 15.0%. With 21 of 400 clients anomalous at all, any per-entity feature is close
to an identity lookup.

⚠ **The delay is deliberately wrong for the domain.** Amaretto is AML, but its 83-day span
cannot carry an AML clock: a realistic 30-day median censors **53.4%** of train frauds and
15 days censors 33.6% — degenerate either way. It therefore keeps the 7-day card-fraud
default, which is too fast for money laundering. Documented, not repaired — see
[`../label-delay.md`](../label-delay.md).

- **Anomalies are violently bursty**: 0.7-minute median inter-arrival, 97.7% of
  consecutive pairs within an hour — three orders of magnitude burstier than card fraud.
  Hence the **1-hour** campaign gap; at the 1-day default this collapses to 490 episodes
  with a 3,870-row maximum. Even at 1h the largest campaign is 1,145 rows.
- `Originator_ID` is the constant `'_XID'` in every row and carries no information.
- **Distributed as a 34-part split zip inside a git repository.** The adapter reassembles
  and extracts it once, caching ~3.6 GB under `data/raw/amaretto/_extracted/`
  (gitignored). Deleting the cache costs one reassembly.
- Upstream's README undercounts its own anomalies (states 81,262; the file has 81,268).
- Its 29.7M rows make it the second-largest dataset here — 1.6 GB processed.
