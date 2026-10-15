# Synthetic label delay for any transaction table

`add_label_delay.py` is one standalone file: copy it into your project. It needs
`numpy`, `pandas` and, for Parquet, `pyarrow`, and nothing from this repository. It
adds the same `campaign_id` and `reported_at` columns that the
[fraud-benchmark](../../README.md) pipeline uses for its Sparkov label-delay study,
and for the same inputs and parameters it reproduces them exactly.

## Why

A detector trained at some cutoff only knows the frauds that were **reported** by
then. Frauds committed before the cutoff but reported after it are not missing from
the training data; they sit in it looking legitimate. No public fraud dataset records
when a fraud became known, so this script synthesises that timestamp.

## Method

1. **Campaigns.** Frauds on one entity (card, user, account), each within
   `--campaign-gap` of the previous one, form a campaign: an investigation
   discovers them together.
2. **Delay.** One delay per campaign is drawn from a lognormal in days, seeded, and
   added to the campaign's **last** transaction, so a campaign is never reported
   before it ends.
3. **Timestamp.** Every fraud in the campaign gets that `reported_at`. Non-fraud
   rows get none (empty / NaT).

**Every `reported_at` produced here is synthetic.** The presets are calibrated to be
plausible for a domain, not fitted to observed reporting behaviour. Describe them
that way when you publish results.

## Usage

```bash
python add_label_delay.py transactions.parquet \
    --entity card_id --time timestamp --fraud is_fraud
```

This writes `transactions.label_delay.parquet`, which is the input plus `campaign_id`
and `reported_at`, with the row order and other columns unchanged. It also writes
`transactions.label_delay.json`, which records the parameters, seed, script version
and the realised delays. CSV works the same way. Use `-o` to choose the output path.

| flag | meaning |
|---|---|
| `--entity`, `--time`, `--fraud` | your column names. The fraud column must hold 0/1 or booleans. |
| `--time-origin`, `--time-unit` | for a numeric time column, e.g. `--time-origin 2017-12-01 --time-unit s` |
| `--preset` | `card` (default), `card-slow` or `aml`, see below |
| `--median-days`, `--sigma`, `--max-delay-days` | override the preset. `--max-delay-days 0` removes its cap. |
| `--regime NAME:MEDIAN,SIGMA[,CAP]` | an extra regime in `reported_at_NAME`. Repeatable. |
| `--campaign-gap` | longest gap inside one campaign. The default is `1D`. |
| `--seed` | default `0` |
| `--cutoff` | your training cutoff. The script then prints how many frauds up to it are still unreported. |

### Presets

| preset | median | sigma | cap | rationale |
|---|---:|---:|---:|---|
| `card` | 7 d | 1.0 | none | a cardholder disputes inside the chargeback window; p95 ≈ 36 d |
| `card-slow` | 15 d | 1.665 | 365 d | a stress test with a fat tail (nominal mean 60 d), not an estimate of real behaviour |
| `aml` | 30 d | 1.0 | none | suspicious-activity investigations run on a ~30-day clock |

The Sparkov study in this repository uses `card` for its `on` regime and `card-slow`
for its `slow` regime, both with gap `1D` and seed `0`:

```bash
python add_label_delay.py sparkov.parquet --entity cc_num --time event_time \
    --fraud is_fraud --preset card --regime slow:15,1.665,365
```

### Several regimes on the same rows

Each `--regime` writes its own column. All regimes share the campaigns and the seed,
so they differ only in the delay distribution. That makes a paired comparison
possible: train on the same rows under each regime and attribute the difference to
the delay alone. With a shared seed the regimes are also *rank-aligned*: a campaign
reported late under one regime is reported late under the others.

## Choosing parameters: check the censored share

How much a delay matters depends on the delay relative to your **training window**,
not on the delay alone. The same 7-day delay leaves 2.2% of Sparkov's training frauds
unreported (a 487-day window), 0.0% of IBM CCF's (9,629 days) and 47% of PaySim's
(14 days). Pass `--cutoff` and read the printed share. Without `--cutoff`, the script
reports the share at the 50th, 70th and 80th percentiles of time as a guide.

To censor more, raise `sigma` before raising the median: a fatter tail censors more
without claiming an implausible typical reporting time. For bursty data, shrink
`--campaign-gap`, or a 1-day gap will fuse many frauds into a few huge campaigns.

## Using the column: censor training labels only

At training time, a fraud reported after the cutoff becomes label `0`. Keep the row;
do not drop it. At that moment the row is indistinguishable from a legitimate one,
and dropping it would presume knowledge of which rows to distrust:

```python
def censored_labels(train: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    known = train['is_fraud'].astype(bool) & (train['reported_at'] <= cutoff)
    return known.to_numpy().astype(int)
```

Fit on these labels, but **score validation and test rows against the true fraud
labels**. Censoring the evaluation labels would measure the reporting delay rather
than the detector. Do not feed `reported_at` or `campaign_id` to a model as features,
since both are derived from the labels.

## Details that affect reproducibility

- Entities are compared as strings, so `123` and `"123"` give the same result.
  Campaign numbering follows the sorted `(entity, time)` order, and draw *i* goes to
  campaign *i*. The same rows, gap and seed therefore always give the same output.
  Changing the rows or the gap reshuffles which campaign receives which draw.
- Timezone-aware times are converted to UTC and written without a timezone.
- Rows with equal timestamps on one entity keep their input order.
