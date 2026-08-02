# Per-dataset label delay, and an IBM CCF subsample

**Status:** approved 2026-08-02. Follows Plan 4 (label delay), which is complete.

## Why

Plan 4 shipped one global lognormal delay for all seven datasets. Verifying it on real
data exposed two datasets where a single global distribution gives a degenerate result.

**PaySim is over-delayed.** Its entire span is 30 simulated days, so the train window is
14 days. Against a 7-day median delay, 2,090 of 3,963 train frauds are still unreported at
the cutoff — only 47.3% of train labels are usable. The delay is not wrong in itself; it is
simply enormous relative to a 30-day clock.

**IBM CCF is effectively undelayed.** Its span is 10,649 days (1991–2020), so a 7-day delay
censors 3 frauds out of 24,924 — 100.0% known at the cutoff. It is the most important
dataset in the suite by size, and it exercises delay-aware evaluation not at all.

## The measured obstacle

IBM CCF's fraud labelling stops on **2019-10-27 14:54**, while transactions continue to
2020-02-28. The final four months carry 645,180 transactions and **zero** frauds:

| month | rows | frauds |
|---|---:|---:|
| 2019-09 | 141,744 | 142 |
| 2019-10 | 145,074 | 275 |
| 2019-11 | 141,946 | 0 |
| 2019-12 | 146,579 | 0 |
| 2020-01 | 170,731 | 0 |
| 2020-02 | 165,769 | 0 |

This is a labelling-window artifact, not a fraud-free quarter. It matters because the obvious
construction — take the last three years of transactions — puts the entire test split inside
the dead zone, yielding **0 frauds in test**. The window must therefore be measured backwards
from the last labelled fraud, not from the last transaction.

## What we are building

### 1. Per-dataset delay overrides

`Config` gains `delay_for(name)`, mirroring the existing `campaign_gap_for(name)`. A
dataset's `delay:` block overrides only the keys it names; everything else inherits the
global block, so a change to the global seed still propagates everywhere.

```yaml
delay:                      # global
  median_days: 7.0
  sigma: 1.0
  seed: 0
  max_delay_days: null

datasets:
  paysim:
    delay:
      median_days: 1.0      # sigma, seed, max_delay_days inherited
```

Implemented with `dataclasses.replace(self.delay, **override)`, which re-runs
`DelayParams.__post_init__`. An invalid override is therefore rejected by exactly the same
validation the global block gets, and an unknown key raises rather than being silently
ignored. Both are wrapped into `ConfigError` for consistency with the rest of config loading.

`pipeline.prepare` uses `config.delay_for(name)` in place of `config.delay`, and
`_describe_delay` records the **resolved** parameters so a dataset card never misreports what
produced its timestamps.

### 2. PaySim: median delay of 1 day

Config only — no code. `sigma` and `seed` inherit, so the tail shape is unchanged and only
the scale moves.

### 3. `ibm_ccf_subsample`: a new dataset

An eighth registered dataset, existing alongside the full `ibm_ccf` rather than replacing it.

Window: cropped on the right at the last labelled fraud, extending `window_years` back.

```
(2016-10-27 14:54, 2019-10-27 14:54]
```

Measured contents at `window_years: 3`:

| | rows | frauds | span |
|---|---:|---:|---:|
| total | 5,166,342 | 5,327 (0.103%) | 1,095d |
| train | 4,133,075 | 3,753 | 876d |
| val | 516,633 | 795 | 109d |
| test | 516,634 | 779 | 109d |

Delay: `sigma: 1.706` = `sqrt(2*ln(30/7))`, which places the mean at 30 days with the median
at 7. Expected label availability at the train cutoff: **~94.5%**, against 100.0% for the
full dataset.

The adapter subclasses `IbmCcfAdapter`, inheriting the two table joins, the money-string
parsing and the `entity_key` logic, and overriding only `to_canonical` to crop:

```python
@register
class IbmCcfSubsampleAdapter(IbmCcfAdapter):
    name = "ibm_ccf_subsample"
    raw_name = "ibm_ccf"

    def to_canonical(self, raw_dir, options):
        df = super().to_canonical(raw_dir, options)
        last = df.loc[df["is_fraud"], "event_time"].max()
        start = last - pd.DateOffset(years=int(options.get("window_years", 3)))
        return df[(df["event_time"] > start) & (df["event_time"] <= last)].reset_index(drop=True)
```

The crop-at-last-fraud rule lives in code because it is a property of this dataset's
labelling; only the window length is configurable, so retuning to two years is a one-line
change. That trade-off is live: two years gives 4,833 frauds in 3.45M rows against 5,327 in
5.17M — nearly the same fraud count for a third fewer rows.

The reason for the crop is recorded in the adapter's `caveats`, so it reaches the dataset
card.

### 4. Sharing the raw download

`pipeline.prepare` currently fetches into `config.raw_dir / name`, keyed on the dataset name.
Left alone, `ibm_ccf_subsample` would re-download the same 3.0 GB Kaggle archive into a
second directory.

`DatasetAdapter` gains `raw_name: str`, defaulting to `name`; the pipeline fetches into
`config.raw_dir / adapter.raw_name`. `ibm_ccf_subsample` sets it to `ibm_ccf`. This is the
general answer for any future variant sharing a source, and it is inert for the seven
existing adapters.

## Testing

- **Config:** partial merge, inheritance of unnamed keys, unknown-key rejection,
  invalid-value rejection, and that a dataset with no `delay:` block gets the global params.
- **Adapter:** reuses `tests/fixtures/ibm_ccf` (5 rows, 2002–2011, last fraud 2011-01-01).
  At `window_years: 3` only that final row survives, which pins both edges of the window.
  Also: rows after the last fraud are dropped, and `entity_key` still works through the
  subclass.
- **Registry:** `ibm_ccf_subsample` is registered and `raw_name` resolves to `ibm_ccf`.
- **Pipeline:** the card records resolved rather than global delay params.
- **Live:** prepare the three affected datasets and record new label-availability figures in
  `docs/verification-notes.md`.

## Costs and known limits

- **+230 MB** processed disk. Home is quota-constrained; check headroom before the live run.
- **Peak memory is unchanged from `ibm_ccf`.** The adapter materialises the full 24.4M-row
  joined frame before cropping. The existing note about IBM CCF's tens-of-GB peak applies
  equally here. Cropping before the joins would fix both and is deliberately out of scope.
- **PaySim's delay remains large relative to its clock** even at a 1-day median; a 30-day
  simulated span cannot host a realistic multi-week reporting delay. The override makes the
  dataset usable, not realistic.

## Not in this scope

- Changing the global delay distribution or the campaign gaps. Both stay as verified.
- Per-dataset delay for the other five datasets. Their figures (82.5%–97.8% known) are
  already in the intended range.
- Reworking `ibm_ccf`'s memory profile or its `category` dtypes. Tracked separately in
  `docs/verification-notes.md` under "Open, non-blocking".
