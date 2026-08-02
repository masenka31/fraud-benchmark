# Per-dataset label delay, and two IBM CCF subsamples

**Status:** approved 2026-08-02. Follows Plan 4 (label delay), which is complete.

## Why

Plan 4 shipped one global lognormal delay for all seven datasets. Verifying it on real
data exposed two datasets where a single global distribution gives a degenerate result.

**PaySim is over-delayed.** Its entire span is 30 simulated days, so the train window is
14 days. Against a 7-day median delay, 2,090 of 3,963 train frauds are still unreported at
the cutoff — only 47.3% of train labels are usable.

**IBM CCF is effectively undelayed.** Its span is 10,649 days (1991–2020), so a 7-day delay
censors 3 frauds out of 24,924 — 100.0% known at the cutoff. It is the largest dataset in
the suite, and it exercises delay-aware evaluation not at all.

## Two measured obstacles in IBM CCF

**Labelling stops on 2019-10-27 14:54**, while transactions continue to 2020-02-28. The
final four months carry 645,180 transactions and zero frauds:

| month | rows | frauds |
|---|---:|---:|
| 2019-09 | 141,744 | 142 |
| 2019-10 | 145,074 | 275 |
| 2019-11 | 141,946 | 0 |
| 2019-12 | 146,579 | 0 |
| 2020-01 | 170,731 | 0 |
| 2020-02 | 165,769 | 0 |

A labelling-window artifact, not a fraud-free quarter. The obvious construction — take the
last N years of transactions — puts the entire test split inside the dead zone, yielding
0 frauds in test. The right edge must therefore be the last labelled fraud.

**2017 is a near-empty fraud year**, a 14× dip against both neighbours:

| year | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---:|---:|---:|---:|---:|
| frauds | 3,281 | 3,579 | **255** | 2,491 | 2,087 |

Almost certainly a second labelling artifact. Any window covering 2016–2019 contains it. At
the chosen left crop it sits inside the train split, degrading training density but leaving
val and test — both in 2019 — unaffected. Accepted rather than worked around, but it means
the effective fraud supply is lumpier than a single rate suggests.

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
`DelayParams.__post_init__`. An invalid override is rejected by exactly the same validation
the global block gets, and an unknown key raises rather than being silently ignored. Both
are wrapped into `ConfigError` for consistency with the rest of config loading.

`pipeline.prepare` uses `config.delay_for(name)` in place of `config.delay`, and
`_describe_delay` records the **resolved** parameters so a dataset card never misreports
what produced its timestamps.

### 2. PaySim: median delay of 1 day

Config only, no code. `sigma` and `seed` inherit, so the tail shape is unchanged and only
the scale moves.

### 3. Two new datasets

Both crop IBM CCF to `[2016-01-01, last labelled fraud]` and differ **only** in their delay
distribution, so a model's performance can be compared across delay regimes on identical
rows.

| | `ibm_ccf_subsample_fast` | `ibm_ccf_subsample_slow` |
|---|---|---|
| median | 7 d | 15 d |
| nominal mean | 30 d | 60 d |
| sigma | 1.7060 | 1.6651 |
| max_delay_days | 365 | 730 |
| realised mean | 25.3 d | 51.3 d |
| realised p95 | 115 d | 231 d |
| known @ train cutoff | 97.0% | 94.8% |

`sigma = sqrt(2*ln(mean/median))` is the closed form that places a lognormal's mean and
median where we want them.

Shared window contents:

| | rows | frauds | span |
|---|---:|---:|---:|
| total | 6,569,157 | 8,412 (0.128%) | 1,395d |
| train | 5,255,327 | 6,441 | 1,117d |
| val | 656,919 | 989 | 139d |
| test | 656,911 | 982 | 139d |

The left crop is a fixed `start_date`, matching the option paysim, banksim and ieee_cis
already use. 2016-01-01 rather than a relative three-year window because the three-year
window starts in late October 2016 and discards most of 2016, a heavy fraud year: +27% rows
buys +58% frauds.

**Truncation is deliberate and its cost is recorded.** Untruncated, these distributions
sample maxima of 1,813 and 3,400 days — a fraud reported nine years after it happened, past
the end of the dataset. The caps bite ~1% of campaigns and pull the realised mean about 15%
below nominal. The 2× ratio between the regimes survives, which is what the comparison needs.
Cards and docs record realised figures, not nominal ones.

Structure: a non-registered `IbmCcfSubsampleAdapter(IbmCcfAdapter)` holds the crop, and two
registered subclasses differ only in `name`. The join, money parsing and `entity_key` logic
are inherited, not copied.

```python
class IbmCcfSubsampleAdapter(IbmCcfAdapter):
    """Not registered: the two concrete variants below differ only in delay config."""
    raw_name = "ibm_ccf"

    def to_canonical(self, raw_dir, options):
        df = super().to_canonical(raw_dir, options)
        last = df.loc[df["is_fraud"], "event_time"].max()
        start = pd.Timestamp(options["start_date"])
        return df[(df["event_time"] >= start) & (df["event_time"] <= last)].reset_index(drop=True)


@register
class IbmCcfSubsampleFastAdapter(IbmCcfSubsampleAdapter):
    name = "ibm_ccf_subsample_fast"


@register
class IbmCcfSubsampleSlowAdapter(IbmCcfSubsampleAdapter):
    name = "ibm_ccf_subsample_slow"
```

Both crops are recorded in `caveats`, so the reason reaches the dataset card.

### 4. Sharing the raw download

`pipeline.prepare` fetches into `config.raw_dir / name`, keyed on the dataset name. Left
alone, each new variant would re-download the same 3.0 GB Kaggle archive.

`DatasetAdapter` gains `raw_name: str`, defaulting to `name`; the pipeline fetches into
`config.raw_dir / adapter.raw_name`. Both variants set it to `ibm_ccf`. Inert for the seven
existing adapters.

## Testing

- **Config:** partial merge, inheritance of unnamed keys, unknown-key rejection,
  invalid-value rejection, and that a dataset with no `delay:` block gets the global params.
- **Adapter:** reuses `tests/fixtures/ibm_ccf` (5 rows, 2002–2011, last fraud 2011-01-01).
  A `start_date` inside that range pins both edges — rows after the last fraud are dropped,
  rows before the start are dropped. `entity_key` still works through the subclass.
- **Registry:** both variants registered, `raw_name` resolves to `ibm_ccf` for both, and the
  default for every other adapter equals its `name`.
- **Pipeline:** the card records resolved rather than global delay params.
- **Live:** prepare the affected datasets and record new figures in
  `docs/verification-notes.md`, including that fast and slow are row-identical and differ
  only in `reported_at`.

## Costs and known limits

- **+580 MB** processed disk (~290 MB per variant). Home is quota-constrained; check
  headroom before the live run.
- **Peak memory is unchanged from `ibm_ccf`,** and now paid twice. Each variant materialises
  the full 24.4M-row joined frame before cropping. Cropping before the joins would fix it and
  is deliberately out of scope.
- **PaySim's delay remains large relative to its clock** even at a 1-day median. The override
  makes the dataset usable, not realistic.
- **The two variants duplicate 6.5M rows of identical feature data.** Justified by keeping the
  canonical one-`reported_at`-per-row schema intact; the alternative of a second timestamp
  column would break Plan 4's validation.

## Not in this scope

- Changing the global delay distribution or the campaign gaps. Both stay as verified.
- Per-dataset delay for the other five datasets. Their figures (82.5%–97.8% known) are
  already in the intended range.
- Reworking `ibm_ccf`'s memory profile or its `category` dtypes. Tracked in
  `docs/verification-notes.md` under "Open, non-blocking".
- Any attempt to repair the 2017 fraud hole or the 2019-10 labelling cliff. Both are
  documented and worked around, not fixed.
