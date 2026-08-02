# Label Delay Implementation Plan (Plan 4 of 4)

**Status:** done, merged in `93414ec` (branch `feat/label-delay`).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every fraudulent transaction a synthetic `reported_at` timestamp — the moment its label would realistically have become known — with transactions discovered together sharing one timestamp.

**Architecture:** Two new pure modules (campaign grouping, delay sampling) plus one pipeline stage. No adapter changes. `reported_at` is already reserved in `schema.CORE_ORDER`, so column ordering needs no work.

**Tech Stack:** Python 3.13, pandas 3.0.5, numpy, pytest.

---

## Why this exists

The benchmark's point is evaluating fraud detection under realistic label delay. A model
predicting on day *D* can only have trained on labels reported by day *D* — not on labels
that existed in hindsight. Without `reported_at`, every experiment on these datasets
silently trains on the future.

---

## Decisions already made, and the one still open

**Settled with the user:**
- Non-fraud rows get a null `reported_at`. Only frauds are ever "reported".
- Transactions belonging to one discovered campaign share a single `reported_at`.

**Open — flagged for the user, with a recommended default this plan implements:**

1. **Delay distribution.** The user said "something like Poisson or lognormal". This plan
   implements **lognormal**, and the reasoning is worth stating: Poisson models counts of
   events, not a continuous waiting time, and its variance is tied to its mean. Reporting
   delay is a continuous, strongly right-skewed quantity — most frauds surface within days,
   a minority take months — which is the shape lognormal is for. The parameters are
   configurable, so switching is a config change, not a code change.
2. **Campaign gap threshold.** ~~Defaulted to 7 days.~~ **Now settled by measurement** — see
   the table below. Default **1 day globally, overridden to 1 hour for Amaretto.**

The distribution choice is a single config line and does not block implementation.

---

## Measured facts that constrain the design

From `docs/verification-notes.md`, measured on all seven real datasets:

| dataset | frauds | fraud entities | % grouped | max/entity | median group |
|---|---:|---:|---:|---:|---:|
| paysim | 8,213 | 8,213 | 0.0% | 1 | – |
| banksim | 7,200 | 1,483 | 94.1% | 144 | 3 |
| sparkov | 9,651 | 976 | 100.0% | 19 | 10 |
| saml_d | 9,873 | 4,950 | 60.3% | 37 | 5 |
| ibm_ccf | 29,757 | 1,343 | 99.9% | 113 | 19 |
| ieee_cis | 20,663 | 13,320 | 46.2% | 72 | 3 |
| amaretto | 81,268 | **21** | 100.0% | **36,673** | 756 |

**The grouping rule cannot be entity-only.** Two datasets prove it from opposite ends:

- **PaySim has no campaigns at all** — every fraud has a unique entity. Campaigns there are
  all size 1, and that is correct, not a bug to engineer around.
- **Amaretto would break an entity-only rule.** Its 81,268 anomalies belong to just 21 of
  400 clients, each spanning **76–83 days** — the dataset's whole period. Its anomalies are
  client-level *behaviours* (five FATF typologies), not bursts. Entity-only grouping would
  collapse 36,673 transactions into one reported timestamp.
- **IBM CCF spans 1991–2020.** A card with 19 frauds across three decades is plainly not one
  campaign.

So: group by entity **and** a time gap. The gap does the real work.

### Choosing the gap — measured, not guessed

Campaigns produced at each threshold, as `count / median size / max size`:

| dataset | 1h | 6h | **1d** | 7d | 30d |
|---|---|---|---|---|---|
| paysim | 8,213/1/1 | 8,213/1/1 | **8,213/1/1** | 8,213/1/1 | 8,213/1/1 |
| banksim | 6,515/1/5 | 6,515/1/5 | **5,551/1/35** | 3,884/1/144 | 2,434/1/144 |
| sparkov | 5,479/1/10 | 3,105/3/16 | **1,005/10/19** | 976/10/19 | 976/10/19 |
| saml_d | 9,783/1/4 | 9,371/1/4 | **7,887/1/10** | 5,714/1/10 | 5,437/1/17 |
| ibm_ccf | 20,271/1/9 | 12,800/2/15 | **9,769/2/20** | 4,940/5/29 | 4,545/6/32 |
| ieee_cis | 17,580/1/15 | 16,948/1/15 | **16,170/1/18** | 14,455/1/53 | 13,592/1/72 |
| amaretto | **1,852/8/1,145** | 918/6/3,870 | 490/15/3,870 | 27/657/36,673 | 21/756/36,673 |

**1 day is the right global default.** Sparkov saturates there (1,005 campaigns, unchanged
at 7d and 30d) — its per-card fraud runs genuinely complete within a day. Every other card
dataset keeps a plausible tail: banksim max 35, ibm_ccf max 20, ieee_cis max 18. PaySim
stays all-singletons at every threshold, which is correct rather than a failure.

**Amaretto needs 1 hour and must override the default.** Its anomalies are far burstier than
card fraud: the median gap between consecutive anomalies on one client is **0.7 minutes**,
with 60.7% inside one minute and 97.7% inside one hour. At 1 day it yields 490 campaigns
with a 3,870-row maximum; at 7 days it collapses to 27. At 1 hour it gives 1,852 campaigns,
median 8, max 1,145 — a defensible reading of "one behavioural episode".

Caveat worth carrying: Amaretto's five anomaly classes differ by two orders of magnitude in
burstiness (median consecutive gaps — class 4: 0.6 min, class 5: 0.6 min, class 3: 2.2 min,
class 1: 8.8 min, class 2: 28.3 min). No single gap serves all five. 1 hour keeps the three
fast classes intact and fragments the two slow ones; a per-class rule is possible later if
that fragmentation turns out to matter.

Separately, and not a campaign issue: only **21 of Amaretto's 400 clients** ever carry an
anomaly, and they are indistinguishable from clean clients by transaction volume (median
45,456 vs 44,340). Within those 21, a median of just 2.1% of their own transactions are
anomalous, so transaction-level detection remains a real task — but `entity_id` is close to
a label there, and should not be handed to a model as a raw feature for this dataset.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fraud_benchmark/campaigns.py` | Group fraud rows into campaigns by entity + time gap |
| `src/fraud_benchmark/delay.py` | Sample reporting delays; assign one `reported_at` per campaign |
| `src/fraud_benchmark/pipeline.py` | + the `reported_at` stage, + card provenance |
| `src/fraud_benchmark/config.py` | + `delay` config block |
| `configs/default.yaml` | + delay defaults |
| `src/fraud_benchmark/schema.py` | + validate `reported_at` when present |

---

### Task 1: Campaign grouping

**Why:** this is the piece the measurements above constrain. It must produce size-1
campaigns for PaySim without special-casing, and must not fuse Amaretto's 36,673-row
clients.

**Files:**
- Create: `src/fraud_benchmark/campaigns.py`
- Test: `tests/test_campaigns.py`

- [x] **Step 1: Write the failing tests** — `tests/test_campaigns.py`

```python
import pandas as pd
import pytest

from fraud_benchmark.campaigns import assign_campaigns


def frame(rows):
    """rows: list of (entity_id, 'YYYY-MM-DD', is_fraud)."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series([r[0] for r in rows], dtype="string"),
            "event_time": pd.to_datetime([r[1] for r in rows]),
            "is_fraud": [r[2] for r in rows],
        }
    )


def test_non_fraud_rows_get_no_campaign():
    df = frame([("a", "2023-01-01", False), ("a", "2023-01-02", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert pd.isna(ids.iloc[0])
    assert not pd.isna(ids.iloc[1])


def test_frauds_close_together_share_a_campaign():
    df = frame([("a", "2023-01-01", True), ("a", "2023-01-03", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] == ids.iloc[1]


def test_frauds_far_apart_are_separate_campaigns():
    df = frame([("a", "2023-01-01", True), ("a", "2023-03-01", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] != ids.iloc[1]


def test_the_gap_is_measured_between_consecutive_frauds_not_from_the_first():
    """A drawn-out run of frauds is one campaign if each step is within the gap.

    This is the behaviour that keeps a slow-burn campaign together, and it is
    also why the gap must be chosen with care on datasets like Amaretto.
    """
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-06", True),
        ("a", "2023-01-11", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.nunique() == 1


def test_different_entities_never_share_a_campaign():
    df = frame([("a", "2023-01-01", True), ("b", "2023-01-01", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] != ids.iloc[1]


def test_every_fraud_belongs_to_exactly_one_campaign():
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
        ("b", "2023-01-01", True),
        ("b", "2023-06-01", True),
        ("c", "2023-01-01", False),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.notna().sum() == 4
    assert ids.dropna().nunique() == 3


def test_unsorted_input_is_handled():
    df = frame([
        ("a", "2023-03-01", True),
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    # The two January rows group; March stands alone.
    assert ids.iloc[1] == ids.iloc[2]
    assert ids.iloc[0] != ids.iloc[1]


def test_result_is_aligned_to_the_input_index():
    df = frame([("a", "2023-01-01", True), ("a", "2023-01-02", True)])
    shuffled = df.iloc[::-1]
    ids = assign_campaigns(shuffled, gap=pd.Timedelta(days=7))
    assert ids.index.equals(shuffled.index)


def test_a_zero_gap_makes_every_fraud_its_own_campaign_unless_simultaneous():
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(0))
    assert ids.iloc[0] == ids.iloc[1]
    assert ids.iloc[2] != ids.iloc[0]


def test_negative_gap_is_rejected():
    df = frame([("a", "2023-01-01", True)])
    with pytest.raises(ValueError, match="gap"):
        assign_campaigns(df, gap=pd.Timedelta(days=-1))


def test_no_frauds_yields_all_null():
    df = frame([("a", "2023-01-01", False)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.isna().all()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_campaigns.py -v`
Expected: `ModuleNotFoundError: No module named 'fraud_benchmark.campaigns'`

- [x] **Step 3: Write the implementation** — `src/fraud_benchmark/campaigns.py`

```python
"""Grouping fraudulent transactions into campaigns.

A campaign is a run of frauds on one entity, each within `gap` of the previous.
Real investigations discover these together, so the label-delay stage gives every
transaction in a campaign the same reported_at.

Grouping by entity alone would be wrong. Measured on the real datasets: Amaretto's
81,268 anomalies belong to 21 clients spanning 76-83 days each, and IBM CCF cards
carry frauds decades apart. The time gap is what keeps those apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Name of the column this module produces.
CAMPAIGN_COLUMN = "campaign_id"


def assign_campaigns(df: pd.DataFrame, gap: pd.Timedelta) -> pd.Series:
    """Return a campaign id per row, null for non-fraud rows.

    Ids are integers, dense from 0, in no meaningful order. The result is aligned
    to `df`'s index.
    """
    if gap < pd.Timedelta(0):
        raise ValueError(f"gap must not be negative, got {gap}")

    ids = pd.Series(pd.NA, index=df.index, dtype="Int64", name=CAMPAIGN_COLUMN)
    frauds = df.loc[df["is_fraud"], ["entity_id", "event_time"]]
    if frauds.empty:
        return ids

    ordered = frauds.sort_values(["entity_id", "event_time"], kind="stable")
    entity = ordered["entity_id"]
    elapsed = ordered["event_time"].diff()

    # A new campaign starts at each entity change, or when the wait since the
    # previous fraud on the same entity exceeds the gap.
    #
    # Two pandas-3 traps here, both reproduced on real data before writing this:
    #  * `entity_id` is `string` dtype, so `entity.ne(entity.shift())` yields <NA>
    #    on the first row rather than True. Hence `.fillna(True)`.
    #  * The resulting mask is `bool[pyarrow]`, and `.cumsum()` on that raises
    #    `ArrowNotImplementedError`. Hence the conversion to a numpy bool array.
    starts = (entity.ne(entity.shift()) | elapsed.gt(gap)).fillna(True)
    numbering = np.cumsum(starts.to_numpy(dtype="bool")) - 1
    ids.loc[ordered.index] = pd.array(numbering, dtype="Int64")
    return ids


def campaign_sizes(ids: pd.Series) -> pd.Series:
    """Rows per campaign, for reporting. Ignores nulls."""
    return ids.dropna().value_counts()
```

- [x] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_campaigns.py -v` — expect 11 passed.
Run: `.venv/bin/pytest` — expect 213 passed, 1 deselected.

- [x] **Step 5: Commit**

```bash
git add src/fraud_benchmark/campaigns.py tests/test_campaigns.py
git commit -m "feat: group frauds into campaigns by entity and time gap"
```

---

### Task 2: Delay sampling

**Why:** the campaign is the unit of discovery, so the delay is sampled once per campaign and
measured from the campaign's LAST transaction — an investigation cannot conclude before the
last fraud it covers has happened.

**Files:**
- Create: `src/fraud_benchmark/delay.py`
- Test: `tests/test_delay.py`

- [x] **Step 1: Write the failing tests** — `tests/test_delay.py`

```python
import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.delay import DelayParams, assign_reported_at


def frame(rows):
    """rows: list of (entity_id, 'YYYY-MM-DD', is_fraud, campaign_id)."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series([r[0] for r in rows], dtype="string"),
            "event_time": pd.to_datetime([r[1] for r in rows]),
            "is_fraud": [r[2] for r in rows],
            "campaign_id": pd.Series([r[3] for r in rows], dtype="Int64"),
        }
    )


PARAMS = DelayParams(median_days=7.0, sigma=1.0, seed=0)


def test_non_fraud_rows_have_no_reported_at():
    df = frame([("a", "2023-01-01", False, pd.NA)])
    out = assign_reported_at(df, PARAMS)
    assert out.isna().all()


def test_every_fraud_gets_a_reported_at():
    df = frame([("a", "2023-01-01", True, 0)])
    out = assign_reported_at(df, PARAMS)
    assert out.notna().all()


def test_reported_at_is_never_before_the_transaction():
    df = frame([("a", f"2023-01-{d:02d}", True, d) for d in range(1, 20)])
    out = assign_reported_at(df, PARAMS)
    assert (out >= df["event_time"]).all()


def test_one_campaign_shares_a_single_reported_at():
    df = frame([
        ("a", "2023-01-01", True, 0),
        ("a", "2023-01-02", True, 0),
        ("a", "2023-01-03", True, 0),
    ])
    out = assign_reported_at(df, PARAMS)
    assert out.nunique() == 1


def test_the_shared_timestamp_follows_the_last_transaction_in_the_campaign():
    """A campaign cannot be reported before its final fraud has happened."""
    df = frame([
        ("a", "2023-01-01", True, 0),
        ("a", "2023-06-01", True, 0),
    ])
    out = assign_reported_at(df, PARAMS)
    assert (out >= pd.Timestamp("2023-06-01")).all()


def test_separate_campaigns_get_independent_timestamps():
    df = frame([
        ("a", "2023-01-01", True, 0),
        ("b", "2023-01-01", True, 1),
    ])
    out = assign_reported_at(df, PARAMS)
    assert out.iloc[0] != out.iloc[1]


def test_the_same_seed_reproduces_the_same_timestamps():
    df = frame([("a", f"2023-01-{d:02d}", True, d) for d in range(1, 15)])
    first = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=42))
    second = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=42))
    assert first.equals(second)


def test_a_different_seed_gives_different_timestamps():
    df = frame([("a", f"2023-01-{d:02d}", True, d) for d in range(1, 15)])
    first = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=1))
    second = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=2))
    assert not first.equals(second)


def test_the_median_delay_matches_the_configured_median():
    """Lognormal's median is exp(mu), so median_days should come out directly."""
    df = frame([("a", "2023-01-01", True, i) for i in range(20_000)])
    out = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=0))
    delays = (out - df["event_time"]).dt.total_seconds() / 86_400
    assert 6.6 < delays.median() < 7.4


def test_the_distribution_is_right_skewed():
    """Mean well above median is the property that makes lognormal the right choice."""
    df = frame([("a", "2023-01-01", True, i) for i in range(20_000)])
    out = assign_reported_at(df, DelayParams(median_days=7.0, sigma=1.0, seed=0))
    delays = (out - df["event_time"]).dt.total_seconds() / 86_400
    assert delays.mean() > delays.median() * 1.3


def test_max_delay_days_truncates_the_tail():
    df = frame([("a", "2023-01-01", True, i) for i in range(5_000)])
    params = DelayParams(median_days=7.0, sigma=2.0, seed=0, max_delay_days=30.0)
    out = assign_reported_at(df, params)
    delays = (out - df["event_time"]).dt.total_seconds() / 86_400
    assert delays.max() <= 30.0 + 1e-9


def test_invalid_params_are_rejected():
    for kwargs in (
        {"median_days": 0.0, "sigma": 1.0},
        {"median_days": -1.0, "sigma": 1.0},
        {"median_days": 7.0, "sigma": 0.0},
        {"median_days": 7.0, "sigma": -1.0},
    ):
        with pytest.raises(ValueError):
            DelayParams(seed=0, **kwargs)


def test_result_is_aligned_to_the_input_index():
    df = frame([("a", "2023-01-01", True, 0), ("a", "2023-01-02", True, 0)])
    shuffled = df.iloc[::-1]
    out = assign_reported_at(shuffled, PARAMS)
    assert out.index.equals(shuffled.index)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_delay.py -v`
Expected: `ModuleNotFoundError: No module named 'fraud_benchmark.delay'`

- [x] **Step 3: Write the implementation** — `src/fraud_benchmark/delay.py`

```python
"""Synthetic label-availability delay.

Each campaign is discovered once, some time after its last fraudulent transaction,
and every transaction in it is labelled at that moment. This module samples that
delay and turns it into a `reported_at` timestamp.

Lognormal rather than Poisson: Poisson models a count of events with variance tied
to its mean, whereas reporting delay is a continuous, strongly right-skewed waiting
time — most frauds surface within days, a minority take months.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPORTED_AT_COLUMN = "reported_at"


@dataclass(frozen=True)
class DelayParams:
    """Lognormal delay in days.

    `median_days` is the distribution's median (exp(mu)), which is the intuitive
    handle; `sigma` controls how heavy the tail is. `max_delay_days` optionally
    truncates it.
    """

    median_days: float
    sigma: float
    seed: int
    max_delay_days: float | None = None

    def __post_init__(self) -> None:
        if self.median_days <= 0:
            raise ValueError(f"median_days must be positive, got {self.median_days}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.max_delay_days is not None and self.max_delay_days <= 0:
            raise ValueError(
                f"max_delay_days must be positive, got {self.max_delay_days}"
            )


def assign_reported_at(df: pd.DataFrame, params: DelayParams) -> pd.Series:
    """Return a reported_at timestamp per row, null for non-fraud rows.

    Requires `campaign_id` (from `campaigns.assign_campaigns`), `event_time` and
    `is_fraud`. One delay is drawn per campaign and measured from that campaign's
    last transaction, so a campaign is never reported before it has finished.
    """
    reported = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[us]", name=REPORTED_AT_COLUMN
    )
    labelled = df["campaign_id"].notna()
    if not labelled.any():
        return reported

    # The campaign is discovered after its final transaction.
    last_seen = df.loc[labelled].groupby("campaign_id")["event_time"].max()

    rng = np.random.default_rng(params.seed)
    days = rng.lognormal(
        mean=math.log(params.median_days), sigma=params.sigma, size=len(last_seen)
    )
    if params.max_delay_days is not None:
        days = np.minimum(days, params.max_delay_days)

    per_campaign = last_seen + pd.to_timedelta(days, unit="D")
    reported.loc[labelled] = df.loc[labelled, "campaign_id"].map(per_campaign)
    return reported
```

Note `last_seen` is sorted by `campaign_id` because `groupby` sorts by default, so the
draws are reproducible for a given seed regardless of row order.

- [x] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_delay.py -v` — expect 13 passed.
Run: `.venv/bin/pytest` — expect 226 passed, 1 deselected.

- [x] **Step 5: Commit**

```bash
git add src/fraud_benchmark/delay.py tests/test_delay.py
git commit -m "feat: lognormal label-delay sampling, one draw per campaign"
```

---

### Task 3: Configuration

**Files:**
- Modify: `src/fraud_benchmark/config.py`, `configs/default.yaml`
- Test: `tests/test_config.py`

- [x] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
import pandas as pd


def test_default_config_has_delay_settings():
    config = load_config()
    assert config.delay.median_days == 7.0
    assert config.delay.sigma == 1.0
    assert config.delay.seed == 0


def test_default_campaign_gap_is_one_day():
    assert load_config().campaign_gap_for("paysim") == pd.Timedelta(days=1)


def test_amaretto_overrides_the_campaign_gap_to_one_hour():
    """Measured: Amaretto's anomalies have a 0.7-minute median inter-arrival.

    At the 1-day default it yields 490 episodes with a 3,870-row maximum; at
    1 hour, 1,852 episodes with a median of 8.
    """
    assert load_config().campaign_gap_for("amaretto") == pd.Timedelta(hours=1)


def test_delay_settings_can_be_overridden(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "delay:\n"
        "  median_days: 30.0\n"
        "  sigma: 1.5\n"
        "  seed: 99\n"
        "  max_delay_days: 180.0\n"
        "campaign:\n"
        "  gap: 2d\n"
    )
    config = load_config(path)
    assert config.delay.median_days == 30.0
    assert config.delay.max_delay_days == 180.0
    assert config.campaign_gap_for("paysim") == pd.Timedelta(days=2)


def test_a_per_dataset_gap_override_wins(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "campaign:\n"
        "  gap: 2d\n"
        "datasets:\n"
        "  banksim:\n"
        "    campaign_gap: 30min\n"
    )
    config = load_config(path)
    assert config.campaign_gap_for("banksim") == pd.Timedelta(minutes=30)
    assert config.campaign_gap_for("paysim") == pd.Timedelta(days=2)


def test_invalid_delay_settings_raise_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("delay:\n  median_days: -1.0\n")
    with pytest.raises(ConfigError, match="median_days"):
        load_config(path)


def test_an_unparseable_gap_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("campaign:\n  gap: not-a-duration\n")
    with pytest.raises(ConfigError, match="gap"):
        load_config(path)


def test_a_negative_gap_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("campaign:\n  gap: -3d\n")
    with pytest.raises(ConfigError, match="negative"):
        load_config(path)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: `AttributeError: 'Config' object has no attribute 'delay'`.

- [x] **Step 3: Extend `src/fraud_benchmark/config.py`**

Import the params type at the top:

```python
from fraud_benchmark.delay import DelayParams
```

Add two fields to the frozen `Config` dataclass, after `split_ratios`:

```python
    delay: DelayParams
    campaign_gap: pd.Timedelta
```

`Config` also gains a resolver beside `for_dataset`, because the right gap is not the same
for every dataset — Amaretto's anomalies are three orders of magnitude burstier than card
fraud:

```python
    def campaign_gap_for(self, name: str) -> pd.Timedelta:
        """The campaign gap for one dataset, honouring a per-dataset override."""
        override = self.for_dataset(name).get("campaign_gap")
        return _parse_gap(override) if override is not None else self.campaign_gap
```

`import pandas as pd` at the top of `config.py`.

Add these builders beside `_validate_ratios`:

```python
def _parse_gap(value) -> pd.Timedelta:
    """Parse a duration like '1d', '1h', '30min' into a Timedelta."""
    try:
        gap = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"campaign gap {value!r} is not a valid duration: {exc}") from exc
    if gap != gap:  # NaT, which pd.Timedelta yields for some bad input
        raise ConfigError(f"campaign gap {value!r} is not a valid duration")
    if gap < pd.Timedelta(0):
        raise ConfigError(f"campaign gap must not be negative, got {value!r}")
    return gap


def _build_delay(data: dict) -> DelayParams:
    delay = data.get("delay") or {}
    try:
        return DelayParams(
            median_days=float(delay.get("median_days", 7.0)),
            sigma=float(delay.get("sigma", 1.0)),
            seed=int(delay.get("seed", 0)),
            max_delay_days=(
                float(delay["max_delay_days"])
                if delay.get("max_delay_days") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid delay settings: {exc}") from exc
```

And populate them in `load_config`'s returned `Config`:

```python
        delay=_build_delay(data),
        campaign_gap=_parse_gap((data.get("campaign") or {}).get("gap", "1d")),
```

Note `pd.Timedelta("not-a-duration")` raises `ValueError`, so the try/except covers the
unparseable case; the `gap != gap` check catches inputs that yield `NaT` instead.

- [x] **Step 4: Add defaults to `configs/default.yaml`**

Insert above the `datasets:` block:

```yaml
# A campaign is a run of frauds on one entity, each within `gap` of the last.
# Entity alone is not enough: IBM CCF cards carry frauds decades apart, and
# Amaretto's anomalies would collapse into 27 episodes at a 7-day gap.
# 1 day is where Sparkov saturates (1,005 episodes, unchanged at 7d and 30d)
# while keeping every other card dataset's tail plausible.
campaign:
  gap: 1d

# Reporting delay, lognormal in days. median_days is the distribution's median;
# sigma controls the tail. Seeded so runs are reproducible.
delay:
  median_days: 7.0
  sigma: 1.0
  seed: 0
  max_delay_days: null
```

And add the Amaretto override inside the existing `datasets:` block, beside its other keys:

```yaml
  amaretto:
    # Anomalies here are far burstier than card fraud: 0.7-minute median
    # inter-arrival, 97.7% of consecutive pairs within an hour. At the 1-day
    # default this yields 490 episodes with a 3,870-row maximum; at 1 hour,
    # 1,852 episodes with a median of 8.
    campaign_gap: 1h
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_config.py -v` — all pass.
Run: `.venv/bin/pytest` — expect 230 passed, 1 deselected.

Every existing test that constructs `Config(...)` directly will now fail for missing
arguments. Fix those constructions by passing
`delay=DelayParams(median_days=7.0, sigma=1.0, seed=0), campaign_gap=pd.Timedelta(days=1)`. If more
than the test fixtures need changing, STOP and report — that would mean `Config` is
constructed in production code somewhere it should not be.

- [x] **Step 6: Commit**

```bash
git add src/fraud_benchmark/config.py configs/default.yaml tests/test_config.py \
        tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: campaign gap and delay distribution configuration"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `src/fraud_benchmark/pipeline.py`, `src/fraud_benchmark/schema.py`
- Test: `tests/test_pipeline.py`, `tests/test_schema.py`

- [x] **Step 1: Write the failing tests**

Add to `tests/test_schema.py`:

```python
def test_validate_rejects_reported_at_before_event_time():
    df = make_valid_frame()
    # Set it ONLY on the fraud row (row 0). Setting it on both would trip the
    # non-fraud check first, and this test would pass without ever exercising
    # the ordering check it is named for.
    df["reported_at"] = pd.Series(
        [df["event_time"].iloc[0] - pd.Timedelta(days=1), pd.NaT],
        dtype="datetime64[us]",
    )
    with pytest.raises(SchemaError, match="precedes event_time"):
        validate_canonical(df)


def test_validate_rejects_reported_at_on_a_non_fraud_row():
    df = make_valid_frame()
    # Later than event_time, so the ordering check cannot be what fires here.
    df["reported_at"] = df["event_time"] + pd.Timedelta(days=1)
    # Row 1 is is_fraud=False, so it must not carry a report timestamp.
    with pytest.raises(SchemaError, match="non-fraud"):
        validate_canonical(df)


def test_validate_accepts_a_correct_reported_at():
    df = make_valid_frame()
    df["reported_at"] = pd.Series(
        [pd.Timestamp("2023-01-05"), pd.NaT], dtype="datetime64[us]"
    )
    validate_canonical(df)


def test_reported_at_is_optional():
    validate_canonical(make_valid_frame())
```

Add to `tests/test_pipeline.py`:

```python
def test_reported_at_is_written(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert "reported_at" in df.columns
    assert df.loc[df.is_fraud, "reported_at"].notna().all()
    assert df.loc[~df.is_fraud, "reported_at"].isna().all()


def test_campaign_id_is_written(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert "campaign_id" in df.columns
    assert df.loc[~df.is_fraud, "campaign_id"].isna().all()


def test_reported_at_never_precedes_the_transaction(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    fraud = df[df.is_fraud]
    assert (fraud["reported_at"] >= fraud["event_time"]).all()


def test_card_records_the_delay_parameters(config, no_download):
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["label_delay"]["distribution"] == "lognormal"
    assert card["label_delay"]["median_days"] == 7.0
    assert card["label_delay"]["seed"] == 0
    assert card["label_delay"]["campaign_gap"] == "1 days 00:00:00"
    assert card["label_delay"]["n_campaigns"] >= 1
    assert card["label_delay"]["largest_campaign"] >= 1
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_schema.py tests/test_pipeline.py -v`
Expected: the new tests fail — no `reported_at` column and no `label_delay` card key.

- [x] **Step 3: Extend validation in `src/fraud_benchmark/schema.py`**

At the end of `validate_canonical`, before it returns:

```python
    if "reported_at" in df.columns:
        reported = df["reported_at"]
        if not pd.api.types.is_datetime64_dtype(reported):
            raise SchemaError(
                f"column 'reported_at' has dtype {str(reported.dtype)!r}; expected a "
                "timezone-naive datetime64 column (any resolution)"
            )
        stray = reported.notna() & ~df["is_fraud"]
        if stray.any():
            raise SchemaError(
                f"column 'reported_at' is set on {int(stray.sum())} non-fraud row(s); "
                "only frauds are ever reported"
            )
        early = reported.notna() & (reported < df["event_time"])
        if early.any():
            raise SchemaError(
                f"column 'reported_at' precedes event_time on {int(early.sum())} "
                "row(s); a fraud cannot be reported before it happens"
            )
```

- [x] **Step 4: Add the stage in `src/fraud_benchmark/pipeline.py`**

Add imports:

```python
from fraud_benchmark.campaigns import assign_campaigns, campaign_sizes
from fraud_benchmark.delay import assign_reported_at
```

In `prepare`, after `df["split"] = ...` and BEFORE `order_columns`:

```python
    df["campaign_id"] = assign_campaigns(df, gap=config.campaign_gap_for(name))
    df["reported_at"] = assign_reported_at(df, config.delay)
```

Validation already ran earlier against the adapter's output, so re-validate the finished
frame to catch a delay-stage bug before anything is written:

```python
    validate_canonical(df)
    df = order_columns(df)
```

Add the card entry inside `_build_card`, after `"split"`:

```python
        "label_delay": _describe_delay(df, config, config.campaign_gap_for(name)),
```

and the helper beside it:

```python
def _describe_delay(df, config, gap) -> dict:
    sizes = campaign_sizes(df["campaign_id"])
    fraud = df.loc[df["is_fraud"]]
    delays = (fraud["reported_at"] - fraud["event_time"]).dt.total_seconds() / 86_400
    return {
        "distribution": "lognormal",
        "median_days": config.delay.median_days,
        "sigma": config.delay.sigma,
        "seed": config.delay.seed,
        "max_delay_days": config.delay.max_delay_days,
        "campaign_gap": str(gap),
        "n_campaigns": int(sizes.size),
        "largest_campaign": int(sizes.max()) if sizes.size else 0,
        "median_campaign_size": float(sizes.median()) if sizes.size else 0.0,
        "observed_median_delay_days": float(delays.median()) if len(delays) else 0.0,
    }
```

- [x] **Step 5: Run tests**

Run: `.venv/bin/pytest` — expect 238 passed, 1 deselected. Report the ACTUAL number.

- [x] **Step 6: Commit**

```bash
git add src/fraud_benchmark/pipeline.py src/fraud_benchmark/schema.py \
        tests/test_pipeline.py tests/test_schema.py
git commit -m "feat: reported_at and campaign_id in the pipeline output"
```

---

### Task 5: Live verification and threshold evidence

Raw data for all seven datasets is cached; preparing all seven takes about 8 minutes.

- [x] **Step 1: Prepare everything**

```bash
.venv/bin/fraud-benchmark prepare --all
```

- [x] **Step 2: Check the invariants on real data**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
NAMES = ("paysim","banksim","sparkov","saml_d","ibm_ccf","ieee_cis","amaretto")
for name in NAMES:
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["event_time","is_fraud","reported_at","campaign_id"])
    f = df[df.is_fraud]
    bad_early = int((f.reported_at < f.event_time).sum())
    bad_null = int(f.reported_at.isna().sum())
    stray = int(df.loc[~df.is_fraud, "reported_at"].notna().sum())
    delays = (f.reported_at - f.event_time).dt.total_seconds() / 86_400
    print(f"{name:9s} frauds={len(f):>7,} early={bad_early} null={bad_null} stray={stray} "
          f"delay p50={delays.median():6.2f}d p95={delays.quantile(0.95):7.2f}d "
          f"max={delays.max():8.2f}d")
PY
```

Every dataset must show `early=0 null=0 stray=0`.

- [x] **Step 3: Measure campaign sizes — the evidence for the gap default**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
NAMES = ("paysim","banksim","sparkov","saml_d","ibm_ccf","ieee_cis","amaretto")
print(f"{'dataset':9s} {'campaigns':>10s} {'size p50':>9s} {'p95':>6s} {'max':>8s} {'singletons':>11s}")
for name in NAMES:
    ids = pd.read_parquet(f"data/processed/{name}/data.parquet",
                          columns=["campaign_id"])["campaign_id"].dropna()
    sizes = ids.value_counts()
    singles = int((sizes == 1).sum()) / sizes.size * 100
    print(f"{name:9s} {sizes.size:>10,} {sizes.median():>9.1f} "
          f"{sizes.quantile(0.95):>6.1f} {int(sizes.max()):>8,} {singles:>10.1f}%")
PY
```

**Judge the 7-day default against this table and report a recommendation.** The figure that
matters most is Amaretto's largest campaign: at an entity-only rule it would be 36,673. If
7 days still yields campaigns in the thousands there, the default is too loose and the
number to propose is one that brings Amaretto's tail into a plausible range without
collapsing Sparkov's and IBM CCF's genuine multi-fraud runs into singletons.

Do NOT quietly change the default — report the evidence and the recommendation.

- [x] **Step 4: Confirm the delay is usable for its purpose**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
# The point of the exercise: at the end of the train split, how many train-split
# frauds would actually be known?
for name in ("banksim","sparkov","ibm_ccf","ieee_cis"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["split","is_fraud","reported_at","event_time"])
    cutoff = df.loc[df.split == "train", "event_time"].max()
    tf = df[(df.split == "train") & df.is_fraud]
    known = (tf.reported_at <= cutoff).mean() * 100
    print(f"{name:9s} train frauds={len(tf):>7,}  known at train cutoff: {known:5.1f}%")
PY
```

Expect high but not 100% — frauds late in the train window are still unreported at the
cutoff, which is exactly the effect the benchmark exists to expose. Report the numbers.

- [x] **Step 5: Update the documentation and commit**

Add a short section to `docs/verification-notes.md` recording the campaign-size table from
Step 3, the delay quantiles from Step 2, and the label-availability figures from Step 4.

```bash
git add -u
git commit -m "test: verify label delay against all seven datasets"
```

---

## Done criteria

- [x] `.venv/bin/pytest` passes
- [x] Every dataset has `reported_at` non-null for frauds, null otherwise, never before `event_time`
- [x] Every dataset has `campaign_id`, null for non-fraud rows
- [x] Dataset cards record the distribution, its parameters, the seed and the campaign stats
- [x] Re-running with the same seed reproduces identical timestamps
- [x] Campaign-size distributions measured on all seven, with a gap recommendation reported

## Not in this plan

- Any evaluation harness or baseline model. This plan produces the labels a delay-aware
  evaluation needs; it does not perform one.
- Per-dataset delay distributions. One global distribution is used, configurable per run.
  If evidence later shows, say, AML reporting is slower than card-fraud reporting, the
  config already supports per-dataset overrides being added.
