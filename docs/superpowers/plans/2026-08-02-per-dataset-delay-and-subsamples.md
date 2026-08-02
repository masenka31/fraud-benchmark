# Per-Dataset Delay and IBM CCF Subsamples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each dataset override the global label-delay distribution, and add two IBM CCF subsamples — row-identical, differing only in delay regime — cropped to a window that actually exercises delay-aware evaluation.

**Architecture:** Three small, independent changes plus one new adapter. `Config` gains `delay_for(name)` beside the existing `campaign_gap_for(name)`. `DatasetAdapter` gains `raw_name` so variants share a raw download. A non-registered `IbmCcfSubsampleAdapter` holds the crop; two thin registered subclasses differ only in name, with their delay set in config.

**Tech Stack:** Python 3.13, pandas 3.0.5, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-02-per-dataset-delay-and-ibm-ccf-subsample-design.md`

---

## Why this exists

Plan 4 shipped one global delay for all seven datasets. Measured against real data, two
of them are degenerate: PaySim's 30-day span leaves only 47.3% of train labels known at
the cutoff, and IBM CCF's 10,649-day span leaves 100.0% known — no delay effect at all.

Two IBM CCF labelling artifacts constrain the fix. Labelling stops on **2019-10-27** while
transactions run to 2020-02-28, so a naive tail window puts the whole test split in a
fraud-free dead zone (measured: 0 frauds in test). And **2017 has 255 frauds** against
~3,000 in its neighbours. Both are documented, worked around, not repaired.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fraud_benchmark/datasets/base.py` | + `raw_name` attribute |
| `src/fraud_benchmark/pipeline.py` | fetch by `raw_name`; resolve delay per dataset |
| `src/fraud_benchmark/config.py` | + `delay_for(name)` |
| `src/fraud_benchmark/datasets/ibm_ccf_subsample.py` | Crop IBM CCF to a labelled window; two variants |
| `src/fraud_benchmark/datasets/__init__.py` | Register the new module |
| `configs/default.yaml` | PaySim delay override; both variants' config |
| `tests/fixtures/ibm_ccf_subsample/` | Fixture with rows on both sides of the window |

Tasks 1–3 are independent of each other and of Task 5. Task 4 depends on Task 2; Task 6
depends on Tasks 2 and 5.

---

### Task 1: Share the raw download between variants

**Why:** `prepare` fetches into `config.raw_dir / name`. Left alone, each new IBM CCF
variant re-downloads the same 3.0 GB Kaggle archive into its own directory.

**Files:**
- Modify: `src/fraud_benchmark/datasets/base.py`, `src/fraud_benchmark/pipeline.py:29`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_registry.py`:

```python
def test_raw_name_defaults_to_the_dataset_name(fake_adapter):
    assert get_adapter("fake_for_tests").raw_name == "fake_for_tests"


def test_every_registered_adapter_has_a_raw_name():
    """A variant may share another dataset's raw files, but never by accident."""
    for name in list_datasets():
        adapter = get_adapter(name)
        assert isinstance(adapter.raw_name, str) and adapter.raw_name


def test_raw_name_can_be_overridden():
    cls = make_fake_class(name="fake_variant")
    cls.raw_name = "fake_for_tests"
    assert cls().raw_name == "fake_for_tests"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: `AttributeError: '_FakeAdapter' object has no attribute 'raw_name'`

- [ ] **Step 3: Add the attribute**

In `src/fraud_benchmark/datasets/base.py`, inside `class DatasetAdapter`, after the
`name` attribute and its comment:

```python
    #: Directory under data/raw to fetch into. Defaults to `name`. A variant that
    #: reuses another dataset's raw files sets this to that dataset's name, so the
    #: download is shared rather than duplicated.
    raw_name: str = ""
```

An empty-string default plus resolution in `register` keeps `raw_name` correct even for
adapters that never mention it. In `register`, after the existing name check and before
the duplicate check:

```python
    if not getattr(cls, "raw_name", ""):
        cls.raw_name = name
```

- [ ] **Step 4: Use it in the pipeline**

In `src/fraud_benchmark/pipeline.py`, replace line 29:

```python
    raw_dir = fetch(adapter.source, config.raw_dir / adapter.raw_name, force=force)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_registry.py tests/test_pipeline.py -v` — all pass.

Note `test_raw_name_can_be_overridden` uses an unregistered class, so `register` never
runs on it; it asserts the class attribute directly.

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/datasets/base.py src/fraud_benchmark/pipeline.py \
        tests/test_registry.py
git commit -m "feat: let an adapter share another dataset's raw download"
```

---

### Task 2: Per-dataset delay overrides

**Why:** PaySim needs a 1-day median and the subsamples need their own sigma and cap.
Partial merge means a change to the global seed still propagates everywhere.

**Files:**
- Modify: `src/fraud_benchmark/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_a_dataset_without_an_override_gets_the_global_delay():
    config = load_config()
    assert config.delay_for("banksim") == config.delay


def test_a_delay_override_merges_over_the_global_block(tmp_path):
    """Only the named keys move; the rest inherit."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "delay:\n"
        "  median_days: 7.0\n"
        "  sigma: 1.0\n"
        "  seed: 5\n"
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: 2.0\n"
    )
    config = load_config(path)
    delay = config.delay_for("banksim")
    assert delay.median_days == 2.0
    assert delay.sigma == 1.0
    assert delay.seed == 5


def test_a_delay_override_does_not_leak_to_other_datasets(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: 2.0\n"
    )
    config = load_config(path)
    assert config.delay_for("banksim").median_days == 2.0
    assert config.delay_for("sparkov").median_days == config.delay.median_days


def test_a_delay_override_can_set_max_delay_days(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      max_delay_days: 365\n"
    )
    assert load_config(path).delay_for("banksim").max_delay_days == 365.0


def test_an_unknown_delay_key_is_rejected(tmp_path):
    """A typo must fail loudly, not silently inherit the global value."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_day: 2.0\n"
    )
    with pytest.raises(ConfigError, match="median_day"):
        load_config(path)


def test_an_invalid_delay_override_is_rejected(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      median_days: -1.0\n"
    )
    with pytest.raises(ConfigError, match="median_days"):
        load_config(path)


def test_a_non_mapping_delay_override_is_rejected(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay: 7\n"
    )
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_delay_overrides_are_validated_at_load_time(tmp_path):
    """Errors must surface on load_config, not on the later delay_for call.

    A bad override that only raises when a dataset is prepared would let
    `prepare --all` fail halfway through, after writing other datasets.
    """
    path = tmp_path / "custom.yaml"
    path.write_text(
        "datasets:\n"
        "  banksim:\n"
        "    delay:\n"
        "      sigma: 0\n"
    )
    with pytest.raises(ConfigError, match="sigma"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: `AttributeError: 'Config' object has no attribute 'delay_for'`

- [ ] **Step 3: Implement it**

In `src/fraud_benchmark/config.py`, extend the dataclasses import at the top:

```python
from dataclasses import dataclass, field, fields, replace
```

Add this method to `Config`, after `campaign_gap_for`:

```python
    def delay_for(self, name: str) -> DelayParams:
        """The delay parameters for one dataset, honouring a partial override.

        A dataset's `delay:` block overrides only the keys it names; the rest
        inherit the global block, so changing the global seed still moves every
        dataset. One distribution does not fit every dataset: PaySim's whole span
        is 30 days, while IBM CCF's is 10,649.
        """
        override = self.for_dataset(name).get("delay")
        if override is None:
            return self.delay
        return _merge_delay(self.delay, override, name)
```

Add the merge helper beside `_build_delay`:

```python
_DELAY_FIELDS = {f.name for f in fields(DelayParams)}


def _merge_delay(base: DelayParams, override: Any, name: str) -> DelayParams:
    """Overlay a per-dataset `delay:` block onto the global one."""
    if not isinstance(override, dict):
        raise ConfigError(
            f"datasets.{name}.delay must be a mapping, got {override!r}"
        )
    unknown = sorted(set(override) - _DELAY_FIELDS)
    if unknown:
        # Silently ignoring a typo would leave the dataset on the global default
        # while the config claims otherwise.
        raise ConfigError(
            f"unknown delay setting(s) for {name}: {', '.join(unknown)}; "
            f"valid keys are {', '.join(sorted(_DELAY_FIELDS))}"
        )
    try:
        coerced = {
            key: (
                int(value)
                if key == "seed"
                else None
                if value is None
                else float(value)
            )
            for key, value in override.items()
        }
        # replace() re-runs DelayParams.__post_init__, so an override gets exactly
        # the validation the global block gets.
        return replace(base, **coerced)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid delay settings for {name}: {exc}") from exc
```

- [ ] **Step 4: Validate every override at load time**

`delay_for` alone would only raise when a dataset is prepared. In `load_config`, after
building the `Config` but before returning it, resolve every dataset's delay so a bad
override fails immediately:

```python
    config = Config(
        raw_dir=Path(paths["raw"]),
        processed_dir=Path(paths["processed"]),
        split_ratios=_validate_ratios(data.get("split", {}).get("ratios")),
        delay=_build_delay(data),
        campaign_gap=_parse_gap((data.get("campaign") or {}).get("gap", "1d")),
        datasets=data.get("datasets") or {},
    )
    for name in config.datasets:
        config.delay_for(name)
    return config
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_config.py -v` — all pass.
Run: `.venv/bin/pytest` — report the ACTUAL count.

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/config.py tests/test_config.py
git commit -m "feat: per-dataset label-delay overrides"
```

---

### Task 3: Use the resolved delay in the pipeline

**Why:** without this the override is inert, and — worse — a dataset card would report the
global parameters while its timestamps came from the override.

**Files:**
- Modify: `src/fraud_benchmark/pipeline.py:42`, `:51-53`, `:78`, `:103-118`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
@pytest.fixture
def config_with_override(tmp_path):
    return Config(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        split_ratios=(0.6, 0.2, 0.2),
        delay=DelayParams(median_days=7.0, sigma=1.0, seed=0),
        campaign_gap=pd.Timedelta(days=1),
        datasets={
            "paysim": {
                "start_date": "2023-01-01",
                "delay": {"median_days": 1.0},
            }
        },
    )


def test_the_override_changes_the_timestamps(config, config_with_override, no_download):
    """The same seed must still produce different delays under a different median."""
    base = pd.read_parquet(prepare("paysim", config) / "data.parquet")
    other = pd.read_parquet(prepare("paysim", config_with_override) / "data.parquet")
    assert not base["reported_at"].equals(other["reported_at"])


def test_the_card_records_the_resolved_delay(config_with_override, no_download):
    out = prepare("paysim", config_with_override)
    card = json.loads((out / "dataset_card.json").read_text())
    # 1.0, not the global 7.0 — a card must never misreport what produced it.
    assert card["label_delay"]["median_days"] == 1.0
    assert card["label_delay"]["sigma"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: both fail — the override is ignored, so timestamps and card are unchanged.

- [ ] **Step 3: Thread the resolved delay through**

In `src/fraud_benchmark/pipeline.py`, in `prepare`, replace lines 40-42:

```python
    gap = config.campaign_gap_for(name)
    delay = config.delay_for(name)
    df["campaign_id"] = assign_campaigns(df, gap=gap)
    df["reported_at"] = assign_reported_at(df, delay)
```

Replace the `_build_card` call (lines 51-53):

```python
    card = _build_card(
        name,
        adapter,
        options,
        df,
        config,
        custom_split=supplied is not None,
        gap=gap,
        delay=delay,
    )
```

Change the `_build_card` signature and its `label_delay` entry:

```python
def _build_card(
    name,
    adapter,
    options,
    df,
    config,
    *,
    custom_split: bool,
    gap: pd.Timedelta,
    delay: DelayParams,
) -> dict:
```

```python
        "label_delay": _describe_delay(df, delay, gap),
```

Rewrite `_describe_delay` to read from the resolved params rather than `config`:

```python
def _describe_delay(df, delay: DelayParams, gap) -> dict:
    sizes = campaign_sizes(df["campaign_id"])
    fraud = df.loc[df["is_fraud"]]
    delays = (fraud["reported_at"] - fraud["event_time"]).dt.total_seconds() / 86_400
    return {
        "distribution": "lognormal",
        "median_days": delay.median_days,
        "sigma": delay.sigma,
        "seed": delay.seed,
        "max_delay_days": delay.max_delay_days,
        "campaign_gap": str(gap),
        "n_campaigns": int(sizes.size),
        "largest_campaign": int(sizes.max()) if sizes.size else 0,
        "median_campaign_size": float(sizes.median()) if sizes.size else 0.0,
        "observed_median_delay_days": float(delays.median()) if len(delays) else 0.0,
        "observed_mean_delay_days": float(delays.mean()) if len(delays) else 0.0,
    }
```

`observed_mean_delay_days` is added because truncation pulls the realised mean ~15% below
nominal, and the card should show what actually happened.

Add the import at the top of `pipeline.py`:

```python
from fraud_benchmark.delay import DelayParams, assign_reported_at
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest` — report the ACTUAL count.

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/pipeline.py tests/test_pipeline.py
git commit -m "feat: resolve label delay per dataset in the pipeline"
```

---

### Task 4: PaySim's 1-day median

**Why:** PaySim's entire span is 30 simulated days. Against a 7-day median only 47.3% of
train labels are known at the cutoff.

**Files:**
- Modify: `configs/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_paysim_overrides_the_delay_to_one_day():
    """PaySim's whole span is 30 simulated days; a 7-day median censors 53% of
    its train labels. sigma and seed still inherit."""
    config = load_config()
    delay = config.delay_for("paysim")
    assert delay.median_days == 1.0
    assert delay.sigma == config.delay.sigma
    assert delay.seed == config.delay.seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_paysim_overrides_the_delay_to_one_day -v`
Expected: FAIL, `assert 7.0 == 1.0`

- [ ] **Step 3: Add the override**

In `configs/default.yaml`, replace the `paysim:` block:

```yaml
  # PaySim's `step` column is an hour offset, not a real date, so it is
  # anchored to this start date.
  paysim:
    start_date: "2023-01-01"
    # Its whole span is 30 simulated days, so the train window is 14. At the
    # global 7-day median only 47.3% of train frauds are reported by the train
    # cutoff. This keeps the dataset usable; it does not make a 30-day clock
    # realistic. sigma and seed inherit.
    delay:
      median_days: 1.0
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_config.py -v` — all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml tests/test_config.py
git commit -m "feat: one-day median reporting delay for PaySim"
```

---

### Task 5: The IBM CCF subsample adapter

**Why:** the crop must land on the last *labelled* fraud, not the last transaction.
Cropping on the last transaction yields a test split with zero frauds — measured.

**Files:**
- Create: `src/fraud_benchmark/datasets/ibm_ccf_subsample.py`
- Create: `tests/fixtures/ibm_ccf_subsample/{credit_card_transactions-ibm_v2.csv,sd254_cards.csv,sd254_users.csv}`
- Modify: `src/fraud_benchmark/datasets/__init__.py`
- Test: `tests/test_ibm_ccf_subsample.py`

- [ ] **Step 1: Create the fixture**

A separate fixture rather than extending `tests/fixtures/ibm_ccf` — that one is pinned by
`test_all_rows_survive` and the `entity_id` assertions in `tests/test_ibm_ccf.py`.

`tests/fixtures/ibm_ccf_subsample/credit_card_transactions-ibm_v2.csv`:

```csv
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
0,0,2002,9,1,06:21,$134.09,Swipe Transaction,3527213246127876953,La Verne,CA,91750.0,5300,,No
0,1,2010,6,1,12:00,$77.10,Online Transaction,123456789,Anywhere,CA,91750.0,5411,,Yes
1,0,2010,12,31,23:59,$1200.55,Chip Transaction,987654321,Boston,MA,2101.0,5812,,No
1,0,2011,1,1,00:01,$3.99,Swipe Transaction,987654321,Boston,MA,2101.0,5812,,Yes
1,0,2011,6,1,09:30,$42.00,Swipe Transaction,987654321,Boston,MA,2101.0,5812,,No
```

Row 1 is before the window, row 5 is after the last fraud. Both must be dropped, which is
what pins the two edges.

`tests/fixtures/ibm_ccf_subsample/sd254_cards.csv`:

```csv
User,CARD INDEX,Card Brand,Card Type,Card Number,Expires,CVV,Has Chip,Cards Issued,Credit Limit,Acct Open Date,Year PIN last Changed,Card on Dark Web
0,0,Visa,Debit,4344676511950444,12/2022,623,YES,2,$24295,09/2002,2008,No
0,1,Visa,Credit,4956965974959986,12/2020,393,YES,1,$21968,04/2014,2014,No
1,0,Amex,Credit,340071811951482,02/2024,693,NO,1,$12400,01/2009,2009,No
```

`tests/fixtures/ibm_ccf_subsample/sd254_users.csv`:

```csv
Person,Current Age,Retirement Age,Birth Year,Birth Month,Gender,Address,Apartment,City,State,Zipcode,Latitude,Longitude,Per Capita Income - Zipcode,Yearly Income - Person,Total Debt,FICO Score,Num Credit Cards
Hazel Robinson,53,66,1966,11,Female,462 Rose Lane,,La Verne,CA,91750,34.15,-117.76,$29278,$59696,$127613,787,5
Sasha Sadr,53,68,1966,12,Female,3606 Federal Boulevard,,Boston,MA,2101,42.34,-71.09,$37891,$77254,$191349,701,5
```

- [ ] **Step 2: Write the failing tests** — `tests/test_ibm_ccf_subsample.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter, list_datasets
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_ccf_subsample"
OPTIONS = {"entity_key": "user", "start_date": "2010-01-01"}
VARIANTS = ("ibm_ccf_subsample_fast", "ibm_ccf_subsample_slow")


@pytest.fixture(params=VARIANTS)
def frame(request):
    return get_adapter(request.param).to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_rows_before_the_start_date_are_dropped(frame):
    assert frame["event_time"].min() >= pd.Timestamp("2010-01-01")


def test_rows_after_the_last_labelled_fraud_are_dropped(frame):
    """The 2011-06-01 row is unlabelled tail, exactly like IBM CCF's real
    2019-11 to 2020-02 dead zone. Keeping it would put a fraud-free block at
    the end of the dataset, and the whole test split inside it."""
    assert frame["event_time"].max() == pd.Timestamp("2011-01-01 00:01")


def test_the_window_keeps_everything_in_between(frame):
    assert len(frame) == 3
    assert frame["is_fraud"].sum() == 2


def test_the_last_row_is_a_fraud(frame):
    """The right edge is the last fraud, so the frame must end on one."""
    assert bool(frame["is_fraud"].iloc[-1])


def test_both_variants_produce_identical_rows():
    """They differ only in delay, which is applied later by the pipeline."""
    fast = get_adapter("ibm_ccf_subsample_fast").to_canonical(FIXTURE, OPTIONS)
    slow = get_adapter("ibm_ccf_subsample_slow").to_canonical(FIXTURE, OPTIONS)
    pd.testing.assert_frame_equal(fast, slow)


def test_entity_key_still_works_through_the_subclass():
    frame = get_adapter("ibm_ccf_subsample_fast").to_canonical(
        FIXTURE, {"entity_key": "card", "start_date": "2010-01-01"}
    )
    assert frame["entity_id"].tolist() == ["0-1", "1-0", "1-0"]


def test_joined_columns_survive_the_crop(frame):
    assert frame["Card Brand"].tolist() == ["Visa", "Amex", "Amex"]


def test_a_missing_start_date_is_a_clear_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("ibm_ccf_subsample_fast").to_canonical(
            FIXTURE, {"entity_key": "user"}
        )


@pytest.mark.parametrize("name", VARIANTS)
def test_variants_share_the_ibm_ccf_raw_download(name):
    assert get_adapter(name).raw_name == "ibm_ccf"


@pytest.mark.parametrize("name", VARIANTS)
def test_variants_are_registered(name):
    assert name in list_datasets()


def test_the_index_is_reset(frame):
    """The crop drops leading rows; a stale index would misalign the later
    campaign and delay stages, which write back by index."""
    assert frame.index.tolist() == [0, 1, 2]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ibm_ccf_subsample.py -v`
Expected: `UnknownDatasetError: unknown dataset 'ibm_ccf_subsample_fast'`

- [ ] **Step 4: Write the adapter** — `src/fraud_benchmark/datasets/ibm_ccf_subsample.py`

```python
"""IBM CCF cropped to a recent, fully-labelled window.

Two measured artifacts in the full dataset motivate this.

Labelling stops on 2019-10-27 while transactions continue to 2020-02-28, so the
final 645,180 rows carry no fraud at all. Cropping on the last transaction would
put the entire test split inside that dead zone — 0 frauds in test, measured.
The right edge is therefore the last labelled fraud, found in the data rather
than hardcoded.

And the full span is 10,649 days, against which a realistic reporting delay
censors nothing: 100.0% of train labels are known at the train cutoff. Cropping
to 2016-01-01 onward gives 6.57M rows and 8,412 frauds, where the same delay
censors a real 3%.

The two registered variants below are row-identical and differ only in the delay
distribution configured for them, so a model can be compared across delay
regimes on the same data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import register
from fraud_benchmark.datasets.ibm_ccf import IbmCcfAdapter

#: Artifacts documented in docs/verification-notes.md, recorded on every card.
SUBSAMPLE_CAVEATS = (
    "Cropped to [start_date, last labelled fraud]. IBM CCF's fraud labelling "
    "stops on 2019-10-27 while transactions run to 2020-02-28; those final "
    "645,180 rows are dropped because they are unlabelled, not fraud-free.",
    "2017 carries 255 frauds against roughly 3,000 in each neighbouring year — "
    "very likely a second labelling artifact. It falls inside the train split, "
    "so it thins training data without affecting validation or test.",
    "Merchant State is close to a label here: in this window 'Italy' is 76.8% "
    "fraud and accounts for 55.7% of all frauds. Do not hand it to a model as a "
    "raw feature without understanding that.",
)


class IbmCcfSubsampleAdapter(IbmCcfAdapter):
    """Shared crop. Not registered: the concrete variants below are.

    Inherits the two table joins, the money-string parsing and the entity_key
    logic from IbmCcfAdapter; only the window is new.
    """

    #: Both variants read the raw files already downloaded for ibm_ccf.
    raw_name = "ibm_ccf"
    caveats = IbmCcfAdapter.caveats + SUBSAMPLE_CAVEATS

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        if "start_date" not in options:
            raise ValueError(
                f"{self.name} requires a 'start_date' option giving the left edge "
                "of the window"
            )
        df = super().to_canonical(raw_dir, options)

        # Found in the data, not hardcoded: the point is to end where labelling
        # ends, whatever date that turns out to be.
        last_fraud = df.loc[df["is_fraud"], "event_time"].max()
        if pd.isna(last_fraud):
            raise ValueError(
                f"{self.name}: the source has no fraudulent rows, so the window "
                "has no right edge"
            )
        start = pd.Timestamp(options["start_date"])
        if start > last_fraud:
            raise ValueError(
                f"{self.name}: start_date {start.date()} is after the last "
                f"labelled fraud {last_fraud.date()}, leaving an empty window"
            )

        window = (df["event_time"] >= start) & (df["event_time"] <= last_fraud)
        return df.loc[window].reset_index(drop=True)


@register
class IbmCcfSubsampleFastAdapter(IbmCcfSubsampleAdapter):
    """Median 7 / mean 30 day reporting delay. Parameters in configs/default.yaml."""

    name = "ibm_ccf_subsample_fast"


@register
class IbmCcfSubsampleSlowAdapter(IbmCcfSubsampleAdapter):
    """Median 15 / mean 60 day reporting delay. Parameters in configs/default.yaml."""

    name = "ibm_ccf_subsample_slow"
```

- [ ] **Step 5: Register the module**

In `src/fraud_benchmark/datasets/__init__.py`, add `ibm_ccf_subsample` to the import list,
keeping it alphabetical:

```python
from fraud_benchmark.datasets import (  # noqa: F401
    amaretto,
    banksim,
    ibm_ccf,
    ibm_ccf_subsample,
    ieee_cis,
    paysim,
    saml_d,
    sparkov,
)
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/test_ibm_ccf_subsample.py -v` — expect 21 passed (seven
fixture-parametrised tests run twice, two name-parametrised tests run twice, three run once).
Run: `.venv/bin/pytest` — report the ACTUAL count.

`tests/test_cli.py` lists datasets; if a test pins the set of names, update it to include
both variants. If it pins a *count*, prefer changing it to a membership assertion.

- [ ] **Step 7: Commit**

```bash
git add src/fraud_benchmark/datasets/ibm_ccf_subsample.py \
        src/fraud_benchmark/datasets/__init__.py \
        tests/fixtures/ibm_ccf_subsample tests/test_ibm_ccf_subsample.py
git commit -m "feat: IBM CCF subsample adapters cropped to a labelled window"
```

---

### Task 6: Configure the two variants

**Files:**
- Modify: `configs/default.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_the_fast_subsample_has_a_thirty_day_mean_delay():
    """sigma = sqrt(2*ln(30/7)) places a lognormal's mean at 30 with median 7."""
    delay = load_config().delay_for("ibm_ccf_subsample_fast")
    assert delay.median_days == 7.0
    assert delay.sigma == pytest.approx(1.706, abs=5e-4)
    assert delay.max_delay_days == 365.0


def test_the_slow_subsample_has_a_sixty_day_mean_delay():
    """sigma = sqrt(2*ln(60/15))."""
    delay = load_config().delay_for("ibm_ccf_subsample_slow")
    assert delay.median_days == 15.0
    assert delay.sigma == pytest.approx(1.665, abs=5e-4)
    assert delay.max_delay_days == 730.0


def test_both_subsamples_share_one_window():
    """They must be row-identical; only the delay may differ."""
    config = load_config()
    fast = config.for_dataset("ibm_ccf_subsample_fast")
    slow = config.for_dataset("ibm_ccf_subsample_slow")
    assert fast["start_date"] == slow["start_date"] == "2016-01-01"
    assert fast["entity_key"] == slow["entity_key"]


def test_the_subsamples_keep_the_default_campaign_gap():
    config = load_config()
    for name in ("ibm_ccf_subsample_fast", "ibm_ccf_subsample_slow"):
        assert config.campaign_gap_for(name) == pd.Timedelta(days=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: `assert 7.0 == 15.0` and similar — the datasets have no config yet, so both
resolve to the global delay.

- [ ] **Step 3: Add the config**

In `configs/default.yaml`, inside `datasets:`, after the `ibm_ccf:` block:

```yaml
  # Two crops of ibm_ccf to [2016-01-01, last labelled fraud 2019-10-27]:
  # 6,569,157 rows, 8,412 frauds, 982 of them in test. Row-identical to each
  # other; they differ only in reporting delay, so a model can be compared
  # across delay regimes on the same data.
  #
  # Why crop at all: ibm_ccf spans 10,649 days, so a realistic delay censors
  # 3 of its 24,924 train frauds — no delay effect to measure. Why 2016-01-01
  # rather than a three-year window: the latter starts in late October 2016 and
  # discards most of a heavy fraud year. +27% rows buys +58% frauds.
  ibm_ccf_subsample_fast:
    entity_key: user
    start_date: "2016-01-01"
    delay:
      # sigma = sqrt(2*ln(30/7)): median 7 days, mean 30. Capped at a year —
      # untruncated this samples delays out to 1,813 days, well past the end of
      # the data. The cap bites ~1% of campaigns and pulls the realised mean to
      # about 25 days.
      sigma: 1.706
      max_delay_days: 365
  ibm_ccf_subsample_slow:
    entity_key: user
    start_date: "2016-01-01"
    delay:
      # sigma = sqrt(2*ln(60/15)): median 15 days, mean 60. Roughly doubles the
      # censoring of the fast variant. Realised mean about 51 days after the cap.
      median_days: 15.0
      sigma: 1.665
      max_delay_days: 730
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest` — report the ACTUAL count.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml tests/test_config.py
git commit -m "feat: configure the fast and slow IBM CCF subsamples"
```

---

### Task 7: Live verification

**Check disk headroom first — the home directory is NFS-quota-limited and a previous run
failed with ENOSPC.** The two variants add roughly 580 MB.

```bash
df -h .
dd if=/dev/zero of=data/processed/_spacetest bs=1M count=2048 && rm data/processed/_spacetest
```

If that fails, stop and report rather than starting a 20-minute run that cannot finish.

- [ ] **Step 1: Prepare the affected datasets**

Do NOT pass `--force` — in this CLI it means re-download, and it deletes the raw directory
before re-fetching.

```bash
.venv/bin/fraud-benchmark prepare paysim
.venv/bin/fraud-benchmark prepare ibm_ccf_subsample_fast
.venv/bin/fraud-benchmark prepare ibm_ccf_subsample_slow
```

- [ ] **Step 2: Confirm PaySim's labels are usable**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/processed/paysim/data.parquet",
                     columns=["split","is_fraud","reported_at","event_time"])
tr = df[df.split == "train"]
cutoff = tr.event_time.max()
tf = tr[tr.is_fraud]
delays = (tf.reported_at - tf.event_time).dt.total_seconds()/86_400
print(f"train frauds={len(tf):,} known at cutoff={((tf.reported_at<=cutoff).mean()*100):.1f}% "
      f"(was 47.3%)  delay p50={delays.median():.2f}d p95={delays.quantile(.95):.2f}d")
PY
```

- [ ] **Step 3: Confirm the two variants are row-identical**

This is the property the whole two-dataset design rests on. If it fails, the comparison
between delay regimes is not controlled and the results are meaningless.

```bash
.venv/bin/python - <<'PY'
import pandas as pd
fast = pd.read_parquet("data/processed/ibm_ccf_subsample_fast/data.parquet")
slow = pd.read_parquet("data/processed/ibm_ccf_subsample_slow/data.parquet")
shared = [c for c in fast.columns if c != "reported_at"]
pd.testing.assert_frame_equal(fast[shared], slow[shared])
print(f"identical on {len(shared)} columns, {len(fast):,} rows")
print("reported_at differs:", not fast.reported_at.equals(slow.reported_at))
PY
```

- [ ] **Step 4: Measure both regimes**

```bash
.venv/bin/python - <<'PY'
import pandas as pd
for name in ("ibm_ccf_subsample_fast","ibm_ccf_subsample_slow"):
    df = pd.read_parquet(f"data/processed/{name}/data.parquet",
                         columns=["split","is_fraud","reported_at","event_time"])
    f = df[df.is_fraud]
    d = (f.reported_at - f.event_time).dt.total_seconds()/86_400
    tr = df[df.split=="train"]; cutoff = tr.event_time.max(); tf = tr[tr.is_fraud]
    counts = df.groupby("split", observed=True).is_fraud.sum()
    print(f"{name}")
    print(f"  rows={len(df):,} frauds={len(f):,} "
          f"train/val/test frauds={counts.get('train',0):,}/{counts.get('val',0):,}/{counts.get('test',0):,}")
    print(f"  delay p50={d.median():.1f}d mean={d.mean():.1f}d p95={d.quantile(.95):.1f}d max={d.max():.1f}d")
    print(f"  known at train cutoff: {((tf.reported_at<=cutoff).mean()*100):.1f}%")
    print(f"  early={int((f.reported_at<f.event_time).sum())} null={int(f.reported_at.isna().sum())} "
          f"stray={int(df.loc[~df.is_fraud,'reported_at'].notna().sum())}")
PY
```

Expected, from the spec's offline measurements: 6,569,157 rows, 8,412 frauds,
6,441/989/982 by split, fast ~97.0% and slow ~94.8% known at cutoff, `early=0 null=0
stray=0` for both. **Report the actual numbers.** If the split fraud counts differ from
these, stop and report — it means the window moved.

- [ ] **Step 5: Update the docs**

In `docs/verification-notes.md`, in the "Label delay as shipped" section, add the three
new datasets to the invariants, campaign-size and label-availability tables, and record
PaySim's new figure beside its old 47.3%. Note explicitly that fast and slow are
row-identical.

In `README.md`, add both variants wherever the seven datasets are listed.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "test: verify per-dataset delay and the IBM CCF subsamples"
```

---

## Done criteria

- [ ] `.venv/bin/pytest` passes
- [ ] `config.delay_for(name)` merges partially and rejects unknown keys at load time
- [ ] Dataset cards record resolved delay parameters, plus the observed mean
- [ ] PaySim's known-at-cutoff figure is reported, up from 47.3%
- [ ] Both subsamples prepare, are row-identical outside `reported_at`, and satisfy
      `early=0 null=0 stray=0`
- [ ] Both subsamples' known-at-cutoff figures are reported and differ as designed
- [ ] `docs/verification-notes.md` and `README.md` updated

## Not in this plan

- The IBM CCF artifact audit. Separate note: `docs/superpowers/plans/2026-08-02-ibm-ccf-artifact-audit.md`.
- Reworking IBM CCF's memory profile. Each variant still materialises the full 24.4M-row
  joined frame before cropping.
- Any evaluation harness or baseline model.
