# Leakage Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SLURM-dispatched harness that scores real classifiers on five dataset variants with and without their leaky columns, so the leakage the audit found can be reported as a number rather than an anecdote.

**Architecture:** A new `src/fraud_benchmark/ablation/` package, read-only with respect to `data/processed/`. Stage 1 builds causal velocity features once per dataset and caches them to `data/features/`. Stage 2 runs one SLURM job per (dataset, feature set, label regime) cell, fitting every model for that cell and appending one JSON record per evaluation to `results/runs.jsonl`. Nothing in the existing pipeline changes.

**Tech Stack:** Python 3.13, pandas, pyarrow, numpy, scikit-learn 1.9, XGBoost, pytest, SLURM.

**Spec:** `docs/superpowers/specs/2026-08-02-leakage-ablation-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fraud_benchmark/ablation/__init__.py` | Package marker |
| `src/fraud_benchmark/ablation/columns.py` | The two exclusion sets and the logic that resolves a feature list for a cell |
| `src/fraud_benchmark/ablation/features.py` | Causal per-entity velocity features |
| `src/fraud_benchmark/ablation/encoding.py` | Train-fit categorical encoder and scaler |
| `src/fraud_benchmark/ablation/metrics.py` | Average precision, and best-F1 threshold picked on val |
| `src/fraud_benchmark/ablation/models.py` | Trivial rule, logistic regression, XGBoost |
| `src/fraud_benchmark/ablation/cell.py` | Run one (dataset, feature set, label regime) cell end to end |
| `src/fraud_benchmark/ablation/build_features.py` | Stage 1 entry point |
| `src/fraud_benchmark/ablation/summarize.py` | `results/runs.jsonl` → `results/summary.md` |
| `scripts/slurm/generate.py` | Emits all `.sbatch` files and `submit_all.sh` |
| `tests/ablation/test_columns.py` | Exclusion-set tests |
| `tests/ablation/test_features.py` | Hand-computed fixture and the no-lookahead property test |
| `tests/ablation/test_encoding.py` | Train-only fitting, unseen categories |
| `tests/ablation/test_metrics.py` | Metric behaviour |
| `tests/ablation/test_cell.py` | Cell wiring, JSONL records, evaluation integrity |

**Why a package rather than more top-level modules:** the existing top-level modules are pipeline stages that every `prepare` run touches. This is a separate study that reads the pipeline's output. Keeping it in its own package means nothing here can accidentally change what `fraud-benchmark prepare` produces.

---

## A note on grouping vs. using a column

`entity_id` is in SAML-D's leaky set, and the velocity features are *grouped by* `entity_id`. These are not the same thing. Grouping by an identifier to compute "how many transactions has this account made in the last hour" does not put the account's identity into the feature matrix — the output is a count, not an id. The `clean` condition drops `entity_id` as a *column* while still grouping by it, and that is correct. Task 2's tests pin this down so nobody later "fixes" it.

---

## Task 1: Dependency and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/fraud_benchmark/ablation/__init__.py`
- Create: `tests/ablation/__init__.py`

- [ ] **Step 1: Add xgboost to the dev extra**

In `pyproject.toml`, replace the `[project.optional-dependencies]` block with:

```toml
[project.optional-dependencies]
# Neither is used by the pipeline. They back the leakage ablation study in
# src/fraud_benchmark/ablation/; 1.4 is the floor for passing string column names
# to HistGradientBoostingClassifier(categorical_features=...).
dev = ["pytest>=8", "scikit-learn>=1.4", "xgboost>=2.0"]
```

- [ ] **Step 2: Create the package markers**

`src/fraud_benchmark/ablation/__init__.py`:

```python
"""Leakage ablation study.

Reads the pipeline's output under data/processed/ and never writes to it. See
docs/superpowers/specs/2026-08-02-leakage-ablation-design.md.
"""
```

`tests/ablation/__init__.py`: empty file.

- [ ] **Step 3: Install and verify**

Run: `.venv/bin/pip install -q -e '.[dev]' && .venv/bin/python -c "import xgboost; print(xgboost.__version__)"`
Expected: a version `2.x` or higher printed, no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/fraud_benchmark/ablation/__init__.py tests/ablation/__init__.py
git commit -m "chore: add xgboost and the ablation package skeleton"
```

---

## Task 2: Column sets

**Files:**
- Create: `src/fraud_benchmark/ablation/columns.py`
- Test: `tests/ablation/test_columns.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_columns.py`:

```python
import pandas as pd
import pytest

from fraud_benchmark.ablation.columns import (
    ALWAYS_EXCLUDED,
    LEAKY_COLUMNS,
    ExcludedColumnError,
    assert_no_excluded,
    feature_columns,
)


def frame(*names):
    return pd.DataFrame({n: [0] for n in names})


def test_label_and_its_derivatives_are_always_excluded():
    for name in ["is_fraud", "reported_at", "campaign_id", "split"]:
        assert name in ALWAYS_EXCLUDED


def test_source_raw_labels_are_always_excluded():
    for name in ["Is Fraud?", "Is_laundering", "Laundering_type"]:
        assert name in ALWAYS_EXCLUDED


def test_row_identifiers_and_provenance_are_always_excluded():
    for name in ["trans_num", "source_file"]:
        assert name in ALWAYS_EXCLUDED


def test_leaky_columns_are_kept_in_the_leaky_condition():
    df = frame("Merchant State", "amount", "is_fraud")
    cols = feature_columns(df, dataset="ibm_ccf", feature_set="leaky")
    assert "Merchant State" in cols


def test_leaky_columns_are_dropped_in_the_clean_condition():
    df = frame("Merchant State", "Merchant Name", "MCC", "amount", "is_fraud")
    cols = feature_columns(df, dataset="ibm_ccf", feature_set="clean")
    assert "Merchant State" not in cols
    assert "Merchant Name" not in cols
    assert "MCC" not in cols
    assert "amount" in cols


def test_the_label_never_survives_either_condition():
    df = frame("amount", "is_fraud", "reported_at", "campaign_id", "split")
    for feature_set in ["leaky", "clean"]:
        cols = feature_columns(df, dataset="ibm_ccf", feature_set=feature_set)
        assert cols == ["amount"]


def test_sparkov_has_no_leaky_columns_so_both_conditions_match():
    df = frame("merchant", "category", "amount", "is_fraud")
    leaky = feature_columns(df, dataset="sparkov", feature_set="leaky")
    clean = feature_columns(df, dataset="sparkov", feature_set="clean")
    assert leaky == clean


def test_saml_d_drops_entity_id_as_a_column_in_the_clean_condition():
    df = frame("entity_id", "Sender_account", "amount", "is_fraud")
    cols = feature_columns(df, dataset="saml_d", feature_set="clean")
    assert "entity_id" not in cols
    assert "Sender_account" not in cols


def test_velocity_features_survive_the_clean_condition_for_saml_d():
    """Grouping by entity_id is not the same as using it as a feature."""
    df = frame("entity_id", "txn_count_1h", "merchant_novelty", "amount", "is_fraud")
    cols = feature_columns(df, dataset="saml_d", feature_set="clean")
    assert "txn_count_1h" in cols
    assert "merchant_novelty" in cols


def test_the_guard_rejects_a_hand_built_list_containing_the_label():
    """The guard is a tripwire on the filter, so it is tested directly.

    Routing it through feature_columns would be tautological: that function
    builds its output by removing exactly these names, so the check could never
    fire there no matter what it was passed.
    """
    with pytest.raises(ExcludedColumnError, match="is_fraud"):
        assert_no_excluded(["amount", "is_fraud"])


def test_the_guard_reports_every_offending_column():
    with pytest.raises(ExcludedColumnError, match="reported_at"):
        assert_no_excluded(["amount", "is_fraud", "reported_at"])


def test_the_guard_accepts_a_clean_list():
    assert_no_excluded(["amount", "txn_count_1h", "merchant_novelty"])


def test_feature_columns_output_always_passes_the_guard():
    df = frame("amount", "is_fraud", "reported_at", "Merchant State")
    for feature_set in ["leaky", "clean"]:
        assert_no_excluded(feature_columns(df, dataset="ibm_ccf", feature_set=feature_set))


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        feature_columns(frame("amount"), dataset="nope", feature_set="leaky")


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError, match="feature_set"):
        feature_columns(frame("amount"), dataset="ibm_ccf", feature_set="sideways")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_columns.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.columns'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/columns.py`:

```python
"""Which columns a model may see.

Two separate sets, deliberately not merged. ALWAYS_EXCLUDED is the label and
anything derived from it -- leaving one of these in invalidates every number in
the study, so it is enforced with an assertion rather than trusted to a list.
LEAKY_COLUMNS is the ablation: the generation artifacts the audit measured, which
are present in the `leaky` condition and absent in the `clean` one.
"""

from __future__ import annotations

import pandas as pd

FEATURE_SETS = ("leaky", "clean")

# The label, anything derived from it, row identifiers, and split provenance.
# `source_file` is here because Sparkov's splits come from separate upstream
# files, so it predicts the split perfectly. `Laundering_type` describes the
# label and is non-null only for laundering rows.
ALWAYS_EXCLUDED = frozenset(
    {
        "is_fraud",
        "reported_at",
        "campaign_id",
        "split",
        "Is Fraud?",
        "Is_laundering",
        "Laundering_type",
        "trans_num",
        "source_file",
    }
)

# From docs/verification-notes.md, "## Known leakage". `Merchant Name` is added
# beyond the tuple the audit suggested: a merchant id encodes its own location,
# so retaining it would reintroduce the geography artifact under another name.
_IBM_CCF_LEAKY = (
    "Merchant State",
    "Merchant City",
    "Zip",
    "MCC",
    "Errors?",
    "Merchant Name",
)

LEAKY_COLUMNS: dict[str, tuple[str, ...]] = {
    "ibm_ccf": _IBM_CCF_LEAKY,
    "ibm_ccf_subsample_fast": _IBM_CCF_LEAKY,
    "ibm_ccf_subsample_slow": _IBM_CCF_LEAKY,
    # Account identifiers only. Sender_bank_location and Receiver_bank_location
    # are kept: the audit found their effects directional and plausible for money
    # laundering -- the phenomenon, not an artifact.
    "saml_d": ("Sender_account", "Receiver_account", "entity_id"),
    # The audit found zero flagged values. Sparkov is the negative control: its
    # two conditions are identical by construction, so the gap between them
    # measures this harness's own noise floor.
    "sparkov": (),
}


class ExcludedColumnError(AssertionError):
    """Raised when a column that must never reach a model is about to."""


def assert_no_excluded(columns: list[str]) -> None:
    """Raise if any always-excluded column is in `columns`.

    A tripwire for callers that assemble a column list themselves rather than
    taking `feature_columns` output verbatim -- which is every consumer that
    adds, renames, or re-orders columns downstream. Calling it on
    `feature_columns` output cannot fail today; it is there so that a future
    change to the filter is caught by a test rather than by a wrong result.
    """
    leaked = sorted(set(ALWAYS_EXCLUDED) & set(columns))
    if leaked:
        raise ExcludedColumnError(
            f"columns that must never reach a model are present: {leaked}"
        )


def feature_columns(df: pd.DataFrame, dataset: str, feature_set: str) -> list[str]:
    """The columns a model may see for one cell, in stable order."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    leaky = LEAKY_COLUMNS[dataset]

    dropped = set(ALWAYS_EXCLUDED)
    if feature_set == "clean":
        dropped |= set(leaky)

    columns = [c for c in df.columns if c not in dropped]
    assert_no_excluded(columns)
    return columns
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_columns.py -q`
Expected: PASS, 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/columns.py tests/ablation/test_columns.py
git commit -m "feat: the two column-exclusion sets for the ablation"
```

---

## Task 3: Causal velocity features

This is the task most likely to introduce a silent bug, because a lookahead error produces *better* numbers and no error message. It gets a hand-computed fixture and a property test.

**Files:**
- Create: `src/fraud_benchmark/ablation/features.py`
- Test: `tests/ablation/test_features.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.ablation.features import VELOCITY_COLUMNS, add_velocity_features


def frame(rows):
    """rows: list of (entity_id, 'YYYY-MM-DD HH:MM', amount, merchant)."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series([r[0] for r in rows], dtype="string"),
            "event_time": pd.to_datetime([r[1] for r in rows]),
            "amount": [float(r[2]) for r in rows],
            "merchant": pd.Series([r[3] for r in rows], dtype="string"),
        }
    )


def test_the_first_transaction_of_an_entity_has_no_history():
    df = frame([("a", "2023-01-01 00:00", 10, "m1")])
    out = add_velocity_features(df, merchant_col="merchant")
    assert out.loc[0, "txn_count_1h"] == 0
    assert out.loc[0, "txn_count_24h"] == 0
    assert out.loc[0, "amount_sum_24h"] == 0.0
    assert pd.isna(out.loc[0, "seconds_since_prev_txn"])


def test_counts_are_of_strictly_previous_transactions():
    """Three transactions an hour apart: counts must be 0, 1, 2 -- never 1, 2, 3."""
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:00", 20, "m1"),
            ("a", "2023-01-01 02:00", 30, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["txn_count_24h"]) == [0, 1, 2]


def test_the_current_amount_is_excluded_from_its_own_window():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:00", 20, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["amount_sum_24h"]) == [0.0, 10.0]


def test_a_window_only_counts_inside_its_span():
    """The 1h window must not see a transaction 90 minutes earlier."""
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:30", 20, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["txn_count_1h"]) == [0, 0]
    assert list(out["txn_count_24h"]) == [0, 1]


def test_entities_do_not_see_each_other():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("b", "2023-01-01 00:30", 20, "m1"),
            ("b", "2023-01-01 00:45", 30, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["txn_count_24h"]) == [0, 0, 1]


def test_seconds_since_prev_txn_is_measured_per_entity():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 00:10", 20, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert out.loc[1, "seconds_since_prev_txn"] == 600.0


def test_merchant_novelty_marks_only_the_first_visit():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 00:10", 20, "m1"),
            ("a", "2023-01-01 00:20", 30, "m2"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["merchant_novelty"]) == [1, 0, 1]


def test_novelty_is_per_entity_not_global():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("b", "2023-01-01 00:10", 20, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["merchant_novelty"]) == [1, 1]


def test_zscore_is_zero_when_history_has_no_spread():
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:00", 10, "m1"),
            ("a", "2023-01-01 02:00", 10, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert (out["amount_zscore_vs_7d"] == 0.0).all()


def test_zscore_uses_only_prior_amounts():
    """History is 10 and 20 -> mean 15, sample std 7.0710678. Current is 50."""
    df = frame(
        [
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:00", 20, "m1"),
            ("a", "2023-01-01 02:00", 50, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert out.loc[2, "amount_mean_7d"] == pytest.approx(15.0)
    assert out.loc[2, "amount_zscore_vs_7d"] == pytest.approx((50 - 15) / 7.0710678, rel=1e-6)


def test_output_row_order_matches_the_input():
    """Rows arrive unsorted; features must land on the right rows."""
    df = frame(
        [
            ("a", "2023-01-01 02:00", 30, "m1"),
            ("a", "2023-01-01 00:00", 10, "m1"),
            ("a", "2023-01-01 01:00", 20, "m1"),
        ]
    )
    out = add_velocity_features(df, merchant_col="merchant")
    assert list(out["amount"]) == [30.0, 10.0, 20.0]
    assert list(out["txn_count_24h"]) == [2, 0, 1]


def test_every_declared_column_is_produced():
    df = frame([("a", "2023-01-01 00:00", 10, "m1")])
    out = add_velocity_features(df, merchant_col="merchant")
    for name in VELOCITY_COLUMNS:
        assert name in out.columns


def test_no_lookahead_property():
    """The direct statement of causality.

    A row's features must be identical whether computed from the whole frame or
    from a frame truncated just after that row. Anything that peeks forward --
    an off-by-one on a window edge, a centred window, a global sort -- breaks
    this and nothing else in the suite would catch it.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "entity_id": pd.Series(rng.choice(["a", "b", "c"], n), dtype="string"),
            "event_time": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 500_000, n)), unit="s"),
            "amount": rng.lognormal(3, 1, n),
            "merchant": pd.Series(rng.choice(["m1", "m2", "m3"], n), dtype="string"),
        }
    )
    full = add_velocity_features(df, merchant_col="merchant")
    for i in [17, 88, 150, 199]:
        truncated = add_velocity_features(df.iloc[: i + 1].copy(), merchant_col="merchant")
        for col in VELOCITY_COLUMNS:
            a, b = full.loc[i, col], truncated.loc[i, col]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (
                f"{col} at row {i} changed when later rows were removed: {a} vs {b}"
            )


def test_a_missing_merchant_column_yields_no_novelty_feature():
    df = frame([("a", "2023-01-01 00:00", 10, "m1")]).drop(columns=["merchant"])
    out = add_velocity_features(df, merchant_col=None)
    assert "merchant_novelty" not in out.columns
    assert "txn_count_24h" in out.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_features.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.features'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/features.py`:

```python
"""Causal per-entity velocity features.

Every window is left-closed and excludes the current row. A feature that includes
the transaction it describes is lookahead within the row -- it would leak the
amount into its own z-score -- and it makes results better, not noisier, so it
would not show up as a failure anywhere else.

Computed over the full timeline rather than per split. A per-split computation
gives every split a cold-start artifact at its left edge, where entities look new
purely because the window was truncated there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VELOCITY_COLUMNS = (
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_24h",
    "amount_sum_7d",
    "amount_mean_7d",
    "amount_zscore_vs_7d",
    "seconds_since_prev_txn",
    "merchant_novelty",
)

# Label -> pandas offset. The labels name the output columns and stay lowercase;
# the offsets must use "D", since lowercase "d" is deprecated.
_WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7D"}


def add_velocity_features(
    df: pd.DataFrame, merchant_col: str | None
) -> pd.DataFrame:
    """Return `df` with velocity features appended, in the input's row order.

    `merchant_col` may be None for a dataset with no merchant-like column, in
    which case `merchant_novelty` is not produced.
    """
    out = df.copy()

    # Sort by (entity, time) for the rolling windows, but remember where each row
    # came from so the results land back on the right rows.
    work = out[["entity_id", "event_time", "amount"]].copy()
    work["_pos"] = np.arange(len(work))
    work = work.sort_values(["entity_id", "event_time"], kind="mergesort")
    pos = work["_pos"].to_numpy()

    indexed = work.set_index("event_time")
    grouped = indexed.groupby("entity_id", observed=True)["amount"]

    def scatter(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(work), dtype="float64")
        result[pos] = values
        return result

    for label, window in _WINDOWS.items():
        rolled = grouped.rolling(window, closed="left")
        # An empty left-closed window counts as NaN, not 0 -- so a row with no
        # prior history would otherwise carry NaN into every count column.
        out[f"txn_count_{label}"] = scatter(np.nan_to_num(rolled.count().to_numpy()))

    for label in ("24h", "7d"):
        rolled = grouped.rolling(_WINDOWS[label], closed="left")
        # A window with no prior rows sums to NaN; 0.0 is the honest value.
        out[f"amount_sum_{label}"] = scatter(np.nan_to_num(rolled.sum().to_numpy()))

    week = grouped.rolling(_WINDOWS["7d"], closed="left")
    mean_7d = week.mean().to_numpy()
    std_7d = week.std().to_numpy()
    out["amount_mean_7d"] = scatter(mean_7d)

    # Zero rather than NaN when the history is empty or flat: "no evidence of
    # deviation" is the correct reading, and it keeps the column dense.
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (work["amount"].to_numpy() - mean_7d) / std_7d
    z[~np.isfinite(z)] = 0.0
    out["amount_zscore_vs_7d"] = scatter(z)

    gaps = (
        work.reset_index(drop=True)
        .groupby("entity_id", observed=True)["event_time"]
        .diff()
        .dt.total_seconds()
        .to_numpy()
    )
    out["seconds_since_prev_txn"] = scatter(gaps)

    if merchant_col is not None:
        novelty = out.copy()
        novelty["_pos"] = np.arange(len(novelty))
        novelty = novelty.sort_values(["entity_id", "event_time"], kind="mergesort")
        first_visit = (
            novelty.groupby(["entity_id", merchant_col], observed=True).cumcount() == 0
        )
        flags = np.empty(len(novelty), dtype="int64")
        flags[novelty["_pos"].to_numpy()] = first_visit.to_numpy().astype("int64")
        out["merchant_novelty"] = flags

    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_features.py -q`
Expected: PASS, 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/features.py tests/ablation/test_features.py
git commit -m "feat: causal per-entity velocity features"
```

---

## Task 4: Train-fit encoding

**Files:**
- Create: `src/fraud_benchmark/ablation/encoding.py`
- Test: `tests/ablation/test_encoding.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_encoding.py`:

```python
import numpy as np
import pandas as pd

from fraud_benchmark.ablation.encoding import UNSEEN, Encoder


def frame(cats, nums):
    return pd.DataFrame({"cat": pd.Series(cats, dtype="string"), "num": nums})


def test_categories_come_from_train_only():
    train = frame(["a", "b"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    assert set(enc.categories_["cat"]) == {"a", "b"}


def test_a_category_only_in_val_maps_to_the_unseen_bucket():
    train = frame(["a", "b"], [1.0, 2.0])
    val = frame(["a", "z"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    out = enc.transform(val)
    assert out.loc[0, "cat"] == enc.code_of("cat", "a")
    assert out.loc[1, "cat"] == enc.code_of("cat", UNSEEN)


def test_the_unseen_bucket_does_not_collide_with_a_real_category():
    train = frame(["a", "b"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    unseen_code = enc.code_of("cat", UNSEEN)
    assert unseen_code not in {enc.code_of("cat", c) for c in ["a", "b"]}


def test_scaling_statistics_come_from_train_only():
    train = frame(["a", "a"], [0.0, 10.0])          # mean 5, std 5
    val = frame(["a", "a"], [5.0, 15.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=True)
    out = enc.transform(val)
    assert out.loc[0, "num"] == 0.0                  # (5 - 5) / 5
    assert out.loc[1, "num"] == 2.0                  # (15 - 5) / 5


def test_scaling_is_skipped_when_not_requested():
    train = frame(["a", "a"], [0.0, 10.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=False)
    out = enc.transform(train)
    assert list(out["num"]) == [0.0, 10.0]


def test_a_zero_variance_column_does_not_produce_nan():
    train = frame(["a", "a"], [3.0, 3.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=True)
    out = enc.transform(train)
    assert np.isfinite(out["num"]).all()


def test_nulls_in_a_categorical_column_get_their_own_code():
    train = pd.DataFrame({"cat": pd.Series(["a", None], dtype="string")})
    enc = Encoder().fit(train, categorical=["cat"])
    out = enc.transform(train)
    assert out["cat"].notna().all()


def test_transform_preserves_row_count_and_order():
    train = frame(["a", "b", "a"], [1.0, 2.0, 3.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"])
    out = enc.transform(train)
    assert len(out) == 3
    assert list(out["num"]) == [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_encoding.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.encoding'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/encoding.py`:

```python
"""Categorical encoding and scaling, fitted on train only.

Fitting on anything but train is lookahead: the val and test rows would have
contributed to the category vocabulary and to the scaling statistics. Values that
appear only in val or test go to a dedicated bucket rather than silently onto a
real level, which would otherwise make an unseen merchant indistinguishable from
a specific known one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNSEEN = "__unseen__"
_NULL = "__null__"


class Encoder:
    """Ordinal category codes plus optional standardisation."""

    def __init__(self) -> None:
        self.categories_: dict[str, list[str]] = {}
        self.numeric_: list[str] = []
        self.scale_: bool = False
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(
        self,
        train: pd.DataFrame,
        categorical: list[str] | None = None,
        numeric: list[str] | None = None,
        scale: bool = False,
    ) -> "Encoder":
        self.scale_ = scale
        self.numeric_ = list(numeric or [])
        for col in categorical or []:
            values = train[col].astype("string").fillna(_NULL)
            # UNSEEN last, so its code cannot collide with a real category.
            self.categories_[col] = sorted(set(values.tolist())) + [UNSEEN]
        if scale:
            for col in self.numeric_:
                values = pd.to_numeric(train[col], errors="coerce")
                mean = float(values.mean())
                std = float(values.std())
                self.mean_[col] = 0.0 if not np.isfinite(mean) else mean
                # A constant column has no spread; dividing by 1 leaves it at 0.
                self.std_[col] = std if np.isfinite(std) and std > 0 else 1.0
        return self

    def code_of(self, column: str, value: str) -> int:
        return self.categories_[column].index(value)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, categories in self.categories_.items():
            lookup = {c: i for i, c in enumerate(categories)}
            unseen = lookup[UNSEEN]
            values = out[col].astype("string").fillna(_NULL)
            out[col] = values.map(lookup).fillna(unseen).astype("int32")
        if self.scale_:
            for col in self.numeric_:
                values = pd.to_numeric(out[col], errors="coerce").fillna(self.mean_[col])
                out[col] = (values - self.mean_[col]) / self.std_[col]
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_encoding.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/encoding.py tests/ablation/test_encoding.py
git commit -m "feat: train-only categorical encoding and scaling"
```

---

## Task 5: Metrics

**Files:**
- Create: `src/fraud_benchmark/ablation/metrics.py`
- Test: `tests/ablation/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_metrics.py`:

```python
import numpy as np
import pytest

from fraud_benchmark.ablation.metrics import best_f1_threshold, score


def test_a_perfect_ranking_gets_average_precision_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert score(y, s, threshold=0.5)["average_precision"] == pytest.approx(1.0)


def test_average_precision_of_a_random_scorer_approaches_the_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.01).astype(int)
    s = rng.random(20_000)
    assert score(y, s, threshold=0.5)["average_precision"] == pytest.approx(0.01, abs=0.005)


def test_roc_auc_is_not_reported():
    """Base rates here are 0.10-0.52%; ROC AUC would flatter a useless model."""
    y = np.array([0, 1])
    s = np.array([0.1, 0.9])
    assert "roc_auc" not in score(y, s, threshold=0.5)


def test_precision_recall_and_f1_come_from_the_given_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.6, 0.9])
    out = score(y, s, threshold=0.5)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(0.8)


def test_best_f1_threshold_finds_a_separating_cut():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_f1_threshold(y, s)
    assert score(y, s, threshold=threshold)["f1"] == pytest.approx(1.0)


def test_score_reports_the_positive_count_it_was_given():
    y = np.array([0, 0, 1])
    s = np.array([0.1, 0.2, 0.9])
    out = score(y, s, threshold=0.5)
    assert out["n_rows"] == 3
    assert out["n_positive"] == 1


def test_a_column_with_no_positives_does_not_crash():
    y = np.zeros(5, dtype=int)
    s = np.linspace(0, 1, 5)
    out = score(y, s, threshold=0.5)
    assert out["n_positive"] == 0
    assert np.isnan(out["average_precision"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_metrics.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.metrics'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/metrics.py`:

```python
"""Scoring.

Average precision throughout. ROC AUC is deliberately absent: base rates in this
suite run from 0.10% to 0.52%, where ROC AUC is dominated by the negative class
and stays high for a model with no useful precision -- exactly the differences
this study exists to measure.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def score(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Average precision, plus precision/recall/F1 at `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype="float64")
    n_positive = int(y_true.sum())

    if n_positive == 0:
        ap = float("nan")
    else:
        ap = float(average_precision_score(y_true, y_score))

    predicted = y_score >= threshold
    tp = int((predicted & (y_true == 1)).sum())
    fp = int((predicted & (y_true == 0)).sum())
    fn = int((~predicted & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_rows": int(len(y_true)),
        "n_positive": n_positive,
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The threshold maximising F1. Chosen on validation, applied unchanged to test."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_metrics.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/metrics.py tests/ablation/test_metrics.py
git commit -m "feat: average-precision scoring for the ablation"
```

---

## Task 6: Models

**Files:**
- Create: `src/fraud_benchmark/ablation/models.py`
- Test: `tests/ablation/test_models.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_models.py`:

```python
import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.ablation.models import (
    TRIVIAL_RULES,
    fit_logistic,
    fit_xgboost,
    trivial_rule_scores,
)


def xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (x["a"] + rng.normal(scale=0.1, size=n) > 0).astype(int).to_numpy()
    return x, y


def test_logistic_learns_a_separable_signal():
    x, y = xy()
    model = fit_logistic(x, y)
    assert model.predict_proba(x)[:, 1].shape == (len(x),)
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, model.predict_proba(x)[:, 1]) > 0.9


def test_xgboost_learns_a_separable_signal():
    x, y = xy()
    model = fit_xgboost(x, y, seed=0)
    from sklearn.metrics import average_precision_score

    assert average_precision_score(y, model.predict_proba(x)[:, 1]) > 0.9


def test_xgboost_is_deterministic_for_a_fixed_seed():
    x, y = xy()
    a = fit_xgboost(x, y, seed=3).predict_proba(x)[:, 1]
    b = fit_xgboost(x, y, seed=3).predict_proba(x)[:, 1]
    assert np.allclose(a, b)


def test_xgboost_differs_across_seeds():
    """If seeds do not move the model, the seed-noise floor is meaningless."""
    x, y = xy()
    a = fit_xgboost(x, y, seed=0).predict_proba(x)[:, 1]
    b = fit_xgboost(x, y, seed=1).predict_proba(x)[:, 1]
    assert not np.allclose(a, b)


def test_the_trivial_rule_scores_one_for_matching_rows():
    df = pd.DataFrame({"Merchant State": pd.Series(["Italy", "CA"], dtype="string")})
    scores = trivial_rule_scores(df, dataset="ibm_ccf")
    assert list(scores) == [1.0, 0.0]


def test_datasets_without_a_rule_return_none():
    for dataset in ["saml_d", "sparkov"]:
        assert dataset not in TRIVIAL_RULES
        assert trivial_rule_scores(pd.DataFrame({"x": [1]}), dataset=dataset) is None


def test_the_rule_needs_no_fitting_and_ignores_labels():
    """It is a fixed rule, which is why it is evaluated once per dataset."""
    df = pd.DataFrame({"Merchant State": pd.Series(["Italy"], dtype="string")})
    assert list(trivial_rule_scores(df, dataset="ibm_ccf_subsample_fast")) == [1.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_models.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.models'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/models.py`:

```python
"""The three models.

Hyperparameters are fixed across every condition. Tuning per condition would
confound the ablation with tuning effort: a gap between `leaky` and `clean` would
no longer be attributable to the columns.

The trivial rule is the floor and matters more than it looks. An earlier probe
found a boosted model scoring *below* the one-line rule on the IBM CCF subsample;
without the floor in the table that result is invisible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# (column, value) pairs that alone identify most of a dataset's frauds.
TRIVIAL_RULES: dict[str, tuple[str, str]] = {
    "ibm_ccf": ("Merchant State", "Italy"),
    "ibm_ccf_subsample_fast": ("Merchant State", "Italy"),
    "ibm_ccf_subsample_slow": ("Merchant State", "Italy"),
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 4,
    "eval_metric": "aucpr",
}


def fit_logistic(x: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    """L2 logistic regression. Deterministic, so it needs no seeds."""
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=4
    )
    model.fit(x, y)
    return model


def fit_xgboost(x: pd.DataFrame, y: np.ndarray, seed: int) -> XGBClassifier:
    positive = int(np.sum(y))
    negative = int(len(y) - positive)
    scale = (negative / positive) if positive else 1.0
    model = XGBClassifier(random_state=seed, scale_pos_weight=scale, **XGB_PARAMS)
    model.fit(x, y)
    return model


def trivial_rule_scores(df: pd.DataFrame, dataset: str) -> np.ndarray | None:
    """1.0 where the rule fires, 0.0 elsewhere. None if the dataset has no rule.

    Reads the raw frame, not a feature matrix, so it is unaffected by the feature
    set and label regime -- which is why it is evaluated once per dataset.
    """
    rule = TRIVIAL_RULES.get(dataset)
    if rule is None:
        return None
    column, value = rule
    return (df[column].astype("string") == value).to_numpy().astype("float64")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_models.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/models.py tests/ablation/test_models.py
git commit -m "feat: trivial rule, logistic and xgboost models"
```

---

## Task 7: Feature build stage

**Files:**
- Create: `src/fraud_benchmark/ablation/build_features.py`
- Test: `tests/ablation/test_build_features.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_build_features.py`:

```python
import pandas as pd
import pytest

from fraud_benchmark.ablation.build_features import MERCHANT_COLUMNS, build


def source(tmp_path, name, df):
    directory = tmp_path / "processed" / name
    directory.mkdir(parents=True)
    df.to_parquet(directory / "data.parquet", index=False)
    return tmp_path


def sample():
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"]
            ),
            "entity_id": pd.Series(["a", "a", "b"], dtype="string"),
            "amount": [10.0, 20.0, 30.0],
            "is_fraud": [False, True, False],
            "split": pd.Series(["train", "val", "test"], dtype="string"),
            "reported_at": pd.to_datetime([None, "2023-01-08", None]),
            "merchant": pd.Series(["m1", "m2", "m1"], dtype="string"),
        }
    )


def test_build_writes_a_parquet_with_the_velocity_columns(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    result = pd.read_parquet(out)
    assert "txn_count_24h" in result.columns
    assert "merchant_novelty" in result.columns


def test_build_preserves_every_row(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    assert len(pd.read_parquet(out)) == 3


def test_build_preserves_the_split_column_and_its_counts(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    result = pd.read_parquet(out)
    assert result["split"].value_counts().to_dict() == {"train": 1, "val": 1, "test": 1}


def test_every_configured_dataset_names_a_merchant_column_or_none(tmp_path):
    for name in [
        "ibm_ccf",
        "ibm_ccf_subsample_fast",
        "ibm_ccf_subsample_slow",
        "saml_d",
        "sparkov",
    ]:
        assert name in MERCHANT_COLUMNS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_build_features.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.build_features'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/build_features.py`:

```python
"""Stage 1: build velocity features once per dataset and cache them.

The 18 stage-2 jobs read this cache. Rebuilding 24M rows of aggregates inside
each of them would be the obvious way to waste the day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_benchmark.ablation.features import add_velocity_features

DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_FEATURES = Path("data/features")

# The merchant-like column each dataset uses for `merchant_novelty`.
MERCHANT_COLUMNS: dict[str, str | None] = {
    "ibm_ccf": "Merchant Name",
    "ibm_ccf_subsample_fast": "Merchant Name",
    "ibm_ccf_subsample_slow": "Merchant Name",
    "saml_d": "Receiver_account",
    "sparkov": "merchant",
}


def build(
    dataset: str,
    processed_dir: Path = DEFAULT_PROCESSED,
    features_dir: Path = DEFAULT_FEATURES,
) -> Path:
    """Read the processed parquet, append velocity features, write the cache."""
    source = Path(processed_dir) / dataset / "data.parquet"
    df = pd.read_parquet(source)
    before = len(df)

    df = add_velocity_features(df, merchant_col=MERCHANT_COLUMNS[dataset])
    if len(df) != before:
        raise AssertionError(f"{dataset}: feature build changed the row count")

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    destination = features_dir / f"{dataset}.parquet"
    df.to_parquet(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation features for one dataset")
    parser.add_argument("dataset")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args()
    destination = build(args.dataset, args.processed_dir, args.features_dir)
    print(f"{args.dataset}: wrote {destination}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_build_features.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/build_features.py tests/ablation/test_build_features.py
git commit -m "feat: stage-1 feature build with caching"
```

---

## Task 8: Cell runner

**Files:**
- Create: `src/fraud_benchmark/ablation/cell.py`
- Test: `tests/ablation/test_cell.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_cell.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.ablation.cell import censored_mask, run_cell


def features(n=600, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n), unit="h")
    signal = rng.normal(size=n)
    is_fraud = (signal > 1.5)
    split = np.array(["train"] * (n - 200) + ["val"] * 100 + ["test"] * 100)
    return pd.DataFrame(
        {
            "event_time": times,
            "entity_id": pd.Series(rng.choice(["a", "b"], n), dtype="string"),
            "amount": rng.lognormal(3, 1, n),
            "signal": signal,
            "merchant": pd.Series(rng.choice(["m1", "m2"], n), dtype="string"),
            "Merchant State": pd.Series(
                np.where(is_fraud, "Italy", "CA"), dtype="string"
            ),
            "is_fraud": is_fraud,
            "split": pd.Series(split, dtype="string"),
            "reported_at": pd.Series(
                np.where(is_fraud, times + pd.Timedelta(days=30), pd.NaT)
            ),
            "txn_count_24h": rng.integers(0, 5, n).astype("float64"),
            "merchant_novelty": rng.integers(0, 2, n),
        }
    )


def test_censored_mask_hides_labels_reported_after_the_cutoff():
    df = pd.DataFrame(
        {
            "is_fraud": [True, True, False],
            "reported_at": pd.to_datetime(["2023-01-02", "2023-02-01", None]),
        }
    )
    mask = censored_mask(df, cutoff=pd.Timestamp("2023-01-15"))
    assert list(mask) == [True, False, True]


def test_censored_mask_keeps_every_non_fraud_row():
    """A non-fraud row's label is known immediately; only frauds are delayed."""
    df = pd.DataFrame(
        {"is_fraud": [False, False], "reported_at": pd.to_datetime([None, None])}
    )
    assert censored_mask(df, cutoff=pd.Timestamp("2023-01-01")).all()


def test_run_cell_writes_one_record_per_model(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["model"] for r in records} == {"logistic", "xgboost"}


def test_each_record_carries_both_split_scores(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "val" in record["scores"]
    assert "test" in record["scores"]
    assert "average_precision" in record["scores"]["val"]


def test_the_test_threshold_is_the_one_chosen_on_val(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scores"]["test"]["threshold"] == record["scores"]["val"]["threshold"]


def test_evaluation_splits_are_never_downsampled(tmp_path):
    out = tmp_path / "runs.jsonl"
    df = features()
    run_cell(
        df,
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scores"]["val"]["n_rows"] == int((df["split"] == "val").sum())
    assert record["scores"]["test"]["n_rows"] == int((df["split"] == "test").sum())


def test_the_trivial_rule_is_recorded_for_ibm_ccf(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="ibm_ccf",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
        include_rule=True,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    rule = [r for r in records if r["model"] == "trivial_rule"]
    assert len(rule) == 1
    assert rule[0]["feature_set"] is None
    assert rule[0]["label_regime"] is None


def test_the_censored_regime_uses_fewer_labels_than_oracle(tmp_path):
    out = tmp_path / "runs.jsonl"
    df = features()
    for regime in ["oracle", "censored"]:
        run_cell(
            df,
            dataset="sparkov",
            feature_set="leaky",
            label_regime=regime,
            seeds=(0,),
            results_path=out,
        )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    oracle = next(r for r in records if r["label_regime"] == "oracle")
    censored = next(r for r in records if r["label_regime"] == "censored")
    assert censored["n_train_rows"] < oracle["n_train_rows"]


def test_records_append_rather_than_overwrite(tmp_path):
    out = tmp_path / "runs.jsonl"
    for feature_set in ["leaky", "clean"]:
        run_cell(
            features(),
            dataset="sparkov",
            feature_set=feature_set,
            label_regime="oracle",
            seeds=(0,),
            results_path=out,
        )
    assert len(out.read_text().splitlines()) == 4


def test_datetime_columns_never_reach_a_model(tmp_path):
    """Neither model accepts datetime64, and coercing one would hand the model
    the split boundary as an integer threshold."""
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "event_time" not in record["features"]


def test_the_resolved_feature_list_is_recorded(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="ibm_ccf",
        feature_set="clean",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "Merchant State" not in record["features"]
    assert "is_fraud" not in record["features"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_cell.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.cell'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/cell.py`:

```python
"""Run one (dataset, feature set, label regime) cell.

Every evaluation appends a record the moment it is scored, so a job that dies
loses at most the fit in flight. The record carries the resolved feature list, so
a result can always be traced to the exact columns that produced it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.ablation.build_features import DEFAULT_FEATURES
from fraud_benchmark.ablation.columns import feature_columns
from fraud_benchmark.ablation.encoding import Encoder
from fraud_benchmark.ablation.metrics import best_f1_threshold, score
from fraud_benchmark.ablation.models import fit_logistic, fit_xgboost, trivial_rule_scores

DEFAULT_RESULTS = Path("results/runs.jsonl")


def censored_mask(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Rows whose label a model training at `cutoff` would actually have.

    Non-fraud rows are always known: only a fraud's label waits on a report.
    """
    reported = pd.to_datetime(df["reported_at"])
    return (~df["is_fraud"].astype(bool)) | (reported <= cutoff)


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def run_cell(
    df: pd.DataFrame,
    dataset: str,
    feature_set: str,
    label_regime: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    results_path: Path = DEFAULT_RESULTS,
    include_rule: bool = False,
) -> None:
    results_path = Path(results_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    if label_regime == "censored":
        train = train[censored_mask(train, cutoff=train["event_time"].max())]
    elif label_regime != "oracle":
        raise ValueError(f"unknown label_regime {label_regime!r}")

    columns = feature_columns(df, dataset=dataset, feature_set=feature_set)

    # Datetime columns are dropped outright: neither model accepts datetime64, and
    # silently coercing one to an integer nanosecond count would hand the model the
    # split boundary as a threshold. This covers event_time and each dataset's own
    # date columns (sparkov's trans_date_trans_time and dob, saml_d's Date).
    # A dataset's *numeric* time parts -- IBM CCF's Year/Month/Day/Time -- are kept:
    # they are ordinary columns of the source data, and the regime shift they encode
    # is a documented property of the dataset rather than an artifact this study
    # ablates.
    columns = [c for c in columns if not pd.api.types.is_datetime64_any_dtype(df[c])]

    categorical = [
        c for c in columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = [c for c in columns if c not in categorical]

    y_train = train["is_fraud"].to_numpy().astype(int)
    y_val = val["is_fraud"].to_numpy().astype(int)
    y_test = test["is_fraud"].to_numpy().astype(int)

    if include_rule:
        rule_val = trivial_rule_scores(val, dataset=dataset)
        if rule_val is not None:
            rule_test = trivial_rule_scores(test, dataset=dataset)
            threshold = best_f1_threshold(y_val, rule_val)
            _append(
                results_path,
                {
                    "dataset": dataset,
                    "feature_set": None,
                    "label_regime": None,
                    "model": "trivial_rule",
                    "seed": None,
                    "features": [],
                    "n_train_rows": 0,
                    "fit_seconds": 0.0,
                    "scores": {
                        "val": score(y_val, rule_val, threshold),
                        "test": score(y_test, rule_test, threshold),
                    },
                },
            )

    encoder_common = dict(categorical=categorical, numeric=numeric)
    tree_encoder = Encoder().fit(train[columns], scale=False, **encoder_common)
    linear_encoder = Encoder().fit(train[columns], scale=True, **encoder_common)

    def emit(model_name: str, seed: int | None, model, encoder) -> None:
        val_scores = model.predict_proba(encoder.transform(val[columns]))[:, 1]
        test_scores = model.predict_proba(encoder.transform(test[columns]))[:, 1]
        threshold = best_f1_threshold(y_val, val_scores)
        _append(
            results_path,
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "label_regime": label_regime,
                "model": model_name,
                "seed": seed,
                "features": columns,
                "n_train_rows": int(len(train)),
                "fit_seconds": round(elapsed, 2),
                "scores": {
                    "val": score(y_val, val_scores, threshold),
                    "test": score(y_test, test_scores, threshold),
                },
            },
        )

    started = time.monotonic()
    linear = fit_logistic(linear_encoder.transform(train[columns]), y_train)
    elapsed = time.monotonic() - started
    emit("logistic", None, linear, linear_encoder)

    x_train = tree_encoder.transform(train[columns])
    for seed in seeds:
        started = time.monotonic()
        booster = fit_xgboost(x_train, y_train, seed=seed)
        elapsed = time.monotonic() - started
        emit("xgboost", seed, booster, tree_encoder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ablation cell")
    parser.add_argument("dataset")
    parser.add_argument("feature_set", choices=["leaky", "clean"])
    parser.add_argument("label_regime", choices=["oracle", "censored"])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--include-rule", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.features_dir) / f"{args.dataset}.parquet")
    run_cell(
        df,
        dataset=args.dataset,
        feature_set=args.feature_set,
        label_regime=args.label_regime,
        results_path=args.results,
        include_rule=args.include_rule,
    )
    print(f"{args.dataset}/{args.feature_set}/{args.label_regime}: done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_cell.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, all previous tests plus the new ones

- [ ] **Step 6: Commit**

```bash
git add src/fraud_benchmark/ablation/cell.py tests/ablation/test_cell.py
git commit -m "feat: run one ablation cell and append its records"
```

---

## Task 9: SLURM scripts

**Files:**
- Create: `scripts/slurm/generate.py`
- Test: `tests/ablation/test_slurm.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_slurm.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "slurm"))

from generate import CELLS, FEATURE_JOBS, render_cell_job, render_feature_job, write_all


def test_ibm_ccf_has_no_censored_cell():
    """3 censored labels of 24,924 -- the contrast is arithmetically null."""
    assert ("ibm_ccf", "leaky", "censored") not in CELLS
    assert ("ibm_ccf", "clean", "censored") not in CELLS


def test_ibm_ccf_keeps_both_oracle_cells():
    assert ("ibm_ccf", "leaky", "oracle") in CELLS
    assert ("ibm_ccf", "clean", "oracle") in CELLS


def test_there_are_eighteen_cells():
    assert len(CELLS) == 18


def test_every_other_dataset_has_four_cells():
    for dataset in ["ibm_ccf_subsample_fast", "ibm_ccf_subsample_slow", "saml_d", "sparkov"]:
        assert len([c for c in CELLS if c[0] == dataset]) == 4


def test_a_cell_job_declares_cpu_and_memory():
    script = render_cell_job("sparkov", "leaky", "oracle")
    assert "--cpus-per-task=4" in script
    assert "--mem=128G" in script


def test_the_ibm_ccf_jobs_use_the_long_partition():
    """24.4M rows will not finish inside the 1-day cpu limit reliably."""
    assert "--partition=cpulong" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--partition=cpu\n" in render_cell_job("sparkov", "leaky", "oracle")


def test_jobs_invoke_the_venv_python_by_absolute_path():
    """There is no module system on this cluster."""
    assert "/.venv/bin/python" in render_cell_job("sparkov", "leaky", "oracle")


def test_only_the_leaky_oracle_cell_requests_the_rule():
    """The rule is fixed, so it is evaluated once per dataset."""
    assert "--include-rule" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--include-rule" not in render_cell_job("ibm_ccf", "clean", "oracle")


def test_feature_jobs_exist_for_every_dataset():
    assert set(FEATURE_JOBS) == {
        "ibm_ccf",
        "ibm_ccf_subsample_fast",
        "ibm_ccf_subsample_slow",
        "saml_d",
        "sparkov",
    }


def test_write_all_emits_a_submit_script_with_dependencies(tmp_path):
    write_all(tmp_path)
    submit = (tmp_path / "submit_all.sh").read_text()
    assert "--dependency=afterok" in submit
    assert len(list(tmp_path.glob("*.sbatch"))) == 23


def test_feature_job_writes_to_the_features_dir():
    assert "build_features" in render_feature_job("sparkov")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_slurm.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'generate'`

- [ ] **Step 3: Write the implementation**

`scripts/slurm/generate.py`:

```python
#!/usr/bin/env python
"""Emit the sbatch files for the leakage ablation, plus a submit script.

Cluster facts this encodes: `cpu` is 48-core / 384 GB nodes with a 1-day limit,
`cpulong` allows 3 days, the default account is `smidlva1`, and there is no
module system -- jobs call the venv's Python by absolute path.

Jobs are small and numerous on purpose. The grid is 18 model jobs that run in
parallel across the cluster, so wall-clock is bounded by the slowest single job
rather than by their sum.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "bin" / "python"
ACCOUNT = "smidlva1"

DATASETS = (
    "ibm_ccf",
    "ibm_ccf_subsample_fast",
    "ibm_ccf_subsample_slow",
    "saml_d",
    "sparkov",
)

# ibm_ccf runs oracle only: it has 3 censored train labels out of 24,924, so the
# censored regime is identical to oracle and would cost roughly eight hours of
# the most expensive fits in the study to confirm arithmetic.
CELLS = tuple(
    (dataset, feature_set, regime)
    for dataset in DATASETS
    for feature_set in ("leaky", "clean")
    for regime in (("oracle",) if dataset == "ibm_ccf" else ("oracle", "censored"))
)

FEATURE_JOBS = DATASETS

_BIG = {"ibm_ccf"}


def _resources(dataset: str, stage: str) -> tuple[str, str]:
    if dataset in _BIG:
        return "cpulong", "24:00:00" if stage == "cell" else "12:00:00"
    return "cpu", "12:00:00" if stage == "cell" else "08:00:00"


def render_feature_job(dataset: str) -> str:
    partition, walltime = _resources(dataset, "features")
    return f"""#!/bin/bash
#SBATCH --job-name=feat_{dataset}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/feat_{dataset}_%j.out
#SBATCH --error={REPO}/results/logs/feat_{dataset}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
{PYTHON} -m fraud_benchmark.ablation.build_features {dataset}
"""


def render_cell_job(dataset: str, feature_set: str, regime: str) -> str:
    partition, walltime = _resources(dataset, "cell")
    name = f"{dataset}_{feature_set}_{regime}"
    # The trivial rule is fixed, so it is emitted once per dataset -- from the
    # leaky/oracle cell, which every dataset has.
    rule = " --include-rule" if (feature_set, regime) == ("leaky", "oracle") else ""
    return f"""#!/bin/bash
#SBATCH --job-name=abl_{name}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/abl_{name}_%j.out
#SBATCH --error={REPO}/results/logs/abl_{name}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
{PYTHON} -m fraud_benchmark.ablation.cell {dataset} {feature_set} {regime}{rule}
"""


def write_all(directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {directory}", ""]
    for dataset in FEATURE_JOBS:
        (directory / f"feat_{dataset}.sbatch").write_text(render_feature_job(dataset))
        lines.append(f'feat_{dataset}=$(sbatch --parsable feat_{dataset}.sbatch)')
    lines.append("")
    for dataset, feature_set, regime in CELLS:
        name = f"{dataset}_{feature_set}_{regime}"
        (directory / f"abl_{name}.sbatch").write_text(
            render_cell_job(dataset, feature_set, regime)
        )
        lines.append(
            f'sbatch --dependency=afterok:$feat_{dataset} abl_{name}.sbatch'
        )
    lines.append("")
    submit = directory / "submit_all.sh"
    submit.write_text("\n".join(lines))
    submit.chmod(0o755)


if __name__ == "__main__":
    target = REPO / "scripts" / "slurm" / "jobs"
    write_all(target)
    (REPO / "results" / "logs").mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(FEATURE_JOBS) + len(CELLS)} sbatch files to {target}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_slurm.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Generate the scripts and eyeball one**

Run: `.venv/bin/python scripts/slurm/generate.py && cat scripts/slurm/jobs/abl_sparkov_leaky_oracle.sbatch`
Expected: 23 sbatch files written; the printed script names `--partition=cpu`, `--mem=128G`, and the venv Python by absolute path.

- [ ] **Step 6: Commit**

```bash
git add scripts/slurm/generate.py tests/ablation/test_slurm.py
git commit -m "feat: generate the ablation sbatch files"
```

---

## Task 10: Summary table

**Files:**
- Create: `src/fraud_benchmark/ablation/summarize.py`
- Test: `tests/ablation/test_summarize.py`

- [ ] **Step 1: Write the failing tests**

`tests/ablation/test_summarize.py`:

```python
import json

from fraud_benchmark.ablation.summarize import summarize


def record(dataset, feature_set, regime, model, ap, seed=None):
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "label_regime": regime,
        "model": model,
        "seed": seed,
        "features": [],
        "n_train_rows": 100,
        "fit_seconds": 1.0,
        "scores": {
            "val": {"average_precision": ap, "f1": 0.5, "threshold": 0.5,
                     "precision": 0.5, "recall": 0.5, "n_rows": 10, "n_positive": 1},
            "test": {"average_precision": ap, "f1": 0.5, "threshold": 0.5,
                      "precision": 0.5, "recall": 0.5, "n_rows": 10, "n_positive": 1},
        },
    }


def write(tmp_path, records):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_seeds_are_averaged(tmp_path):
    path = write(
        tmp_path,
        [
            record("sparkov", "leaky", "oracle", "xgboost", 0.4, seed=0),
            record("sparkov", "leaky", "oracle", "xgboost", 0.6, seed=1),
        ],
    )
    text = summarize(path)
    assert "0.500" in text


def test_the_leakage_gap_is_reported(tmp_path):
    path = write(
        tmp_path,
        [
            record("ibm_ccf", "leaky", "oracle", "xgboost", 0.80, seed=0),
            record("ibm_ccf", "clean", "oracle", "xgboost", 0.20, seed=0),
        ],
    )
    text = summarize(path)
    assert "0.600" in text


def test_average_precision_is_named_not_abbreviated_as_auc(tmp_path):
    path = write(tmp_path, [record("sparkov", "leaky", "oracle", "xgboost", 0.4, seed=0)])
    text = summarize(path)
    assert "average precision" in text.lower()
    assert "roc" not in text.lower()


def test_an_empty_results_file_produces_a_message_not_a_crash(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("")
    assert "no results" in summarize(path).lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ablation/test_summarize.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fraud_benchmark.ablation.summarize'`

- [ ] **Step 3: Write the implementation**

`src/fraud_benchmark/ablation/summarize.py`:

```python
"""results/runs.jsonl -> results/summary.md.

Seeds are averaged and their spread reported: the seed standard deviation is the
noise floor any leakage gap has to clear before it means anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS = Path("results/runs.jsonl")
DEFAULT_SUMMARY = Path("results/summary.md")


def _load(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for split in ("val", "test"):
            scores = record["scores"][split]
            rows.append(
                {
                    "dataset": record["dataset"],
                    "feature_set": record["feature_set"] or "n/a",
                    "label_regime": record["label_regime"] or "n/a",
                    "model": record["model"],
                    "split": split,
                    "average_precision": scores["average_precision"],
                    "f1": scores["f1"],
                }
            )
    return pd.DataFrame(rows)


def summarize(results_path: Path = DEFAULT_RESULTS) -> str:
    df = _load(results_path)
    if df.empty:
        return "# Leakage ablation\n\nNo results yet.\n"

    grouped = (
        df.groupby(["dataset", "split", "label_regime", "model", "feature_set"])[
            "average_precision"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    lines = [
        "# Leakage ablation",
        "",
        "Average precision. ROC AUC is not reported: base rates here run from 0.10% to",
        "0.52%, where it stays high for a model with no useful precision.",
        "",
        "`gap` is leaky minus clean — how much of the score the leaky columns supply.",
        "Sparkov has no leaky columns, so its gap is a pure measurement of noise and",
        "calibrates how much of every other row to believe.",
        "",
        "| dataset | split | labels | model | leaky | clean | gap | seed sd |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    keys = ["dataset", "split", "label_regime", "model"]
    for (dataset, split, regime, model), block in grouped.groupby(keys, sort=True):
        by_set = block.set_index("feature_set")
        leaky = by_set["mean"].get("leaky")
        clean = by_set["mean"].get("clean")
        spread = block["std"].max()
        gap = (leaky - clean) if (leaky is not None and clean is not None) else None

        def cell(value):
            return "—" if value is None or pd.isna(value) else f"{value:.3f}"

        if model == "trivial_rule":
            leaky_cell = cell(by_set["mean"].get("n/a"))
            clean_cell, gap_cell = "—", "—"
        else:
            leaky_cell, clean_cell, gap_cell = cell(leaky), cell(clean), cell(gap)

        lines.append(
            f"| {dataset} | {split} | {regime} | {model} | "
            f"{leaky_cell} | {clean_cell} | {gap_cell} | {cell(spread)} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise ablation results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    text = summarize(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ablation/test_summarize.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fraud_benchmark/ablation/summarize.py tests/ablation/test_summarize.py
git commit -m "feat: summarise ablation results into a markdown table"
```

---

## Task 11: Smoke run on Sparkov

Sparkov is 1.85M rows — the smallest dataset — so it is the cheapest way to find a wiring bug before committing the cluster to 23 jobs.

- [ ] **Step 1: Run the whole test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, everything green

- [ ] **Step 2: Build Sparkov's features locally**

Run: `.venv/bin/python -m fraud_benchmark.ablation.build_features sparkov`
Expected: `sparkov: wrote data/features/sparkov.parquet`

- [ ] **Step 3: Check the features look causal on real data**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/features/sparkov.parquet')
print('rows:', len(df))
print(df[['txn_count_24h','amount_sum_7d','seconds_since_prev_txn','merchant_novelty']].describe())
print('split counts:', df['split'].value_counts().to_dict())
"
```
Expected: 1,852,394 rows; split counts `{'train': 1167008, 'test': 555719, 'val': 129667}`; `txn_count_24h` minimum 0 (never negative, never null); `merchant_novelty` in {0, 1}.

- [ ] **Step 4: Run one cell locally**

Run: `.venv/bin/python -m fraud_benchmark.ablation.cell sparkov leaky oracle --include-rule`
Expected: `sparkov/leaky/oracle: done`, and `results/runs.jsonl` containing 4 records (logistic + 3 XGBoost seeds; no rule, since Sparkov has none).

- [ ] **Step 5: Confirm the evaluation splits were not downsampled**

Run:
```bash
.venv/bin/python -c "
import json
for line in open('results/runs.jsonl'):
    r = json.loads(line)
    print(r['model'], r['seed'],
          'val_rows', r['scores']['val']['n_rows'],
          'test_rows', r['scores']['test']['n_rows'],
          'val_AP', round(r['scores']['val']['average_precision'], 4))
"
```
Expected: every row shows `val_rows 129667` and `test_rows 555719`, matching the dataset card exactly.

- [ ] **Step 6: Commit the smoke-run evidence**

```bash
git add -A results/.gitignore 2>/dev/null || true
git commit --allow-empty -m "test: sparkov smoke run passes end to end"
```

---

## Task 12: Dispatch the grid

- [ ] **Step 1: Add results and features to .gitignore**

Append to `.gitignore`:

```
data/features/
results/logs/
```

`results/runs.jsonl` and `results/summary.md` stay tracked — they are the study's output.

- [ ] **Step 2: Generate the sbatch files**

Run: `.venv/bin/python scripts/slurm/generate.py`
Expected: `wrote 23 sbatch files to .../scripts/slurm/jobs`

- [ ] **Step 3: Clear the smoke-run results**

Run: `rm -f results/runs.jsonl`
Expected: no output. The grid must start from an empty file so the smoke run's records are not mixed into the study.

- [ ] **Step 4: Submit**

Run: `bash scripts/slurm/jobs/submit_all.sh`
Expected: 23 job ids printed, the 18 cell jobs each held behind their dataset's feature job.

- [ ] **Step 5: Watch until the queue drains**

Run: `squeue -u $USER`
Expected: eventually empty. Check `results/logs/*.err` for any non-empty file before trusting the results.

- [ ] **Step 6: Summarise**

Run: `.venv/bin/python -m fraud_benchmark.ablation.summarize && cat results/summary.md`
Expected: the ablation table, with a row per (dataset, split, label regime, model).

- [ ] **Step 7: Commit the results**

```bash
git add results/runs.jsonl results/summary.md .gitignore
git commit -m "results: leakage ablation across the five candidate datasets"
```

---

## Task 13: Write up the findings

- [ ] **Step 1: Append a `## Leakage ablation` section to `docs/verification-notes.md`**

Follow the style of the existing `## Known leakage` section: what was measured, the
numbers, what it means, what it changes. It must cover, with the measured values:

1. The average-precision gap between `leaky` and `clean` for each dataset, on test.
2. **Sparkov's gap as the noise floor**, stated before any other gap is interpreted —
   it has no leaky columns, so its gap is measurement error and bounds what the others
   mean.
3. Whether XGBoost beat the trivial rule on the IBM CCF variants, on both val and test.
4. Whether the delay effect (oracle minus censored) exceeded the seed standard deviation
   on the two subsamples — the question the earlier quick probe answered "below noise".
5. The `Merchant Name` deviation from the audit's suggested leaky tuple, and its effect.
6. Novelty's standalone lift per dataset (13–15× IBM CCF, 8.4× SAML-D, 1.60× Sparkov) and
   the open question of whether the first two are a separate generation artifact.

- [ ] **Step 2: Update the plan and spec status headers**

Add to the top of both `docs/superpowers/plans/2026-08-02-leakage-ablation.md` and
`docs/superpowers/specs/2026-08-02-leakage-ablation-design.md`:

```markdown
**Status:** done, <date>. Results in `results/summary.md` and the `## Leakage ablation`
section of `docs/verification-notes.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/verification-notes.md docs/superpowers/plans/2026-08-02-leakage-ablation.md docs/superpowers/specs/2026-08-02-leakage-ablation-design.md
git commit -m "docs: record the leakage ablation results"
```

---

## Done criteria

- [ ] `.venv/bin/python -m pytest -q` passes
- [ ] `data/features/` holds one parquet per dataset, each with the same row count as its
      source
- [ ] `results/runs.jsonl` holds 75 records: 72 fits (18 cells × (1 logistic + 3 XGBoost))
      plus 3 trivial-rule evaluations
- [ ] Every record's `val` and `test` `n_rows` match the dataset card's split counts —
      evaluation was never downsampled
- [ ] No record's `features` list contains any name in `ALWAYS_EXCLUDED`
- [ ] No `clean` record's `features` list contains any of its dataset's leaky columns
- [ ] `results/summary.md` reports average precision and contains no ROC AUC
- [ ] `docs/verification-notes.md` has a `## Leakage ablation` section that states
      Sparkov's noise floor before interpreting any other gap
