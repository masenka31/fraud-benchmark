import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.features import VELOCITY_COLUMNS, add_velocity_features


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
