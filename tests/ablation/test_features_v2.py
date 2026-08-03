"""Causality tests for the richer feature set.

Same reasoning as the v1 velocity tests: a lookahead here raises the score and
raises no error, so it must be asserted directly rather than eyeballed.
"""
import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.features_v2 import V2_COLUMNS, add_v2_features

KW = dict(merchant_col="merchant", state_col="state", city_col="city", mcc_col="mcc",
          channel_col="channel", error_col="error", amount_col="amount")


def frame(rows):
    """rows: (entity, 'YYYY-MM-DD HH:MM', amount, state, mcc, error)"""
    df = pd.DataFrame({
        "entity_id": pd.Series([r[0] for r in rows], dtype="string"),
        "event_time": pd.to_datetime([r[1] for r in rows]),
        "amount": [float(r[2]) for r in rows],
        "state": pd.Series([r[3] for r in rows], dtype="string"),
        "mcc": pd.Series([str(r[4]) for r in rows], dtype="string"),
        "error": pd.Series([r[5] for r in rows], dtype="string"),
        "city": pd.Series(["c"] * len(rows), dtype="string"),
        "merchant": pd.Series(["m"] * len(rows), dtype="string"),
        "channel": pd.Series(["swipe"] * len(rows), dtype="string"),
    })
    for c in ["txn_count_1h", "txn_count_24h", "txn_count_7d",
              "amount_sum_24h", "amount_sum_7d"]:
        df[c] = 1.0
    return df


def test_first_occurrence_marks_only_the_entitys_first_visit():
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None),
                ("a", "2023-01-02 00:00", 10, "CA", 1, None),
                ("a", "2023-01-03 00:00", 10, "NY", 1, None)])
    out = add_v2_features(df, **KW)
    assert list(out["first_state_for_entity"]) == [1.0, 0.0, 1.0]


def test_first_occurrence_is_per_entity():
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None),
                ("b", "2023-01-02 00:00", 10, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    assert list(out["first_state_for_entity"]) == [1.0, 1.0]


def test_first_foreign_marks_only_the_first_trip_abroad():
    """A 2-letter code is US; anything else is foreign."""
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None),
                ("a", "2023-01-02 00:00", 10, "Italy", 1, None),
                ("a", "2023-01-03 00:00", 10, "Spain", 1, None),
                ("a", "2023-01-04 00:00", 10, "TX", 1, None)])
    out = add_v2_features(df, **KW)
    assert list(out["first_foreign_for_entity"]) == [0.0, 1.0, 0.0, 0.0]


def test_errors_window_excludes_the_current_row():
    """Three declines an hour apart: the counts must be 0, 1, 2."""
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, "Bad PIN"),
                ("a", "2023-01-01 01:00", 10, "CA", 1, "Bad PIN"),
                ("a", "2023-01-01 02:00", 10, "CA", 1, "Bad PIN")])
    out = add_v2_features(df, **KW)
    assert list(out["errors_24h"]) == [0.0, 1.0, 2.0]


def test_entity_baseline_excludes_the_current_amount():
    """Prior amounts 10 and 20 -> mean 15; the third row of 60 is 4x it."""
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None),
                ("a", "2023-01-02 00:00", 20, "CA", 1, None),
                ("a", "2023-01-03 00:00", 60, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    assert out.loc[2, "amount_over_entity_mean"] == pytest.approx(60 / 15)
    assert out.loc[2, "amount_over_entity_max"] == pytest.approx(60 / 20)
    assert list(out["entity_txn_ordinal"]) == [0.0, 1.0, 2.0]


def test_the_first_row_of_an_entity_has_no_baseline():
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    assert out.loc[0, "amount_over_entity_mean"] == 0.0
    assert out.loc[0, "entity_txn_ordinal"] == 0.0
    assert pd.isna(out.loc[0, "secs_since_same_state"])


def test_amount_shape_flags():
    df = frame([("a", "2023-01-01 00:00", 100.0, "CA", 1, None),
                ("a", "2023-01-02 00:00", 20.0, "CA", 1, None),
                ("a", "2023-01-03 00:00", 0.05, "CA", 1, None),
                ("a", "2023-01-04 00:00", 12.34, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    assert list(out["amount_is_round_100"]) == [1.0, 0.0, 0.0, 0.0]
    assert list(out["amount_is_round_10"]) == [1.0, 1.0, 0.0, 0.0]
    assert list(out["amount_is_micro"]) == [0.0, 0.0, 1.0, 0.0]
    assert out.loc[3, "amount_cents"] == 34.0


def test_output_row_order_matches_input():
    df = frame([("a", "2023-01-03 00:00", 30, "NY", 1, None),
                ("a", "2023-01-01 00:00", 10, "CA", 1, None),
                ("a", "2023-01-02 00:00", 20, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    assert list(out["amount"]) == [30.0, 10.0, 20.0]
    assert list(out["first_state_for_entity"]) == [1.0, 1.0, 0.0]


def test_every_declared_column_is_produced():
    df = frame([("a", "2023-01-01 00:00", 10, "CA", 1, None)])
    out = add_v2_features(df, **KW)
    for name in V2_COLUMNS:
        assert name in out.columns, name


def test_no_lookahead_property():
    """A row's features must not change when later rows are removed."""
    rng = np.random.default_rng(0)
    n = 240
    rows = [(str(rng.integers(0, 4)),
             f"2023-01-{1 + i // 12:02d} {i % 12:02d}:00",
             float(rng.lognormal(3, 1)),
             rng.choice(["CA", "NY", "Italy", "Spain"]),
             int(rng.integers(1000, 1010)),
             rng.choice([None, "Bad PIN", "Bad CVV"]))
            for i in range(n)]
    df = frame(rows)
    full = add_v2_features(df, **KW)
    for i in (23, 97, 180, 239):
        part = add_v2_features(df.iloc[: i + 1].copy(), **KW)
        for col in V2_COLUMNS:
            if col.startswith("days_"):
                continue        # derived from static card dates, not history
            a, b = full.loc[i, col], part.loc[i, col]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (
                f"{col} at row {i} changed when later rows were removed: {a} vs {b}"
            )
