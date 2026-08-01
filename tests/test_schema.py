import pandas as pd
import pytest

from fraud_benchmark.schema import SchemaError, order_columns, validate_canonical


def make_valid_frame():
    return pd.DataFrame(
        {
            "amount": pd.Series([1.0, 2.0], dtype="float64"),
            "event_time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "extra": ["a", "b"],
            "entity_id": pd.Series(["c1", "c2"], dtype="string"),
            "is_fraud": pd.Series([True, False], dtype="bool"),
        }
    )


def test_validate_accepts_a_valid_frame():
    validate_canonical(make_valid_frame())


def test_validate_rejects_missing_column():
    df = make_valid_frame().drop(columns=["entity_id"])
    with pytest.raises(SchemaError, match="entity_id"):
        validate_canonical(df)


def test_validate_rejects_wrong_dtype():
    df = make_valid_frame()
    df["amount"] = df["amount"].astype("int64")
    with pytest.raises(SchemaError, match="amount"):
        validate_canonical(df)


def test_validate_rejects_nulls_in_required_column():
    df = make_valid_frame()
    df.loc[0, "entity_id"] = None
    with pytest.raises(SchemaError, match="null"):
        validate_canonical(df)


def test_validate_rejects_empty_frame():
    df = make_valid_frame().iloc[:0]
    with pytest.raises(SchemaError, match="empty"):
        validate_canonical(df)


def test_validate_rejects_implausible_timestamps():
    df = make_valid_frame()
    df["event_time"] = pd.to_datetime(["1900-01-01", "1900-01-02"])
    with pytest.raises(SchemaError, match="implausible"):
        validate_canonical(df)


def test_validate_accepts_any_datetime_resolution():
    # pandas 3 yields datetime64[us] from to_datetime; older code paths yield [ns].
    # Both are valid canonical frames.
    df = make_valid_frame()
    df["event_time"] = df["event_time"].astype("datetime64[ns]")
    validate_canonical(df)
    df["event_time"] = df["event_time"].astype("datetime64[s]")
    validate_canonical(df)


def test_validate_rejects_timezone_aware_event_time():
    df = make_valid_frame()
    df["event_time"] = df["event_time"].dt.tz_localize("UTC")
    with pytest.raises(SchemaError, match="event_time"):
        validate_canonical(df)


def test_order_columns_puts_core_first_and_keeps_the_rest():
    ordered = order_columns(make_valid_frame())
    assert list(ordered.columns) == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "extra",
    ]


def test_order_columns_includes_split_when_present():
    df = make_valid_frame()
    df["split"] = pd.Series(["train", "test"], dtype="category")
    ordered = order_columns(df)
    assert list(ordered.columns)[:5] == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "split",
    ]
