"""The canonical schema every processed dataset conforms to."""

from __future__ import annotations

import pandas as pd

# Core columns, in the order they appear in the output. `split` is added by the
# pipeline and `reported_at` by the label-delay stage, so both may be absent.
CORE_ORDER = ("event_time", "entity_id", "amount", "is_fraud", "split", "reported_at")

# Non-temporal columns every adapter must produce, mapped to their exact dtype.
# `event_time` is checked separately: pandas datetime resolution varies by version
# and by how the column was built (pandas 3 yields [us], not [ns]), so any
# timezone-naive datetime64 resolution is accepted.
REQUIRED_DTYPES = {
    "entity_id": "string",
    "amount": "float64",
    "is_fraud": "bool",
}

REQUIRED_COLUMNS = ("event_time", *REQUIRED_DTYPES)

# Any real transaction dataset falls inside this window. Outside it, the adapter
# almost certainly mis-parsed a relative offset.
MIN_PLAUSIBLE_TIME = pd.Timestamp("1990-01-01")
MAX_PLAUSIBLE_TIME = pd.Timestamp("2050-01-01")


class SchemaError(ValueError):
    """Raised when a frame does not conform to the canonical schema."""


def validate_canonical(df: pd.DataFrame) -> None:
    """Raise SchemaError unless `df` is a valid canonical frame.

    Checks that the frame is non-empty, that every required column is present with
    the expected dtype and no nulls, and that event_time falls in a plausible range.
    `reported_at`, when present, must be datetime64, set only on frauds, and never
    earlier than the event it labels.
    """
    if len(df) == 0:
        raise SchemaError("dataset is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")

    if not pd.api.types.is_datetime64_dtype(df["event_time"]):
        raise SchemaError(
            f"column 'event_time' has dtype {str(df['event_time'].dtype)!r}; expected a "
            "timezone-naive datetime64 column (any resolution)"
        )

    for column, expected in REQUIRED_DTYPES.items():
        actual = str(df[column].dtype)
        if actual != expected:
            raise SchemaError(
                f"column {column!r} has dtype {actual!r}, expected {expected!r}"
            )

    for column in REQUIRED_COLUMNS:
        null_count = int(df[column].isna().sum())
        if null_count:
            raise SchemaError(f"column {column!r} has {null_count} null value(s)")

    times = df["event_time"]
    if times.min() < MIN_PLAUSIBLE_TIME or times.max() > MAX_PLAUSIBLE_TIME:
        raise SchemaError(
            f"column 'event_time' has implausible range "
            f"{times.min()} to {times.max()}; expected between "
            f"{MIN_PLAUSIBLE_TIME.date()} and {MAX_PLAUSIBLE_TIME.date()}"
        )

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


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder `df` to CORE_ORDER first, then the remaining columns as they were."""
    core = [c for c in CORE_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in core]
    return df[core + rest]
