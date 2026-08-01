"""Temporal train/validation/test splitting.

Cuts land on timestamp values rather than row positions, so transactions sharing an
event_time always end up in the same split. Datasets with coarse time resolution
(PaySim's hours, BankSim's days) have heavy ties, and slicing through a tied block
would leak same-instant information across the boundary.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def _cumulative_fraction(df: pd.DataFrame) -> pd.Series:
    """Fraction of rows at or before each distinct timestamp, indexed by timestamp."""
    counts = df["event_time"].value_counts().sort_index()
    return counts.cumsum() / len(df)


def _cut_at(cumulative: pd.Series, target: float) -> pd.Timestamp:
    """The last timestamp belonging to a split ending at `target`.

    Returns the first timestamp whose cumulative fraction reaches `target`, so a split
    is never empty even when a single tied block is larger than its target share.
    """
    index = int(np.searchsorted(cumulative.to_numpy(), target, side="left"))
    index = min(index, len(cumulative) - 1)
    return cumulative.index[index]


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3:
        raise ValueError(f"expected three ratios, got {len(ratios)}")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"ratios must all be positive, got {ratios}")


def _validate_frame(df: pd.DataFrame) -> None:
    """Reject input that would otherwise be split silently and wrongly."""
    if "event_time" not in df.columns:
        raise ValueError("cannot split: frame has no 'event_time' column")
    if len(df) == 0:
        raise ValueError("cannot split an empty frame")
    missing = int(df["event_time"].isna().sum())
    if missing:
        # NaT compares False against every boundary, so it would land in the
        # last split regardless of when it belongs.
        raise ValueError(
            f"cannot split: 'event_time' has {missing} missing value(s), which would "
            "be silently assigned to the last split"
        )


def _warn_if_any_split_is_empty(
    splits: pd.Series, ratios: tuple[float, float, float], n_distinct: int
) -> None:
    counts = splits.value_counts()
    empty = [name for name in SPLIT_NAMES if int(counts.get(name, 0)) == 0]
    if empty:
        warnings.warn(
            f"empty split(s): {', '.join(empty)}. Requested ratios {ratios} could not be "
            f"honoured because the data has only {n_distinct} distinct timestamp(s). "
            "Splits are cut on timestamp values, so heavily tied data limits granularity.",
            UserWarning,
            stacklevel=3,
        )


def split_boundaries(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> dict[str, pd.Timestamp]:
    """The last timestamp in the train and val splits."""
    _validate_ratios(ratios)
    _validate_frame(df)
    cumulative = _cumulative_fraction(df)
    return {
        "train_end": _cut_at(cumulative, ratios[0]),
        "val_end": _cut_at(cumulative, ratios[0] + ratios[1]),
    }


def assign_splits(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> pd.Series:
    """Return a categorical split label per row, aligned to `df`'s index."""
    bounds = split_boundaries(df, ratios)
    times = df["event_time"]

    labels = np.where(
        times <= bounds["train_end"],
        "train",
        np.where(times <= bounds["val_end"], "val", "test"),
    )
    result = pd.Series(
        pd.Categorical(labels, categories=SPLIT_NAMES), index=df.index, name="split"
    )
    _warn_if_any_split_is_empty(result, ratios, df["event_time"].nunique())
    return result
