"""Temporal train/validation/test splitting.

Cuts land on timestamp values rather than row positions, so transactions sharing an
event_time always end up in the same split. Datasets with coarse time resolution
(PaySim's hours, BankSim's days) have heavy ties, and slicing through a tied block
would leak same-instant information across the boundary.
"""

from __future__ import annotations

import math

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


def split_boundaries(
    df: pd.DataFrame, ratios: tuple[float, float, float]
) -> dict[str, pd.Timestamp]:
    """The last timestamp in the train and val splits."""
    _validate_ratios(ratios)
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
    return pd.Series(
        pd.Categorical(labels, categories=SPLIT_NAMES), index=df.index, name="split"
    )
