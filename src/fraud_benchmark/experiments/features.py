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
