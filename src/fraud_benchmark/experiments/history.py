"""Concatenating an entity's previous transactions onto the target row.

The target row's features, then the same features from its previous transaction,
then the one before that, out to `n_lags`. A model with no notion of sequence gets
the history as extra columns.

Lags are per entity and strictly backwards: lag *k* of a row is that entity's *k*th
previous transaction, and rows without one carry `MISSING` rather than another
entity's data. An entity's first transaction therefore has every lag missing, which
is correct -- it has no history -- and is why `MISSING` is a sentinel a model can
learn rather than a zero that reads as a real measurement of zero.

Cost is why this is a subset rather than every column. IBM CCF's 82 features at
`n_lags=10` would be 902 columns over 24.4M rows -- 88 GB of float32 -- so each
dataset module declares a short `HISTORY_COLUMNS` naming what "what the previous
transaction looked like" means for it. `--history-columns all` overrides that when
the dataset is small enough to afford it.

An earlier version of this experiment measured lagged columns earning only 13% of a
tree's total gain, so a wide history is unlikely to be worth its memory. The
narrow default is not only a budget decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Filled in wherever a lag does not exist. Not 0.0: a gap of zero seconds and "no
#: previous transaction" are different facts, and a model given 0.0 cannot tell them
#: apart. Chosen negative and far outside every feature's range so that a tree can
#: isolate it with one threshold.
MISSING = -1.0


def lag_columns(names: list[str], n_lags: int) -> list[str]:
    """The column names `lag_matrix` produces, in the order it produces them."""
    out = list(names)
    for lag in range(1, n_lags + 1):
        out.extend(f"{name}_lag{lag}" for name in names)
    return out


def previous_positions(entity_id: pd.Series, event_time: pd.Series, n_lags: int) -> np.ndarray:
    """Row positions of each row's previous transactions, `-1` where none exists.

    Shape `(len, n_lags)`; column *k* holds the position of the (k+1)th previous
    transaction of the same entity, in the caller's row order. Separated from
    `lag_matrix` because it is the whole of the correctness argument and is worth
    testing on its own.
    """
    n = len(entity_id)
    work = pd.DataFrame(
        {
            "entity": entity_id.to_numpy(),
            "event_time": pd.to_datetime(event_time).to_numpy(),
            "_pos": np.arange(n),
        }
    )
    # Stable, so rows tied on (entity, time) keep their input order -- IBM CCF has
    # minute resolution and no seconds, which makes ties routine.
    work = work.sort_values(["entity", "event_time"], kind="mergesort")
    pos = work["_pos"].to_numpy()
    # How many earlier transactions this entity has: the k-th lag exists only where
    # this is at least k.
    depth = work.groupby("entity", observed=True).cumcount().to_numpy()

    out = np.full((n, n_lags), -1, dtype="int64")
    for lag in range(1, n_lags + 1):
        exists = depth >= lag
        source = np.roll(pos, lag)
        out[pos[exists], lag - 1] = source[exists]
    return out


def lag_matrix(
    values: np.ndarray,
    entity_id: pd.Series,
    event_time: pd.Series,
    n_lags: int,
) -> np.ndarray:
    """`values` with `n_lags` per-entity lagged copies appended.

    `values` is `(rows, features)` in the caller's row order; the result is
    `(rows, features * (n_lags + 1))`, target block first. `n_lags=0` returns
    `values` unchanged, which is the no-history condition.
    """
    values = np.asarray(values, dtype="float32")
    if n_lags == 0:
        return values

    rows, features = values.shape
    previous = previous_positions(entity_id, event_time, n_lags)

    out = np.empty((rows, features * (n_lags + 1)), dtype="float32")
    out[:, :features] = values
    for lag in range(1, n_lags + 1):
        source = previous[:, lag - 1]
        found = source >= 0
        block = out[:, lag * features : (lag + 1) * features]
        # Gather with the missing rows pointed at position 0, then overwrite them --
        # cheaper than a masked assignment over a 24M-row block.
        block[:] = values[np.where(found, source, 0)]
        block[~found] = MISSING
    return out
