"""Dataset transformations that break IBM CCF's geography oracle.

`geo_diluted` is the literal question: replace Italy with some other location in
95% of Italy rows. Doing it to `Merchant State` alone would measure nothing --
the audit found `Merchant City` mirrors the country exactly (Rome, Algiers, Port
au Prince), `Zip` is flagged in its own right, and a merchant id encodes its own
location. So the whole merchant identity is reassigned jointly, sampled from real
non-Italy rows in proportion to their volume, which scatters the diluted frauds
into common locations rather than concentrating them somewhere new.

`no_geo_keep_mcc` isolates MCC cleanly by dropping every location and identity
column and keeping MCC plus everything else. The audit measured MCC as an
independent artifact (MCC 5732: 843 frauds at 6.7%, none in Italy), so this asks
how much that is worth on its own.

Both are dataset transformations, not fitting steps, so they are applied
identically to train, val and test before anything is fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_benchmark.experiments.ablation.columns import (
    ABSOLUTE_TIME_COLUMNS,
    ALWAYS_EXCLUDED,
)


# The columns that jointly carry "where and who", all reassigned together.
IDENTITY = ["Merchant State", "Merchant City", "Zip", "Merchant Name"]
DILUTE_FRACTION = 0.95


def dilute_geography(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Reassign the merchant identity of 95% of Italy rows to a real non-Italy one."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    is_italy = (out["Merchant State"].astype("string") == "Italy").fillna(False)
    donors = out.index[~is_italy]
    targets = out.index[is_italy]
    chosen = targets[rng.random(len(targets)) < DILUTE_FRACTION]

    # Sampling donors uniformly from actual rows reproduces the real location
    # mix, so diluted frauds land mostly in high-volume places.
    picks = rng.choice(donors, size=len(chosen), replace=True)
    for col in IDENTITY:
        out.loc[chosen, col] = out.loc[picks, col].to_numpy()
    return out


def matrix_columns(df: pd.DataFrame, drop: list[str]) -> list[str]:
    banned = set(ALWAYS_EXCLUDED) | set(ABSOLUTE_TIME_COLUMNS) | set(drop)
    cols = [c for c in df.columns if c not in banned]
    return [c for c in cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
