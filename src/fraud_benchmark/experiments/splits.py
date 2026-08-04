"""Splits derived from an already-prepared dataset.

Neither is the pipeline's split. `standard_split` recomputes the temporal 80/10/10
so an experiment can run on the feature cache, which carries `split` from a
possibly older preparation; `italy_holdout_split` deliberately places its boundary
to isolate the IBM CCF regime shift the audit found.

Both are strictly temporal: no row is moved across time, only the boundaries are
placed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The first IBM CCF fraud carrying `Merchant State == 'Italy'`. The train half of
#: the Italy holdout ends one second before it, so it contains zero Italy frauds by
#: construction.
FIRST_ITALY_FRAUD = pd.Timestamp("2017-11-19 12:06:00")

#: IBM CCF's last labelled fraud. Rows after it are dropped from the holdout.
LAST_LABELLED_FRAUD = pd.Timestamp("2019-10-27 14:54:00")

#: The left crop that makes the Italy holdout exactly 80/10/10 against its tail.
TRAIN_ROWS = 13_350_884


def standard_split(df: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's own temporal 80/10/10, matching the reported ~0.5 baseline."""
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    t = df["event_time"]
    val_start, test_start = t.quantile(0.8), t.quantile(0.9)
    df["split"] = np.where(t < val_start, "train", np.where(t < test_start, "val", "test"))
    return df


def italy_holdout_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal split whose train half predates the first Italy fraud."""
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    cut = FIRST_ITALY_FRAUD - pd.Timedelta(seconds=1)
    head = df[df["event_time"] < cut]
    tail = df[(df["event_time"] >= cut) & (df["event_time"] <= LAST_LABELLED_FRAUD)]

    train = head.iloc[-TRAIN_ROWS:]
    half = len(tail) // 2
    val, test = tail.iloc[:half], tail.iloc[half:]

    out = pd.concat([train, val, test], ignore_index=True)
    out["split"] = ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
    return out
