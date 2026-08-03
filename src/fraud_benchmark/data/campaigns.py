"""Grouping fraudulent transactions into campaigns.

A campaign is a run of frauds on one entity, each within `gap` of the previous.
Real investigations discover these together, so the label-delay stage gives every
transaction in a campaign the same reported_at.

Grouping by entity alone would be wrong. Measured on the real datasets: Amaretto's
81,268 anomalies belong to 21 clients and would form 21 campaigns of up to 36,673
rows, and IBM CCF cards carry frauds decades apart. The time gap is what keeps
those apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Name of the column this module produces.
CAMPAIGN_COLUMN = "campaign_id"


def assign_campaigns(df: pd.DataFrame, gap: pd.Timedelta) -> pd.Series:
    """Campaign id per fraud row, <NA> elsewhere, aligned to `df`'s index.

    Requires `is_fraud`, `entity_id` and `event_time`. Consecutive frauds on one
    entity share a campaign while at most `gap` apart. Ids are Int64, dense from
    0, in no meaningful order.

    Raises ValueError if `gap` is negative.
    """
    if gap < pd.Timedelta(0):
        raise ValueError(f"gap must not be negative, got {gap}")

    ids = pd.Series(pd.NA, index=df.index, dtype="Int64", name=CAMPAIGN_COLUMN)
    frauds = df.loc[df["is_fraud"], ["entity_id", "event_time"]]
    if frauds.empty:
        return ids

    ordered = frauds.sort_values(["entity_id", "event_time"], kind="stable")
    entity = ordered["entity_id"]
    elapsed = ordered["event_time"].diff()

    # Two pandas-3 traps, both reproduced on real data: comparing `string` dtype
    # yields <NA> on the first row rather than True, hence `.fillna(True)`; and
    # `.cumsum()` on the resulting `bool[pyarrow]` mask raises TypeError, hence
    # the numpy round-trip.
    is_campaign_start = (entity.ne(entity.shift()) | elapsed.gt(gap)).fillna(True)
    campaign_number = np.cumsum(is_campaign_start.to_numpy(dtype="bool")) - 1
    ids.loc[ordered.index] = pd.array(campaign_number, dtype="Int64")
    return ids


def campaign_sizes(ids: pd.Series) -> pd.Series:
    """Row count per campaign id, indexed by id. Nulls are ignored."""
    return ids.dropna().value_counts()
