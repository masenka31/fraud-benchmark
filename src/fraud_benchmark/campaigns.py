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
    """Return a campaign id per row, null for non-fraud rows.

    Ids are integers, dense from 0, in no meaningful order. The result is aligned
    to `df`'s index.
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

    # A new campaign starts at each entity change, or when the wait since the
    # previous fraud on the same entity exceeds the gap.
    #
    # Two pandas-3 traps here, both reproduced on real data before writing this:
    #  * `entity_id` is `string` dtype, so `entity.ne(entity.shift())` yields <NA>
    #    on the first row rather than True. Hence `.fillna(True)`.
    #  * The resulting mask is `bool[pyarrow]`, and `.cumsum()` on that raises
    #    `ArrowNotImplementedError`. Hence the conversion to a numpy bool array.
    starts = (entity.ne(entity.shift()) | elapsed.gt(gap)).fillna(True)
    numbering = np.cumsum(starts.to_numpy(dtype="bool")) - 1
    ids.loc[ordered.index] = pd.array(numbering, dtype="Int64")
    return ids


def campaign_sizes(ids: pd.Series) -> pd.Series:
    """Rows per campaign, for reporting. Ignores nulls."""
    return ids.dropna().value_counts()
