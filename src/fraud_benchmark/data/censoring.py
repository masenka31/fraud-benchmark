"""Consuming `reported_at`: the labels a model training at a cutoff would have.

Lives beside `delay.py` rather than with the models. The delay stage writes
`reported_at`; this is the only correct way to read it back, and an experiment that
reimplemented it slightly differently would silently measure something else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def censored_labels(df: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """0/1 labels as known at `cutoff`, one per row of `df`.

    Requires `is_fraud` and `reported_at`. A fraud reported after `cutoff` is
    labelled 0 rather than dropped: at that moment it is indistinguishable from a
    legitimate transaction, and dropping it would presume knowledge of which rows
    to distrust.
    """
    reported = pd.to_datetime(df['reported_at'])
    known_fraud = df['is_fraud'].astype(bool) & (reported <= cutoff)
    return known_fraud.to_numpy().astype(int)
