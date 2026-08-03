"""Consuming `reported_at`: the labels a model training at a cutoff would have.

Lives beside `delay.py` rather than with the models. The delay stage writes
`reported_at`; this is the only correct way to read it back, and an experiment that
reimplemented it slightly differently would silently measure something else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def censored_labels(df: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """The labels a model training at `cutoff` would actually have.

    A fraud not yet reported is NOT missing from the training data -- it sits in
    it looking like a legitimate transaction. So the unreported frauds are
    relabelled 0, not dropped. Dropping them would model a system that somehow
    knows which rows to distrust, which is precisely the knowledge label delay
    denies it, and would understate the harm: the damage is wrong labels, not
    fewer of them.
    """
    reported = pd.to_datetime(df["reported_at"])
    known_fraud = df["is_fraud"].astype(bool) & (reported <= cutoff)
    return known_fraud.to_numpy().astype(int)
