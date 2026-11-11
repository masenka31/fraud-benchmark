"""Chronological, label-free counts over earlier training rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_count(values: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    """Count earlier training occurrences for every temporally ordered row."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(values) != len(train_mask):
        raise ValueError('values and train_mask must have equal length')
    codes, _ = pd.factorize(values.astype('string').fillna('__null__'), sort=False)
    updates = pd.Series(train_mask.astype('int64'))
    cumulative = updates.groupby(codes, sort=False).cumsum().to_numpy()
    return cumulative - train_mask.astype('int64')
