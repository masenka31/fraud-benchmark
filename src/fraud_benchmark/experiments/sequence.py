"""Label-free, bounded transaction histories for the LSTM protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 30


def _entity_order(entity: pd.Series, event_time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Entity-major order, preserving input order for equal timestamps."""
    codes, _ = pd.factorize(entity, sort=False)
    if (codes < 0).any():
        raise ValueError('entity_id must not be missing')
    times = pd.to_datetime(event_time).to_numpy(dtype='datetime64[ns]')
    if np.isnat(times).any():
        raise ValueError('event_time must not be missing')
    order = np.lexsort((np.arange(len(codes)), times, codes)).astype('int32')
    return order, codes[order]


@dataclass(frozen=True)
class Chunks:
    """Disjoint, complete IBM chunks in source-row coordinates."""

    rows: np.ndarray
    n_dropped: int

    @property
    def targets(self) -> np.ndarray:
        return self.rows[:, -1]


def complete_chunks(entity: pd.Series, event_time: pd.Series) -> Chunks:
    """Keep one final-row target per complete 30-transaction entity chunk."""
    order, codes = _entity_order(entity, event_time)
    boundaries = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(order)]
    n_chunks = int(sum((end - start) // WINDOW for start, end in zip(boundaries[:-1], boundaries[1:])))
    rows = np.empty((n_chunks, WINDOW), dtype='int32')
    cursor = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        count = (end - start) // WINDOW
        rows[cursor : cursor + count] = order[start : start + count * WINDOW].reshape(
            count, WINDOW
        )
        cursor += count
    return Chunks(rows=rows, n_dropped=len(order) - n_chunks * WINDOW)


def iid_chunk_split(n_chunks: int, seed: int) -> np.ndarray:
    """One deterministic 80/10/10 draw, independent of estimator seeds."""
    order = np.random.default_rng(seed).permutation(n_chunks)
    train_end = int(0.8 * n_chunks)
    val_end = int(0.9 * n_chunks)
    split = np.empty(n_chunks, dtype='int8')
    split[order[:train_end]] = 0
    split[order[train_end:val_end]] = 1
    split[order[val_end:]] = 2
    return split


@dataclass(frozen=True)
class WindowIndex:
    """Compact Sparkov history index over an entity-major event matrix."""

    order: np.ndarray
    position: np.ndarray
    entity_start: np.ndarray
    tie_start: np.ndarray

    def batch(self, source_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Right-padded rows and lengths, excluding same-timestamp predecessors."""
        pos = self.position[np.asarray(source_rows, dtype='int32')]
        prior_end = self.tie_start[pos]
        count = np.minimum(prior_end - self.entity_start[pos], WINDOW - 1)
        start = prior_end - count
        offsets = np.arange(WINDOW, dtype='int32')
        valid_prior = offsets[None, :] < count[:, None]
        positions = np.where(valid_prior, start[:, None] + offsets[None, :], 0)
        positions[np.arange(len(pos)), count] = pos
        return positions.astype('int32'), count + 1


def window_index(entity: pd.Series, event_time: pd.Series) -> WindowIndex:
    """Index strict-past, same-entity windows without storing their contents."""
    order, codes = _entity_order(entity, event_time)
    n = len(order)
    pos = np.empty(n, dtype='int32')
    pos[order] = np.arange(n, dtype='int32')
    times = pd.to_datetime(event_time).to_numpy(dtype='datetime64[ns]')[order]
    group_start = np.r_[True, codes[1:] != codes[:-1]]
    entity_start = np.maximum.accumulate(
        np.where(group_start, np.arange(n, dtype='int32'), 0)
    )
    tie_start = np.maximum.accumulate(
        np.where(
            np.r_[True, (codes[1:] != codes[:-1]) | (times[1:] != times[:-1])],
            np.arange(n, dtype='int32'),
            0,
        )
    )
    return WindowIndex(order, pos, entity_start, tie_start)
