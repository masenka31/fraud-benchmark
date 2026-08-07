"""Splits derived from an already-prepared dataset.

Neither is the pipeline's split. `standard_split` recomputes the temporal 80/10/10
so an experiment can run on the feature cache, which carries `split` from a
possibly older preparation; `italy_holdout_split` deliberately places its boundary
to isolate the IBM CCF regime shift the audit found.

The standard and Italy-holdout splits are strictly temporal.  The two IID splits
exist for the IBM CCF temporal-vs-random evaluation only.  Their assignment is
deterministic and deliberately separate from model seeds, so five fitted models
see exactly the same rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The first IBM CCF fraud carrying `Merchant State == 'Italy'`. The train half of
#: the Italy holdout ends one second before it, so it contains zero Italy frauds by
#: construction.
FIRST_ITALY_FRAUD = pd.Timestamp('2017-11-19 12:06:00')

#: IBM CCF's last labelled fraud. Rows after it are dropped from the holdout.
LAST_LABELLED_FRAUD = pd.Timestamp('2019-10-27 14:54:00')

#: The left crop that makes the Italy holdout exactly 80/10/10 against its tail.
TRAIN_ROWS = 13_350_884

#: One fixed split draw, independent of the estimator seeds.  Changing this is a
#: change to the experimental cell, not another model replicate.
IID_SPLIT_SEED = 20260806

_SPLIT_NAMES = np.asarray(('train', 'val', 'test'))
_SPLIT_RATIOS = np.asarray((0.8, 0.1, 0.1), dtype='float64')


def standard_split(df: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's own temporal 80/10/10, matching the reported ~0.5 baseline."""
    df = df.sort_values('event_time', kind='mergesort').reset_index(drop=True)
    t = df['event_time']
    val_start, test_start = t.quantile(0.8), t.quantile(0.9)
    df['split'] = np.where(t < val_start, 'train', np.where(t < test_start, 'val', 'test'))
    return df


def italy_holdout_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal split whose train half predates the first Italy fraud."""
    df = df.sort_values('event_time', kind='mergesort').reset_index(drop=True)
    cut = FIRST_ITALY_FRAUD - pd.Timedelta(seconds=1)
    head = df[df['event_time'] < cut]
    tail = df[(df['event_time'] >= cut) & (df['event_time'] <= LAST_LABELLED_FRAUD)]

    train = head.iloc[-TRAIN_ROWS:]
    half = len(tail) // 2
    val, test = tail.iloc[:half], tail.iloc[half:]

    out = pd.concat([train, val, test], ignore_index=True)
    out['split'] = ['train'] * len(train) + ['val'] * len(val) + ['test'] * len(test)
    return out


def pre_italy_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal 80/10/10 split over rows strictly before the first Italy fraud."""
    out = _pre_italy_rows(df)
    train_end = int(0.8 * len(out))
    val_end = int(0.9 * len(out))
    out['split'] = np.where(
        out.index < train_end,
        'train',
        np.where(out.index < val_end, 'val', 'test'),
    )
    return out


def _pre_italy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the shared chronological population for the pre-Italy paper study."""
    out = df.sort_values('event_time', kind='mergesort').reset_index(drop=True)
    return out.loc[out['event_time'] < FIRST_ITALY_FRAUD].reset_index(drop=True)


def iid_row_split(df: pd.DataFrame, seed: int = IID_SPLIT_SEED) -> pd.DataFrame:
    """Deterministic exact 80/10/10 assignment of individual transaction rows.

    The frame is first put in a stable canonical order, making the split invariant
    to the parquet's current row order.  Rows are returned in temporal order because
    history construction and reporting remain easier to audit that way.
    """
    out = df.sort_values(['event_time', 'entity_id'], kind='mergesort').reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(out))
    train_end = int(0.8 * len(out))
    val_end = int(0.9 * len(out))
    labels = np.empty(len(out), dtype=object)
    labels[order[:train_end]] = 'train'
    labels[order[train_end:val_end]] = 'val'
    labels[order[val_end:]] = 'test'
    out['split'] = labels
    return out


def iid_customer_split(df: pd.DataFrame, seed: int = IID_SPLIT_SEED) -> pd.DataFrame:
    """Deterministic customer-disjoint split balanced on transaction counts.

    Customers are assigned whole.  Largest histories are placed first into the
    split that yields the smallest squared deviation from the 80/10/10 transaction
    targets.  A seeded random key resolves equal-size customers and equivalent
    destinations, retaining a random IID allocation while preventing one unusually
    long history from dominating a small split.
    """
    out = df.sort_values(['event_time', 'entity_id'], kind='mergesort').reset_index(drop=True)
    sizes = out.groupby('entity_id', observed=True, sort=False).size()
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame(
        {
            'entity_id': sizes.index,
            'rows': sizes.to_numpy(dtype='int64'),
            'tie': rng.random(len(sizes)),
        }
    ).sort_values(['rows', 'tie'], ascending=[False, True], kind='mergesort')

    targets = _SPLIT_RATIOS * len(out)
    customer_targets = np.asarray(
        (int(0.8 * len(customers)), int(0.1 * len(customers)), 0), dtype='int64'
    )
    customer_targets[2] = len(customers) - customer_targets[:2].sum()
    assigned = np.zeros(3, dtype='int64')
    assigned_customers = np.zeros(3, dtype='int64')
    customer_ids = customers['entity_id'].to_numpy()
    customer_rows = customers['rows'].to_numpy(dtype='int64')
    destinations = np.empty(len(customers), dtype='int8')
    split_ties = rng.random((len(customers), 3))
    for position, customer in enumerate(customers.itertuples(index=False)):
        candidates = assigned[None, :] + np.eye(3, dtype='int64') * customer.rows
        error = (((candidates - targets) / targets) ** 2).sum(axis=1)
        error[assigned_customers >= customer_targets] = np.inf
        chosen = int(np.lexsort((split_ties[position], error))[0])
        assigned[chosen] += customer.rows
        assigned_customers[chosen] += 1
        destinations[position] = chosen

    # Preserve the exact customer counts while swapping pairs that improve the
    # transaction ratios.  The largest-first pass gets close; this local repair
    # removes avoidable drift caused by an early irrevocable placement.
    for _ in range(200):
        current_error = float((((assigned - targets) / targets) ** 2).sum())
        best: tuple[float, int, int, int, int] | None = None
        for left in range(3):
            left_index = np.flatnonzero(destinations == left)
            for right in range(left + 1, 3):
                right_index = np.flatnonzero(destinations == right)
                delta = customer_rows[right_index][None, :] - customer_rows[left_index][:, None]
                errors = (
                    ((assigned[left] + delta - targets[left]) / targets[left]) ** 2
                    + ((assigned[right] - delta - targets[right]) / targets[right]) ** 2
                )
                errors += current_error - (
                    ((assigned[left] - targets[left]) / targets[left]) ** 2
                    + ((assigned[right] - targets[right]) / targets[right]) ** 2
                )
                flat = int(np.argmin(errors))
                candidate = float(errors.flat[flat])
                if best is None or candidate < best[0]:
                    i, j = np.unravel_index(flat, errors.shape)
                    best = (candidate, left, right, int(left_index[i]), int(right_index[j]))
        if best is None or best[0] >= current_error - 1e-15:
            break
        _, left, right, left_customer, right_customer = best
        change = customer_rows[right_customer] - customer_rows[left_customer]
        assigned[left] += change
        assigned[right] -= change
        destinations[left_customer], destinations[right_customer] = right, left

    customer_split = {
        customer_id: str(_SPLIT_NAMES[destination])
        for customer_id, destination in zip(customer_ids, destinations)
    }

    out['split'] = out['entity_id'].map(customer_split)
    return out


def pre_italy_iid_row_split(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction-IID split over the shared pre-Italy paper population."""
    return iid_row_split(_pre_italy_rows(df))


def pre_italy_iid_customer_split(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-IID split over the shared pre-Italy paper population."""
    return iid_customer_split(_pre_italy_rows(df))
