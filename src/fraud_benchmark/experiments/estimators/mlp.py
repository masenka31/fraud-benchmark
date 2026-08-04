"""A 3-layer MLP, with the categoricals one-hot expanded per batch.

Known weakness, stated rather than hidden: the pipeline hands over ordinal codes,
which imply an ordering that does not exist (merchant state 5 is not "more" than 4).
Trees are indifferent; a dense layer is not. `OneHotMLP` removes the false ordering
by expanding the codes, which is why it is the default here -- the retired
experiments measured one-hot beating ordinal codes by 3.4x on IBM CCF (0.1804 against
0.0537), the largest single effect anyone found in this project. `MLP` keeps the
ordinal path for that comparison and nothing else. Learned embeddings per categorical
are the untried next step.

Expansion happens per batch rather than in the matrix: IBM CCF's one-hot width was
1,425, which dense over 24.4M rows is 129 GiB, and per 8192-row batch is 45 MiB.
Mathematically identical to feeding a one-hot matrix.

Streaming is deliberate throughout. The design matrix reaches tens of GiB, so
standardisation is computed chunk-wise and batches are gathered by index rather than
by slicing per-split copies out of the array.

NaN needs a value here, as it does for any dense layer: a feature parquet uses it to
mean "this history does not exist", and it is replaced with the train mean, which is
the least-committal choice and is fitted on train like the standardisation.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from fraud_benchmark.experiments.metrics import best_f1_threshold
from fraud_benchmark.experiments.metrics import score

BATCH = 8192
EVAL_BATCH = 65536
MAX_EPOCHS = 15
PATIENCE = 3
CHUNK = 1_000_000


class OneHotMLP(nn.Module):
    """Same depth, but categoricals arrive as codes and are one-hot expanded here.

    Expansion happens per batch, not in the data matrix: the one-hot width is
    1,425, which dense over 24.4M rows would be 129 GiB. Per 8192-row batch it is
    45 MiB. Mathematically identical to feeding a one-hot matrix.
    """

    def __init__(
        self,
        n_continuous: int,
        cardinalities: list[int],
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        width = n_continuous + sum(cardinalities)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(width, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, continuous: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        parts = [continuous]
        for i, n_levels in enumerate(self.cardinalities):
            parts.append(
                nn.functional.one_hot(
                    codes[:, i].long().clamp(0, n_levels - 1), num_classes=n_levels
                ).float()
            )
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


class MLP(nn.Module):
    """161 -> 256 -> 128 -> 1. Three Linear layers."""

    def __init__(
        self, n_features: int, hidden: tuple[int, int] = (256, 128), dropout: float = 0.2
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_statistics(
    x: np.ndarray, rows: np.ndarray, cols: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over `rows` only, accumulated in chunks.

    Fitting on anything but train would leak; materialising x[rows] would copy
    12 GiB, so the sums are accumulated a million rows at a time.
    """
    cols = np.arange(x.shape[1]) if cols is None else cols
    total = np.zeros(len(cols), dtype='float64')
    total_sq = np.zeros(len(cols), dtype='float64')
    for start in range(0, len(rows), CHUNK):
        block = x[np.ix_(rows[start : start + CHUNK], cols)].astype('float64')
        total += block.sum(axis=0)
        total_sq += (block**2).sum(axis=0)
    n = len(rows)
    mean = total / n
    var = np.maximum(total_sq / n - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0  # a constant column contributes nothing
    return mean.astype('float32'), std.astype('float32')


@torch.no_grad()
def predict(
    model: nn.Module,
    x: np.ndarray,
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    cont_cols: np.ndarray,
    cat_cols: np.ndarray | None,
) -> np.ndarray:
    model.eval()
    out = np.empty(len(rows), dtype='float32')
    for start in range(0, len(rows), EVAL_BATCH):
        idx = rows[start : start + EVAL_BATCH]
        cont = (x[np.ix_(idx, cont_cols)] - mean) / std
        if cat_cols is None:
            logits = model(torch.from_numpy(cont))
        else:
            logits = model(torch.from_numpy(cont), torch.from_numpy(x[np.ix_(idx, cat_cols)]))
        out[start : start + len(idx)] = torch.sigmoid(logits).numpy()
    return out


def run_seed(
    x,
    y,
    tr_rows,
    va_rows,
    te_rows,
    mean,
    std,
    seed: int,
    cont_cols=None,
    cat_cols=None,
    cardinalities=None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    cont_cols = np.arange(x.shape[1]) if cont_cols is None else cont_cols
    if cat_cols is None:
        model = MLP(len(cont_cols))
    else:
        model = OneHotMLP(len(cont_cols), cardinalities)
    positive = int(y[tr_rows].sum())
    negative = len(tr_rows) - positive
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / max(positive, 1)], dtype=torch.float32)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best = {'val_ap': -1.0, 'epoch': -1, 'state': None}
    started = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(tr_rows))
        running, batches = 0.0, 0
        for start in range(0, len(order), BATCH):
            idx = tr_rows[order[start : start + BATCH]]
            if len(idx) < 2:  # BatchNorm needs >1 row
                continue
            cont = (x[np.ix_(idx, cont_cols)] - mean) / std
            target = torch.from_numpy(y[idx].astype('float32'))
            opt.zero_grad(set_to_none=True)
            if cat_cols is None:
                logits = model(torch.from_numpy(cont))
            else:
                logits = model(torch.from_numpy(cont), torch.from_numpy(x[np.ix_(idx, cat_cols)]))
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            batches += 1
        val_scores = predict(model, x, va_rows, mean, std, cont_cols, cat_cols)
        from sklearn.metrics import average_precision_score

        val_ap = float(average_precision_score(y[va_rows], val_scores))
        print(
            f'  seed={seed} epoch={epoch:02d} loss={running / max(batches, 1):.4f} '
            f'val_AP={val_ap:.4f}',
            flush=True,
        )
        if val_ap > best['val_ap']:
            best = {
                'val_ap': val_ap,
                'epoch': epoch,
                'state': {k: v.clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best['epoch'] >= PATIENCE:
            print(f'  seed={seed} early stop (no val gain in {PATIENCE} epochs)', flush=True)
            break

    model.load_state_dict(best['state'])
    val_scores = predict(model, x, va_rows, mean, std, cont_cols, cat_cols)
    test_scores = predict(model, x, te_rows, mean, std, cont_cols, cat_cols)
    threshold = best_f1_threshold(y[va_rows], val_scores)
    return {
        'seed': seed,
        'best_epoch': best['epoch'],
        'fit_seconds': round(time.monotonic() - started, 1),
        'scores': {
            'val': score(y[va_rows], val_scores, threshold),
            'test': score(y[te_rows], test_scores, threshold),
        },
    }


# --- the estimator interface ------------------------------------------------


def fill_missing(x: np.ndarray, train_rows: np.ndarray) -> int:
    """Replace NaN with the train-row mean of its column, in place. Returns the count.

    A dense layer has no NaN path, so the sentinel a feature parquet uses for "this
    history does not exist" needs a value. The train mean is the least-committal one
    available, and it is fitted on train like everything else here.

    In place, and idempotent because filling a column with its own mean does not move
    that mean: the alternative is a second copy of a matrix that reaches tens of GiB.
    Later seeds therefore see an already-filled matrix and compute the same statistics.
    """
    missing = np.isnan(x)
    if not missing.any():
        return 0
    columns = np.flatnonzero(missing.any(axis=0))
    for column in columns:
        values = x[train_rows, column]
        mean = np.nanmean(values)
        x[missing[:, column], column] = 0.0 if not np.isfinite(mean) else mean
    return int(missing.sum())


def fit_and_score(prepared, seed: int, ordinal: bool = False) -> dict:
    """Fit on train, select the epoch on val, report val and test.

    `ordinal` feeds the codes as numbers instead of one-hot expanding them. It exists
    to reproduce the measured 3.4x encoding gap, not because it is a sane default.
    """
    filled = fill_missing(prepared.x, prepared.train_rows)

    one_hot = bool(len(prepared.categorical)) and not ordinal
    if one_hot:
        mean, std = train_statistics(prepared.x, prepared.train_rows, prepared.continuous)
        record = run_seed(
            prepared.x,
            prepared.y,
            prepared.train_rows,
            prepared.val_rows,
            prepared.test_rows,
            mean,
            std,
            seed,
            cont_cols=prepared.continuous,
            cat_cols=prepared.categorical,
            cardinalities=prepared.cardinalities,
        )
    else:
        # Everything standardised together: the codes, where there are any, ride along
        # as numbers, false ordering included.
        columns = np.sort(np.concatenate([prepared.continuous, prepared.categorical])).astype(
            'int64'
        )
        mean, std = train_statistics(prepared.x, prepared.train_rows, columns)
        record = run_seed(
            prepared.x,
            prepared.y,
            prepared.train_rows,
            prepared.val_rows,
            prepared.test_rows,
            mean,
            std,
            seed,
            cont_cols=columns,
        )

    record['encoding'] = 'onehot' if one_hot else 'ordinal'
    record['onehot_width'] = int(sum(prepared.cardinalities)) if one_hot else 0
    record['filled_missing'] = filled
    return record
