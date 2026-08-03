"""A 3-layer MLP on the same flattened sequence window as seq_window.py.

Same features, same splits, same metric -- only the model differs, so the
comparison against XGBoost isolates architecture.

Known weakness, stated rather than hidden: the categorical columns arrive as
ordinal codes, which imply an ordering that does not exist (merchant state 5 is
not "more" than 4). Trees are indifferent to this; a dense layer is not.
Standardising conditions them but cannot remove the false ordering. Learned
embeddings per categorical would be the fair treatment and are the obvious next
step if this underperforms -- the gap would then be attributable to input
encoding rather than to the architecture.

Streaming is deliberate: the design matrix is 10-15 GiB, so features are
standardised chunk-wise and batches are gathered by index rather than by slicing
the array into per-split copies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fraud_benchmark.ablation.metrics import best_f1_threshold, score
from italy_holdout import build_split as italy_holdout_split
from seq_window import build, standard_split

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

    def __init__(self, n_continuous: int, cardinalities: list[int],
                 hidden: tuple[int, int] = (256, 128), dropout: float = 0.2) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        width = n_continuous + sum(cardinalities)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(width, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, continuous: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        parts = [continuous]
        for i, n_levels in enumerate(self.cardinalities):
            parts.append(
                nn.functional.one_hot(codes[:, i].long().clamp(0, n_levels - 1),
                                      num_classes=n_levels).float()
            )
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


class MLP(nn.Module):
    """161 -> 256 -> 128 -> 1. Three Linear layers."""

    def __init__(self, n_features: int, hidden: tuple[int, int] = (256, 128),
                 dropout: float = 0.2) -> None:
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


def train_statistics(x: np.ndarray, rows: np.ndarray,
                     cols: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over `rows` only, accumulated in chunks.

    Fitting on anything but train would leak; materialising x[rows] would copy
    12 GiB, so the sums are accumulated a million rows at a time.
    """
    cols = np.arange(x.shape[1]) if cols is None else cols
    total = np.zeros(len(cols), dtype="float64")
    total_sq = np.zeros(len(cols), dtype="float64")
    for start in range(0, len(rows), CHUNK):
        block = x[np.ix_(rows[start : start + CHUNK], cols)].astype("float64")
        total += block.sum(axis=0)
        total_sq += (block ** 2).sum(axis=0)
    n = len(rows)
    mean = total / n
    var = np.maximum(total_sq / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0          # a constant column contributes nothing
    return mean.astype("float32"), std.astype("float32")


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, rows: np.ndarray, mean: np.ndarray,
            std: np.ndarray, cont_cols: np.ndarray,
            cat_cols: np.ndarray | None) -> np.ndarray:
    model.eval()
    out = np.empty(len(rows), dtype="float32")
    for start in range(0, len(rows), EVAL_BATCH):
        idx = rows[start : start + EVAL_BATCH]
        cont = (x[np.ix_(idx, cont_cols)] - mean) / std
        if cat_cols is None:
            logits = model(torch.from_numpy(cont))
        else:
            logits = model(torch.from_numpy(cont),
                           torch.from_numpy(x[np.ix_(idx, cat_cols)]))
        out[start : start + len(idx)] = torch.sigmoid(logits).numpy()
    return out


def run_seed(x, y, tr_rows, va_rows, te_rows, mean, std, seed: int,
             cont_cols=None, cat_cols=None, cardinalities=None) -> dict:
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

    best = {"val_ap": -1.0, "epoch": -1, "state": None}
    started = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(tr_rows))
        running, batches = 0.0, 0
        for start in range(0, len(order), BATCH):
            idx = tr_rows[order[start : start + BATCH]]
            if len(idx) < 2:                     # BatchNorm needs >1 row
                continue
            cont = (x[np.ix_(idx, cont_cols)] - mean) / std
            target = torch.from_numpy(y[idx].astype("float32"))
            opt.zero_grad(set_to_none=True)
            if cat_cols is None:
                logits = model(torch.from_numpy(cont))
            else:
                logits = model(torch.from_numpy(cont),
                               torch.from_numpy(x[np.ix_(idx, cat_cols)]))
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            batches += 1
        val_scores = predict(model, x, va_rows, mean, std, cont_cols, cat_cols)
        from sklearn.metrics import average_precision_score
        val_ap = float(average_precision_score(y[va_rows], val_scores))
        print(f"  seed={seed} epoch={epoch:02d} loss={running/max(batches,1):.4f} "
              f"val_AP={val_ap:.4f}", flush=True)
        if val_ap > best["val_ap"]:
            best = {"val_ap": val_ap, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= PATIENCE:
            print(f"  seed={seed} early stop (no val gain in {PATIENCE} epochs)", flush=True)
            break

    model.load_state_dict(best["state"])
    val_scores = predict(model, x, va_rows, mean, std, cont_cols, cat_cols)
    test_scores = predict(model, x, te_rows, mean, std, cont_cols, cat_cols)
    threshold = best_f1_threshold(y[va_rows], val_scores)
    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "fit_seconds": round(time.monotonic() - started, 1),
        "scores": {
            "val": score(y[va_rows], val_scores, threshold),
            "test": score(y[te_rows], test_scores, threshold),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["standard", "italy_holdout"], default="standard")
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/seq_mlp.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--encoding", choices=["ordinal", "onehot"], default="onehot",
                    help="onehot expands categorical codes per batch inside the model")
    args = ap.parse_args()

    torch.set_num_threads(4)
    df = pd.read_parquet(args.features)
    df = standard_split(df) if args.split == "standard" else italy_holdout_split(df)
    x, names, y, split = build(df)
    del df

    tr_rows = np.flatnonzero(split == "train")
    va_rows = np.flatnonzero(split == "val")
    te_rows = np.flatnonzero(split == "test")
    print(f"split={args.split}  X={x.shape} ({x.nbytes/2**30:.1f} GiB)  "
          f"train={len(tr_rows):,} val={len(va_rows):,} test={len(te_rows):,}", flush=True)

    # Categoricals are every column whose name carries the cat_ prefix. The
    # binary flags (same_state, merchant_novelty, ...) stay continuous -- a
    # one-hot of a 0/1 column is just that column plus its complement.
    if args.encoding == "onehot":
        cat_cols = np.array([i for i, n in enumerate(names) if "cat_" in n])
        cont_cols = np.array([i for i in range(len(names)) if i not in set(cat_cols)])
        cardinalities = [int(x[:, i].max()) + 1 for i in cat_cols]
        print(f"one-hot: {len(cat_cols)} categorical columns -> "
              f"{sum(cardinalities):,} indicator columns; "
              f"{len(cont_cols)} continuous; model input width "
              f"{len(cont_cols) + sum(cardinalities):,}", flush=True)
    else:
        cat_cols, cont_cols, cardinalities = None, np.arange(len(names)), None
        print(f"ordinal: all {len(names)} columns standardised", flush=True)

    mean, std = train_statistics(x, tr_rows, cont_cols)
    print("standardisation fitted on train rows only", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        rec = run_seed(x, y, tr_rows, va_rows, te_rows, mean, std, seed,
                       cont_cols, cat_cols, cardinalities)
        rec.update(experiment=f"seq_mlp_{args.split}_{args.encoding}",
                   model="mlp_3layer", encoding=args.encoding,
                   n_features=len(names),
                   input_width=len(cont_cols) + (sum(cardinalities) if cardinalities else 0))
        results.append(rec)
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed={seed} best_epoch={rec['best_epoch']} "
              f"val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    v = [r["scores"]["val"]["average_precision"] for r in results]
    t = [r["scores"]["test"]["average_precision"] for r in results]
    print(f"\nMLP {args.split} ({args.encoding}): val AP {np.mean(v):.4f}+-{np.std(v):.4f}  "
          f"test AP {np.mean(t):.4f}+-{np.std(t):.4f}", flush=True)


if __name__ == "__main__":
    main()
