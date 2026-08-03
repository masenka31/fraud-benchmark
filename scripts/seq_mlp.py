"""A 3-layer MLP on the same flattened sequence window as seq_window.py.

Same features, same splits, same metric -- only the model differs, so the
comparison against XGBoost isolates architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_benchmark.experiments.mlp import run_seed, train_statistics
from fraud_benchmark.experiments.seq_window import build
from fraud_benchmark.experiments.splits import italy_holdout_split, standard_split


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
