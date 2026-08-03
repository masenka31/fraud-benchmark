"""MLP with one-hot categoricals on the sequence window PLUS the v2 features.

The two things that moved the needle so far, combined:

  * one-hot encoding, which took a 3-layer MLP from 0.054 to 0.180 test AP with
    no other change,
  * the 26 history-relative v2 features (context novelty, decline velocity,
    entity-baseline deviation), whose value on trees is still being measured.

Feature set = seq_window's 161 (target + 9 lagged transactions) + 26 v2 = 187.
Directly comparable to the 0.180 run: same model, same encoding, same split,
26 extra columns. So the difference isolates what the v2 features add on top of
the best configuration found so far.

Ordering matters here. seq_window.build sorts by (User, event_time) internally
and returns arrays in that order, so the v2 block is computed on a frame sorted
the same way before being concatenated -- otherwise the two halves of every row
would describe different transactions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_benchmark.experiments.features_v2 import V2_COLUMNS, add_v2_features
from fraud_benchmark.experiments.splits import italy_holdout_split, standard_split
from seq_mlp import run_seed, train_statistics
from seq_window import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["standard", "italy_holdout"], default="standard")
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/seq_mlp_v2.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    torch.set_num_threads(4)
    df = pd.read_parquet(args.features)
    df = standard_split(df) if args.split == "standard" else italy_holdout_split(df)

    # Sort once, here, so both feature blocks are built on the same row order.
    # build() sorts by the same keys, which is then a no-op.
    order = np.lexsort((df["event_time"].to_numpy(), df["User"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)

    x_seq, names, y, split = build(df)
    enriched = add_v2_features(
        df, merchant_col="Merchant Name", state_col="Merchant State",
        city_col="Merchant City", mcc_col="MCC", channel_col="Use Chip",
        error_col="Errors?", amount_col="Amount",
        acct_open_col="Acct Open Date", pin_year_col="Year PIN last Changed",
        expires_col="Expires",
    )
    v2 = enriched[list(V2_COLUMNS)].astype("float32").fillna(-1.0).to_numpy()
    del df, enriched

    x = np.concatenate([x_seq, v2], axis=1)
    names = list(names) + [f"v2_{c}" for c in V2_COLUMNS]
    del x_seq, v2
    assert x.shape[1] == len(names)

    tr_rows = np.flatnonzero(split == "train")
    va_rows = np.flatnonzero(split == "val")
    te_rows = np.flatnonzero(split == "test")
    print(f"split={args.split}  X={x.shape} ({x.nbytes / 2**30:.1f} GiB)  "
          f"train={len(tr_rows):,} val={len(va_rows):,} test={len(te_rows):,}", flush=True)

    # The v2 block is all numeric, so the categorical set is unchanged from the
    # sequence window: the cat_ columns at each of the 10 positions.
    cat_cols = np.array([i for i, n in enumerate(names) if "cat_" in n])
    cont_cols = np.array([i for i in range(len(names)) if i not in set(cat_cols)])
    cardinalities = [int(x[:, i].max()) + 1 for i in cat_cols]
    print(f"one-hot: {len(cat_cols)} categorical -> {sum(cardinalities):,} indicators; "
          f"{len(cont_cols)} continuous; input width "
          f"{len(cont_cols) + sum(cardinalities):,}", flush=True)

    mean, std = train_statistics(x, tr_rows, cont_cols)
    print("standardisation fitted on train rows only", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        rec = run_seed(x, y, tr_rows, va_rows, te_rows, mean, std, seed,
                       cont_cols, cat_cols, cardinalities)
        rec.update(experiment=f"seq_mlp_v2_{args.split}", model="mlp_3layer",
                   encoding="onehot", n_features=len(names),
                   input_width=len(cont_cols) + sum(cardinalities))
        results.append(rec)
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed={seed} best_epoch={rec['best_epoch']} "
              f"val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    v = [r["scores"]["val"]["average_precision"] for r in results]
    t = [r["scores"]["test"]["average_precision"] for r in results]
    print(f"\nMLP onehot + v2, {args.split}: val AP {np.mean(v):.4f}+-{np.std(v):.4f}  "
          f"test AP {np.mean(t):.4f}+-{np.std(t):.4f}", flush=True)
    print("reference (same model/encoding, without v2): test AP 0.1804+-0.0274", flush=True)


if __name__ == "__main__":
    main()
