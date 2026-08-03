"""Flattened sequence window: target transaction + its previous 9, same user.

Tests whether the gap between a sequential model (~0.5 test AP reported on IBM
CCF) and a plain tabular one (0.041 in our ablation) is architecture rather than
features. If concatenating the recent history closes it, the missing ingredient
was sequence context; if not, the difference lies elsewhere.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from fraud_benchmark.experiments.metrics import best_f1_threshold, score
from fraud_benchmark.experiments.seq_window import N_LAGS, build
from fraud_benchmark.experiments.splits import italy_holdout_split, standard_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["standard", "italy_holdout"], default="standard")
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/seq_window.jsonl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df = standard_split(df) if args.split == "standard" else italy_holdout_split(df)

    x, names, y, split = build(df)
    del df                      # the 60-column source frame is no longer needed
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"split={args.split}  X={x.shape}  ({x.nbytes/2**30:.1f} GiB)", flush=True)
    for s, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {s:5s} rows={int(m.sum()):>10,} frauds={int(y[m].sum()):>6,} "
              f"base={y[m].mean()*100:.4f}%", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gains = []
    for seed in (0, 1, 2):
        started = time.monotonic()
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=4,
            eval_metric="aucpr", random_state=seed,
            scale_pos_weight=float((y[tr] == 0).sum() / max(y[tr].sum(), 1)),
        )
        # numpy, not a DataFrame: wrapping a 15 GiB float32 block would copy it.
        # The booster then keys importances as f0, f1, ... which we map back below.
        model.fit(x[tr], y[tr])
        pv = model.predict_proba(x[va])[:, 1]
        pt = model.predict_proba(x[te])[:, 1]
        th = best_f1_threshold(y[va], pv)
        rec = {
            "experiment": f"seq_window_{args.split}",
            "n_lags": N_LAGS, "seed": seed, "n_features": len(names),
            "fit_seconds": round(time.monotonic() - started, 1),
            "scores": {"val": score(y[va], pv, th), "test": score(y[te], pt, th)},
            "importance_gain": {
                names[int(k[1:])]: v
                for k, v in model.get_booster()
                .get_score(importance_type="total_gain").items()
            },
        }
        gains.append(rec["importance_gain"])
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed={seed} val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    total: dict[str, float] = {}
    for g in gains:
        for k, v in g.items():
            total[k] = total.get(k, 0.0) + v / len(gains)
    share = sum(total.values()) or 1.0
    print("\ntop 20 by mean total gain:", flush=True)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:34s} {v/share*100:6.2f}%", flush=True)
    lag_share = sum(v for k, v in total.items() if k.startswith("t-")) / share * 100
    print(f"\nshare of gain from lagged (t-1..t-9) features: {lag_share:.1f}%", flush=True)


if __name__ == "__main__":
    main()
