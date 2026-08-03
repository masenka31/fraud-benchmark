"""IBM CCF with every Italy fraud held out of training.

The question: can a model trained only on non-Italy fraud detect the Italy fraud
that makes up almost all of val and test? This isolates the regime shift the
audit found, deliberately rather than as an accident of where the temporal cut
happened to land.

Split. Train ends one second before the first Italy fraud (2017-11-19 12:06), so
it contains zero of them by construction. The left crop is then chosen to make
the ratio exactly 80/10/10: 2009-09-12 onward gives 13,350,884 train rows against
a 3,337,721-row tail, halved into val and test. Still strictly temporal -- no row
is moved across time to achieve this, only the boundaries are placed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from fraud_benchmark.experiments.ibm_features import build_features
from fraud_benchmark.experiments.metrics import best_f1_threshold, score
from fraud_benchmark.experiments.splits import italy_holdout_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("results/italy_holdout.jsonl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.features_dir / "ibm_ccf.parquet")
    df = italy_holdout_split(df)

    italy = (df["Merchant State"].astype("string") == "Italy").fillna(False)
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        fr = m & df["is_fraud"]
        print(f"{sp:5s} rows={int(m.sum()):>10,} frauds={int(fr.sum()):>6,} "
              f"italy_frauds={int((fr & italy).sum()):>6,} "
              f"span {df.loc[m,'event_time'].min().date()}..{df.loc[m,'event_time'].max().date()}",
              flush=True)
    assert int((( df['split']=='train') & df['is_fraud'] & italy).sum()) == 0, \
        "train must contain no Italy frauds"

    x, names = build_features(df)
    y = df["is_fraud"].to_numpy().astype(int)
    tr, va, te = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy(), (df["split"] == "test").to_numpy()
    print(f"\n{len(names)} features; train positives {int(y[tr].sum()):,}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gains: list[dict] = []
    for seed in (0, 1, 2):
        started = time.monotonic()
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=4, eval_metric="aucpr", random_state=seed,
            scale_pos_weight=float((y[tr] == 0).sum() / max(y[tr].sum(), 1)),
        )
        model.fit(x[tr], y[tr])
        pv, pt = model.predict_proba(x[va])[:, 1], model.predict_proba(x[te])[:, 1]
        th = best_f1_threshold(y[va], pv)
        rec = {
            "experiment": "italy_holdout",
            "seed": seed,
            "n_features": len(names),
            "fit_seconds": round(time.monotonic() - started, 1),
            "scores": {"val": score(y[va], pv, th), "test": score(y[te], pt, th)},
            # total_gain is keyed by the DataFrame's column names; features the
            # trees never split on are simply absent.
            "importance_gain": dict(
                model.get_booster().get_score(importance_type="total_gain")
            ),
        }
        gains.append(rec["importance_gain"])
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed={seed} val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    total = {}
    for g in gains:
        for k, v in g.items():
            total[k] = total.get(k, 0.0) + v / len(gains)
    share = sum(total.values()) or 1.0
    print("\ntop 20 features by mean total gain:", flush=True)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:28s} {v/share*100:6.2f}%", flush=True)


if __name__ == "__main__":
    main()
