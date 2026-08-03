"""Does IBM CCF survive on relative geography alone?

Drops the merchant location identity -- Merchant State, and everything derived
from it or from Merchant City -- and keeps only the relative pair `same_state`
and `same_city`, which say "away from home" without naming a place.

Run on the pipeline's own temporal 80/10/10 split, the one the sequential-model
table uses, so the numbers are comparable to it.

Note on what cannot be removed this way: `Use Chip == "Online Transaction"` is
equivalent to `Merchant City == "ONLINE"`, so the online channel survives in the
card column regardless. Channel is not location, and removing it would change a
different thing, so it stays -- but it is why the drop is not "all geography".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_benchmark.ablation.metrics import best_f1_threshold, score
from italy_holdout import build_features

# Merchant location identity, and the flags derived from it.
LOCATION = ["cat_merchant_state", "merchant_is_online", "merchant_state_missing"]
# What the experiment keeps instead.
RELATIVE = ["same_state", "same_city"]


def standard_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    t = df["event_time"]
    val_start, test_start = t.quantile(0.8), t.quantile(0.9)
    df["split"] = np.where(t < val_start, "train", np.where(t < test_start, "val", "test"))
    return df


def evaluate(x: pd.DataFrame, y, split, condition: str, out: Path) -> None:
    tr, va, te = split == "train", split == "val", split == "test"
    aps = []
    for seed in (0, 1, 2):
        started = time.monotonic()
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=4,
            eval_metric="aucpr", random_state=seed,
            scale_pos_weight=float((y[tr] == 0).sum() / max(y[tr].sum(), 1)),
        )
        model.fit(x[tr], y[tr])
        pv = model.predict_proba(x[va])[:, 1]
        pt = model.predict_proba(x[te])[:, 1]
        th = best_f1_threshold(y[va], pv)
        rec = {
            "condition": condition, "seed": seed, "n_features": x.shape[1],
            "features": list(x.columns),
            "fit_seconds": round(time.monotonic() - started, 1),
            "scores": {"val": score(y[va], pv, th), "test": score(y[te], pt, th)},
            "importance_gain": dict(
                model.get_booster().get_score(importance_type="total_gain")
            ),
        }
        aps.append(rec["scores"]["test"]["average_precision"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {condition:22s} seed={seed} "
              f"val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)
        if seed == 2:
            top = sorted(rec["importance_gain"].items(), key=lambda kv: -kv[1])[:10]
            share = sum(rec["importance_gain"].values()) or 1.0
            print(f"  top features ({condition}):", flush=True)
            for k, v in top:
                print(f"     {k:28s} {v / share * 100:6.2f}%", flush=True)
    print(f"  -> {condition}: test AP {np.mean(aps):.4f}+-{np.std(aps):.4f}\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/ibm_relative_geo.jsonl"))
    args = ap.parse_args()

    df = standard_split(pd.read_parquet(args.features))
    italy = (df["Merchant State"].astype("string") == "Italy").fillna(False)
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        fr = m & df["is_fraud"]
        print(f"{sp:5s} rows={int(m.sum()):>10,} frauds={int(fr.sum()):>6,} "
              f"italy_frauds={int((fr & italy).sum()):>6,}", flush=True)

    x_full, _ = build_features(df)
    y = df["is_fraud"].to_numpy().astype(int)
    split = df["split"].to_numpy()
    del df

    for name in LOCATION + RELATIVE:
        assert name in x_full.columns, f"{name} missing from the feature frame"

    print(f"\nfull feature set: {x_full.shape[1]} columns", flush=True)
    evaluate(x_full, y, split, "with_location", args.out)

    x_rel = x_full.drop(columns=LOCATION)
    print(f"dropped {LOCATION}; kept {RELATIVE}  -> {x_rel.shape[1]} columns", flush=True)
    evaluate(x_rel, y, split, "relative_geo_only", args.out)


if __name__ == "__main__":
    main()
