"""Does richer feature engineering close the gap to the sequential models?

Three conditions through identical code on IBM CCF's standard temporal 80/10/10,
so the only variable is the feature set:

  v1        -- the 36-column set: raw columns, clock, relative geography, and the
               original velocity aggregates. Scored 0.041 test AP.
  v1_plus   -- v1 with `Merchant State` and its derived flags removed, keeping
               only same_state / same_city. Tests whether location identity is
               carrying anything on this split.
  v2        -- v1 plus 26 history-relative features: context novelty, context
               recency, decline velocity, burst ratios, entity baseline
               deviation, 30-day memory, amount shape, card age.

The comparison of interest is v2 against the sequential models' 0.335 (RNN) and
0.565 (CAST fine-tuned). If hand-crafted summaries reach them, the sequential
advantage was expressible as features; if they plateau near the 0.052 the
flattened window reached, it was not.
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
from features_v2 import V2_COLUMNS, add_v2_features
from italy_holdout import build_features

LOCATION = ["cat_merchant_state", "merchant_is_online", "merchant_state_missing"]


def standard_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    t = df["event_time"]
    val_start, test_start = t.quantile(0.8), t.quantile(0.9)
    df["split"] = np.where(t < val_start, "train", np.where(t < test_start, "val", "test"))
    return df


def evaluate(x: pd.DataFrame, y, split, condition: str, out: Path) -> float:
    tr, va, te = split == "train", split == "val", split == "test"
    aps, gains = [], []
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
        gain = dict(model.get_booster().get_score(importance_type="total_gain"))
        rec = {
            "condition": condition, "seed": seed, "n_features": x.shape[1],
            "fit_seconds": round(time.monotonic() - started, 1),
            "scores": {"val": score(y[va], pv, th), "test": score(y[te], pt, th)},
            "importance_gain": gain,
        }
        aps.append(rec["scores"]["test"]["average_precision"])
        gains.append(gain)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {condition:9s} seed={seed} "
              f"val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    total: dict[str, float] = {}
    for g in gains:
        for k, v in g.items():
            total[k] = total.get(k, 0.0) + v / len(gains)
    share = sum(total.values()) or 1.0
    print(f"  top 15 features ({condition}):", flush=True)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:15]:
        mark = " *" if k in V2_COLUMNS else ""
        print(f"     {k:30s} {v / share * 100:6.2f}%{mark}", flush=True)
    v2_share = sum(v for k, v in total.items() if k in V2_COLUMNS) / share * 100
    print(f"  gain from v2 features: {v2_share:.1f}%", flush=True)
    print(f"  -> {condition}: test AP {np.mean(aps):.4f}+-{np.std(aps):.4f}\n", flush=True)
    return float(np.mean(aps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/ibm_features_v2.jsonl"))
    args = ap.parse_args()

    raw = standard_split(pd.read_parquet(args.features))
    y = raw["is_fraud"].to_numpy().astype(int)
    split = raw["split"].to_numpy()

    x_v1, _ = build_features(raw)
    print(f"v1: {x_v1.shape[1]} features", flush=True)

    enriched = add_v2_features(
        raw, merchant_col="Merchant Name", state_col="Merchant State",
        city_col="Merchant City", mcc_col="MCC", channel_col="Use Chip",
        error_col="Errors?", amount_col="Amount",
        acct_open_col="Acct Open Date", pin_year_col="Year PIN last Changed",
        expires_col="Expires",
    )
    x_v2 = pd.concat(
        [x_v1, enriched[list(V2_COLUMNS)].astype("float32").fillna(-1.0)], axis=1
    )
    del raw, enriched
    print(f"v2: {x_v2.shape[1]} features ({len(V2_COLUMNS)} new)\n", flush=True)

    results = {}
    results["v1"] = evaluate(x_v1, y, split, "v1", args.out)
    results["v1_plus"] = evaluate(x_v1.drop(columns=LOCATION), y, split, "v1_plus", args.out)
    results["v2"] = evaluate(x_v2, y, split, "v2", args.out)

    print("summary (test average precision):", flush=True)
    for k, v in results.items():
        print(f"  {k:9s} {v:.4f}", flush=True)
    print("  reference: flattened window 0.0516, RNN 0.335, CAST fine-tuned 0.565",
          flush=True)


if __name__ == "__main__":
    main()
