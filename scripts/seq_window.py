"""Flattened sequence window: target transaction + its previous 9, same user.

Tests whether the gap between a sequential model (~0.5 test AP reported on IBM
CCF) and a plain tabular one (0.041 in our ablation) is architecture rather than
features. If concatenating the recent history closes it, the missing ingredient
was sequence context; if not, the difference lies elsewhere.

Two deliberate departures from "just concatenate 10x everything":

* Customer-static columns (FICO, credit limit, incomes, age, card count, gender,
  home state) are constant per user, so lagging them yields nine identical copies.
  They appear ONCE, for the target row. Repeating them would waste memory and
  dilute colsample_bytree across duplicates.
* Velocity aggregates are already summaries of the same history the window now
  carries explicitly, so they too stay on the target row only.

Lag direction is the one thing that must not be wrong: lag k is the k-th
PREVIOUS transaction of that user. Shifting the other way leaks the future, which
would raise the score and raise no error -- hence test_lag_matrix_is_causal.
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
from italy_holdout import build_split as italy_holdout_split
from italy_holdout import encode_mcc, _money, _signed_log1p

N_LAGS = 9

# Per-transaction: meaningful to look back at.
DYNAMIC = [
    "amount_signed", "amount_usd", "hour", "weekday",
    "same_state", "same_city", "merchant_is_online", "merchant_state_missing",
    "cat_use_chip", "cat_mcc_encoded", "cat_merchant_state", "cat_error",
    "merchant_novelty", "log_seconds_since_prev",
]
# Constant per user, or already a summary of the window: target row only.
STATIC = [
    "minute", "month", "day", "credit_limit", "total_debt", "income_person",
    "income_zip", "fico", "num_cards", "age", "cat_card_brand", "cat_card_type",
    "cat_gender", "cat_customer_state",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_24h", "amount_sum_7d", "amount_mean_7d", "amount_zscore_vs_7d",
]

RARE_COVERAGE = 0.99
MAX_LEVELS = 256


def standard_split(df: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's own temporal 80/10/10, matching the reported ~0.5 baseline."""
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    t = df["event_time"]
    val_start, test_start = t.quantile(0.8), t.quantile(0.9)
    df["split"] = np.where(t < val_start, "train", np.where(t < test_start, "val", "test"))
    return df


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per transaction, before any lagging."""
    f = pd.DataFrame(index=df.index)
    hhmm = df["Time"].astype("string").str.split(":", n=1, expand=True)
    f["hour"] = pd.to_numeric(hhmm[0], errors="coerce").fillna(0)
    f["minute"] = pd.to_numeric(hhmm[1], errors="coerce").fillna(0)
    f["weekday"] = df["event_time"].dt.weekday
    f["month"] = df["event_time"].dt.month
    f["day"] = df["event_time"].dt.day

    amount = _money(df["Amount"])
    f["amount_signed"] = amount
    f["amount_usd"] = amount.abs()
    f["credit_limit"] = _money(df["Credit Limit"])
    f["total_debt"] = _money(df["Total Debt"])
    f["income_person"] = _money(df["Yearly Income - Person"])
    f["income_zip"] = _money(df["Per Capita Income - Zipcode"])
    f["fico"] = pd.to_numeric(df["FICO Score"], errors="coerce")
    f["num_cards"] = pd.to_numeric(df["Num Credit Cards"], errors="coerce")
    f["age"] = pd.to_numeric(df["Current Age"], errors="coerce")

    f["same_state"] = (df["State"].astype("string") == df["Merchant State"].astype("string")).fillna(False)
    f["same_city"] = (df["City"].astype("string") == df["Merchant City"].astype("string")).fillna(False)
    f["merchant_is_online"] = (df["Merchant City"].astype("string") == "ONLINE").fillna(False)
    f["merchant_state_missing"] = df["Merchant State"].isna()

    for c in ["txn_count_1h", "txn_count_24h", "txn_count_7d", "amount_sum_24h",
              "amount_sum_7d", "amount_mean_7d", "amount_zscore_vs_7d"]:
        f[c] = df[c]
    f["log_seconds_since_prev"] = _signed_log1p(
        df["seconds_since_prev_txn"].fillna(-1).to_numpy()
    )
    f["merchant_novelty"] = df["merchant_novelty"]

    cats = {
        "use_chip": df["Use Chip"].astype("string"),
        "mcc_encoded": df["MCC"].map(encode_mcc).astype("string"),
        "merchant_state": df["Merchant State"].astype("string").fillna("unknown"),
        "error": df["Errors?"].astype("string").fillna("No Error"),
        "card_brand": df["Card Brand"].astype("string"),
        "card_type": df["Card Type"].astype("string"),
        "gender": df["Gender"].astype("string"),
        "customer_state": df["State"].astype("string"),
    }
    is_train = (df["split"] == "train").to_numpy()
    for name, values in cats.items():
        counts = values[is_train].value_counts(normalize=True)
        keep = counts.cumsum().shift(1).fillna(0.0) < RARE_COVERAGE
        vocab = list(counts[keep].index[:MAX_LEVELS])
        lookup = {v: i for i, v in enumerate(vocab)}
        f[f"cat_{name}"] = values.map(lookup).fillna(len(vocab))

    for c in ["amount_signed", "amount_usd", "credit_limit", "total_debt",
              "income_person", "income_zip", "amount_sum_24h", "amount_sum_7d",
              "amount_mean_7d"]:
        t = _signed_log1p(f[c].fillna(0.0).to_numpy())
        q25, q50, q75 = np.quantile(t[is_train], [0.25, 0.5, 0.75])
        f[c] = (t - q50) / ((q75 - q25) + 1e-8)

    return f.fillna(0.0).astype("float32")


def lag_matrix(values: np.ndarray, user: np.ndarray, n_lags: int) -> np.ndarray:
    """Concatenate lag 1..n_lags of `values` along axis 1, zero where unavailable.

    `values` and `user` must be sorted by (user, time). Lag k of row i is row i-k,
    kept only when row i-k belongs to the same user -- so a user's first rows get
    zeros rather than another user's transactions.
    """
    n, width = values.shape
    out = np.zeros((n, width * n_lags), dtype="float32")
    for k in range(1, n_lags + 1):
        same_user = np.zeros(n, dtype=bool)
        same_user[k:] = user[k:] == user[:-k]
        rows = np.flatnonzero(same_user)
        out[rows, (k - 1) * width : k * width] = values[rows - k]
    return out


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return (X, feature names, y, split labels), all aligned to `df`'s order."""
    order = np.lexsort((df["event_time"].to_numpy(), df["User"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)

    f = base_features(df)
    user = df["User"].to_numpy()
    dyn = f[DYNAMIC].to_numpy(dtype="float32")

    blocks = [f[DYNAMIC + STATIC].to_numpy(dtype="float32"),
              lag_matrix(dyn, user, N_LAGS)]
    names = [f"t0_{c}" for c in DYNAMIC + STATIC]
    names += [f"t-{k}_{c}" for k in range(1, N_LAGS + 1) for c in DYNAMIC]

    x = np.concatenate(blocks, axis=1)
    return x, names, df["is_fraud"].to_numpy().astype(int), df["split"].to_numpy()


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
