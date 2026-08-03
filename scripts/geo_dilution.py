"""Is there anything left in IBM CCF once the geography oracle is broken?

Two conditions beyond the main ablation's `leaky` and `clean`.

`geo_diluted` is the literal question: replace Italy with some other location in
95% of Italy rows. Doing it to `Merchant State` alone would measure nothing --
the audit found `Merchant City` mirrors the country exactly (Rome, Algiers, Port
au Prince), `Zip` is flagged in its own right, and a merchant id encodes its own
location. So the whole merchant identity is reassigned jointly, sampled from real
non-Italy rows in proportion to their volume, which scatters the diluted frauds
into common locations rather than concentrating them somewhere new.

`no_geo_keep_mcc` isolates MCC cleanly by dropping every location and identity
column and keeping MCC plus everything else. The audit measured MCC as an
independent artifact (MCC 5732: 843 frauds at 6.7%, none in Italy), so this asks
how much that is worth on its own.

Both are dataset transformations, not fitting steps, so they are applied
identically to train, val and test before anything is fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.ablation.columns import ALWAYS_EXCLUDED, ABSOLUTE_TIME_COLUMNS
from fraud_benchmark.ablation.encoding import Encoder
from fraud_benchmark.ablation.metrics import best_f1_threshold, score
from fraud_benchmark.ablation.models import fit_logistic, fit_xgboost

# The columns that jointly carry "where and who", all reassigned together.
IDENTITY = ["Merchant State", "Merchant City", "Zip", "Merchant Name"]
DILUTE_FRACTION = 0.95


def dilute_geography(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Reassign the merchant identity of 95% of Italy rows to a real non-Italy one."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    is_italy = (out["Merchant State"].astype("string") == "Italy").fillna(False)
    donors = out.index[~is_italy]
    targets = out.index[is_italy]
    chosen = targets[rng.random(len(targets)) < DILUTE_FRACTION]

    # Sampling donors uniformly from actual rows reproduces the real location
    # mix, so diluted frauds land mostly in high-volume places.
    picks = rng.choice(donors, size=len(chosen), replace=True)
    for col in IDENTITY:
        out.loc[chosen, col] = out.loc[picks, col].to_numpy()
    return out


def matrix_columns(df: pd.DataFrame, drop: list[str]) -> list[str]:
    banned = set(ALWAYS_EXCLUDED) | set(ABSOLUTE_TIME_COLUMNS) | set(drop)
    cols = [c for c in df.columns if c not in banned]
    return [c for c in cols if not pd.api.types.is_datetime64_any_dtype(df[c])]


def run(df: pd.DataFrame, condition: str, columns: list[str], out_path: Path) -> None:
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    y_train = train["is_fraud"].to_numpy().astype(int)
    y_val = val["is_fraud"].to_numpy().astype(int)
    y_test = test["is_fraud"].to_numpy().astype(int)

    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in columns if c not in categorical]
    common = dict(categorical=categorical, numeric=numeric)
    tree = Encoder().fit(train[columns], scale=False, **common)
    linear = Encoder().fit(train[columns], scale=True, **common)

    def emit(name, seed, model, enc, elapsed):
        v = model.predict_proba(enc.transform(val[columns]))[:, 1]
        t = model.predict_proba(enc.transform(test[columns]))[:, 1]
        th = best_f1_threshold(y_val, v)
        rec = {
            "condition": condition,
            "model": name,
            "seed": seed,
            "n_features": len(columns),
            "fit_seconds": round(elapsed, 1),
            "scores": {"val": score(y_val, v, th), "test": score(y_test, t, th)},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{condition:18s} {name:9s} seed={seed} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    s = time.monotonic()
    emit("logistic", None, fit_logistic(linear.transform(train[columns]), y_train),
         linear, time.monotonic() - s)

    x = tree.transform(train[columns])
    for seed in (0, 1, 2):
        s = time.monotonic()
        emit("xgboost", seed, fit_xgboost(x, y_train, seed=seed), tree,
             time.monotonic() - s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ibm_ccf_subsample_fast")
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("results/geo_dilution.jsonl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.features_dir / f"{args.dataset}.parquet")

    # Condition 1: keep everything, but break the geography oracle.
    diluted = dilute_geography(df)
    before = (df["Merchant State"].astype("string") == "Italy").fillna(False)
    after = (diluted["Merchant State"].astype("string") == "Italy").fillna(False)
    print(f"Italy rows {int(before.sum()):,} -> {int(after.sum()):,}", flush=True)
    print(f"Italy frauds {int((before & df.is_fraud).sum()):,} -> "
          f"{int((after & diluted.is_fraud).sum()):,}", flush=True)
    run(diluted, "geo_diluted", matrix_columns(diluted, drop=[]), args.out)

    # Condition 2: no location or identity at all, but MCC retained.
    run(df, "no_geo_keep_mcc", matrix_columns(df, drop=IDENTITY), args.out)


if __name__ == "__main__":
    main()
