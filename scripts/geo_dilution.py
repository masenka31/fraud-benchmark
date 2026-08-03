"""Is there anything left in IBM CCF once the geography oracle is broken?

Two conditions beyond the main ablation's `leaky` and `clean`.

The recorded numbers in results/geo_dilution.jsonl (geo_diluted 0.195,
no_geo_keep_mcc 0.057) were measured on `ibm_ccf_subsample_fast`, which has since
been retired -- see docs/verification-notes.md, "## The IBM CCF subsamples are
retired". The default is now the full `ibm_ccf`, so a re-run is comparable to the
rest of the IBM CCF table but NOT to those two figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fraud_benchmark.experiments.encoding import Encoder
from fraud_benchmark.experiments.geo import IDENTITY, dilute_geography, matrix_columns
from fraud_benchmark.experiments.metrics import best_f1_threshold, score
from fraud_benchmark.experiments.models import fit_logistic, fit_xgboost


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
    ap.add_argument("--dataset", default="ibm_ccf")
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
