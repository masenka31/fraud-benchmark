"""Run one (dataset, feature set, label regime) cell.

Every evaluation appends a record the moment it is scored, so a job that dies
loses at most the fit in flight. The record carries the resolved feature list, so
a result can always be traced to the exact columns that produced it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.ablation.build_features import DEFAULT_FEATURES
from fraud_benchmark.ablation.columns import feature_columns
from fraud_benchmark.ablation.encoding import Encoder
from fraud_benchmark.ablation.metrics import best_f1_threshold, score
from fraud_benchmark.ablation.models import fit_logistic, fit_xgboost, trivial_rule_scores

DEFAULT_RESULTS = Path("results/runs.jsonl")


def censored_labels(df: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """The labels a model training at `cutoff` would actually have.

    A fraud not yet reported is NOT missing from the training data -- it sits in
    it looking like a legitimate transaction. So the unreported frauds are
    relabelled 0, not dropped. Dropping them would model a system that somehow
    knows which rows to distrust, which is precisely the knowledge label delay
    denies it, and would understate the harm: the damage is wrong labels, not
    fewer of them.
    """
    reported = pd.to_datetime(df["reported_at"])
    known_fraud = df["is_fraud"].astype(bool) & (reported <= cutoff)
    return known_fraud.to_numpy().astype(int)


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def run_cell(
    df: pd.DataFrame,
    dataset: str,
    feature_set: str,
    label_regime: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    results_path: Path = DEFAULT_RESULTS,
    include_rule: bool = False,
) -> None:
    results_path = Path(results_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    if label_regime == "oracle":
        y_train = train["is_fraud"].to_numpy().astype(int)
    elif label_regime == "censored":
        # Same rows, fewer known positives -- see censored_labels.
        y_train = censored_labels(train, cutoff=train["event_time"].max())
    else:
        raise ValueError(f"unknown label_regime {label_regime!r}")

    if y_train.sum() == 0:
        raise ValueError(
            f"{dataset}/{feature_set}/{label_regime}: no positive labels in train. "
            "Under the censored regime this means every fraud in the train window "
            "is reported after the cutoff, so there is nothing to learn. Fitting "
            "would fail deep inside sklearn with a much less informative message."
        )

    columns = feature_columns(df, dataset=dataset, feature_set=feature_set)

    # Datetime columns are dropped outright: neither model accepts datetime64, and
    # silently coercing one to an integer nanosecond count would hand the model the
    # split boundary as a threshold. This covers event_time and each dataset's own
    # date columns (sparkov's trans_date_trans_time and dob, saml_d's Date).
    # A dataset's *numeric* time parts -- IBM CCF's Year/Month/Day/Time -- are kept:
    # they are ordinary columns of the source data, and the regime shift they encode
    # is a documented property of the dataset rather than an artifact this study
    # ablates.
    columns = [c for c in columns if not pd.api.types.is_datetime64_any_dtype(df[c])]

    categorical = [
        c for c in columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = [c for c in columns if c not in categorical]

    # Evaluation labels are never censored: val and test are scored against the
    # truth, whatever the model was allowed to learn from.
    y_val = val["is_fraud"].to_numpy().astype(int)
    y_test = test["is_fraud"].to_numpy().astype(int)

    if include_rule:
        rule_val = trivial_rule_scores(val, dataset=dataset)
        if rule_val is not None:
            rule_test = trivial_rule_scores(test, dataset=dataset)
            threshold = best_f1_threshold(y_val, rule_val)
            _append(
                results_path,
                {
                    "dataset": dataset,
                    "feature_set": None,
                    "label_regime": None,
                    "model": "trivial_rule",
                    "seed": None,
                    "features": [],
                    "n_train_rows": 0,
                    "fit_seconds": 0.0,
                    "scores": {
                        "val": score(y_val, rule_val, threshold),
                        "test": score(y_test, rule_test, threshold),
                    },
                },
            )

    encoder_common = dict(categorical=categorical, numeric=numeric)
    tree_encoder = Encoder().fit(train[columns], scale=False, **encoder_common)
    linear_encoder = Encoder().fit(train[columns], scale=True, **encoder_common)

    def emit(model_name: str, seed: int | None, model, encoder, elapsed: float) -> None:
        val_scores = model.predict_proba(encoder.transform(val[columns]))[:, 1]
        test_scores = model.predict_proba(encoder.transform(test[columns]))[:, 1]
        threshold = best_f1_threshold(y_val, val_scores)
        _append(
            results_path,
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "label_regime": label_regime,
                "model": model_name,
                "seed": seed,
                "features": columns,
                "n_train_rows": int(len(train)),
                # Positives the model was allowed to see -- lower than the true
                # count under the censored regime, which is the point of it.
                "n_train_positive": int(y_train.sum()),
                "fit_seconds": round(elapsed, 2),
                "scores": {
                    "val": score(y_val, val_scores, threshold),
                    "test": score(y_test, test_scores, threshold),
                },
            },
        )

    started = time.monotonic()
    linear = fit_logistic(linear_encoder.transform(train[columns]), y_train)
    emit("logistic", None, linear, linear_encoder, time.monotonic() - started)

    x_train = tree_encoder.transform(train[columns])
    for seed in seeds:
        started = time.monotonic()
        booster = fit_xgboost(x_train, y_train, seed=seed)
        emit("xgboost", seed, booster, tree_encoder, time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ablation cell")
    parser.add_argument("dataset")
    parser.add_argument("feature_set", choices=["leaky", "clean"])
    parser.add_argument("label_regime", choices=["oracle", "censored"])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--include-rule", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.features_dir) / f"{args.dataset}.parquet")
    run_cell(
        df,
        dataset=args.dataset,
        feature_set=args.feature_set,
        label_regime=args.label_regime,
        results_path=args.results,
        include_rule=args.include_rule,
    )
    print(f"{args.dataset}/{args.feature_set}/{args.label_regime}: done")


if __name__ == "__main__":
    main()
