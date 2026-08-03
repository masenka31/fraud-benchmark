import json

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.ablation.cell import censored_labels, run_cell


def features(n=600, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n), unit="h")
    signal = rng.normal(size=n)
    is_fraud = (signal > 1.5)
    split = np.array(["train"] * (n - 200) + ["val"] * 100 + ["test"] * 100)
    return pd.DataFrame(
        {
            "event_time": times,
            "entity_id": pd.Series(rng.choice(["a", "b"], n), dtype="string"),
            "amount": rng.lognormal(3, 1, n),
            "signal": signal,
            "merchant": pd.Series(rng.choice(["m1", "m2"], n), dtype="string"),
            "Merchant State": pd.Series(
                np.where(is_fraud, "Italy", "CA"), dtype="string"
            ),
            "is_fraud": is_fraud,
            "split": pd.Series(split, dtype="string"),
            # Mixed reporting speed. A single delay longer than the train window
            # would censor every fraud, leaving the censored regime with no
            # positives at all -- which is a degenerate case, not the partial
            # censoring this study is about.
            "reported_at": pd.Series(
                np.where(
                    is_fraud,
                    times + pd.to_timedelta(np.where(np.arange(n) % 2 == 0, 1, 60), unit="D"),
                    pd.NaT,
                )
            ),
            "txn_count_24h": rng.integers(0, 5, n).astype("float64"),
            "merchant_novelty": rng.integers(0, 2, n),
        }
    )


def test_censored_labels_hide_frauds_reported_after_the_cutoff():
    df = pd.DataFrame(
        {
            "is_fraud": [True, True, False],
            "reported_at": pd.to_datetime(["2023-01-02", "2023-02-01", None]),
        }
    )
    y = censored_labels(df, cutoff=pd.Timestamp("2023-01-15"))
    assert list(y) == [1, 0, 0]


def test_censored_labels_keep_every_row():
    """An unreported fraud is not missing from the data -- it looks legitimate.

    Dropping it would model a system that knows which rows to distrust, which
    is exactly what label delay denies it.
    """
    df = pd.DataFrame(
        {
            "is_fraud": [True, True],
            "reported_at": pd.to_datetime(["2023-01-02", "2023-02-01"]),
        }
    )
    y = censored_labels(df, cutoff=pd.Timestamp("2023-01-15"))
    assert len(y) == 2
    assert list(y) == [1, 0]


def test_censored_labels_never_invent_a_fraud():
    """A non-fraud row stays 0 no matter what reported_at says."""
    df = pd.DataFrame(
        {"is_fraud": [False, False], "reported_at": pd.to_datetime(["2023-01-01", None])}
    )
    assert list(censored_labels(df, cutoff=pd.Timestamp("2023-06-01"))) == [0, 0]


def test_run_cell_writes_one_record_per_model(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["model"] for r in records} == {"logistic", "xgboost"}


def test_each_record_carries_both_split_scores(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "val" in record["scores"]
    assert "test" in record["scores"]
    assert "average_precision" in record["scores"]["val"]


def test_the_test_threshold_is_the_one_chosen_on_val(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scores"]["test"]["threshold"] == record["scores"]["val"]["threshold"]


def test_evaluation_splits_are_never_downsampled(tmp_path):
    out = tmp_path / "runs.jsonl"
    df = features()
    run_cell(
        df,
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scores"]["val"]["n_rows"] == int((df["split"] == "val").sum())
    assert record["scores"]["test"]["n_rows"] == int((df["split"] == "test").sum())


def test_the_trivial_rule_is_recorded_for_ibm_ccf(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="ibm_ccf",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
        include_rule=True,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    rule = [r for r in records if r["model"] == "trivial_rule"]
    assert len(rule) == 1
    assert rule[0]["feature_set"] is None
    assert rule[0]["label_regime"] is None


def test_the_censored_regime_keeps_the_rows_but_loses_positives(tmp_path):
    """The harm of label delay is wrong labels, not fewer rows."""
    out = tmp_path / "runs.jsonl"
    df = features()
    for regime in ["oracle", "censored"]:
        run_cell(
            df,
            dataset="sparkov",
            feature_set="leaky",
            label_regime=regime,
            seeds=(0,),
            results_path=out,
        )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    oracle = next(r for r in records if r["label_regime"] == "oracle")
    censored = next(r for r in records if r["label_regime"] == "censored")
    assert censored["n_train_rows"] == oracle["n_train_rows"]
    assert censored["n_train_positive"] < oracle["n_train_positive"]


def test_evaluation_labels_are_never_censored(tmp_path):
    """Whatever the model was allowed to learn from, it is scored on the truth."""
    out = tmp_path / "runs.jsonl"
    df = features()
    for regime in ["oracle", "censored"]:
        run_cell(
            df,
            dataset="sparkov",
            feature_set="leaky",
            label_regime=regime,
            seeds=(0,),
            results_path=out,
        )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    positives = {r["scores"]["test"]["n_positive"] for r in records}
    assert len(positives) == 1, "test positives must not vary with the label regime"


def test_records_append_rather_than_overwrite(tmp_path):
    out = tmp_path / "runs.jsonl"
    for feature_set in ["leaky", "clean"]:
        run_cell(
            features(),
            dataset="sparkov",
            feature_set=feature_set,
            label_regime="oracle",
            seeds=(0,),
            results_path=out,
        )
    assert len(out.read_text().splitlines()) == 4


def test_datetime_columns_never_reach_a_model(tmp_path):
    """Neither model accepts datetime64, and coercing one would hand the model
    the split boundary as an integer threshold."""
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="sparkov",
        feature_set="leaky",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "event_time" not in record["features"]


def test_the_resolved_feature_list_is_recorded(tmp_path):
    out = tmp_path / "runs.jsonl"
    run_cell(
        features(),
        dataset="ibm_ccf",
        feature_set="clean",
        label_regime="oracle",
        seeds=(0,),
        results_path=out,
    )
    record = json.loads(out.read_text().splitlines()[0])
    assert "Merchant State" not in record["features"]
    assert "is_fraud" not in record["features"]


def test_a_regime_with_no_known_positives_fails_loudly(tmp_path):
    """A clear message beats sklearn's 'needs samples of at least 2 classes'.

    This is reachable on real data: a dataset whose reporting delay exceeds its
    train window censors every fraud. Failing here names the cell and the cause;
    failing inside the fitter names neither, six hours into a cluster job.
    """
    df = features()
    df.loc[df["is_fraud"], "reported_at"] = pd.Timestamp("2030-01-01")
    with pytest.raises(ValueError, match="no positive labels in train"):
        run_cell(
            df,
            dataset="sparkov",
            feature_set="leaky",
            label_regime="censored",
            seeds=(0,),
            results_path=tmp_path / "runs.jsonl",
        )
