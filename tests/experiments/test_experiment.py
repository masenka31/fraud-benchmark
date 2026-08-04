"""The experiment pipeline: what each axis actually does to the data.

Built on a synthetic feature parquet rather than the real ones, so these run in
seconds and assert the semantics rather than any measured score.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.experiment import (
    ExperimentConfig,
    ExperimentError,
    append_record,
    describe,
    prepare,
    run,
)
from fraud_benchmark.experiments.features.util import KEY_COLUMNS, write_features
from fraud_benchmark.experiments.history import MISSING

BASE = pd.Timestamp("2019-01-01 00:00:00")
N = 600


def make_parquet(directory, dataset="sparkov", n=N, slow=True):
    """A parquet with the columns `features/sparkov.py` produces, in miniature.

    Only the columns the pipeline reads by name need to be real: the key columns, the
    dataset's HISTORY_COLUMNS, and at least one artifact_ column.
    """
    rng = np.random.default_rng(0)
    time = [BASE + pd.Timedelta(hours=i) for i in range(n)]
    fraud = np.zeros(n, dtype=bool)
    fraud[rng.choice(n, size=n // 10, replace=False)] = True

    keys = pd.DataFrame(
        {
            "entity_id": pd.Series([f"card{i % 5}" for i in range(n)], dtype="string"),
            "event_time": time,
            # Half the frauds are reported quickly, half only after the whole span,
            # so a cutoff at the train boundary hides a measurable number of them.
            "reported_at": [
                (t + pd.Timedelta(days=1 if i % 2 else 400)) if f else pd.NaT
                for i, (t, f) in enumerate(zip(time, fraud))
            ],
            "is_fraud": fraud,
            "split": pd.Series(["train"] * n, dtype="string"),
        }
    )
    extra = ()
    if slow:
        # Strictly later than `reported_at`, but not so late that every train label
        # disappears -- an all-negative training set is rejected outright, and it
        # would not be a delay regime worth measuring anyway.
        keys["reported_at_slow"] = [
            (t + pd.Timedelta(days=10 if i % 2 else 800)) if f else pd.NaT
            for i, (t, f) in enumerate(zip(time, fraud))
        ]
        extra = ("reported_at_slow",)

    features = pd.DataFrame(
        {
            "amount_log1p": rng.normal(size=n) + fraud * 3.0,
            "seconds_since_prev_txn": rng.random(n) * 1000,
            "hour": [t.hour for t in time],
            "distance_from_home_km": rng.random(n) * 100,
            "distance_over_entity_mean": rng.random(n) * 2,
            "amount_over_entity_mean": rng.random(n) * 2,
            "txn_count_24h": rng.integers(0, 9, n).astype("float64"),
            "first_merchant_for_entity": rng.integers(0, 2, n).astype("float64"),
            "category": pd.Series([f"cat{i % 4}" for i in range(n)], dtype="string").astype(
                "category"
            ),
            "an_extra_numeric": rng.random(n),
            "artifact_merchant": pd.Series(
                [f"shop{i % 7}" for i in range(n)], dtype="string"
            ).astype("category"),
        }
    )
    write_features(dataset, keys, features, features_dir=directory, extra_keys=extra)
    return directory


@pytest.fixture
def features_dir(tmp_path):
    return make_parquet(tmp_path)


def config(**overrides):
    base = {"dataset": "sparkov", "model": "xgboost", "seeds": (0,)}
    return ExperimentConfig(**{**base, **overrides})


# --- validation rejects what cannot mean what it says ----------------------


def test_slow_delay_is_rejected_where_there_is_no_second_timestamp():
    with pytest.raises(ExperimentError, match="only sparkov carries"):
        config(dataset="saml_d", label_delay="slow").validate()


def test_italy_holdout_is_rejected_off_ibm_ccf():
    with pytest.raises(ExperimentError, match="means nothing on sparkov"):
        config(split="italy_holdout").validate()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"dataset": "nope"}, "unknown dataset"),
        ({"label_delay": "maybe"}, "label_delay must be"),
        ({"artifacts": "some"}, "artifacts must be"),
        ({"split": "random"}, "split must be"),
        ({"history": -1}, "must not be negative"),
    ],
)
def test_a_bad_axis_names_what_was_expected(overrides, message):
    with pytest.raises(ExperimentError, match=message):
        config(**overrides).validate()


def test_a_missing_parquet_says_how_to_build_it(tmp_path):
    with pytest.raises(ExperimentError, match="python -m fraud_benchmark"):
        prepare(config(), tmp_path)


# --- the artifact axis ----------------------------------------------------


def test_dropping_artifacts_removes_exactly_the_prefixed_columns(features_dir):
    kept = prepare(config(artifacts="keep"), features_dir)
    dropped = prepare(config(artifacts="drop"), features_dir)
    assert "artifact_merchant" in kept.names
    assert "artifact_merchant" not in dropped.names
    assert len(kept.names) == len(dropped.names) + 1


def test_no_key_column_becomes_a_feature(features_dir):
    prepared = prepare(config(artifacts="keep"), features_dir)
    assert not set(KEY_COLUMNS) & set(prepared.names)
    assert "reported_at_slow" not in prepared.names


# --- the history axis ----------------------------------------------------


def test_history_widens_the_matrix_by_the_declared_columns(features_dir):
    flat = prepare(config(history=0), features_dir)
    lagged = prepare(config(history=3), features_dir)
    # 9 HISTORY_COLUMNS for sparkov, 3 lags of them.
    assert len(lagged.names) == len(flat.names) + 3 * 9
    assert lagged.names[: len(flat.names)] == flat.names


def test_the_target_block_keeps_the_full_feature_set(features_dir):
    """Only the lags are narrow; the row itself is never reduced."""
    lagged = prepare(config(history=2), features_dir)
    assert "an_extra_numeric" in lagged.names
    assert "an_extra_numeric_lag1" not in lagged.names
    assert "amount_log1p_lag1" in lagged.names


def test_history_all_lags_every_feature(features_dir):
    flat = prepare(config(history=0), features_dir)
    every = prepare(config(history=2, history_columns="all"), features_dir)
    assert len(every.names) == len(flat.names) * 3
    assert "an_extra_numeric_lag2" in every.names


def test_a_lagged_categorical_shares_its_base_vocabulary(features_dir):
    prepared = prepare(config(history=2), features_dir)
    cardinalities = dict(
        zip([prepared.names[i] for i in prepared.categorical], prepared.cardinalities)
    )
    assert cardinalities["category"] == cardinalities["category_lag1"]
    assert cardinalities["category"] == cardinalities["category_lag2"]


def test_the_first_transaction_of_an_entity_has_missing_lags(features_dir):
    prepared = prepare(config(history=1), features_dir)
    first = prepared.names.index("amount_log1p_lag1")
    # 5 entities, so the 5 earliest rows are each an entity's first.
    order = np.argsort(prepared.x[:, prepared.names.index("hour")], kind="stable")
    assert (prepared.x[:, first] == MISSING).sum() == 5
    assert order is not None  # ordering is incidental; the count is the claim


def test_history_columns_must_all_be_present(features_dir):
    """Dropping a column that HISTORY_COLUMNS names must fail, not silently narrow."""
    from fraud_benchmark.experiments import experiment

    module = experiment.DATASET_MODULES["sparkov"]
    original = module.HISTORY_COLUMNS
    module.HISTORY_COLUMNS = (*original, "not_a_column")
    try:
        with pytest.raises(ExperimentError, match="not_a_column"):
            prepare(config(history=1), features_dir)
    finally:
        module.HISTORY_COLUMNS = original


# --- the label-delay axis -------------------------------------------------


def test_delay_off_uses_the_true_labels(features_dir):
    prepared = prepare(config(label_delay="off"), features_dir)
    assert prepared.censored_train_labels == 0
    assert np.array_equal(prepared.y, prepared.y_true)


def test_delay_on_hides_train_frauds_reported_after_the_cutoff(features_dir):
    prepared = prepare(config(label_delay="on"), features_dir)
    assert prepared.censored_train_labels > 0
    train = prepared.train_rows
    # Hidden, not dropped: the row count is unchanged and the label became 0.
    assert len(prepared.y) == len(prepared.y_true)
    assert prepared.y[train].sum() < prepared.y_true[train].sum()


def test_delay_never_touches_val_or_test(features_dir):
    prepared = prepare(config(label_delay="on"), features_dir)
    for rows in (prepared.val_rows, prepared.test_rows):
        assert np.array_equal(prepared.y[rows], prepared.y_true[rows])


def test_slow_delay_hides_at_least_as_much_as_the_fast_one(features_dir):
    fast = prepare(config(label_delay="on"), features_dir)
    slow = prepare(config(label_delay="slow"), features_dir)
    assert slow.censored_train_labels >= fast.censored_train_labels


def test_the_delay_axis_changes_no_feature(features_dir):
    """Only labels move. A regime that changed the matrix would confound itself."""
    off = prepare(config(label_delay="off"), features_dir)
    on = prepare(config(label_delay="on"), features_dir)
    assert np.array_equal(np.nan_to_num(off.x), np.nan_to_num(on.x))
    assert off.names == on.names


# --- splits and preparation invariants -----------------------------------


def test_the_three_splits_partition_every_row(features_dir):
    prepared = prepare(config(), features_dir)
    total = len(prepared.train_rows) + len(prepared.val_rows) + len(prepared.test_rows)
    assert total == len(prepared.y)
    assert not set(prepared.train_rows) & set(prepared.val_rows)
    assert not set(prepared.val_rows) & set(prepared.test_rows)


def test_the_splits_stay_in_time_order(features_dir):
    prepared = prepare(config(), features_dir)
    assert prepared.train_rows.max() < prepared.val_rows.min()
    assert prepared.val_rows.max() < prepared.test_rows.min()


def test_every_run_uses_every_row(features_dir):
    """There is no subsampling, and no field that could request it."""
    from dataclasses import fields

    assert "max_rows" not in {f.name for f in fields(ExperimentConfig)}
    prepared = prepare(config(), features_dir)
    assert len(prepared.y) == N


def test_continuous_and_categorical_indices_partition_the_columns(features_dir):
    prepared = prepare(config(history=2, artifacts="keep"), features_dir)
    both = set(prepared.continuous.tolist()) | set(prepared.categorical.tolist())
    assert both == set(range(len(prepared.names)))
    assert not set(prepared.continuous.tolist()) & set(prepared.categorical.tolist())
    assert len(prepared.cardinalities) == len(prepared.categorical)


def test_the_matrix_is_float32(features_dir):
    assert prepare(config(), features_dir).x.dtype == np.dtype("float32")


# --- the unlabelled tail --------------------------------------------------


def ibm_frame(directory, n=600, tail=120):
    """An ibm_ccf-shaped parquet whose last `tail` rows carry no frauds at all."""
    rng = np.random.default_rng(2)
    time = [BASE + pd.Timedelta(hours=i) for i in range(n)]
    fraud = np.zeros(n, dtype=bool)
    # Frauds spread through the labelled part only, as IBM CCF's are.
    fraud[rng.choice(n - tail, size=(n - tail) // 8, replace=False)] = True

    keys = pd.DataFrame(
        {
            "entity_id": pd.Series([f"user{i % 5}" for i in range(n)], dtype="string"),
            "event_time": time,
            "reported_at": [
                (t + pd.Timedelta(days=1)) if f else pd.NaT for t, f in zip(time, fraud)
            ],
            "is_fraud": fraud,
            "split": pd.Series(["train"] * n, dtype="string"),
        }
    )
    names = (
        "amount_log1p",
        "seconds_since_prev_txn",
        "hour",
        "same_state",
        "same_city",
        "merchant_is_online",
        "merchant_is_foreign",
        "amount_over_credit_limit",
        "amount_over_entity_mean",
        "txn_count_24h",
        "errors_24h",
        "first_merchant_for_entity",
    )
    features = pd.DataFrame({name: rng.random(n) for name in names})
    for name in ("mcc_group", "use_chip"):
        features[name] = pd.Series([f"{name}{i % 3}" for i in range(n)], dtype="string").astype(
            "category"
        )
    features["artifact_merchant_state"] = pd.Series(
        ["Italy" if i % 9 == 0 else "CA" for i in range(n)], dtype="string"
    ).astype("category")
    write_features("ibm_ccf", keys, features, features_dir=directory)
    return directory


def test_the_unlabelled_tail_is_dropped_from_ibm_ccf(tmp_path):
    """IBM CCF stops generating fraud four months before the data ends."""
    directory = ibm_frame(tmp_path)
    prepared = prepare(ExperimentConfig(dataset="ibm_ccf", model="xgboost", seeds=(0,)), directory)
    # The cut lands on the last fraud, which is somewhere in the labelled 480 -- so
    # at least the 120-row tail goes, and the surviving frame ends *on* a fraud.
    assert len(prepared.y) <= 480
    assert prepared.y_true[-1] == 1
    assert any("after the last labelled fraud" in note for note in prepared.notes)


def test_dropping_the_tail_leaves_frauds_in_every_split(tmp_path):
    """The point of the cut: an unlabelled tail in test cannot be scored at all."""
    directory = ibm_frame(tmp_path)
    prepared = prepare(ExperimentConfig(dataset="ibm_ccf", model="xgboost", seeds=(0,)), directory)
    for rows in (prepared.train_rows, prepared.val_rows, prepared.test_rows):
        assert int(prepared.y_true[rows].sum()) > 0


def test_other_datasets_keep_every_row(tmp_path):
    """Only IBM CCF has a tail its generator left unlabelled."""
    directory = make_parquet(tmp_path)
    prepared = prepare(config(), directory)
    assert len(prepared.y) == N
    assert not any("last labelled fraud" in note for note in prepared.notes)


# --- a run that cannot measure anything must fail, not report NaN ----------


def test_a_split_with_no_frauds_is_rejected(tmp_path):
    """Average precision is undefined there, and a NaN record reads as a result."""
    rng = np.random.default_rng(1)
    n = 400
    time = [BASE + pd.Timedelta(hours=i) for i in range(n)]
    fraud = np.zeros(n, dtype=bool)
    fraud[:20] = True  # every fraud in the earliest rows, so val and test have none

    keys = pd.DataFrame(
        {
            "entity_id": pd.Series([f"card{i % 5}" for i in range(n)], dtype="string"),
            "event_time": time,
            "reported_at": [
                (t + pd.Timedelta(days=1)) if f else pd.NaT for t, f in zip(time, fraud)
            ],
            "is_fraud": fraud,
            "split": pd.Series(["train"] * n, dtype="string"),
        }
    )
    features = pd.DataFrame(
        {
            name: rng.random(n)
            for name in (
                "amount_log1p",
                "seconds_since_prev_txn",
                "hour",
                "distance_from_home_km",
                "distance_over_entity_mean",
                "amount_over_entity_mean",
                "txn_count_24h",
                "first_merchant_for_entity",
            )
        }
    )
    features["category"] = pd.Series(["a", "b"] * (n // 2), dtype="string").astype("category")
    features["artifact_merchant"] = pd.Series(["m"] * n, dtype="string").astype("category")
    write_features("sparkov", keys, features, features_dir=tmp_path)

    with pytest.raises(ExperimentError, match="contains no frauds"):
        prepare(config(), tmp_path)


def test_a_delay_that_hides_every_train_label_is_rejected(tmp_path):
    """Training on all-negative labels fits nothing and would report it as a score."""
    directory = make_parquet(tmp_path)
    frame = pd.read_parquet(directory / "sparkov.parquet")
    # Every fraud reported long after the whole span ends.
    frame["reported_at"] = frame["reported_at"].where(
        ~frame["is_fraud"], BASE + pd.Timedelta(days=9999)
    )
    frame.to_parquet(directory / "sparkov.parquet", index=False)

    with pytest.raises(ExperimentError, match="no fraud is labelled in train"):
        prepare(config(label_delay="on"), directory)


def test_append_record_refuses_to_write_a_nan(tmp_path):
    """Python writes NaN as a bare literal that json.loads accepts and pandas does not."""
    with pytest.raises(ValueError):
        append_record({"aggregate": float("nan")}, tmp_path / "out.jsonl")


# --- end to end -----------------------------------------------------------


@pytest.mark.parametrize("model", ["xgboost", "logistic"])
def test_a_run_produces_a_complete_record(features_dir, model, tmp_path):
    record = run(config(model=model, history=2, seeds=(0,)), features_dir)
    assert record["config"]["model"] == model
    assert record["rows"]["train"] > 0
    assert len(record["seeds"]) == 1
    for split in ("val", "test"):
        assert 0.0 <= record["aggregate"][split]["average_precision"]["mean"] <= 1.0
    assert "average precision" in describe(record)

    destination = append_record(record, tmp_path / "out.jsonl")
    assert json.loads(destination.read_text().strip())["n_features"] == record["n_features"]


def test_multiple_seeds_are_aggregated_with_a_spread(features_dir):
    record = run(config(seeds=(0, 1)), features_dir)
    assert len(record["seeds"]) == 2
    assert record["aggregate"]["test"]["average_precision"]["sd"] >= 0.0


def test_append_record_adds_one_line_per_run(features_dir, tmp_path):
    out = tmp_path / "runs.jsonl"
    for _ in range(3):
        append_record(run(config(seeds=(0,)), features_dir), out)
    assert len(out.read_text().strip().splitlines()) == 3


def test_a_dataset_without_a_trivial_rule_reports_none(features_dir):
    assert prepare(config(), features_dir).rule_scores is None


def test_the_trivial_rule_is_read_before_artifacts_are_dropped(tmp_path):
    """It is a reference point, so it must not depend on the artifact setting."""
    from fraud_benchmark.experiments import models

    original = models.TRIVIAL_RULES.get("sparkov")
    models.TRIVIAL_RULES["sparkov"] = ("artifact_merchant", "shop1")
    try:
        directory = make_parquet(tmp_path)
        prepared = prepare(config(artifacts="drop"), directory)
        assert prepared.rule_scores is not None
        assert prepared.rule_scores.sum() > 0
        assert "artifact_merchant" not in prepared.names
    finally:
        if original is None:
            del models.TRIVIAL_RULES["sparkov"]
        else:
            models.TRIVIAL_RULES["sparkov"] = original
