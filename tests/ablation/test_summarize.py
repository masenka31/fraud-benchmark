import json

from fraud_benchmark.ablation.summarize import summarize


def record(dataset, feature_set, regime, model, ap, seed=None):
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "label_regime": regime,
        "model": model,
        "seed": seed,
        "features": [],
        "n_train_rows": 100,
        "n_train_positive": 10,
        "fit_seconds": 1.0,
        "scores": {
            "val": {"average_precision": ap, "f1": 0.5, "threshold": 0.5,
                    "precision": 0.5, "recall": 0.5, "n_rows": 10, "n_positive": 1},
            "test": {"average_precision": ap, "f1": 0.5, "threshold": 0.5,
                     "precision": 0.5, "recall": 0.5, "n_rows": 10, "n_positive": 1},
        },
    }


def write(tmp_path, records):
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_seeds_are_averaged(tmp_path):
    path = write(
        tmp_path,
        [
            record("sparkov", "leaky", "oracle", "xgboost", 0.4, seed=0),
            record("sparkov", "leaky", "oracle", "xgboost", 0.6, seed=1),
        ],
    )
    assert "0.500" in summarize(path)


def test_the_leakage_gap_is_reported(tmp_path):
    path = write(
        tmp_path,
        [
            record("ibm_ccf", "leaky", "oracle", "xgboost", 0.80, seed=0),
            record("ibm_ccf", "clean", "oracle", "xgboost", 0.20, seed=0),
        ],
    )
    assert "0.600" in summarize(path)


def test_average_precision_is_named_and_roc_auc_is_absent(tmp_path):
    path = write(tmp_path, [record("sparkov", "leaky", "oracle", "xgboost", 0.4, seed=0)])
    text = summarize(path)
    assert "average precision" in text.lower()
    assert "roc" not in text.lower()


def test_an_empty_results_file_produces_a_message_not_a_crash(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text("")
    assert "no results" in summarize(path).lower()
