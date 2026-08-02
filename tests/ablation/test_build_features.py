import pandas as pd
import pytest

from fraud_benchmark.ablation.build_features import MERCHANT_COLUMNS, build


def source(tmp_path, name, df):
    directory = tmp_path / "processed" / name
    directory.mkdir(parents=True)
    df.to_parquet(directory / "data.parquet", index=False)
    return tmp_path


def sample():
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"]
            ),
            "entity_id": pd.Series(["a", "a", "b"], dtype="string"),
            "amount": [10.0, 20.0, 30.0],
            "is_fraud": [False, True, False],
            "split": pd.Series(["train", "val", "test"], dtype="string"),
            "reported_at": pd.to_datetime([None, "2023-01-08", None]),
            "merchant": pd.Series(["m1", "m2", "m1"], dtype="string"),
        }
    )


def test_build_writes_a_parquet_with_the_velocity_columns(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    result = pd.read_parquet(out)
    assert "txn_count_24h" in result.columns
    assert "merchant_novelty" in result.columns


def test_build_preserves_every_row(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    assert len(pd.read_parquet(out)) == 3


def test_build_preserves_the_split_column_and_its_counts(tmp_path):
    root = source(tmp_path, "sparkov", sample())
    out = build("sparkov", processed_dir=root / "processed", features_dir=tmp_path / "features")
    result = pd.read_parquet(out)
    assert result["split"].value_counts().to_dict() == {"train": 1, "val": 1, "test": 1}


def test_every_configured_dataset_names_a_merchant_column(tmp_path):
    for name in [
        "ibm_ccf",
        "ibm_ccf_subsample_fast",
        "ibm_ccf_subsample_slow",
        "saml_d",
        "sparkov",
    ]:
        assert name in MERCHANT_COLUMNS
