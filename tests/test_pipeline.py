import json
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.config import Config
from fraud_benchmark.pipeline import prepare

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"


@pytest.fixture
def config(tmp_path):
    return Config(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        split_ratios=(0.6, 0.2, 0.2),
        datasets={"paysim": {"start_date": "2023-01-01"}},
    )


@pytest.fixture
def no_download(monkeypatch):
    """Pretend the raw files are already downloaded, using the test fixture."""

    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr("fraud_benchmark.pipeline.fetch", fake_fetch)


def test_prepare_writes_parquet(config, no_download):
    out = prepare("paysim", config)
    assert (out / "data.parquet").exists()
    assert (out / "dataset_card.json").exists()


def test_output_has_core_columns_first(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert list(df.columns)[:5] == [
        "event_time",
        "entity_id",
        "amount",
        "is_fraud",
        "split",
    ]


def test_output_row_count_matches_fixture(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert len(df) == 5


def test_every_row_has_a_split(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert df["split"].notna().all()
    assert set(df["split"]) <= {"train", "val", "test"}


def test_output_is_sorted_by_event_time(config, no_download):
    out = prepare("paysim", config)
    df = pd.read_parquet(out / "data.parquet")
    assert df["event_time"].is_monotonic_increasing


def test_dataset_card_records_counts_and_provenance(config, no_download):
    out = prepare("paysim", config)
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["name"] == "paysim"
    assert card["n_rows"] == 5
    assert card["n_fraud"] == 2
    assert card["source"]["handle"] == "ealaxi/paysim1"
    assert card["column_mapping"]["entity_id"] == "nameOrig"
    assert card["caveats"]
    assert card["split"]["ratios"] == [0.6, 0.2, 0.2]
    assert "train" in card["split"]["counts"]


def test_rerun_replaces_previous_output(config, no_download):
    first = prepare("paysim", config)
    stale = first / "stale.txt"
    stale.write_text("left over from a previous run")
    prepare("paysim", config)
    assert not stale.exists()
    assert (first / "data.parquet").exists()


def test_no_temp_directory_is_left_behind(config, no_download):
    prepare("paysim", config)
    leftovers = list(config.processed_dir.glob("*.tmp*"))
    assert leftovers == []


def test_validation_failure_writes_nothing(config, monkeypatch):
    def fake_fetch(source, dest, *, force=False):
        return FIXTURE

    monkeypatch.setattr("fraud_benchmark.pipeline.fetch", fake_fetch)
    monkeypatch.setattr(
        "fraud_benchmark.pipeline.validate_canonical",
        lambda df: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(ValueError, match="boom"):
        prepare("paysim", config)
    assert not (config.processed_dir / "paysim").exists()
