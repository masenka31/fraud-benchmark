from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"
OPTIONS = {"start_date": "2023-01-01"}


@pytest.fixture
def frame():
    return get_adapter("paysim").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_step_is_anchored_to_start_date(frame):
    # step 1 is the first hour, so it maps to start_date itself.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00")
    # step 25 is 24 hours later.
    assert frame["event_time"].iloc[4] == pd.Timestamp("2023-01-02 00:00:00")


def test_entity_id_is_the_originating_customer(frame):
    assert frame["entity_id"].iloc[0] == "C1231006815"


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, True, True, False, False]


def test_source_columns_pass_through(frame):
    assert frame["type"].iloc[0] == "PAYMENT"
    assert "isFlaggedFraud" in frame.columns
    assert "oldbalanceOrg" in frame.columns


def test_original_step_column_is_kept(frame):
    assert frame["step"].tolist() == [1, 1, 2, 3, 25]


def test_column_mapping_documents_provenance():
    mapping = get_adapter("paysim").column_mapping(OPTIONS)
    assert mapping["entity_id"] == "nameOrig"
    assert mapping["is_fraud"] == "isFraud"
    assert "step" in mapping["event_time"]


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("paysim").to_canonical(FIXTURE, {})


def test_missing_csv_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSV"):
        get_adapter("paysim").to_canonical(tmp_path, OPTIONS)


def test_paysim_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "paysim" in list_datasets()
