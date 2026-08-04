from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent.parent / "fixtures" / "paysim"
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


def test_the_declared_source_label_is_a_column_this_adapter_produces(frame):
    """The pipeline drops the raw label by the name declared here, so a stale name
    would silently drop nothing. Absence after the drop is a pipeline property and
    is asserted in tests/data/test_pipeline.py -- to_canonical does not drop."""
    adapter = get_adapter("paysim")
    assert adapter.source_label_column == "isFraud"
    assert adapter.source_label_column in frame.columns
    assert adapter.source_label_column not in adapter.label_descriptive_columns
    # And the label derived from it survives with the fixture's values.
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
    from fraud_benchmark.data.adapters.base import list_datasets

    assert "paysim" in list_datasets()


def test_paysim_declares_its_data_license():
    # Read from the Kaggle API on 2026-08-01. See docs/dataset-licenses.md.
    assert get_adapter("paysim").data_license == "CC BY-SA 4.0"


def test_every_registered_adapter_declares_a_data_license():
    # Guards against a future adapter shipping with the "unknown" default,
    # which would silently omit the dataset's terms from its card.
    from fraud_benchmark.data.adapters.base import list_datasets

    undeclared = [name for name in list_datasets() if get_adapter(name).data_license == "unknown"]
    assert undeclared == [], f"adapters missing a data_license: {undeclared}"
