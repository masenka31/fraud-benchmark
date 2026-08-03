from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "banksim"
OPTIONS = {"start_date": "2023-01-01"}


@pytest.fixture
def frame():
    return get_adapter("banksim").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_picks_the_transaction_file_not_the_graph_file(frame):
    # The graph file has a Source column; the transaction file has customer.
    assert "customer" in frame.columns
    assert "Source" not in frame.columns
    assert len(frame) == 5


def test_single_quotes_are_stripped_from_entity_id(frame):
    assert frame["entity_id"].iloc[0] == "C1093826151"
    assert not frame["entity_id"].str.startswith("'").any()


def test_single_quotes_are_stripped_from_every_string_column(frame):
    for column in ("customer", "age", "gender", "merchant", "category",
                   "zipcodeOri", "zipMerchant"):
        values = frame[column].astype(str)
        assert not values.str.startswith("'").any(), f"{column} still quoted"
        assert not values.str.endswith("'").any(), f"{column} still quoted"


def test_step_is_zero_based_days(frame):
    # step 0 is the first day, so it maps to start_date itself.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2023-01-01")
    # step 179 is 179 days later.
    assert frame["event_time"].iloc[4] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=179)


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_amount_is_float(frame):
    assert frame["amount"].iloc[0] == 4.55


def test_banksim_is_noncommercial():
    adapter = get_adapter("banksim")
    assert adapter.commercial_use is False
    assert adapter.data_license == "CC BY-NC-SA 4.0"


def test_column_mapping_documents_provenance():
    mapping = get_adapter("banksim").column_mapping(OPTIONS)
    assert mapping["entity_id"] == "customer"
    assert mapping["is_fraud"] == "fraud"
    assert "step" in mapping["event_time"]


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("banksim").to_canonical(FIXTURE, {})


def test_banksim_is_registered():
    from fraud_benchmark.data.adapters.base import list_datasets
    assert "banksim" in list_datasets()
