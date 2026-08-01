from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_ccf"


@pytest.fixture
def frame():
    return get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "user"})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_all_rows_survive(frame):
    assert len(frame) == 5


def test_event_time_is_built_from_the_date_parts(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2002-09-01 06:21")
    assert frame["event_time"].iloc[4] == pd.Timestamp("2011-01-01 00:01")


def test_dollar_amounts_are_parsed(frame):
    assert frame["amount"].iloc[0] == 134.09
    # Negative amounts appear as "$-25.00".
    assert frame["amount"].iloc[2] == -25.00


def test_is_fraud_comes_from_the_yes_no_column(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_entity_id_defaults_to_the_user(frame):
    assert frame["entity_id"].tolist() == ["0", "0", "0", "1", "1"]


def test_entity_key_can_be_switched_to_card():
    frame = get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "card"})
    # User 0 has two cards, so its rows must now split into two entities.
    assert frame["entity_id"].tolist() == ["0-0", "0-0", "0-1", "1-0", "1-0"]


def test_unknown_entity_key_is_an_error():
    with pytest.raises(ValueError, match="entity_key"):
        get_adapter("ibm_ccf").to_canonical(FIXTURE, {"entity_key": "nonsense"})


def test_card_attributes_are_joined(frame):
    assert frame["Card Brand"].tolist()[:3] == ["Visa", "Visa", "Visa"]
    assert frame["Card Type"].iloc[2] == "Credit"
    assert frame["Card Type"].iloc[0] == "Debit"


def test_user_attributes_are_joined_positionally(frame):
    # sd254_users.csv has no ID column; row 0 is User 0.
    assert frame["Person"].iloc[0] == "Hazel Robinson"
    assert frame["Person"].iloc[3] == "Sasha Sadr"
    assert frame["FICO Score"].iloc[0] == 787


def test_dollar_columns_in_the_joined_tables_are_parsed(frame):
    assert frame["Credit Limit"].iloc[0] == 24295.0
    assert frame["Total Debt"].iloc[0] == 127613.0
    assert frame["Yearly Income - Person"].iloc[0] == 59696.0
    assert frame["Per Capita Income - Zipcode"].iloc[0] == 29278.0


def test_the_card_index_join_key_is_not_duplicated(frame):
    assert "CARD INDEX" not in frame.columns


def test_ibm_ccf_is_commercially_usable():
    adapter = get_adapter("ibm_ccf")
    assert adapter.commercial_use is True


def test_column_mapping_documents_provenance():
    mapping = get_adapter("ibm_ccf").column_mapping({"entity_key": "user"})
    assert mapping["amount"] == "Amount"
    assert mapping["is_fraud"] == "Is Fraud?"
    assert "Year" in mapping["event_time"]


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="credit_card_transactions"):
        get_adapter("ibm_ccf").to_canonical(tmp_path, {"entity_key": "user"})


def test_ibm_ccf_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "ibm_ccf" in list_datasets()
