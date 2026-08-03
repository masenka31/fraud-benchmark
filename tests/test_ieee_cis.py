from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ieee_cis"
OPTIONS = {"start_date": "2017-12-01"}


@pytest.fixture
def frame():
    return get_adapter("ieee_cis").to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_only_labelled_train_rows_are_used(frame):
    # The 2 unlabelled test rows must not appear in the canonical frame.
    assert len(frame) == 5
    assert 3663549 not in set(frame["TransactionID"])


def test_transaction_dt_is_anchored_to_start_date(frame):
    # TransactionDT is a seconds offset; 86400 is one day past the anchor.
    assert frame["event_time"].iloc[0] == pd.Timestamp("2017-12-02 00:00:00")
    assert frame["event_time"].iloc[4] == pd.Timestamp("2017-12-05 00:00:00")


def test_identity_columns_are_left_joined(frame):
    assert "id_31" in frame.columns
    joined = frame.set_index("TransactionID")["id_31"]
    assert joined.loc[2987001] == "samsung browser 6.2"
    # Transactions without identity data keep nulls rather than being dropped.
    assert pd.isna(joined.loc[2987000])


def test_uid_groups_repeat_customers(frame):
    # Rows 0, 1 and 4 share card1 and addr1, and their D1 values track TransactionDT
    # so all three resolve to the same account start — one entity.
    uids = frame.set_index("TransactionID")["entity_id"]
    assert uids.loc[2987000] == uids.loc[2987001] == uids.loc[2987004]
    # A different card must not collide with them.
    assert uids.loc[2987000] != uids.loc[2987003]


def test_rows_missing_a_uid_component_get_a_unique_identity(frame):
    """addr1 or D1 being null must not fuse unrelated rows into one entity.

    Under pandas 3, astype(str) on NaN yields <NA> and propagates through
    concatenation, so the naive heuristic would emit NULL entity_id and fail
    validation outright.
    """
    uids = frame.set_index("TransactionID")["entity_id"]
    assert uids.loc[2987002] == "txn_2987002"  # missing addr1
    assert uids.loc[2987003] == "txn_2987003"  # missing D1


def test_no_entity_id_is_null(frame):
    assert frame["entity_id"].notna().all()


def test_amount_comes_from_transaction_amt(frame):
    assert frame["amount"].iloc[0] == 68.5


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].tolist() == [False, False, True, False, True]


def test_unlabelled_test_set_is_offered_as_an_auxiliary_frame():
    aux = get_adapter("ieee_cis").auxiliary_frames(FIXTURE, OPTIONS)
    assert set(aux) == {"unlabelled_test"}
    test = aux["unlabelled_test"]
    assert len(test) == 2
    assert "isFraud" not in test.columns
    # It gets a usable timestamp even though it is not canonical.
    assert "event_time" in test.columns
    # And its identity table is joined too.
    assert "id_31" in test.columns


def test_missing_start_date_is_an_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("ieee_cis").to_canonical(FIXTURE, {})


def test_ieee_cis_is_registered():
    from fraud_benchmark.datasets.base import list_datasets

    assert "ieee_cis" in list_datasets()


def test_column_mapping_records_the_uid_heuristic():
    mapping = get_adapter("ieee_cis").column_mapping(OPTIONS)
    assert "card1" in mapping["entity_id"]
    assert mapping["is_fraud"] == "isFraud"
