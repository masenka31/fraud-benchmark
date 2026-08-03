from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "sparkov"


@pytest.fixture
def frame():
    return get_adapter("sparkov").to_canonical(FIXTURE, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_both_source_files_are_concatenated(frame):
    # 3 rows from fraudTrain plus 2 from fraudTest.
    assert len(frame) == 5


def test_rows_are_ordered_in_time(frame):
    assert frame["event_time"].is_monotonic_increasing


def test_event_time_comes_from_the_real_timestamp(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2019-01-01 00:00:18")
    assert frame["event_time"].iloc[-1] == pd.Timestamp("2020-12-31 23:59:34")


def test_entity_id_is_the_card_number_as_string(frame):
    assert frame["entity_id"].iloc[0] == "2703186189652095"
    assert str(frame["entity_id"].dtype) == "string"


def test_amount_comes_from_amt(frame):
    assert frame["amount"].iloc[0] == 4.97


def test_is_fraud_is_boolean(frame):
    assert frame["is_fraud"].sum() == 2


def test_row_index_column_is_dropped(frame):
    # 'Unnamed: 0' is a per-file index, meaningless once the files are concatenated.
    assert "Unnamed: 0" not in frame.columns


def test_source_column_is_recorded(frame):
    # Which file each row came from is worth keeping, and the split depends on it.
    assert set(frame["source_file"]) == {"fraudTrain.csv", "fraudTest.csv"}


def test_upstream_test_set_becomes_the_test_split(frame):
    splits = get_adapter("sparkov").custom_splits(frame, {"val_fraction": 0.5})
    from_test_file = frame["source_file"] == "fraudTest.csv"
    assert set(splits[from_test_file]) == {"test"}
    assert "test" not in set(splits[~from_test_file])


def test_validation_is_the_tail_of_the_train_file(frame):
    # val_fraction 0.5 on a 3-row train file puts the later rows in val.
    splits = get_adapter("sparkov").custom_splits(frame, {"val_fraction": 0.5})
    train_rows = frame[frame["source_file"] == "fraudTrain.csv"]
    labels = splits[train_rows.index]
    assert set(labels) == {"train", "val"}
    # Whatever the cut, train must end before val begins.
    assert train_rows[labels == "train"]["event_time"].max() < (
        train_rows[labels == "val"]["event_time"].min()
    )


def test_split_order_is_train_then_val_then_test(frame):
    splits = get_adapter("sparkov").custom_splits(frame, {"val_fraction": 0.5})
    rank = {"train": 0, "val": 1, "test": 2}
    codes = [rank[s] for s in splits]
    assert codes == sorted(codes), "splits are not ordered in time"


def test_custom_splits_returns_all_three_categories(frame):
    splits = get_adapter("sparkov").custom_splits(frame, {"val_fraction": 0.5})
    assert str(splits.dtype) == "category"
    assert set(splits.cat.categories) == {"train", "val", "test"}
    assert splits.index.equals(frame.index)


def test_passthrough_columns_survive(frame):
    for column in ("merchant", "category", "city_pop", "merch_lat"):
        assert column in frame.columns


def test_sparkov_is_commercially_usable():
    adapter = get_adapter("sparkov")
    assert adapter.commercial_use is True
    assert adapter.data_license == "CC0 1.0"


def test_column_mapping_documents_provenance():
    mapping = get_adapter("sparkov").column_mapping({})
    assert mapping["entity_id"] == "cc_num"
    assert mapping["amount"] == "amt"
    assert mapping["event_time"] == "trans_date_trans_time"


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraudTrain.csv"):
        get_adapter("sparkov").to_canonical(tmp_path, {})


def test_sparkov_is_registered():
    from fraud_benchmark.datasets.base import list_datasets
    assert "sparkov" in list_datasets()


def test_sparkov_slow_shares_sparkovs_raw_download():
    from fraud_benchmark.datasets.sparkov import SparkovSlowAdapter

    assert SparkovSlowAdapter.raw_name == "sparkov"


def test_sparkov_slow_records_that_it_is_a_stress_test():
    """A reader must not mistake its delay for a realistic reporting regime."""
    from fraud_benchmark.datasets.sparkov import SparkovSlowAdapter

    joined = " ".join(SparkovSlowAdapter.caveats).lower()
    assert "stress test" in joined
    assert "row-identical" in joined
