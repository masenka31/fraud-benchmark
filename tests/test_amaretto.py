import shutil
from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "amaretto"


@pytest.fixture
def raw_dir(tmp_path):
    """A disposable copy of the committed fixture.

    Copied rather than used in place because extraction writes a cache directory
    beside the archive parts, and one test deletes the parts outright.
    """
    target = tmp_path / "amaretto"
    shutil.copytree(FIXTURE, target)
    return target


@pytest.fixture
def frame(raw_dir):
    return get_adapter("amaretto").to_canonical(raw_dir, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_all_rows_are_read_from_the_split_archive(frame):
    assert len(frame) == 5


def test_event_time_is_the_real_timestamp(frame):
    assert frame["event_time"].iloc[0] == pd.Timestamp("2019-01-01 17:55:33")


def test_entity_id_is_the_originator(frame):
    assert frame["entity_id"].iloc[0] == "Client_087"
    assert str(frame["entity_id"].dtype) == "string"


def test_amount_comes_from_normalized_amount(frame):
    assert frame["amount"].iloc[0] == 10317357.93


def test_any_nonzero_anomaly_class_counts_as_fraud(frame):
    # Anomaly is 0 plus five FATF typologies, not a boolean.
    assert frame["is_fraud"].tolist() == [False, False, True, True, False]


def test_the_anomaly_class_is_preserved(frame):
    # Plan 4 may key campaign grouping on the typology, as with SAML-D.
    assert frame["Anomaly"].tolist() == [0, 0, 3, 5, 0]


def test_passthrough_columns_survive(frame):
    for column in ("Market", "Product Type", "Currency", "InputOutput"):
        assert column in frame.columns


def test_extraction_is_cached_between_calls(raw_dir):
    adapter = get_adapter("amaretto")
    first = adapter.to_canonical(raw_dir, {})
    for part in (raw_dir / "Data").iterdir():
        part.unlink()
    # The extracted CSV is cached under raw_dir, so a second call still works.
    second = adapter.to_canonical(raw_dir, {})
    assert len(first) == len(second)


def test_amaretto_is_mit_licensed_and_commercial():
    adapter = get_adapter("amaretto")
    assert adapter.commercial_use is True
    assert adapter.data_license == "MIT"


def test_source_is_a_git_repo():
    from fraud_benchmark.data.sources import GitRepo

    assert isinstance(get_adapter("amaretto").source, GitRepo)


def test_missing_archive_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="amaretto_dataset_anon.zip"):
        get_adapter("amaretto").to_canonical(tmp_path, {})


def test_column_mapping_documents_provenance():
    mapping = get_adapter("amaretto").column_mapping({})
    assert mapping["entity_id"] == "Originator"
    assert mapping["amount"] == "Normalized Amount"
    assert mapping["is_fraud"] == "Anomaly > 0"


def test_amaretto_is_registered():
    from fraud_benchmark.data.adapters.base import list_datasets

    assert "amaretto" in list_datasets()
