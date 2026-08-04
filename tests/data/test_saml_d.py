from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.schema import validate_canonical

FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'saml_d'


@pytest.fixture
def frame():
    return get_adapter('saml_d').to_canonical(FIXTURE, {})


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_date_and_time_are_combined(frame):
    assert frame['event_time'].iloc[0] == pd.Timestamp('2022-10-07 10:35:19')
    assert frame['event_time'].iloc[-1] == pd.Timestamp('2022-10-09 23:59:59')


def test_entity_id_is_the_sender_account_as_string(frame):
    assert frame['entity_id'].iloc[0] == '8724731955'
    assert str(frame['entity_id'].dtype) == 'string'


def test_is_fraud_comes_from_is_laundering(frame):
    assert frame['is_fraud'].tolist() == [False, False, True, True, False]


def test_the_declared_source_label_is_a_column_this_adapter_produces(frame):
    """The pipeline drops the raw label by the name declared here, so a stale name
    would silently drop nothing. Absence after the drop is a pipeline property and
    is asserted in tests/data/test_pipeline.py -- to_canonical does not drop."""
    adapter = get_adapter('saml_d')
    assert adapter.source_label_column == 'Is_laundering'
    assert adapter.source_label_column in frame.columns
    # Is_laundering is dropped; only the typology beside it is label-descriptive.
    assert adapter.source_label_column not in adapter.label_descriptive_columns
    # And the label derived from it survives with the fixture's values.
    assert frame['is_fraud'].tolist() == [False, False, True, True, False]


def test_laundering_type_is_preserved(frame):
    # Encodes the typology and is likely needed for campaign grouping later.
    assert 'Laundering_type' in frame.columns
    assert frame['Laundering_type'].iloc[2] == 'Smurfing'


def test_currency_columns_are_preserved_without_conversion(frame):
    assert frame['Payment_currency'].iloc[1] == 'Indian rupee'
    assert frame['Received_currency'].iloc[1] == 'Dirham'
    # Amount stays in native units.
    assert frame['amount'].iloc[1] == 6019.64


def test_saml_d_is_noncommercial():
    adapter = get_adapter('saml_d')
    assert adapter.commercial_use is False
    assert adapter.data_license == 'CC BY-NC-SA 4.0'


def test_column_mapping_documents_provenance():
    mapping = get_adapter('saml_d').column_mapping({})
    assert mapping['entity_id'] == 'Sender_account'
    assert mapping['is_fraud'] == 'Is_laundering'
    assert 'Date' in mapping['event_time'] and 'Time' in mapping['event_time']


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='SAML-D.csv'):
        get_adapter('saml_d').to_canonical(tmp_path, {})


def test_saml_d_is_registered():
    from fraud_benchmark.data.adapters.base import list_datasets

    assert 'saml_d' in list_datasets()


def test_receiver_account_is_passed_through(frame):
    # Laundering is multi-party; the counterparty must survive for Plan 4.
    assert 'Receiver_account' in frame.columns
