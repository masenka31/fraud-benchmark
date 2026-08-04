import pandas as pd
import pytest

from fraud_benchmark.data.adapters import base
from fraud_benchmark.data.adapters.base import DatasetAdapter
from fraud_benchmark.data.adapters.base import UnknownDatasetError
from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.adapters.base import list_datasets
from fraud_benchmark.data.adapters.base import register
from fraud_benchmark.data.sources import KaggleDataset


def make_fake_class(name='fake_for_tests', handle='someone/fake'):
    # The abstract methods must be defined in the class body. ABCMeta computes
    # __abstractmethods__ once, at class creation, so assigning them afterwards
    # leaves the class permanently un-instantiable.
    class _FakeAdapter(DatasetAdapter):
        source_label_column = 'is_fraud'
        caveats = ('this dataset is fake',)

        def to_canonical(self, raw_dir, options):
            return pd.DataFrame()

        def column_mapping(self, options):
            return {'event_time': 'ts'}

    _FakeAdapter.name = name
    _FakeAdapter.source = KaggleDataset(handle)
    return _FakeAdapter


@pytest.fixture
def fake_adapter():
    """Register a throwaway adapter, then remove it.

    Registration must not leak: pytest imports every test module during collection,
    so a module-level @register would leave this fake in the global registry and
    break `prepare --all` in tests/test_cli.py.
    """
    cls = make_fake_class()
    register(cls)
    yield cls
    del base._REGISTRY[cls.name]


def test_get_adapter_returns_an_instance(fake_adapter):
    adapter = get_adapter('fake_for_tests')
    assert isinstance(adapter, fake_adapter)
    assert adapter.source.handle == 'someone/fake'


def test_list_datasets_includes_registered_adapter(fake_adapter):
    assert 'fake_for_tests' in list_datasets()


def test_raw_name_defaults_to_the_dataset_name(fake_adapter):
    assert get_adapter('fake_for_tests').raw_name == 'fake_for_tests'


def test_every_registered_adapter_has_a_raw_name():
    """A variant may share another dataset's raw files, but never by accident."""
    for name in list_datasets():
        adapter = get_adapter(name)
        assert isinstance(adapter.raw_name, str) and adapter.raw_name


def test_raw_name_can_be_overridden():
    cls = make_fake_class(name='fake_variant')
    cls.raw_name = 'fake_for_tests'
    assert cls().raw_name == 'fake_for_tests'


def test_registry_is_clean_without_the_fixture():
    assert 'fake_for_tests' not in list_datasets()


def test_list_datasets_is_sorted():
    names = list_datasets()
    assert names == sorted(names)


def test_unknown_dataset_raises_with_available_names(fake_adapter):
    with pytest.raises(UnknownDatasetError, match='fake_for_tests'):
        get_adapter('no_such_dataset')


def test_registering_a_duplicate_name_is_an_error(fake_adapter):
    with pytest.raises(ValueError, match='already registered'):
        register(make_fake_class(handle='someone/other'))


def test_abstract_methods_are_enforced():
    class _Incomplete(DatasetAdapter):
        name = 'incomplete'
        source = KaggleDataset('someone/incomplete')

    with pytest.raises(TypeError, match='abstract'):
        _Incomplete()


def test_registering_without_a_name_is_a_clear_error():
    class _Nameless(DatasetAdapter):
        source = KaggleDataset('someone/nameless')

        def to_canonical(self, raw_dir, options):
            return pd.DataFrame()

        def column_mapping(self, options):
            return {}

    with pytest.raises(ValueError, match="must set a non-empty string 'name'"):
        register(_Nameless)


def test_require_start_date_returns_a_timestamp():
    from fraud_benchmark.data.adapters.base import require_start_date

    anchor = require_start_date({'start_date': '2023-01-01'}, 'paysim', 'step')
    assert anchor == pd.Timestamp('2023-01-01')


def test_require_start_date_names_the_option_and_the_column():
    """The three offset datasets share this; the message must still be specific."""
    from fraud_benchmark.data.adapters.base import require_start_date

    with pytest.raises(ValueError, match='datasets.ieee_cis.start_date'):
        require_start_date({}, 'ieee_cis', 'TransactionDT')
    with pytest.raises(ValueError, match='TransactionDT'):
        require_start_date({}, 'ieee_cis', 'TransactionDT')


def test_an_empty_start_date_is_rejected_like_a_missing_one():
    """An empty string would otherwise anchor everything to the epoch."""
    from fraud_benchmark.data.adapters.base import require_start_date

    with pytest.raises(ValueError, match='start_date'):
        require_start_date({'start_date': ''}, 'banksim', 'step')


def test_every_adapter_declares_its_source_label_column():
    """A new adapter that forgets this would pass its raw label to models."""
    from fraud_benchmark.data.adapters.base import get_adapter
    from fraud_benchmark.data.adapters.base import list_datasets

    for name in list_datasets():
        adapter = get_adapter(name)
        assert isinstance(adapter.source_label_column, str)
        assert adapter.source_label_column, f'{name} declares an empty label column'


def test_registration_rejects_an_adapter_with_no_source_label_column():
    from fraud_benchmark.data.adapters import base

    class Unlabelled(base.DatasetAdapter):
        name = 'unlabelled_probe'
        source = KaggleDataset('x/y')

        def to_canonical(self, raw_dir, options):
            raise NotImplementedError

        def column_mapping(self, options):
            return {}

    with pytest.raises(ValueError, match='source_label_column'):
        base.register(Unlabelled)


def test_label_descriptive_columns_defaults_to_empty():
    from fraud_benchmark.data.adapters.base import get_adapter

    assert get_adapter('banksim').label_descriptive_columns == ()


def test_the_multiclass_labels_are_declared_descriptive():
    """Kept in the frame, never a feature."""
    from fraud_benchmark.data.adapters.base import get_adapter

    assert get_adapter('amaretto').label_descriptive_columns == ('Anomaly',)
    assert get_adapter('saml_d').label_descriptive_columns == ('Laundering_type',)
