import pandas as pd
import pytest

from fraud_benchmark.datasets import base
from fraud_benchmark.datasets.base import (
    DatasetAdapter,
    UnknownDatasetError,
    get_adapter,
    list_datasets,
    register,
)
from fraud_benchmark.sources import KaggleDataset


def make_fake_class(name="fake_for_tests", handle="someone/fake"):
    # The abstract methods must be defined in the class body. ABCMeta computes
    # __abstractmethods__ once, at class creation, so assigning them afterwards
    # leaves the class permanently un-instantiable.
    class _FakeAdapter(DatasetAdapter):
        caveats = ("this dataset is fake",)

        def to_canonical(self, raw_dir, options):
            return pd.DataFrame()

        def column_mapping(self, options):
            return {"event_time": "ts"}

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
    adapter = get_adapter("fake_for_tests")
    assert isinstance(adapter, fake_adapter)
    assert adapter.source.handle == "someone/fake"


def test_list_datasets_includes_registered_adapter(fake_adapter):
    assert "fake_for_tests" in list_datasets()


def test_registry_is_clean_without_the_fixture():
    assert "fake_for_tests" not in list_datasets()


def test_list_datasets_is_sorted():
    names = list_datasets()
    assert names == sorted(names)


def test_unknown_dataset_raises_with_available_names(fake_adapter):
    with pytest.raises(UnknownDatasetError, match="fake_for_tests"):
        get_adapter("no_such_dataset")


def test_registering_a_duplicate_name_is_an_error(fake_adapter):
    with pytest.raises(ValueError, match="already registered"):
        register(make_fake_class(handle="someone/other"))


def test_abstract_methods_are_enforced():
    class _Incomplete(DatasetAdapter):
        name = "incomplete"
        source = KaggleDataset("someone/incomplete")

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()


def test_registering_without_a_name_is_a_clear_error():
    class _Nameless(DatasetAdapter):
        source = KaggleDataset("someone/nameless")

        def to_canonical(self, raw_dir, options):
            return pd.DataFrame()

        def column_mapping(self, options):
            return {}

    with pytest.raises(ValueError, match="must set a non-empty string 'name'"):
        register(_Nameless)
