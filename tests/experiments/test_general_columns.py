"""The general drop-list: what no model may ever see, whatever the experiment.

The point of deriving it from the adapter registry is that adding a dataset cannot
silently add a feature that is really a label.
"""
import pytest

from fraud_benchmark.experiments.columns import (
    ABSOLUTE_TIME_COLUMNS,
    ALWAYS_EXCLUDED,
    ExcludedColumnError,
    assert_no_excluded,
)


def test_the_pipeline_produced_columns_are_excluded():
    for column in ("is_fraud", "reported_at", "campaign_id", "split"):
        assert column in ALWAYS_EXCLUDED


def test_every_adapters_label_descriptive_columns_are_excluded():
    """Derived from the registry, so a new dataset cannot be forgotten."""
    from fraud_benchmark.data.adapters.base import get_adapter, list_datasets

    for name in list_datasets():
        for column in get_adapter(name).label_descriptive_columns:
            assert column in ALWAYS_EXCLUDED, f"{name}: {column} is not excluded"


def test_the_multiclass_labels_are_excluded_by_name():
    """Belt and braces on the two that exist today."""
    assert "Anomaly" in ALWAYS_EXCLUDED
    assert "Laundering_type" in ALWAYS_EXCLUDED


def test_a_dropped_source_label_is_no_longer_listed():
    """The pipeline drops these now, so restating them would be dead weight -- and
    a reader would wrongly infer the frame still carries them."""
    for column in ("Is Fraud?", "Is_laundering", "isFraud", "fraud"):
        assert column not in ALWAYS_EXCLUDED


def test_the_simulator_detector_output_is_excluded():
    """isFlaggedFraud is PaySim's own decision, not an input a model would have."""
    assert "isFlaggedFraud" in ALWAYS_EXCLUDED


def test_absolute_time_columns_are_excluded():
    for column in ("unix_time", "trans_date_trans_time", "Date", "Year"):
        assert column in ABSOLUTE_TIME_COLUMNS


def test_assert_no_excluded_raises_on_a_label():
    with pytest.raises(ExcludedColumnError, match="is_fraud"):
        assert_no_excluded(["amount", "is_fraud"])


def test_assert_no_excluded_passes_a_clean_list():
    assert assert_no_excluded(["amount", "entity_id"]) is None


def test_a_new_adapters_declaration_lands_in_the_list_without_editing_this_module():
    """The whole point: registration is the only place the knowledge lives.

    Registering an adapter whose label-descriptive column is not yet known must
    change what the drop-list computes, with no edit to `experiments/columns.py`.
    """
    from fraud_benchmark.data.adapters import base
    from fraud_benchmark.data.sources import KaggleDataset
    from fraud_benchmark.experiments import columns as general

    class _Probe(base.DatasetAdapter):
        name = "registry_derivation_probe"
        source = KaggleDataset("x/y")
        source_label_column = "raw_label"
        label_descriptive_columns = ("probe_typology",)

        def to_canonical(self, raw_dir, options):
            raise NotImplementedError

        def column_mapping(self, options):
            return {}

    assert "probe_typology" not in general.ALWAYS_EXCLUDED
    base.register(_Probe)
    try:
        assert "probe_typology" in general.build_always_excluded()
    finally:
        del base._REGISTRY[_Probe.name]
