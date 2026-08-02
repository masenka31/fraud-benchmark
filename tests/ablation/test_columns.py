import pandas as pd
import pytest

from fraud_benchmark.ablation.columns import (
    ALWAYS_EXCLUDED,
    LEAKY_COLUMNS,
    ExcludedColumnError,
    assert_no_excluded,
    feature_columns,
)


def frame(*names):
    return pd.DataFrame({n: [0] for n in names})


def test_label_and_its_derivatives_are_always_excluded():
    for name in ["is_fraud", "reported_at", "campaign_id", "split"]:
        assert name in ALWAYS_EXCLUDED


def test_source_raw_labels_are_always_excluded():
    for name in ["Is Fraud?", "Is_laundering", "Laundering_type"]:
        assert name in ALWAYS_EXCLUDED


def test_row_identifiers_and_provenance_are_always_excluded():
    for name in ["trans_num", "source_file"]:
        assert name in ALWAYS_EXCLUDED


def test_leaky_columns_are_kept_in_the_leaky_condition():
    df = frame("Merchant State", "amount", "is_fraud")
    cols = feature_columns(df, dataset="ibm_ccf", feature_set="leaky")
    assert "Merchant State" in cols


def test_leaky_columns_are_dropped_in_the_clean_condition():
    df = frame("Merchant State", "Merchant Name", "MCC", "amount", "is_fraud")
    cols = feature_columns(df, dataset="ibm_ccf", feature_set="clean")
    assert "Merchant State" not in cols
    assert "Merchant Name" not in cols
    assert "MCC" not in cols
    assert "amount" in cols


def test_the_label_never_survives_either_condition():
    df = frame("amount", "is_fraud", "reported_at", "campaign_id", "split")
    for feature_set in ["leaky", "clean"]:
        cols = feature_columns(df, dataset="ibm_ccf", feature_set=feature_set)
        assert cols == ["amount"]


def test_sparkov_has_no_leaky_columns_so_both_conditions_match():
    df = frame("merchant", "category", "amount", "is_fraud")
    leaky = feature_columns(df, dataset="sparkov", feature_set="leaky")
    clean = feature_columns(df, dataset="sparkov", feature_set="clean")
    assert leaky == clean


def test_saml_d_drops_entity_id_as_a_column_in_the_clean_condition():
    df = frame("entity_id", "Sender_account", "amount", "is_fraud")
    cols = feature_columns(df, dataset="saml_d", feature_set="clean")
    assert "entity_id" not in cols
    assert "Sender_account" not in cols


def test_velocity_features_survive_the_clean_condition_for_saml_d():
    """Grouping by entity_id is not the same as using it as a feature."""
    df = frame("entity_id", "txn_count_1h", "merchant_novelty", "amount", "is_fraud")
    cols = feature_columns(df, dataset="saml_d", feature_set="clean")
    assert "txn_count_1h" in cols
    assert "merchant_novelty" in cols


def test_the_guard_rejects_a_hand_built_list_containing_the_label():
    """The guard is a tripwire on the filter, so it is tested directly.

    Routing it through feature_columns would be tautological: that function
    builds its output by removing exactly these names, so the check could never
    fire there no matter what it was passed.
    """
    with pytest.raises(ExcludedColumnError, match="is_fraud"):
        assert_no_excluded(["amount", "is_fraud"])


def test_the_guard_reports_every_offending_column():
    with pytest.raises(ExcludedColumnError, match="reported_at"):
        assert_no_excluded(["amount", "is_fraud", "reported_at"])


def test_the_guard_accepts_a_clean_list():
    assert_no_excluded(["amount", "txn_count_1h", "merchant_novelty"])


def test_feature_columns_output_always_passes_the_guard():
    df = frame("amount", "is_fraud", "reported_at", "Merchant State")
    for feature_set in ["leaky", "clean"]:
        assert_no_excluded(feature_columns(df, dataset="ibm_ccf", feature_set=feature_set))


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        feature_columns(frame("amount"), dataset="nope", feature_set="leaky")


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError, match="feature_set"):
        feature_columns(frame("amount"), dataset="ibm_ccf", feature_set="sideways")


def test_absolute_time_restatements_are_dropped_in_both_conditions():
    """The splits are temporal, so an absolute clock encodes split membership.
    A dtype check misses these: trans_date_trans_time is a string, unix_time an int."""
    df = frame("unix_time", "trans_date_trans_time", "Date", "Year", "amount", "is_fraud")
    for feature_set in ["leaky", "clean"]:
        cols = feature_columns(df, dataset="sparkov", feature_set=feature_set)
        for name in ["unix_time", "trans_date_trans_time", "Date", "Year"]:
            assert name not in cols
        assert "amount" in cols


def test_cyclical_and_customer_date_parts_are_kept():
    """Month/Day/time-of-day generalise forward; birth year describes the
    cardholder, not when the transaction happened."""
    df = frame("Month", "Day", "Time", "Birth Year", "Acct Open Date", "dob", "amount")
    cols = feature_columns(df, dataset="ibm_ccf", feature_set="clean")
    for name in ["Month", "Day", "Time", "Birth Year", "Acct Open Date", "dob"]:
        assert name in cols
