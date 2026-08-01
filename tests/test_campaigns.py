import pandas as pd
import pytest

from fraud_benchmark.campaigns import assign_campaigns


def frame(rows):
    """rows: list of (entity_id, 'YYYY-MM-DD', is_fraud)."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series([r[0] for r in rows], dtype="string"),
            "event_time": pd.to_datetime([r[1] for r in rows]),
            "is_fraud": [r[2] for r in rows],
        }
    )


def test_non_fraud_rows_get_no_campaign():
    df = frame([("a", "2023-01-01", False), ("a", "2023-01-02", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert pd.isna(ids.iloc[0])
    assert not pd.isna(ids.iloc[1])


def test_frauds_close_together_share_a_campaign():
    df = frame([("a", "2023-01-01", True), ("a", "2023-01-03", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] == ids.iloc[1]


def test_frauds_far_apart_are_separate_campaigns():
    df = frame([("a", "2023-01-01", True), ("a", "2023-03-01", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] != ids.iloc[1]


def test_the_gap_is_measured_between_consecutive_frauds_not_from_the_first():
    """A drawn-out run is one campaign if each step is within the gap.

    This keeps a slow-burn campaign together, and is also why the gap must be
    chosen carefully on a dataset like Amaretto.
    """
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-06", True),
        ("a", "2023-01-11", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.nunique() == 1


def test_different_entities_never_share_a_campaign():
    df = frame([("a", "2023-01-01", True), ("b", "2023-01-01", True)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.iloc[0] != ids.iloc[1]


def test_every_fraud_belongs_to_exactly_one_campaign():
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
        ("b", "2023-01-01", True),
        ("b", "2023-06-01", True),
        ("c", "2023-01-01", False),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.notna().sum() == 4
    assert ids.dropna().nunique() == 3


def test_unsorted_input_is_handled():
    df = frame([
        ("a", "2023-03-01", True),
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    # The two January rows group; March stands alone.
    assert ids.iloc[1] == ids.iloc[2]
    assert ids.iloc[0] != ids.iloc[1]


def test_result_is_aligned_to_the_input_index():
    df = frame([("a", "2023-01-01", True), ("a", "2023-01-02", True)])
    shuffled = df.iloc[::-1]
    ids = assign_campaigns(shuffled, gap=pd.Timedelta(days=7))
    assert ids.index.equals(shuffled.index)


def test_a_zero_gap_makes_every_fraud_its_own_campaign_unless_simultaneous():
    df = frame([
        ("a", "2023-01-01", True),
        ("a", "2023-01-01", True),
        ("a", "2023-01-02", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(0))
    assert ids.iloc[0] == ids.iloc[1]
    assert ids.iloc[2] != ids.iloc[0]


def test_negative_gap_is_rejected():
    df = frame([("a", "2023-01-01", True)])
    with pytest.raises(ValueError, match="gap"):
        assign_campaigns(df, gap=pd.Timedelta(days=-1))


def test_no_frauds_yields_all_null():
    df = frame([("a", "2023-01-01", False)])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert ids.isna().all()


def test_ids_are_dense_from_zero():
    df = frame([
        ("a", "2023-01-01", True),
        ("b", "2023-01-01", True),
        ("c", "2023-01-01", True),
    ])
    ids = assign_campaigns(df, gap=pd.Timedelta(days=7))
    assert sorted(ids.dropna().unique()) == [0, 1, 2]
