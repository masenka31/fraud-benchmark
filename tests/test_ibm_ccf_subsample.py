from pathlib import Path

import pandas as pd
import pytest

from fraud_benchmark.datasets.base import get_adapter, list_datasets
from fraud_benchmark.schema import validate_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_ccf_subsample"
OPTIONS = {"entity_key": "user", "start_date": "2010-01-01"}
VARIANTS = ("ibm_ccf_subsample_fast", "ibm_ccf_subsample_slow")


@pytest.fixture(params=VARIANTS)
def frame(request):
    return get_adapter(request.param).to_canonical(FIXTURE, OPTIONS)


def test_output_is_canonical(frame):
    validate_canonical(frame)


def test_rows_before_the_start_date_are_dropped(frame):
    assert frame["event_time"].min() >= pd.Timestamp("2010-01-01")


def test_rows_after_the_last_labelled_fraud_are_dropped(frame):
    """The 2011-06-01 row is unlabelled tail, exactly like IBM CCF's real
    2019-11 to 2020-02 dead zone. Keeping it would put a fraud-free block at
    the end of the dataset, and the whole test split inside it."""
    assert frame["event_time"].max() == pd.Timestamp("2011-01-01 00:01")


def test_the_window_keeps_everything_in_between(frame):
    assert len(frame) == 3
    assert frame["is_fraud"].sum() == 2


def test_the_last_row_is_a_fraud(frame):
    """The right edge is the last fraud, so the frame must end on one."""
    assert bool(frame["is_fraud"].iloc[-1])


def test_both_variants_produce_identical_rows():
    """They differ only in delay, which is applied later by the pipeline."""
    fast = get_adapter("ibm_ccf_subsample_fast").to_canonical(FIXTURE, OPTIONS)
    slow = get_adapter("ibm_ccf_subsample_slow").to_canonical(FIXTURE, OPTIONS)
    pd.testing.assert_frame_equal(fast, slow)


def test_entity_key_still_works_through_the_subclass():
    frame = get_adapter("ibm_ccf_subsample_fast").to_canonical(
        FIXTURE, {"entity_key": "card", "start_date": "2010-01-01"}
    )
    assert frame["entity_id"].tolist() == ["0-1", "1-0", "1-0"]


def test_joined_columns_survive_the_crop(frame):
    assert frame["Card Brand"].tolist() == ["Visa", "Amex", "Amex"]


def test_a_missing_start_date_is_a_clear_error():
    with pytest.raises(ValueError, match="start_date"):
        get_adapter("ibm_ccf_subsample_fast").to_canonical(
            FIXTURE, {"entity_key": "user"}
        )


@pytest.mark.parametrize("name", VARIANTS)
def test_variants_share_the_ibm_ccf_raw_download(name):
    assert get_adapter(name).raw_name == "ibm_ccf"


@pytest.mark.parametrize("name", VARIANTS)
def test_variants_are_registered(name):
    assert name in list_datasets()


def test_the_index_is_reset(frame):
    """The crop drops leading rows; a stale index would misalign the later
    campaign and delay stages, which write back by index."""
    assert frame.index.tolist() == [0, 1, 2]
