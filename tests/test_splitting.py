import pandas as pd
import pytest

from fraud_benchmark.splitting import assign_splits, split_boundaries


def frame_with_times(times):
    return pd.DataFrame({"event_time": pd.to_datetime(times)})


def test_splits_ten_distinct_days_80_10_10():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert splits.tolist() == ["train"] * 8 + ["val"] + ["test"]


def test_result_is_categorical_with_all_three_levels():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert str(splits.dtype) == "category"
    assert set(splits.cat.categories) == {"train", "val", "test"}


def test_rows_sharing_a_timestamp_are_never_split():
    # Nine rows on day 1, one row each on days 2 and 3. A naive positional 80/10/10
    # cut would slice through the day-1 block.
    times = ["2023-01-01"] * 9 + ["2023-01-02", "2023-01-03"]
    df = frame_with_times(times)
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    day_one = splits[:9]
    assert day_one.nunique() == 1


def test_split_is_monotonic_in_time():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 21)])
    splits = assign_splits(df, (0.7, 0.2, 0.1))
    rank = {"train": 0, "val": 1, "test": 2}
    codes = [rank[s] for s in splits]
    assert codes == sorted(codes)


def test_unsorted_input_is_handled():
    df = frame_with_times(
        ["2023-01-05", "2023-01-01", "2023-01-03", "2023-01-02", "2023-01-04"]
    )
    splits = assign_splits(df, (0.6, 0.2, 0.2))
    # The earliest date must be train, the latest must be test.
    assert splits.iloc[1] == "train"
    assert splits.iloc[0] == "test"


def test_single_timestamp_puts_everything_in_train():
    df = frame_with_times(["2023-01-01"] * 5)
    splits = assign_splits(df, (0.8, 0.1, 0.1))
    assert set(splits) == {"train"}


def test_boundaries_are_reported():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    bounds = split_boundaries(df, (0.8, 0.1, 0.1))
    assert bounds["train_end"] == pd.Timestamp("2023-01-08")
    assert bounds["val_end"] == pd.Timestamp("2023-01-09")


def test_ratios_must_sum_to_one():
    df = frame_with_times(["2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(df, (0.5, 0.2, 0.2))
