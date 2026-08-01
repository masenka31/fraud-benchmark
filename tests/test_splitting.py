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
    with pytest.warns(UserWarning, match="empty split"):
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


def test_nat_in_event_time_is_rejected():
    df = frame_with_times(["2023-01-01", None, "2023-01-03"])
    with pytest.raises(ValueError, match="missing value"):
        assign_splits(df, (0.8, 0.1, 0.1))


def test_empty_frame_is_rejected():
    df = frame_with_times([])
    with pytest.raises(ValueError, match="empty frame"):
        assign_splits(df, (0.8, 0.1, 0.1))


def test_missing_event_time_column_is_rejected():
    df = pd.DataFrame({"amount": [1.0, 2.0]})
    with pytest.raises(ValueError, match="event_time"):
        assign_splits(df, (0.8, 0.1, 0.1))


def test_negative_ratios_are_rejected():
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 11)])
    # Sums to 1.0, so the sum check alone would let this through.
    with pytest.raises(ValueError, match="positive"):
        assign_splits(df, (1.2, -0.1, -0.1))


def test_empty_split_warns_with_actionable_detail():
    # Two timestamps cannot be divided into three non-empty splits.
    df = frame_with_times(["2023-01-01"] * 95 + ["2023-01-02"] * 5)
    with pytest.warns(UserWarning, match="distinct timestamp"):
        assign_splits(df, (0.8, 0.1, 0.1))


def test_no_timestamp_appears_in_two_splits():
    # The central guarantee: a tied block must never straddle a boundary.
    times = [f"2023-01-{d:02d}" for d in range(1, 8) for _ in range(20)]
    df = frame_with_times(times)
    df["split"] = assign_splits(df, (0.7, 0.2, 0.1))
    per_timestamp = df.groupby("event_time")["split"].nunique()
    assert (per_timestamp == 1).all()
    train_max = df[df.split == "train"]["event_time"].max()
    val_min = df[df.split == "val"]["event_time"].min()
    assert train_max < val_min


def test_labels_align_to_a_shuffled_index():
    # Assigning the result back to a frame must not silently misalign rows.
    df = frame_with_times([f"2023-01-{d:02d}" for d in range(1, 21)])
    shuffled = df.sample(frac=1, random_state=0)
    splits = assign_splits(shuffled, (0.8, 0.1, 0.1))
    assert splits.index.equals(shuffled.index)
    shuffled["split"] = splits
    # The earliest date must be train and the latest test, whatever the row order.
    assert shuffled.loc[shuffled.event_time.idxmin(), "split"] == "train"
    assert shuffled.loc[shuffled.event_time.idxmax(), "split"] == "test"
