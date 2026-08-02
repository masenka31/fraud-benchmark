import numpy as np
import pandas as pd

from fraud_benchmark.ablation.encoding import UNSEEN, Encoder


def frame(cats, nums):
    return pd.DataFrame({"cat": pd.Series(cats, dtype="string"), "num": nums})


def test_categories_come_from_train_only():
    train = frame(["a", "b"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    assert set(enc.categories_["cat"]) == {"a", "b", UNSEEN}


def test_a_category_only_in_val_maps_to_the_unseen_bucket():
    train = frame(["a", "b"], [1.0, 2.0])
    val = frame(["a", "z"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    out = enc.transform(val)
    assert out.loc[0, "cat"] == enc.code_of("cat", "a")
    assert out.loc[1, "cat"] == enc.code_of("cat", UNSEEN)


def test_the_unseen_bucket_does_not_collide_with_a_real_category():
    train = frame(["a", "b"], [1.0, 2.0])
    enc = Encoder().fit(train, categorical=["cat"])
    unseen_code = enc.code_of("cat", UNSEEN)
    assert unseen_code not in {enc.code_of("cat", c) for c in ["a", "b"]}


def test_scaling_statistics_come_from_train_only():
    train = frame(["a", "a"], [0.0, 10.0])
    val = frame(["a", "a"], [5.0, 15.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=True)
    out = enc.transform(val)
    mean, std = 5.0, train["num"].std()
    assert out.loc[0, "num"] == (5.0 - mean) / std
    assert out.loc[1, "num"] == (15.0 - mean) / std


def test_scaling_is_skipped_when_not_requested():
    train = frame(["a", "a"], [0.0, 10.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=False)
    out = enc.transform(train)
    assert list(out["num"]) == [0.0, 10.0]


def test_a_zero_variance_column_does_not_produce_nan():
    train = frame(["a", "a"], [3.0, 3.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"], scale=True)
    out = enc.transform(train)
    assert np.isfinite(out["num"]).all()


def test_nulls_in_a_categorical_column_get_their_own_code():
    train = pd.DataFrame({"cat": pd.Series(["a", None], dtype="string")})
    enc = Encoder().fit(train, categorical=["cat"])
    out = enc.transform(train)
    assert out["cat"].notna().all()


def test_transform_preserves_row_count_and_order():
    train = frame(["a", "b", "a"], [1.0, 2.0, 3.0])
    enc = Encoder().fit(train, categorical=["cat"], numeric=["num"])
    out = enc.transform(train)
    assert len(out) == 3
    assert list(out["num"]) == [1.0, 2.0, 3.0]
