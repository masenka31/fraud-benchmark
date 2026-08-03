"""The three dataset modules, on frames shaped like their real processed parquets.

The shared property, checked against every module: tampering with the last row
cannot change any feature on an earlier one. That is the only failure mode nothing
else would catch, because lookahead makes a result better rather than noisier.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_benchmark.experiments.columns import ABSOLUTE_TIME_COLUMNS, assert_no_excluded
from fraud_benchmark.experiments.features import ibm_ccf, saml_d, sparkov
from fraud_benchmark.experiments.features.util import (
    ARTIFACT_PREFIX,
    KEY_COLUMNS,
    FeatureContractError,
    artifact_columns,
    feature_columns,
    write_features,
)

BASE = pd.Timestamp("2018-06-01 09:00:00")
N = 24


def times(n=N):
    return [BASE + pd.Timedelta(hours=3 * i) for i in range(n)]


def cycle(values, n=N):
    return [values[i % len(values)] for i in range(n)]


def core(entities, n=N):
    """The canonical columns the pipeline puts on every prepared frame."""
    return {
        "event_time": times(n),
        "entity_id": pd.Series(cycle(entities, n), dtype="string"),
        "amount": pd.Series([10.0 + 7 * i for i in range(n)], dtype="float64"),
        "is_fraud": [i % 11 == 0 for i in range(n)],
        "split": pd.Series(
            ["train"] * (n - 6) + ["val"] * 3 + ["test"] * 3, dtype="string"
        ),
        "reported_at": [
            BASE + pd.Timedelta(days=9) if i % 11 == 0 else pd.NaT for i in range(n)
        ],
    }


def ibm_frame(n=N):
    df = pd.DataFrame(core(["0", "1", "2"], n))
    df["Card"] = cycle([0, 1], n)
    df["Amount"] = df["amount"]
    df["Use Chip"] = cycle(["Swipe Transaction", "Chip Transaction", "Online Transaction"], n)
    df["Merchant Name"] = cycle([-727612092, 3414527459, -34551508], n)
    df["Merchant City"] = cycle(["Beulah", "ONLINE", "Rome", "La Verne"], n)
    df["Merchant State"] = cycle(["ND", None, "Italy", "CA"], n)
    df["Zip"] = cycle([58523.0, np.nan, np.nan, 91750.0], n)
    df["MCC"] = cycle([5499, 5732, 4829, 7996], n)
    df["Errors?"] = cycle([None, None, "Bad CVV", None], n)
    df["Card Brand"] = cycle(["Visa", "Mastercard"], n)
    df["Card Type"] = cycle(["Debit", "Credit"], n)
    df["Has Chip"] = cycle(["YES", "NO"], n)
    df["Cards Issued"] = cycle([1, 2], n)
    df["Credit Limit"] = cycle([12400.0, 21000.0], n)
    df["Acct Open Date"] = cycle(["09/2002", "04/2014"], n)
    df["Expires"] = cycle(["12/2022", "08/2020"], n)
    df["Year PIN last Changed"] = cycle([2008, 2016], n)
    df["Card on Dark Web"] = "No"
    df["Current Age"] = cycle([53, 34], n)
    df["Retirement Age"] = cycle([66, 68], n)
    df["Gender"] = cycle(["Female", "Male"], n)
    df["FICO Score"] = cycle([787, 692], n)
    df["Num Credit Cards"] = cycle([5, 3], n)
    df["Total Debt"] = cycle([127613.0, 4000.0], n)
    df["Yearly Income - Person"] = cycle([59696.0, 32000.0], n)
    df["Per Capita Income - Zipcode"] = cycle([29278.0, 18000.0], n)
    df["City"] = cycle(["La Verne", "Beulah"], n)
    df["State"] = cycle(["CA", "ND"], n)
    # Present in the source and dropped on purpose; see IBM_DROPPED.
    df["Card Number"] = cycle([4956965974959986, 4582313478255491], n)
    df["CVV"] = cycle([393, 719], n)
    df["Person"] = cycle(["Hazel Robinson", "Sasha Sadr"], n)
    df["Address"] = cycle(["462 Rose Lane", "3606 Federal Boulevard"], n)
    df["Apartment"] = cycle([np.nan, 21.0], n)
    df["Latitude"] = cycle([34.15, 47.29], n)
    df["Longitude"] = cycle([-117.76, -101.78], n)
    df["Zipcode"] = cycle([91750, 58523], n)
    df["Birth Year"] = cycle([1966, 1985], n)
    df["Birth Month"] = cycle([11, 4], n)
    df["Year"] = df["event_time"].dt.year
    df["Month"] = df["event_time"].dt.month
    df["Day"] = df["event_time"].dt.day
    df["Time"] = df["event_time"].dt.strftime("%H:%M")
    df["User"] = df["entity_id"].astype("int64")
    df["campaign_id"] = cycle([-1, 3], n)
    return df


def sparkov_frame(n=N):
    df = pd.DataFrame(core(["2703186189652095", "630423337322"], n))
    df["amt"] = df["amount"]
    df["merchant"] = cycle(["fraud_Rippin", "fraud_Heller", "fraud_Lind"], n)
    df["category"] = cycle(["misc_net", "grocery_pos", "gas_transport"], n)
    df["gender"] = cycle(["F", "M"], n)
    df["city"] = cycle(["Moravian Falls", "Orient"], n)
    df["state"] = cycle(["NC", "WA"], n)
    df["zip"] = cycle([28654, 99160], n)
    df["lat"] = cycle([36.0788, 48.8878], n)
    df["long"] = cycle([-81.1781, -118.2105], n)
    df["city_pop"] = cycle([3495, 149], n)
    df["job"] = cycle(["Psychologist", "Special educational needs teacher"], n)
    df["dob"] = cycle(["1988-03-09", "1978-06-21"], n)
    df["merch_lat"] = cycle([36.011293, 49.159047], n)
    df["merch_long"] = cycle([-82.048315, -118.186462], n)
    df["trans_num"] = [f"hash{i:04d}" for i in range(n)]
    # Present in the source and dropped on purpose; see SPARKOV_DROPPED.
    df["first"] = cycle(["Jennifer", "Stephanie"], n)
    df["last"] = cycle(["Banks", "Gill"], n)
    df["street"] = cycle(["561 Perry Cove", "43039 Riley Greens Suite 393"], n)
    df["cc_num"] = df["entity_id"].astype("int64")
    df["trans_date_trans_time"] = df["event_time"].astype("string")
    df["unix_time"] = (df["event_time"].astype("int64") // 10**9)
    df["source_file"] = ["fraudTrain.csv"] * (n - 3) + ["fraudTest.csv"] * 3
    df["campaign_id"] = cycle([-1, 3], n)
    df["reported_at_slow"] = [
        BASE + pd.Timedelta(days=40) if i % 11 == 0 else pd.NaT for i in range(n)
    ]
    return df


def saml_frame(n=N):
    df = pd.DataFrame(core(["8724731955", "1491989064"], n))
    df["Amount"] = df["amount"]
    df["Receiver_account"] = cycle([2769355426, 1902456840, 8767591678], n)
    df["Payment_currency"] = cycle(["UK pounds", "Euro"], n)
    df["Received_currency"] = cycle(["UK pounds", "US dollar"], n)
    df["Sender_bank_location"] = cycle(["UK", "Germany"], n)
    df["Receiver_bank_location"] = cycle(["UK", "Spain", "Nigeria"], n)
    df["Payment_type"] = cycle(["Cash Deposit", "Cross-border", "Cheque"], n)
    df["Laundering_type"] = cycle(["Normal_Fan_Out", "Smurfing"], n)
    # Present in the source and dropped on purpose; see SAML_DROPPED.
    df["Sender_account"] = df["entity_id"].astype("int64")
    df["Date"] = df["event_time"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["event_time"].dt.strftime("%H:%M:%S")
    df["campaign_id"] = cycle([-1, 3], n)
    return df


#: What each module's docstring says it drops outright. The fixtures above carry
#: every one of these, so a column that started being passed through fails here.
IBM_DROPPED = (
    "Card Number", "CVV", "Person", "Address", "Apartment", "Latitude", "Longitude",
    "Zipcode", "Birth Year", "Birth Month", "Year", "Month", "Day", "Time", "User",
    "campaign_id",
)
SPARKOV_DROPPED = (
    "first", "last", "street", "cc_num", "trans_num", "trans_date_trans_time",
    "unix_time", "source_file", "campaign_id",
)
SAML_DROPPED = ("Sender_account", "Date", "Time", "Laundering_type", "campaign_id")

MODULES = [
    pytest.param(ibm_ccf, ibm_frame, (), IBM_DROPPED, id="ibm_ccf"),
    pytest.param(
        sparkov, sparkov_frame, ("reported_at_slow",), SPARKOV_DROPPED, id="sparkov"
    ),
    pytest.param(saml_d, saml_frame, (), SAML_DROPPED, id="saml_d"),
]


# --- the contract, for every module ---------------------------------------


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_build_returns_one_row_per_input_row(module, make, extra, dropped):
    df = make()
    keys, features = module.build(df)
    assert len(keys) == len(df)
    assert len(features) == len(df)


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_keys_are_exactly_the_key_columns(module, make, extra, dropped):
    keys, _ = module.build(make())
    assert list(keys.columns) == [*KEY_COLUMNS, *extra]


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_the_documented_drop_list_really_is_dropped(module, make, extra, dropped):
    """Each module's docstring names what it discards. This is that claim, checked."""
    df = make()
    _, features = module.build(df)
    assert set(dropped) <= set(df.columns), "the fixture must carry every dropped column"
    survived = sorted(set(dropped) & set(features.columns))
    assert not survived, f"{module.DATASET}: dropped column(s) became features: {survived}"


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_no_absolute_clock_and_no_label_reaches_a_feature(module, make, extra, dropped):
    """The splits are temporal, so an absolute clock lets a model find the boundary."""
    _, features = module.build(make())
    assert not set(ABSOLUTE_TIME_COLUMNS) & set(features.columns)
    assert_no_excluded(list(features.columns))


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_features_are_writable_and_round_trip(module, make, extra, dropped, tmp_path):
    df = make()
    keys, features = module.build(df)
    written = pd.read_parquet(
        write_features(
            module.DATASET, keys, features, features_dir=tmp_path, extra_keys=extra
        )
    )
    assert list(written.columns) == [*KEY_COLUMNS, *extra, *features.columns]
    assert len(feature_columns(written)) == len(features.columns)
    assert artifact_columns(written), "every dataset declares an artifact group"
    for column in feature_columns(written):
        dtype = written[column].dtype
        assert dtype == "float32" or isinstance(dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_no_feature_is_entirely_null(module, make, extra, dropped):
    _, features = module.build(make())
    all_null = [c for c in features.columns if features[c].isna().all()]
    assert not all_null


# --- the property that matters -------------------------------------------


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_tampering_with_the_last_row_cannot_change_an_earlier_one(
    module, make, extra, dropped
):
    """A feature that can see its own row, or a later one, fails here and nowhere else."""
    df = make()
    _, before = module.build(df)

    tampered = df.copy()
    last = tampered.index[-1]
    for column in tampered.columns:
        if column in ("event_time", "entity_id", "split", "reported_at",
                      "reported_at_slow"):
            continue
        values = tampered[column]
        if pd.api.types.is_bool_dtype(values):
            tampered.loc[last, column] = not values.iloc[-1]
        elif pd.api.types.is_numeric_dtype(values):
            tampered.loc[last, column] = 10_000_000.0
        else:
            tampered.loc[last, column] = "tampered"

    _, after = module.build(tampered)
    changed = [
        c
        for c in before.columns
        if not before[c].iloc[:-1].equals(after[c].iloc[:-1])
    ]
    assert not changed, f"{module.DATASET}: later data leaked into {changed}"


@pytest.mark.parametrize("module,make,extra,dropped", MODULES)
def test_a_lone_first_transaction_has_no_history(module, make, extra, dropped):
    """Every count over an empty window is 0, not NaN -- the column stays dense."""
    df = make().iloc[:1].reset_index(drop=True)
    _, features = module.build(df)
    for column in features.columns:
        if column.startswith(("txn_count", "amount_sum", "errors_",
                              "distinct_", "prior_distinct", "receiver_in_count")):
            assert features.loc[0, column] == 0.0, column


# --- what each dataset specifically claims --------------------------------


def test_ibm_first_foreign_fires_only_on_a_foreign_row():
    df = ibm_frame()
    _, features = ibm_ccf.build(df)
    foreign = features["merchant_is_foreign"] == 1.0
    assert (features.loc[~foreign, "first_foreign_for_entity"] == 0.0).all()
    assert features["first_foreign_for_entity"].sum() > 0


def test_ibm_relative_geography_replaces_the_country():
    _, features = ibm_ccf.build(ibm_frame())
    for relative in ("same_state", "same_city", "merchant_is_foreign"):
        assert relative in features.columns
    # The country itself survives only inside the artifact group.
    assert "Merchant State" not in features.columns
    assert "artifact_merchant_state" in features.columns


def test_ibm_error_velocity_is_past_only_and_the_row_outcome_is_an_artifact():
    _, features = ibm_ccf.build(ibm_frame())
    assert features.columns.str.startswith("errors_").sum() == 3
    assert "artifact_error_on_row" in features.columns


def test_ibm_mcc_group_generalises_where_the_raw_code_does_not():
    _, features = ibm_ccf.build(ibm_frame())
    assert features["mcc_group"].nunique() <= features["artifact_mcc"].nunique()
    assert not features["mcc_group"].str.startswith(ARTIFACT_PREFIX).any()


@pytest.mark.parametrize(
    "mcc,group",
    [
        (5499, 54),     # a two-digit range
        (5816, 5815),   # the 5815-5818 merge
        (3010, 3000),   # an airline block
        (0, 0),         # unusable
    ],
)
def test_ibm_mcc_group_collapses_to_the_iso_hierarchy(mcc, group):
    assert ibm_ccf.mcc_group(mcc) == group


def test_ibm_mcc_group_handles_a_missing_code():
    assert ibm_ccf.mcc_group(None) == 0
    assert ibm_ccf.mcc_group(float("nan")) == 0


def test_sparkov_distance_is_a_real_distance():
    _, features = sparkov.build(sparkov_frame())
    assert (features["distance_from_home_km"] >= 0).all()
    assert features["distance_from_home_km"].max() > 1.0


def test_sparkov_carries_both_delay_regimes():
    keys, _ = sparkov.build(sparkov_frame())
    assert "reported_at" in keys.columns
    assert "reported_at_slow" in keys.columns
    frauds = keys["is_fraud"]
    assert (keys.loc[frauds, "reported_at_slow"] > keys.loc[frauds, "reported_at"]).all()


def test_sparkov_attaches_the_slow_regime_by_transaction_id():
    df = sparkov_frame().drop(columns=["reported_at_slow"])
    slow = pd.DataFrame(
        {
            "trans_num": df["trans_num"][::-1].to_numpy(),
            "reported_at": [
                BASE + pd.Timedelta(days=40) if i % 11 == 0 else pd.NaT
                for i in range(len(df))
            ][::-1],
        }
    )
    joined = sparkov.attach_slow_delay(df, slow)
    # Reversed row order in the second frame must not shuffle the timestamps.
    assert list(joined["reported_at_slow"].isna()) == list(df["reported_at"].isna())


def test_sparkov_rejects_a_slow_frame_that_cannot_align():
    df = sparkov_frame().drop(columns=["reported_at_slow"])
    duplicated = pd.DataFrame(
        {"trans_num": ["hash0000"] * 2, "reported_at": [pd.NaT, pd.NaT]}
    )
    with pytest.raises(FeatureContractError, match="not unique"):
        sparkov.attach_slow_delay(df, duplicated)


def test_sparkov_rejects_a_slow_frame_missing_rows():
    df = sparkov_frame().drop(columns=["reported_at_slow"])
    partial = pd.DataFrame({"trans_num": ["hash0000"], "reported_at": [pd.NaT]})
    with pytest.raises(FeatureContractError, match="found no match"):
        sparkov.attach_slow_delay(df, partial)


def test_sparkov_notices_an_unmatched_non_fraud_row():
    """`reported_at` is NaT on every non-fraud row, so a failed join there leaves no
    trace in the nulls. Only membership catches it."""
    df = sparkov_frame().drop(columns=["reported_at_slow"])
    slow = pd.DataFrame(
        {"trans_num": df["trans_num"], "reported_at": df["reported_at"]}
    )
    # Row 1 is not a fraud, so both timestamps would be NaT either way.
    assert not df.loc[1, "is_fraud"]
    slow.loc[1, "trans_num"] = "no-such-hash"
    with pytest.raises(FeatureContractError, match="found no match"):
        sparkov.attach_slow_delay(df, slow)


def test_saml_d_measures_both_sides_of_the_transfer():
    _, features = saml_d.build(saml_frame())
    fan_out = [c for c in features.columns if c.startswith("distinct_receivers")]
    fan_in = [c for c in features.columns if c.startswith("receiver_")]
    assert fan_out and fan_in


def test_saml_d_fan_in_differs_from_fan_out():
    """A receiver-keyed aggregate is not derivable from any sender-keyed one."""
    _, features = saml_d.build(saml_frame())
    assert not features["receiver_in_count_7d"].equals(features["txn_count_7d"])


def test_saml_d_never_carries_the_typology():
    df = saml_frame()
    _, features = saml_d.build(df)
    assert "Laundering_type" in df.columns
    assert "Laundering_type" not in features.columns


def test_saml_d_cross_border_is_relative_and_the_locations_are_artifacts():
    _, features = saml_d.build(saml_frame())
    assert set(features["is_cross_border"].unique()) <= {0.0, 1.0}
    assert features["is_cross_border"].sum() > 0
    assert "artifact_sender_bank_location" in features.columns
    assert "artifact_receiver_bank_location" in features.columns


def test_saml_d_structuring_flag_fires_just_below_the_threshold():
    df = saml_frame()
    df["Amount"] = [9_500.0, 10_000.0, 8_000.0] + [50.0] * (len(df) - 3)
    _, features = saml_d.build(df)
    assert list(features["amount_just_under_10k"][:3]) == [1.0, 0.0, 0.0]
