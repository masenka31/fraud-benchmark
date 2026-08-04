"""IBM CCF features: 24.4M synthetic card transactions, joined card and user tables.

Run as `python -m fraud_benchmark.experiments.features.ibm_ccf`, reads
`data/processed/ibm_ccf/data.parquet`, writes `data/features/ibm_ccf.parquet`.

`entity_id` is the *user*, per configs/default.yaml, and a user may hold several
cards -- which is why `first_card_for_entity` is a feature: a dormant card waking
up is a different event from the same user's next purchase.

## The artifact group

This dataset carries the sharpest artifact in the project: validation frauds are
100% a single merchant country, and a one-line rule on `Merchant State` outscores
every model. `Merchant State`, `Merchant City` (which mirrors the country exactly
-- Rome, Algiers, Port au Prince), `Zip`, `Merchant Name` (a merchant id encodes
its own location) and the raw `MCC` are therefore kept only as `artifact_*`, along
with `Errors?`, whose Bad CVV and Bad Expiration values enrich 25.7x and 14.2x on
non-Italy rows.

What replaces them is the *relative* form: `same_state`, `same_city`,
`merchant_is_foreign` and the `first_*_for_entity` flags say "away from home" and
"somewhere this card has never been" without naming a country, so they should
survive the regime shift the Italy holdout isolates. `mcc_group` is kept outside
the artifact group deliberately: the ISO ranges are a hierarchy a model can
generalise over, where the raw code is 100+ unrelated tokens to memorise.

`errors_1h/24h/7d` are also outside the group. They count the entity's *prior*
failed attempts, which is card testing described as behaviour -- failed attempts
followed by a success -- and the left-closed window keeps this row's own outcome
out. Whether this row errored is `artifact_error_on_row`, and it is in the group,
because knowing a transaction was declined is close to knowing how it ended.

## Dropped outright, not even as artifacts

`Card Number`, `CVV`, `User` and `Person` are identifiers -- and `Card Number`
restates `Card`, so keeping it would smuggle a card identity past the `entity_id`
choice. `Address`, `Apartment`, `Zipcode`, `Latitude` and `Longitude` are the
cardholder's home, which has no relative form here: the source gives no merchant
coordinates, so there is no distance to compute, and `same_state`/`same_city`
already carry what the home address is for.

`Year`, `Month`, `Day` and `Time` are the absolute clock. The splits are temporal,
so a model given a year can isolate the split boundary as a threshold; the
cyclical parts that generalise forward are recomputed from `event_time` instead.
`Birth Year` and `Birth Month` go the same way, replaced by `age`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.experiments.features.util import (
    FEATURE_DIR,
    KEY_COLUMNS,
    EntityHistory,
    amount_shape,
    clock_features,
    days_between,
    parse_money,
    safe_ratio,
    write_features,
)

DATASET = "ibm_ccf"
DEFAULT_PROCESSED = Path("data/processed")

#: What "what the previous transaction looked like" means here, for the flattened
#: history in `experiments.history`. Deliberately short: all 82 features at 10 lags
#: would be 902 columns over 24.4M rows. These are the ones whose *previous value*
#: says something the current row does not -- where it was, what it cost, whether it
#: was declined -- rather than a static cardholder attribute, which is identical on
#: every lag and so pure waste in a lagged copy.
HISTORY_COLUMNS = (
    "amount_log1p",
    "seconds_since_prev_txn",
    "hour",
    "same_state",
    "same_city",
    "merchant_is_online",
    "merchant_is_foreign",
    "amount_over_credit_limit",
    "amount_over_entity_mean",
    "txn_count_24h",
    "errors_24h",
    "first_merchant_for_entity",
    "mcc_group",
    "use_chip",
)

#: "MM/YYYY" in `Acct Open Date` and `Expires`.
_MONTH_YEAR = "%m/%Y"


def _first_n_digits(num: int, n: int) -> int:
    return num // 10 ** (int(math.log10(num)) - n + 1)


def mcc_group(mcc) -> int:
    """Collapse a merchant category code to its ISO category group.

    Raw MCC treats 100+ codes as unrelated tokens; the ISO ranges are a hierarchy,
    so grouping gives a model something it can generalise over. The special cases
    inside 5000-5600 merge codes that name the same trade under different labels.
    """
    if mcc is None or (isinstance(mcc, float) and np.isnan(mcc)):
        return 0
    mcc = int(mcc)
    if mcc <= 0:
        return 0
    two = _first_n_digits(mcc, 2)
    if 0 < mcc < 1499 or 1500 <= mcc < 3000 or 4000 <= mcc < 4800 or 4800 <= mcc < 5000:
        return two
    if 3000 <= mcc < 3300:
        return 3000
    if 3300 <= mcc < 3500:
        return 3300
    if 3500 <= mcc < 4000:
        return 3500
    if 5000 <= mcc < 5600:
        if mcc in (5169, 5172):
            return 5169
        if mcc in (5192, 5122):
            return mcc
        if mcc in (5139, 5137):
            return 5139
        if mcc in (5111, 5044, 5021):
            return 5111
        if mcc in (5099, 5085, 5039, 5074):
            return 5099
        if mcc in (5094, 5047, 5013, 5051, 5046):
            return mcc
        if mcc in (5045, 5065, 5072):
            return 5045
        return two
    if 5600 <= mcc < 5700:
        return two
    if 5700 <= mcc < 7300:
        if 5815 <= mcc <= 5818:
            return 5815
        return _first_n_digits(mcc, 3)
    if 7300 <= mcc < 8000:
        return _first_n_digits(mcc, 3)
    if 8000 <= mcc < 10000:
        return two
    return mcc


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (keys, features) for a prepared IBM CCF frame."""
    f = pd.DataFrame(index=df.index)
    history = EntityHistory(df["entity_id"], df["event_time"])

    amount = parse_money(df["Amount"])
    magnitude = amount.abs()

    # --- when: cyclical clock parts only, never the absolute date
    f[["hour", "minute", "weekday", "day", "month", "is_weekend", "hour_sin",
       "hour_cos"]] = clock_features(df["event_time"])

    # --- how much, and the shape of the number
    f[["amount", "amount_log1p", "amount_is_refund", "amount_cents",
       "amount_is_round_10", "amount_is_round_100",
       "amount_is_micro"]] = amount_shape(amount)

    # --- how fast: the user's own recent volume, every window excluding this row
    f["txn_count_1h"] = history.rolling_count("1h")
    f["txn_count_24h"] = history.rolling_count("24h")
    f["txn_count_7d"] = history.rolling_count("7d")
    f["txn_count_30d"] = history.rolling_count("30d")
    f["amount_sum_24h"] = history.rolling_sum(magnitude, "24h")
    f["amount_sum_7d"] = history.rolling_sum(magnitude, "7d")
    f["amount_sum_30d"] = history.rolling_sum(magnitude, "30d")
    f["amount_mean_7d"] = history.rolling_mean(magnitude, "7d")
    f["amount_zscore_vs_7d"] = history.rolling_zscore(magnitude, "7d")
    f["seconds_since_prev_txn"] = history.gap_seconds()

    # Short window over long: "suddenly busy" rather than "busy".
    f["burst_1h_over_24h"] = safe_ratio(f["txn_count_1h"], f["txn_count_24h"])
    f["burst_24h_over_7d"] = safe_ratio(f["txn_count_24h"], f["txn_count_7d"])
    f["amount_24h_over_7d"] = safe_ratio(f["amount_sum_24h"], f["amount_sum_7d"])

    # --- how unlike this user's own baseline
    f["amount_over_entity_mean"] = safe_ratio(magnitude, history.prior_mean(magnitude))
    f["amount_over_entity_max"] = safe_ratio(magnitude, history.prior_max(magnitude))
    f["entity_txn_ordinal"] = history.ordinal()

    merchant = df["Merchant Name"].astype("string")
    mcc = df["MCC"].astype("string")
    state = df["Merchant State"].astype("string")
    city = df["Merchant City"].astype("string")
    channel = df["Use Chip"].astype("string")
    card = df["Card"].astype("string")

    # Foreign = a merchant state that is present and is not a 2-letter US code.
    is_foreign = ((state.str.len() != 2) & state.notna()).to_numpy().astype("float64")

    # --- first time ever, for this user: the fraud pattern stated relatively
    f["first_merchant_for_entity"] = history.first_occurrence(merchant)
    f["first_mcc_for_entity"] = history.first_occurrence(mcc)
    f["first_state_for_entity"] = history.first_occurrence(state)
    f["first_city_for_entity"] = history.first_occurrence(city)
    f["first_channel_for_entity"] = history.first_occurrence(channel)
    f["first_card_for_entity"] = history.first_occurrence(card)
    # Only the first foreign row counts: first_occurrence also fires on the first
    # domestic one, and multiplying by the flag keeps just the half that is meant.
    f["first_foreign_for_entity"] = history.first_occurrence(
        pd.Series(is_foreign, index=df.index)
    ) * is_foreign

    # --- how long since this user was last in this context
    f["secs_since_same_merchant"] = history.gap_since_same(merchant)
    f["secs_since_same_mcc"] = history.gap_since_same(mcc)
    f["secs_since_same_state"] = history.gap_since_same(state)
    f["secs_since_same_channel"] = history.gap_since_same(channel)
    f["secs_since_same_card"] = history.gap_since_same(card)

    # --- how wide: breadth of the user's recent and lifetime merchant set
    f["distinct_merchants_24h"] = history.rolling_distinct(merchant, "24h")
    f["prior_distinct_merchants"] = history.prior_distinct(merchant)

    # --- decline velocity: prior failed attempts, this row's outcome excluded
    errored = df["Errors?"].notna().astype("float64")
    f["errors_1h"] = history.rolling_sum(errored, "1h")
    f["errors_24h"] = history.rolling_sum(errored, "24h")
    f["errors_7d"] = history.rolling_sum(errored, "7d")

    # --- where, relative to home. No country is named.
    f["same_state"] = (
        (df["State"].astype("string") == state).fillna(False).to_numpy().astype("float64")
    )
    f["same_city"] = (
        (df["City"].astype("string") == city).fillna(False).to_numpy().astype("float64")
    )
    f["merchant_is_online"] = (city == "ONLINE").fillna(False).to_numpy().astype("float64")
    f["merchant_state_missing"] = state.isna().to_numpy().astype("float64")
    f["merchant_is_foreign"] = is_foreign

    # --- what kind of merchant and channel, as generalisable types
    # Mapped over the distinct codes rather than the rows: there are ~110 of the
    # former and 24.4M of the latter.
    groups = {code: mcc_group(code) for code in df["MCC"].unique()}
    f["mcc_group"] = df["MCC"].map(groups).astype("string")
    f["use_chip"] = channel

    # --- the card
    f["card_brand"] = df["Card Brand"].astype("string")
    f["card_type"] = df["Card Type"].astype("string")
    f["has_chip"] = df["Has Chip"].astype("string").eq("YES").to_numpy().astype("float64")
    f["card_on_dark_web"] = (
        df["Card on Dark Web"].astype("string").eq("Yes").to_numpy().astype("float64")
    )
    f["cards_issued"] = pd.to_numeric(df["Cards Issued"], errors="coerce")
    f["days_since_acct_open"] = days_between(
        df["event_time"], pd.to_datetime(df["Acct Open Date"], format=_MONTH_YEAR,
                                        errors="coerce")
    )
    f["days_to_expiry"] = days_between(
        pd.to_datetime(df["Expires"], format=_MONTH_YEAR, errors="coerce"),
        df["event_time"],
    )
    # The source records only the year of the last PIN change, so this is accurate
    # to a year and no better.
    f["days_since_pin_change"] = (
        df["event_time"].dt.year - pd.to_numeric(df["Year PIN last Changed"],
                                                 errors="coerce")
    ) * 365.25

    # --- the cardholder
    f["age"] = pd.to_numeric(df["Current Age"], errors="coerce")
    f["years_to_retirement"] = pd.to_numeric(df["Retirement Age"], errors="coerce") - f["age"]
    f["gender"] = df["Gender"].astype("string")
    f["fico"] = pd.to_numeric(df["FICO Score"], errors="coerce")
    f["num_cards"] = pd.to_numeric(df["Num Credit Cards"], errors="coerce")
    f["credit_limit"] = parse_money(df["Credit Limit"])
    f["total_debt"] = parse_money(df["Total Debt"])
    f["income_person"] = parse_money(df["Yearly Income - Person"])
    f["income_zip"] = parse_money(df["Per Capita Income - Zipcode"])
    f["debt_to_income"] = safe_ratio(f["total_debt"], f["income_person"])
    f["amount_over_credit_limit"] = safe_ratio(magnitude, f["credit_limit"])

    # --- the measured artifacts: kept, prefixed, droppable as a group
    f["artifact_merchant_name"] = merchant
    f["artifact_merchant_state"] = state
    f["artifact_merchant_city"] = city
    f["artifact_zip"] = df["Zip"].astype("string")
    f["artifact_customer_state"] = df["State"].astype("string")
    f["artifact_mcc"] = mcc
    f["artifact_error_type"] = df["Errors?"].astype("string").fillna("No Error")
    f["artifact_error_on_row"] = errored

    return df[list(KEY_COLUMNS)], f


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--features-dir", type=Path, default=FEATURE_DIR)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.processed_dir / DATASET / "data.parquet")
    keys, features = build(df)
    destination = write_features(
        DATASET, keys, features, features_dir=args.features_dir
    )
    print(f"{DATASET}: {len(features.columns)} features over {len(df):,} rows "
          f"-> {destination}")

    return destination


if __name__ == "__main__":
    main()
