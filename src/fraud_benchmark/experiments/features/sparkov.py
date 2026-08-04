"""Sparkov features: 1.85M simulated card transactions with real coordinates.

Run as `python -m fraud_benchmark.experiments.features.sparkov`, reads
`data/processed/sparkov/data.parquet` *and* `data/processed/sparkov_slow/data.parquet`,
writes `data/features/sparkov.parquet`.

## Three label regimes, one file

The two preparations are row-identical apart from `reported_at`, so building
features twice would write 220 MB of duplicate columns to say one thing. This
module reads both and carries both timestamps, joined on `trans_num`:

    no label delay    ignore both -- every label known the moment it happens
    label delay       `reported_at`, the 7-day card-fraud default (2.2% of train
                      labels censored at the cutoff)
    slow label delay  `reported_at_slow`, median 15 days with a tail stretched
                      past a realistic chargeback window (8.9% censored)

The regime is therefore chosen at training time, by picking a column, rather than
by picking an input file.

## What this dataset has that the others do not

Coordinates on both sides -- the cardholder's home (`lat`, `long`) and the
merchant's (`merch_lat`, `merch_long`) -- so `distance_from_home_km` is a real
distance rather than a same/different flag, and `distance_over_entity_mean` asks
the more useful question of whether *this* card has ever shopped that far out.

## The artifact group

Sparkov is the project's negative control: the leakage audit found no flagged
values in it at all. The `artifact_*` columns here are the same *kind* of column
as IBM CCF's -- a specific merchant, an absolute place -- prefixed for the same
reason and so that a downstream filter can treat both datasets identically. There
is no measured artifact behind them.

`category`, `job` and `gender` stay outside the group: they are type-level, name
no place and no counterparty, and are what a real detector would use.

Dropped outright: `first`, `last`, `street` (fabricated personal details with no
relative form), `cc_num` (restates `entity_id`), `trans_num` (a row id, used here
only to align the two preparations), `unix_time` and `trans_date_trans_time` (the
absolute clock, which separates the temporal splits exactly -- train max
1367492969 < val min 1367493022), and `source_file`, which predicts the split
perfectly because the splits come from separate upstream files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.experiments.features.util import (
    FEATURE_DIR,
    KEY_COLUMNS,
    EntityHistory,
    FeatureContractError,
    amount_shape,
    clock_features,
    days_between,
    haversine_km,
    safe_ratio,
    write_features,
)

DATASET = "sparkov"

#: The preparation that supplies `reported_at_slow`. Row-identical to `sparkov`
#: on every other column; see configs/default.yaml for the two delay regimes.
SLOW_DATASET = "sparkov_slow"

#: Aligns the two preparations. A per-transaction hash in the source, so unique --
#: asserted rather than assumed, since a silent many-to-many join would multiply rows.
JOIN_KEY = "trans_num"

DEFAULT_PROCESSED = Path("data/processed")

#: The lagged history for `experiments.history`: what changed between this
#: transaction and the card's last few, not what is static about the cardholder.
HISTORY_COLUMNS = (
    "amount_log1p",
    "seconds_since_prev_txn",
    "hour",
    "distance_from_home_km",
    "distance_over_entity_mean",
    "amount_over_entity_mean",
    "txn_count_24h",
    "first_merchant_for_entity",
    "category",
)


def attach_slow_delay(df: pd.DataFrame, slow: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the `sparkov_slow` report timestamp as `reported_at_slow`."""
    if not slow[JOIN_KEY].is_unique:
        raise FeatureContractError(
            f"{SLOW_DATASET}: {JOIN_KEY} is not unique, so it cannot align the two "
            "delay regimes"
        )
    lookup = slow.set_index(JOIN_KEY)["reported_at"]

    # Checked on membership rather than on the resulting nulls: `reported_at` is NaT
    # on every non-fraud row anyway, so a non-fraud row that found no match would
    # look exactly like a successful join.
    unmatched = int((~df[JOIN_KEY].isin(lookup.index)).sum())
    if unmatched:
        raise FeatureContractError(
            f"{SLOW_DATASET}: {unmatched} row(s) of {DATASET} found no match on "
            f"{JOIN_KEY}; the two preparations are meant to be row-identical"
        )

    out = df.copy()
    out["reported_at_slow"] = out[JOIN_KEY].map(lookup)
    return out


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (keys, features) for a prepared Sparkov frame carrying both regimes."""
    f = pd.DataFrame(index=df.index)
    history = EntityHistory(df["entity_id"], df["event_time"])

    amount = pd.to_numeric(df["amt"], errors="coerce")
    magnitude = amount.abs()

    # --- when: cyclical clock parts only, never the absolute date
    f[["hour", "minute", "weekday", "day", "month", "is_weekend", "hour_sin",
       "hour_cos"]] = clock_features(df["event_time"])

    # --- how much, and the shape of the number
    f[["amount", "amount_log1p", "amount_is_refund", "amount_cents",
       "amount_is_round_10", "amount_is_round_100",
       "amount_is_micro"]] = amount_shape(amount)

    # --- how fast: the card's own recent volume, every window excluding this row
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

    f["burst_1h_over_24h"] = safe_ratio(f["txn_count_1h"], f["txn_count_24h"])
    f["burst_24h_over_7d"] = safe_ratio(f["txn_count_24h"], f["txn_count_7d"])
    f["amount_24h_over_7d"] = safe_ratio(f["amount_sum_24h"], f["amount_sum_7d"])

    # --- how unlike this card's own baseline
    f["amount_over_entity_mean"] = safe_ratio(magnitude, history.prior_mean(magnitude))
    f["amount_over_entity_max"] = safe_ratio(magnitude, history.prior_max(magnitude))
    f["entity_txn_ordinal"] = history.ordinal()

    merchant = df["merchant"].astype("string")
    category = df["category"].astype("string")

    # --- first time ever, for this card
    f["first_merchant_for_entity"] = history.first_occurrence(merchant)
    f["first_category_for_entity"] = history.first_occurrence(category)

    # --- how long since this card was last in this context
    f["secs_since_same_merchant"] = history.gap_since_same(merchant)
    f["secs_since_same_category"] = history.gap_since_same(category)

    # --- how wide: breadth of the card's recent and lifetime merchant set
    f["distinct_merchants_24h"] = history.rolling_distinct(merchant, "24h")
    f["distinct_categories_7d"] = history.rolling_distinct(category, "7d")
    f["prior_distinct_merchants"] = history.prior_distinct(merchant)

    # --- how far from home, the one thing only this dataset can answer
    distance = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    f["distance_from_home_km"] = distance
    f["distance_over_entity_mean"] = safe_ratio(
        distance, history.prior_mean(pd.Series(distance, index=df.index))
    )
    f["distance_over_entity_max"] = safe_ratio(
        distance, history.prior_max(pd.Series(distance, index=df.index))
    )

    # --- what kind of purchase, and who the cardholder is
    f["category"] = category
    f["job"] = df["job"].astype("string")
    f["gender"] = df["gender"].astype("string")
    f["age_at_txn"] = days_between(
        df["event_time"], pd.to_datetime(df["dob"], errors="coerce")
    ) / 365.25
    # Population is heavy-tailed across four orders of magnitude; the log is the
    # scale on which "small town" and "city" are a step apart rather than a ratio.
    f["city_pop_log"] = np.log1p(
        pd.to_numeric(df["city_pop"], errors="coerce").fillna(0.0).to_numpy()
    )

    # --- absolute place and specific counterparty. No measured artifact behind
    # them here -- Sparkov is the negative control -- but the same kind of column.
    f["artifact_merchant"] = merchant
    f["artifact_customer_state"] = df["state"].astype("string")
    f["artifact_customer_city"] = df["city"].astype("string")
    f["artifact_zip"] = df["zip"].astype("string")

    keys = df[[*KEY_COLUMNS, "reported_at_slow"]]
    return keys, f


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--features-dir", type=Path, default=FEATURE_DIR)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.processed_dir / DATASET / "data.parquet")
    slow = pd.read_parquet(
        args.processed_dir / SLOW_DATASET / "data.parquet",
        columns=[JOIN_KEY, "reported_at"],
    )
    df = attach_slow_delay(df, slow)

    keys, features = build(df)
    destination = write_features(
        DATASET,
        keys,
        features,
        features_dir=args.features_dir,
        extra_keys=("reported_at_slow",),
    )
    print(f"{DATASET}: {len(features.columns)} features over {len(df):,} rows, "
          f"both delay regimes -> {destination}")

    return destination


if __name__ == "__main__":
    main()
