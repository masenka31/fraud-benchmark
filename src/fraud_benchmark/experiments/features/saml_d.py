"""SAML-D features: 9.5M synthetic AML transactions between named accounts.

Run as `python -m fraud_benchmark.experiments.features.saml_d`, reads
`data/processed/saml_d/data.parquet`, writes `data/features/saml_d.parquet`.

## Why this file looks different from the card datasets

Card fraud happens to one entity; laundering happens *between* two. `entity_id` is
the sending account, but half the signal is on the receiving side, so this module
builds two histories -- one keyed on the sender, one on the receiver -- and the
features that matter are the shapes those two make together:

    fan-out   `distinct_receivers_24h/7d` -- how many counterparties one account
              pays inside a window
    fan-in    `receiver_distinct_senders_7d`, `receiver_in_count_24h/7d` -- how
              many accounts converge on one

Fan-in is not expressible as a per-row aggregate over the sender alone, which is
why the generic velocity block is the smaller half of this file.

**Both separate in the opposite direction from the textbook story, and the reason
is in how the dataset was generated.** Measured over all 9.5M rows, laundering rows
average 2.90 distinct receivers in 7 days against 6.96 for normal rows, and 1.45
distinct senders per receiver against 3.72. Smurfing would predict the reverse. The
explanation is that 61% of the *normal* rows are generated as explicit fan patterns
-- `Normal_Small_Fan_Out` 3.48M, `Normal_Fan_Out` 2.30M, `Normal_Fan_In` 2.10M --
so the background traffic is fan-heavy by construction, while the 9,873 laundering
rows spread across a dozen typologies of which only some are fan-shaped
(`Structuring` 1,870, `Smurfing` 932, `Layered_Fan_In` 656).

The features are kept, because the gap is large and a model does not care about its
sign, and `receiver_sender_concentration_7d` (0.48 against 0.27) points the way the
narrative expects. But do not read a high fan-out here as evidence of laundering,
and do not carry that reading to another dataset: it is a property of this
generator, not of laundering.

## One feature does most of the work, and it is a generator property

Measured: XGBoost on these 63 features reaches 0.9919 test average precision, against
0.500 for the retired ablation's raw columns. `first_receiver_for_entity` earns **35% of
the total gain** and `payment_type` 23%.

The novelty flag is causally clean -- `EntityHistory.first_occurrence` counts only
preceding rows, and the tests assert it -- but it is predictive here largely by
construction: the laundering typologies create fresh sender->receiver pairs while normal
traffic recurs against established ones. Keep it, because a real AML system would use it
too, but do not read the score as evidence that the task is easy in general.

## Amounts are not comparable across rows

The source spans 13 currencies (`Payment_currency`, `Received_currency`) and does
not convert them. Absolute `amount` is therefore close to meaningless between two
arbitrary rows, and the weight falls on the relative forms -- `amount_over_entity_mean`,
`amount_zscore_vs_7d`, `amount_over_entity_max` -- which compare an account only
against itself and so stay inside one currency in practice. `amount` and
`amount_log1p` are kept anyway; they are cheap, and within a single account they
are informative.

`amount_just_under_10k` is the structuring pattern -- a payment sized to sit below
a reporting threshold. The threshold is stated in one currency, so on the others
this is noise rather than signal; it is included because structuring is one of the
typologies the dataset actually contains and no relative form expresses it.

## The artifact group

The leakage audit found `Sender_bank_location` and `Receiver_bank_location`
directional and plausible for laundering -- the phenomenon rather than an artifact
-- so unlike IBM CCF's geography these are not known to be poisoned. They are
still absolute places, so they carry the prefix and the same downstream filter
handles them: `artifact_sender_bank_location`, `artifact_receiver_bank_location`,
`artifact_receiver_account`. What replaces them in the main set is relative --
`is_cross_border`, `first_receiver_country_for_entity`,
`secs_since_same_receiver_country`.

Currencies stay outside the group. A currency names a monetary zone rather than a
place, is type-level rather than an identity, and cross-currency movement is
itself an AML signal (`is_currency_conversion`).

## Dropped outright

`Laundering_type` is not a feature. It records the typology behind the label --
Smurfing, Normal_Fan_Out -- and is a restatement of `is_fraud` for the positive
rows. The adapter declares it label-descriptive, so `write_features` rejects it if
it ever appears here.

`Sender_account` restates `entity_id`, and `Date` and `Time` are the absolute
clock, which the temporal splits make a boundary detector rather than a feature;
the cyclical parts are recomputed from `event_time`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_benchmark.experiments.features.util import (
    FEATURE_DIR,
    KEY_COLUMNS,
    EntityHistory,
    amount_shape,
    clock_features,
    safe_ratio,
    write_features,
)

DATASET = "saml_d"
DEFAULT_PROCESSED = Path("data/processed")

#: The lagged history for `experiments.history`. Both sides are represented: a
#: sequence of transfers is a laundering pattern only in terms of who received the
#: previous ones and how concentrated they were.
HISTORY_COLUMNS = (
    "amount_log1p",
    "seconds_since_prev_txn",
    "hour",
    "distinct_receivers_7d",
    "new_receiver_rate_7d",
    "receiver_distinct_senders_7d",
    "receiver_sender_concentration_7d",
    "is_cross_border",
    "is_currency_conversion",
    "first_receiver_for_entity",
    "payment_type",
)

#: The currency-transaction reporting threshold structuring is sized against, and
#: the band below it that counts as "just under". Stated in one currency; see the
#: module docstring for what that costs on the other twelve.
REPORTING_THRESHOLD = 10_000.0
JUST_UNDER_BAND = 0.10


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (keys, features) for a prepared SAML-D frame."""
    f = pd.DataFrame(index=df.index)

    sender = EntityHistory(df["entity_id"], df["event_time"])
    # The second key. Laundering is a two-sided phenomenon, and a receiver's
    # inbound history is not derivable from any sender's outbound one.
    receiver_side = EntityHistory(df["Receiver_account"].astype("string"), df["event_time"])

    amount = pd.to_numeric(df["Amount"], errors="coerce")
    magnitude = amount.abs()

    receiver = df["Receiver_account"].astype("string")
    sender_id = df["entity_id"].astype("string")
    receiver_country = df["Receiver_bank_location"].astype("string")
    sender_country = df["Sender_bank_location"].astype("string")
    payment_type = df["Payment_type"].astype("string")
    paid_currency = df["Payment_currency"].astype("string")
    got_currency = df["Received_currency"].astype("string")

    # --- when: cyclical clock parts only, never the absolute Date
    f[["hour", "minute", "weekday", "day", "month", "is_weekend", "hour_sin", "hour_cos"]] = (
        clock_features(df["event_time"])
    )

    # --- how much. Cross-row comparison is not meaningful; see the docstring.
    f[
        [
            "amount",
            "amount_log1p",
            "amount_is_refund",
            "amount_cents",
            "amount_is_round_10",
            "amount_is_round_100",
            "amount_is_micro",
        ]
    ] = amount_shape(amount)
    lower = REPORTING_THRESHOLD * (1 - JUST_UNDER_BAND)
    f["amount_just_under_10k"] = (
        magnitude.between(lower, REPORTING_THRESHOLD, inclusive="left").to_numpy().astype("float64")
    )

    # --- how fast this account sends, every window excluding this row
    f["txn_count_1h"] = sender.rolling_count("1h")
    f["txn_count_24h"] = sender.rolling_count("24h")
    f["txn_count_7d"] = sender.rolling_count("7d")
    f["txn_count_30d"] = sender.rolling_count("30d")
    f["amount_sum_24h"] = sender.rolling_sum(magnitude, "24h")
    f["amount_sum_7d"] = sender.rolling_sum(magnitude, "7d")
    f["amount_sum_30d"] = sender.rolling_sum(magnitude, "30d")
    f["amount_mean_7d"] = sender.rolling_mean(magnitude, "7d")
    f["amount_zscore_vs_7d"] = sender.rolling_zscore(magnitude, "7d")
    f["seconds_since_prev_txn"] = sender.gap_seconds()

    f["burst_1h_over_24h"] = safe_ratio(f["txn_count_1h"], f["txn_count_24h"])
    f["burst_24h_over_7d"] = safe_ratio(f["txn_count_24h"], f["txn_count_7d"])
    f["amount_24h_over_7d"] = safe_ratio(f["amount_sum_24h"], f["amount_sum_7d"])

    # --- how unlike this account's own baseline
    f["amount_over_entity_mean"] = safe_ratio(magnitude, sender.prior_mean(magnitude))
    f["amount_over_entity_max"] = safe_ratio(magnitude, sender.prior_max(magnitude))
    f["entity_txn_ordinal"] = sender.ordinal()

    # --- fan-out: how many counterparties, over a bounded window and ever
    f["distinct_receivers_24h"] = sender.rolling_distinct(receiver, "24h")
    f["distinct_receivers_7d"] = sender.rolling_distinct(receiver, "7d")
    f["prior_distinct_receivers"] = sender.prior_distinct(receiver)
    f["distinct_receiver_countries_7d"] = sender.rolling_distinct(receiver_country, "7d")
    # New counterparty share: a burst of *new* accounts is the pattern, where a
    # burst to the same account is a standing arrangement.
    f["new_receiver_rate_7d"] = safe_ratio(f["distinct_receivers_7d"], f["txn_count_7d"])

    # --- fan-in: the same picture from the receiving account's side
    f["receiver_in_count_24h"] = receiver_side.rolling_count("24h")
    f["receiver_in_count_7d"] = receiver_side.rolling_count("7d")
    f["receiver_in_amount_7d"] = receiver_side.rolling_sum(magnitude, "7d")
    f["receiver_distinct_senders_7d"] = receiver_side.rolling_distinct(sender_id, "7d")
    f["receiver_prior_distinct_senders"] = receiver_side.prior_distinct(sender_id)
    f["receiver_txn_ordinal"] = receiver_side.ordinal()
    f["receiver_amount_over_mean"] = safe_ratio(magnitude, receiver_side.prior_mean(magnitude))
    # A concentration ratio: many senders relative to volume is collection, one
    # sender relative to volume is a salary.
    f["receiver_sender_concentration_7d"] = safe_ratio(
        f["receiver_distinct_senders_7d"], f["receiver_in_count_7d"]
    )

    # --- first time ever, for this sending account
    f["first_receiver_for_entity"] = sender.first_occurrence(receiver)
    f["first_receiver_country_for_entity"] = sender.first_occurrence(receiver_country)
    f["first_payment_type_for_entity"] = sender.first_occurrence(payment_type)
    f["first_currency_pair_for_entity"] = sender.first_occurrence(
        paid_currency + "->" + got_currency
    )
    # And for the receiving account: a dormant account starting to collect.
    f["first_sender_for_receiver"] = receiver_side.first_occurrence(sender_id)

    # --- how long since this account was last in this context
    f["secs_since_same_receiver"] = sender.gap_since_same(receiver)
    f["secs_since_same_receiver_country"] = sender.gap_since_same(receiver_country)
    f["secs_since_same_payment_type"] = sender.gap_since_same(payment_type)
    f["secs_since_receiver_last_paid"] = receiver_side.gap_seconds()

    # --- where the money crosses, stated relatively
    f["is_cross_border"] = (sender_country != receiver_country).to_numpy().astype("float64")
    f["is_currency_conversion"] = (paid_currency != got_currency).to_numpy().astype("float64")
    f["same_currency"] = (paid_currency == got_currency).to_numpy().astype("float64")

    # --- how the money moves, as generalisable types
    f["payment_type"] = payment_type
    f["payment_currency"] = paid_currency
    f["received_currency"] = got_currency

    # --- absolute place and specific counterparty
    f["artifact_sender_bank_location"] = sender_country
    f["artifact_receiver_bank_location"] = receiver_country
    f["artifact_receiver_account"] = receiver

    return df[list(KEY_COLUMNS)], f


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--features-dir", type=Path, default=FEATURE_DIR)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.processed_dir / DATASET / "data.parquet")
    keys, features = build(df)
    destination = write_features(DATASET, keys, features, features_dir=args.features_dir)
    print(f"{DATASET}: {len(features.columns)} features over {len(df):,} rows -> {destination}")

    return destination


if __name__ == "__main__":
    main()
