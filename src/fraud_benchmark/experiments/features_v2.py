"""Richer causal features: what a sequential model gets for free.

The flattened-window experiment showed lagged copies of the raw columns earn only
13% of a tree's gain. The plausible reason is that the useful content of a history
is not the previous rows themselves but *summaries relative to the entity's own
past* -- and a fixed positional slot cannot express "unlike anything this card has
done before".

So this builds those summaries directly. Seven groups, all strictly past-only:

1. Context novelty -- first time this entity has used this state / city / MCC /
   merchant, and first time it has gone abroad at all. "First foreign transaction
   ever for this card" is the fraud pattern in words, and it is *relative*, so
   unlike `Merchant State` it carries no country identity and should survive the
   regime shift.
2. Context recency -- seconds since this entity's last transaction in the same
   state, same MCC, and the same channel.
3. Decline velocity -- errors in the entity's last 1h / 24h / 7d. Card testing is
   failed attempts followed by a success; the audit found Bad CVV and Bad
   Expiration enriched 25.7x and 14.2x on non-Italy rows.
4. Burst ratios -- short window over long window. A ratio separates "busy card"
   from "suddenly busy card", which a raw count cannot.
5. Entity baseline deviation -- amount against this entity's own expanding mean
   and max, and how many transactions it has ever made.
6. Longer memory -- 30-day count and amount sum.
7. Amount shape and card age -- roundness, micro-amounts (the audit measured
   $0.01-$0.09 at 11-35x base), days since account opening and PIN change, days
   to expiry.

Every window is left-closed and every expanding statistic is shifted, so no
feature can see the row it describes. test_features_v2.py asserts that.

Outcome: **the hypothesis did not hold.** On trees these 26 features scored 0.0280
against v1's 0.0333 -- worse, by roughly four times the seed spread. The MLP
measurement is still running. Kept because the features are correct and cheap to
reuse, not because they helped. See docs/experiments.md.

Note the name: this module is a LIBRARY (no `main`). The experiment that uses it is
`ibm_features_v2.py`, one character away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

V2_COLUMNS = (
    # 1. context novelty
    "first_state_for_entity", "first_city_for_entity", "first_mcc_for_entity",
    "first_merchant_for_entity", "first_foreign_for_entity",
    # 2. context recency
    "secs_since_same_state", "secs_since_same_mcc", "secs_since_same_channel",
    # 3. decline velocity
    "errors_1h", "errors_24h", "errors_7d",
    # 4. burst ratios
    "burst_1h_over_24h", "burst_24h_over_7d", "amount_24h_over_7d",
    # 5. entity baseline deviation
    "amount_over_entity_mean", "amount_over_entity_max", "entity_txn_ordinal",
    # 6. longer memory
    "txn_count_30d", "amount_sum_30d",
    # 7. amount shape and card age
    "amount_is_round_10", "amount_is_round_100", "amount_cents", "amount_is_micro",
    "days_since_acct_open", "days_since_pin_change", "days_to_expiry",
)


def _scatter(values: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(n, dtype="float64")
    out[pos] = values
    return out


def _first_occurrence(work: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """1 on an entity's first transaction in a context, 0 after. Past-only by
    construction: cumcount counts preceding rows within the group."""
    return (work.groupby(keys, observed=True).cumcount() == 0).to_numpy().astype("float64")


def _seconds_since_same(work: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Gap to this entity's previous transaction sharing `keys`. NaN on the first."""
    return (
        work.groupby(keys, observed=True)["event_time"].diff().dt.total_seconds().to_numpy()
    )


def add_v2_features(df: pd.DataFrame, *, merchant_col: str, state_col: str,
                    city_col: str, mcc_col: str, channel_col: str,
                    error_col: str, amount_col: str,
                    acct_open_col: str | None = None,
                    pin_year_col: str | None = None,
                    expires_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    work = out[["entity_id", "event_time"]].copy()
    work["_pos"] = np.arange(n)
    work["amount"] = pd.to_numeric(out[amount_col], errors="coerce").abs()
    work["state"] = out[state_col].astype("string").fillna("~na")
    work["city"] = out[city_col].astype("string").fillna("~na")
    work["mcc"] = out[mcc_col].astype("string").fillna("~na")
    work["merchant"] = out[merchant_col].astype("string").fillna("~na")
    work["channel"] = out[channel_col].astype("string").fillna("~na")
    work["is_error"] = out[error_col].notna().astype("float64")
    # Foreign = a merchant state that is neither a 2-letter US code nor absent.
    work["is_foreign"] = (
        (work["state"].str.len() != 2) & (work["state"] != "~na")
    ).astype("float64")

    work = work.sort_values(["entity_id", "event_time"], kind="mergesort")
    pos = work["_pos"].to_numpy()

    def scatter(v):
        return _scatter(np.asarray(v, dtype="float64"), pos, n)

    # --- 1. context novelty
    out["first_state_for_entity"] = scatter(_first_occurrence(work, ["entity_id", "state"]))
    out["first_city_for_entity"] = scatter(_first_occurrence(work, ["entity_id", "city"]))
    out["first_mcc_for_entity"] = scatter(_first_occurrence(work, ["entity_id", "mcc"]))
    out["first_merchant_for_entity"] = scatter(
        _first_occurrence(work, ["entity_id", "merchant"])
    )
    # First time abroad, counted only over the foreign rows themselves.
    foreign_first = np.zeros(len(work), dtype="float64")
    fmask = work["is_foreign"].to_numpy() > 0
    if fmask.any():
        sub = work.loc[fmask]
        foreign_first[fmask] = (
            sub.groupby("entity_id", observed=True).cumcount() == 0
        ).to_numpy().astype("float64")
    out["first_foreign_for_entity"] = scatter(foreign_first)

    # --- 2. context recency
    out["secs_since_same_state"] = scatter(_seconds_since_same(work, ["entity_id", "state"]))
    out["secs_since_same_mcc"] = scatter(_seconds_since_same(work, ["entity_id", "mcc"]))
    out["secs_since_same_channel"] = scatter(
        _seconds_since_same(work, ["entity_id", "channel"])
    )

    # --- 3/6. rolling windows, left-closed so the current row is excluded
    indexed = work.set_index("event_time")
    by_entity = indexed.groupby("entity_id", observed=True)
    for label, window in (("1h", "1h"), ("24h", "24h"), ("7d", "7D")):
        rolled = by_entity["is_error"].rolling(window, closed="left").sum()
        out[f"errors_{label}"] = scatter(np.nan_to_num(rolled.to_numpy()))
    month = by_entity["amount"].rolling("30D", closed="left")
    out["txn_count_30d"] = scatter(np.nan_to_num(month.count().to_numpy()))
    out["amount_sum_30d"] = scatter(np.nan_to_num(month.sum().to_numpy()))

    # --- 4. burst ratios, from the counts the v1 builder already produced
    def ratio(a: str, b: str) -> np.ndarray:
        num = pd.to_numeric(out[a], errors="coerce").to_numpy(dtype="float64")
        den = pd.to_numeric(out[b], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        r[~np.isfinite(r)] = 0.0
        return r

    out["burst_1h_over_24h"] = ratio("txn_count_1h", "txn_count_24h")
    out["burst_24h_over_7d"] = ratio("txn_count_24h", "txn_count_7d")
    out["amount_24h_over_7d"] = ratio("amount_sum_24h", "amount_sum_7d")

    # --- 5. entity baseline: expanding statistics, shifted to exclude this row
    grp = work.groupby("entity_id", observed=True)["amount"]
    prior_sum = (grp.cumsum() - work["amount"]).to_numpy(dtype="float64")
    prior_count = work.groupby("entity_id", observed=True).cumcount().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = prior_sum / prior_count
    prior_mean[~np.isfinite(prior_mean)] = 0.0
    prior_max = grp.cummax().groupby(work["entity_id"], observed=True).shift(1).to_numpy(
        dtype="float64"
    )
    amount = work["amount"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        over_mean = amount / prior_mean
        over_max = amount / prior_max
    over_mean[~np.isfinite(over_mean)] = 0.0
    over_max[~np.isfinite(over_max)] = 0.0
    out["amount_over_entity_mean"] = scatter(over_mean)
    out["amount_over_entity_max"] = scatter(over_max)
    out["entity_txn_ordinal"] = scatter(prior_count.astype("float64"))

    # --- 7. amount shape
    amt = pd.to_numeric(out[amount_col], errors="coerce").abs().fillna(0.0)
    cents = (amt * 100).round().astype("int64") % 100
    out["amount_cents"] = cents.astype("float64")
    out["amount_is_round_10"] = ((amt % 10 == 0) & (amt > 0)).astype("float64")
    out["amount_is_round_100"] = ((amt % 100 == 0) & (amt > 0)).astype("float64")
    out["amount_is_micro"] = ((amt > 0) & (amt < 0.10)).astype("float64")

    # --- 7. card and account age
    event = out["event_time"]
    if acct_open_col is not None and acct_open_col in out.columns:
        opened = pd.to_datetime(out[acct_open_col], format="%m/%Y", errors="coerce")
        out["days_since_acct_open"] = (event - opened).dt.total_seconds() / 86400
    else:
        out["days_since_acct_open"] = np.nan
    if pin_year_col is not None and pin_year_col in out.columns:
        pin_year = pd.to_numeric(out[pin_year_col], errors="coerce")
        out["days_since_pin_change"] = (event.dt.year - pin_year) * 365.25
    else:
        out["days_since_pin_change"] = np.nan
    if expires_col is not None and expires_col in out.columns:
        expires = pd.to_datetime(out[expires_col], format="%m/%Y", errors="coerce")
        out["days_to_expiry"] = (expires - event).dt.total_seconds() / 86400
    else:
        out["days_to_expiry"] = np.nan

    return out
