"""The IBM CCF feature pipeline.

Features follow the user's own IBM pipeline where it is better than ours:
hierarchical MCC grouping rather than raw codes, same_state/same_city (relative,
so they capture "away from home" without naming a country), clock features, and
signed-log1p before robust scaling for the heavy-tailed magnitudes. Categorical
vocabularies are capped the same way: values up to 99% cumulative frequency, at
most 256, everything else pooled -- fitted on train only, which means Italy is
pooled into the rare bucket even where it appears.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


RARE_COVERAGE = 0.99
MAX_LEVELS = 256


def _first_n_digits(num: int, n: int) -> int:
    return num // 10 ** (int(math.log10(num)) - n + 1)


def encode_mcc(mcc) -> int:
    """Collapse an MCC to its category group. Ported from the user's pipeline.

    Raw MCC treats 100+ codes as unrelated tokens; the ISO ranges are a hierarchy,
    so grouping gives the model something it can generalise over.
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
        if mcc == 5192 or mcc == 5122:
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
        if mcc < 5815:
            return _first_n_digits(mcc, 3)
        if 5815 <= mcc <= 5818:
            return 5815
        return _first_n_digits(mcc, 3)
    if 7300 <= mcc < 8000:
        return _first_n_digits(mcc, 3)
    if 8000 <= mcc < 10000:
        return two
    return mcc


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def _money(series: pd.Series) -> pd.Series:
    """'$123.45' -> 123.45. The IBM source keeps money as strings."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the feature frame and the ordered feature names."""
    f = pd.DataFrame(index=df.index)

    # --- clock: cyclical only, never the absolute date
    hhmm = df["Time"].astype("string").str.split(":", n=1, expand=True)
    f["hour"] = pd.to_numeric(hhmm[0], errors="coerce").fillna(0).astype("int16")
    f["minute"] = pd.to_numeric(hhmm[1], errors="coerce").fillna(0).astype("int16")
    f["weekday"] = df["event_time"].dt.weekday.astype("int16")
    f["month"] = df["event_time"].dt.month.astype("int16")
    f["day"] = df["event_time"].dt.day.astype("int16")

    # --- amounts and balances, signed-log1p then robust-scaled on train
    amount = _money(df["Amount"])
    f["amount_signed"] = amount
    f["amount_usd"] = amount.abs()
    f["credit_limit"] = _money(df["Credit Limit"])
    f["total_debt"] = _money(df["Total Debt"])
    f["income_person"] = _money(df["Yearly Income - Person"])
    f["income_zip"] = _money(df["Per Capita Income - Zipcode"])
    f["fico"] = pd.to_numeric(df["FICO Score"], errors="coerce")
    f["num_cards"] = pd.to_numeric(df["Num Credit Cards"], errors="coerce")
    f["age"] = pd.to_numeric(df["Current Age"], errors="coerce")

    # --- relative geography: away-from-home without naming a country
    f["same_state"] = (
        df["State"].astype("string") == df["Merchant State"].astype("string")
    ).fillna(False).astype("int8")
    f["same_city"] = (
        df["City"].astype("string") == df["Merchant City"].astype("string")
    ).fillna(False).astype("int8")
    f["merchant_is_online"] = (
        df["Merchant City"].astype("string") == "ONLINE"
    ).fillna(False).astype("int8")
    f["merchant_state_missing"] = df["Merchant State"].isna().astype("int8")

    # --- velocity, already causal from the main pipeline
    for c in [
        "txn_count_1h", "txn_count_24h", "txn_count_7d",
        "amount_sum_24h", "amount_sum_7d", "amount_mean_7d",
        "amount_zscore_vs_7d", "seconds_since_prev_txn", "merchant_novelty",
    ]:
        f[c] = df[c]
    f["seconds_since_prev_txn"] = f["seconds_since_prev_txn"].fillna(-1)

    # --- categoricals, vocabularies fitted on train only
    f["mcc_encoded"] = df["MCC"].map(encode_mcc).astype("int32")
    cats = {
        "mcc_encoded": f["mcc_encoded"].astype("string"),
        "use_chip": df["Use Chip"].astype("string"),
        "card_brand": df["Card Brand"].astype("string"),
        "card_type": df["Card Type"].astype("string"),
        "error": df["Errors?"].astype("string").fillna("No Error"),
        "gender": df["Gender"].astype("string"),
        "merchant_state": df["Merchant State"].astype("string").fillna("unknown"),
        "customer_state": df["State"].astype("string"),
    }
    is_train = (df["split"] == "train").to_numpy()
    for name, values in cats.items():
        counts = values[is_train].value_counts(normalize=True)
        keep = counts.cumsum().shift(1).fillna(0.0) < RARE_COVERAGE
        vocab = list(counts[keep].index[:MAX_LEVELS])
        lookup = {v: i for i, v in enumerate(vocab)}
        f[f"cat_{name}"] = values.map(lookup).fillna(len(vocab)).astype("int32")

    # heavy-tailed magnitudes: compress, then robust-scale on train statistics
    for c in [
        "amount_signed", "amount_usd", "credit_limit", "total_debt",
        "income_person", "income_zip",
        "amount_sum_24h", "amount_sum_7d", "amount_mean_7d",
        "seconds_since_prev_txn",
    ]:
        t = _signed_log1p(f[c].fillna(0.0).to_numpy())
        q25, q50, q75 = np.quantile(t[is_train], [0.25, 0.5, 0.75])
        f[c] = (t - q50) / ((q75 - q25) + 1e-8)

    return f.astype("float32"), list(f.columns)
