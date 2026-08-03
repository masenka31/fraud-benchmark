"""Flattened sequence window: target transaction + its previous 9, same user.

Two deliberate departures from "just concatenate 10x everything":

* Customer-static columns (FICO, credit limit, incomes, age, card count, gender,
  home state) are constant per user, so lagging them yields nine identical copies.
  They appear ONCE, for the target row. Repeating them would waste memory and
  dilute colsample_bytree across duplicates.
* Velocity aggregates are already summaries of the same history the window now
  carries explicitly, so they too stay on the target row only.

Lag direction is the one thing that must not be wrong: lag k is the k-th
PREVIOUS transaction of that user. Shifting the other way leaks the future, which
would raise the score and raise no error -- hence test_lag_matrix_is_causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_benchmark.experiments.ibm_features import _money, _signed_log1p, encode_mcc


N_LAGS = 9

# Per-transaction: meaningful to look back at.
DYNAMIC = [
    "amount_signed", "amount_usd", "hour", "weekday",
    "same_state", "same_city", "merchant_is_online", "merchant_state_missing",
    "cat_use_chip", "cat_mcc_encoded", "cat_merchant_state", "cat_error",
    "merchant_novelty", "log_seconds_since_prev",
]
# Constant per user, or already a summary of the window: target row only.
STATIC = [
    "minute", "month", "day", "credit_limit", "total_debt", "income_person",
    "income_zip", "fico", "num_cards", "age", "cat_card_brand", "cat_card_type",
    "cat_gender", "cat_customer_state",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_24h", "amount_sum_7d", "amount_mean_7d", "amount_zscore_vs_7d",
]

RARE_COVERAGE = 0.99
MAX_LEVELS = 256


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per transaction, before any lagging."""
    f = pd.DataFrame(index=df.index)
    hhmm = df["Time"].astype("string").str.split(":", n=1, expand=True)
    f["hour"] = pd.to_numeric(hhmm[0], errors="coerce").fillna(0)
    f["minute"] = pd.to_numeric(hhmm[1], errors="coerce").fillna(0)
    f["weekday"] = df["event_time"].dt.weekday
    f["month"] = df["event_time"].dt.month
    f["day"] = df["event_time"].dt.day

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

    f["same_state"] = (df["State"].astype("string") == df["Merchant State"].astype("string")).fillna(False)
    f["same_city"] = (df["City"].astype("string") == df["Merchant City"].astype("string")).fillna(False)
    f["merchant_is_online"] = (df["Merchant City"].astype("string") == "ONLINE").fillna(False)
    f["merchant_state_missing"] = df["Merchant State"].isna()

    for c in ["txn_count_1h", "txn_count_24h", "txn_count_7d", "amount_sum_24h",
              "amount_sum_7d", "amount_mean_7d", "amount_zscore_vs_7d"]:
        f[c] = df[c]
    f["log_seconds_since_prev"] = _signed_log1p(
        df["seconds_since_prev_txn"].fillna(-1).to_numpy()
    )
    f["merchant_novelty"] = df["merchant_novelty"]

    cats = {
        "use_chip": df["Use Chip"].astype("string"),
        "mcc_encoded": df["MCC"].map(encode_mcc).astype("string"),
        "merchant_state": df["Merchant State"].astype("string").fillna("unknown"),
        "error": df["Errors?"].astype("string").fillna("No Error"),
        "card_brand": df["Card Brand"].astype("string"),
        "card_type": df["Card Type"].astype("string"),
        "gender": df["Gender"].astype("string"),
        "customer_state": df["State"].astype("string"),
    }
    is_train = (df["split"] == "train").to_numpy()
    for name, values in cats.items():
        counts = values[is_train].value_counts(normalize=True)
        keep = counts.cumsum().shift(1).fillna(0.0) < RARE_COVERAGE
        vocab = list(counts[keep].index[:MAX_LEVELS])
        lookup = {v: i for i, v in enumerate(vocab)}
        f[f"cat_{name}"] = values.map(lookup).fillna(len(vocab))

    for c in ["amount_signed", "amount_usd", "credit_limit", "total_debt",
              "income_person", "income_zip", "amount_sum_24h", "amount_sum_7d",
              "amount_mean_7d"]:
        t = _signed_log1p(f[c].fillna(0.0).to_numpy())
        q25, q50, q75 = np.quantile(t[is_train], [0.25, 0.5, 0.75])
        f[c] = (t - q50) / ((q75 - q25) + 1e-8)

    return f.fillna(0.0).astype("float32")


def lag_matrix(values: np.ndarray, user: np.ndarray, n_lags: int) -> np.ndarray:
    """Concatenate lag 1..n_lags of `values` along axis 1, zero where unavailable.

    `values` and `user` must be sorted by (user, time). Lag k of row i is row i-k,
    kept only when row i-k belongs to the same user -- so a user's first rows get
    zeros rather than another user's transactions.
    """
    n, width = values.shape
    out = np.zeros((n, width * n_lags), dtype="float32")
    for k in range(1, n_lags + 1):
        same_user = np.zeros(n, dtype=bool)
        same_user[k:] = user[k:] == user[:-k]
        rows = np.flatnonzero(same_user)
        out[rows, (k - 1) * width : k * width] = values[rows - k]
    return out


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return (X, feature names, y, split labels), all aligned to `df`'s order."""
    order = np.lexsort((df["event_time"].to_numpy(), df["User"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)

    f = base_features(df)
    user = df["User"].to_numpy()
    dyn = f[DYNAMIC].to_numpy(dtype="float32")

    blocks = [f[DYNAMIC + STATIC].to_numpy(dtype="float32"),
              lag_matrix(dyn, user, N_LAGS)]
    names = [f"t0_{c}" for c in DYNAMIC + STATIC]
    names += [f"t-{k}_{c}" for k in range(1, N_LAGS + 1) for c in DYNAMIC]

    x = np.concatenate(blocks, axis=1)
    return x, names, df["is_fraud"].to_numpy().astype(int), df["split"].to_numpy()
