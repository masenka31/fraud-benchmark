"""IBM CCF with every Italy fraud held out of training.

The question: can a model trained only on non-Italy fraud detect the Italy fraud
that makes up almost all of val and test? This isolates the regime shift the
audit found, deliberately rather than as an accident of where the temporal cut
happened to land.

Split. Train ends one second before the first Italy fraud (2017-11-19 12:06), so
it contains zero of them by construction. The left crop is then chosen to make
the ratio exactly 80/10/10: 2009-09-12 onward gives 13,350,884 train rows against
a 3,337,721-row tail, halved into val and test. Still strictly temporal -- no row
is moved across time to achieve this, only the boundaries are placed.

Features follow the user's own IBM pipeline where it is better than ours:
hierarchical MCC grouping rather than raw codes, same_state/same_city (relative,
so they capture "away from home" without naming a country), clock features, and
signed-log1p before robust scaling for the heavy-tailed magnitudes. Categorical
vocabularies are capped the same way: values up to 99% cumulative frequency, at
most 256, everything else pooled -- fitted on train only, which means Italy is
pooled into the rare bucket even where it appears.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_benchmark.experiments.metrics import best_f1_threshold, score

FIRST_ITALY_FRAUD = pd.Timestamp("2017-11-19 12:06:00")
LAST_LABELLED_FRAUD = pd.Timestamp("2019-10-27 14:54:00")
TRAIN_ROWS = 13_350_884          # gives exactly 80/10/10 against the tail
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


def build_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal split whose train half predates the first Italy fraud."""
    df = df.sort_values("event_time", kind="mergesort").reset_index(drop=True)
    cut = FIRST_ITALY_FRAUD - pd.Timedelta(seconds=1)
    head = df[df["event_time"] < cut]
    tail = df[(df["event_time"] >= cut) & (df["event_time"] <= LAST_LABELLED_FRAUD)]

    train = head.iloc[-TRAIN_ROWS:]
    half = len(tail) // 2
    val, test = tail.iloc[:half], tail.iloc[half:]

    out = pd.concat([train, val, test], ignore_index=True)
    out["split"] = (
        ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
    )
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--out", type=Path, default=Path("results/italy_holdout.jsonl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.features_dir / "ibm_ccf.parquet")
    df = build_split(df)

    italy = (df["Merchant State"].astype("string") == "Italy").fillna(False)
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        fr = m & df["is_fraud"]
        print(f"{sp:5s} rows={int(m.sum()):>10,} frauds={int(fr.sum()):>6,} "
              f"italy_frauds={int((fr & italy).sum()):>6,} "
              f"span {df.loc[m,'event_time'].min().date()}..{df.loc[m,'event_time'].max().date()}",
              flush=True)
    assert int((( df['split']=='train') & df['is_fraud'] & italy).sum()) == 0, \
        "train must contain no Italy frauds"

    x, names = build_features(df)
    y = df["is_fraud"].to_numpy().astype(int)
    tr, va, te = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy(), (df["split"] == "test").to_numpy()
    print(f"\n{len(names)} features; train positives {int(y[tr].sum()):,}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gains: list[dict] = []
    for seed in (0, 1, 2):
        started = time.monotonic()
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=4, eval_metric="aucpr", random_state=seed,
            scale_pos_weight=float((y[tr] == 0).sum() / max(y[tr].sum(), 1)),
        )
        model.fit(x[tr], y[tr])
        pv, pt = model.predict_proba(x[va])[:, 1], model.predict_proba(x[te])[:, 1]
        th = best_f1_threshold(y[va], pv)
        rec = {
            "experiment": "italy_holdout",
            "seed": seed,
            "n_features": len(names),
            "fit_seconds": round(time.monotonic() - started, 1),
            "scores": {"val": score(y[va], pv, th), "test": score(y[te], pt, th)},
            # total_gain is keyed by the DataFrame's column names; features the
            # trees never split on are simply absent.
            "importance_gain": dict(
                model.get_booster().get_score(importance_type="total_gain")
            ),
        }
        gains.append(rec["importance_gain"])
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed={seed} val_AP={rec['scores']['val']['average_precision']:.4f} "
              f"test_AP={rec['scores']['test']['average_precision']:.4f}", flush=True)

    total = {}
    for g in gains:
        for k, v in g.items():
            total[k] = total.get(k, 0.0) + v / len(gains)
    share = sum(total.values()) or 1.0
    print("\ntop 20 features by mean total gain:", flush=True)
    for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:28s} {v/share*100:6.2f}%", flush=True)


if __name__ == "__main__":
    main()
