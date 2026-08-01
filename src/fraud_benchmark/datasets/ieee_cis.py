"""IEEE-CIS / Vesta: real e-commerce transactions from a Kaggle competition.

https://www.kaggle.com/c/ieee-fraud-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleCompetition

TRAIN_TRANSACTION = "train_transaction.csv"
TRAIN_IDENTITY = "train_identity.csv"
TEST_TRANSACTION = "test_transaction.csv"
TEST_IDENTITY = "test_identity.csv"

SECONDS_PER_DAY = 86_400


def _require_start_date(options: dict[str, Any]) -> pd.Timestamp:
    start_date = options.get("start_date")
    if not start_date:
        raise ValueError(
            "ieee_cis requires a 'start_date' option to anchor its relative "
            "'TransactionDT' offsets; set datasets.ieee_cis.start_date in the config"
        )
    return pd.Timestamp(start_date)


def _load_with_identity(raw_dir: Path, transactions: str, identity: str) -> pd.DataFrame:
    df = pd.read_csv(require_file(raw_dir, transactions))
    ids = pd.read_csv(require_file(raw_dir, identity))
    return df.merge(ids, how="left", on="TransactionID")


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Derive a pseudo card identifier, falling back to a per-row unique id.

    IEEE-CIS has no card identifier. The community heuristic combines card1, addr1
    and a D1-derived account start day. Two things matter here:

    1. addr1 is null in ~11% of rows and D1 in a few thousand. Under pandas 3,
       astype(str) on NaN yields <NA> and concatenation propagates it, so the naive
       heuristic produces NULL entity_id values that fail schema validation.
    2. Bucketing those rows together under a shared "nan" key would be worse than
       useless — it would fabricate campaigns out of unrelated transactions.

    So rows missing any component get their own identity instead.
    """
    day = df["TransactionDT"] / SECONDS_PER_DAY
    account_start = (day - df["D1"]).round()
    uid = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].astype(str)
        + "_"
        + account_start.astype(str)
    )
    incomplete = df["addr1"].isna() | df["D1"].isna()
    return uid.where(~incomplete, "txn_" + df["TransactionID"].astype(str))


@register
class IeeeCisAdapter(DatasetAdapter):
    name = "ieee_cis"
    source = KaggleCompetition("ieee-fraud-detection")
    data_license = "Competition rules (research use)"
    commercial_use = False
    caveats = (
        "The only non-synthetic dataset here: real Vesta e-commerce transactions, "
        "heavily anonymised.",
        "entity_id is a DERIVED pseudo-identifier, not a real card id. It combines "
        "card1, addr1 and a D1-derived account start day — the well-known community "
        "'uid' heuristic. Rows missing addr1 or D1 (~11%) get a per-row unique id "
        "rather than being fused into a shared bucket.",
        "TransactionDT is a seconds offset with no stated origin, so event_time is "
        "anchored to a configured start_date and absolute dates carry no meaning.",
        "The competition test set has no labels and is therefore excluded from the "
        "benchmark; it is written separately as unlabelled_test.parquet.",
        "Identity data covers only about a quarter of transactions; the rest are null.",
        "Use is governed by the Kaggle competition rules, which must be accepted "
        "before download.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        anchor = _require_start_date(options)
        df = _load_with_identity(raw_dir, TRAIN_TRANSACTION, TRAIN_IDENTITY)

        event_time = anchor + pd.to_timedelta(df["TransactionDT"], unit="s")
        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", build_uid(df).astype("string"))
        df.insert(2, "amount", df["TransactionAmt"].astype("float64"))
        df.insert(3, "is_fraud", df["isFraud"].astype(bool))
        return df

    def auxiliary_frames(
        self, raw_dir: Path, options: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """The unlabelled competition test set, timestamped the same way."""
        anchor = _require_start_date(options)
        test = _load_with_identity(raw_dir, TEST_TRANSACTION, TEST_IDENTITY)
        test.insert(
            0, "event_time", anchor + pd.to_timedelta(test["TransactionDT"], unit="s")
        )
        test.insert(1, "entity_id", build_uid(test).astype("string"))
        return {"unlabelled_test": test}

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"TransactionDT (seconds since {options.get('start_date')})",
            "entity_id": "derived uid: card1 + addr1 + (day - D1)",
            "amount": "TransactionAmt",
            "is_fraud": "isFraud",
        }
