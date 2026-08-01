"""Amaretto: a synthetic capital-market dataset for money-laundering detection.

https://github.com/necst/amaretto_dataset

Unlike the other datasets here this is securities trading, not payments: clients
buy and sell instruments on a market.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_split_zip_member
from fraud_benchmark.sources import GitRepo

ARCHIVE_PARTS = "amaretto_dataset_anon.zip.*"
ARCHIVE_MEMBER = "amaretto_dataset_anon.csv"

#: Where the extracted CSV is cached, relative to the raw directory. Kept inside
#: data/raw so it is gitignored and survives between runs.
EXTRACT_DIR = "_extracted"


@register
class AmarettoAdapter(DatasetAdapter):
    name = "amaretto"
    source = GitRepo("https://github.com/necst/amaretto_dataset")
    data_license = "MIT"
    commercial_use = True
    caveats = (
        "Capital-market trading data, not payments: rows are securities buy/sell "
        "orders, so 'amount' is a normalised trade value rather than a transfer.",
        "The label column Anomaly is NOT binary. It is 0 plus five classes matching "
        "the FATF typologies described upstream; is_fraud is Anomaly > 0 and the "
        "class itself is retained.",
        "Originator_ID is the constant '_XID' in every row and carries no information.",
        "Distributed as a 34-part split zip inside a git repository; the adapter "
        "reassembles and extracts it once, caching the result under data/raw.",
        "Fully synthetic, built from aggregate real market parameters.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        csv_path = require_split_zip_member(
            raw_dir, ARCHIVE_PARTS, ARCHIVE_MEMBER, raw_dir / EXTRACT_DIR
        )
        df = pd.read_csv(csv_path)

        df.insert(0, "event_time", pd.to_datetime(df["EntryDate"]))
        df.insert(1, "entity_id", df["Originator"].astype("string"))
        df.insert(2, "amount", df["Normalized Amount"].astype("float64"))
        df.insert(3, "is_fraud", df["Anomaly"].gt(0))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "EntryDate",
            "entity_id": "Originator",
            "amount": "Normalized Amount",
            "is_fraud": "Anomaly > 0",
        }
