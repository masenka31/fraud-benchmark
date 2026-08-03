"""BankSim: an agent-based retail-payment simulator.

https://www.kaggle.com/datasets/ealaxi/banksim1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.data.adapters.base import (
    DatasetAdapter,
    register,
    require_start_date,
)
from fraud_benchmark.data.adapters.files import require_file
from fraud_benchmark.data.sources import KaggleDataset

#: The transaction table. The bundle also ships bsNET140513_032310.csv, which is a
#: graph edge list (Source/Target/Weight) rather than transactions.
TRANSACTIONS_FILE = "bs140513_032310.csv"


@register
class BankSimAdapter(DatasetAdapter):
    name = "banksim"
    source = KaggleDataset("ealaxi/banksim1")
    source_label_column = "fraud"
    data_license = "CC BY-NC-SA 4.0"
    commercial_use = False
    caveats = (
        "BankSim is fully synthetic. Its 'step' column is a 0-based day offset, not a "
        "real date, so event_time is anchored to a configured start_date and the "
        "absolute dates carry no meaning.",
        "Timestamps have daily granularity, so many transactions share an event_time.",
        "Every string column in the source file is wrapped in literal single quotes; "
        "the adapter strips them.",
        "zipcodeOri and zipMerchant are constant ('28007') in the source data.",
        "Licence is CC BY-NC-SA 4.0: NonCommercial and ShareAlike.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        anchor = require_start_date(options, "banksim", "step")

        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))

        # Every string column arrives quoted, e.g. "'C1093826151'".
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]):
                df[column] = df[column].str.strip("'")

        df["amount"] = df["amount"].astype("float64")
        df.insert(0, "event_time", anchor + pd.to_timedelta(df["step"], unit="D"))
        df.insert(1, "entity_id", df["customer"].astype("string"))
        df.insert(2, "is_fraud", df["fraud"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"step (days since {options.get('start_date')})",
            "entity_id": "customer",
            "amount": "amount",
            "is_fraud": "fraud",
        }
