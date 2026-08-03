"""PaySim: an agent-based mobile money transaction simulator.

https://www.kaggle.com/datasets/ealaxi/paysim1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import (
    DatasetAdapter,
    register,
    require_start_date,
)
from fraud_benchmark.datasets.files import find_single_csv
from fraud_benchmark.sources import KaggleDataset


@register
class PaySimAdapter(DatasetAdapter):
    name = "paysim"
    source = KaggleDataset("ealaxi/paysim1")
    data_license = "CC BY-SA 4.0"
    commercial_use = True
    caveats = (
        "PaySim is fully synthetic. Its 'step' column is a 1-based hour offset, not a "
        "real date, so event_time is anchored to a configured start_date and the "
        "absolute dates carry no meaning.",
        "Timestamps have hourly granularity, so many transactions share an event_time.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        anchor = require_start_date(options, "paysim", "step")

        df = pd.read_csv(find_single_csv(raw_dir))

        # PaySim already has a column literally named `amount`, so cast it in place
        # rather than inserting a second one.
        df["amount"] = df["amount"].astype("float64")
        df.insert(0, "event_time", anchor + pd.to_timedelta(df["step"] - 1, unit="h"))
        df.insert(1, "entity_id", df["nameOrig"].astype("string"))
        df.insert(2, "is_fraud", df["isFraud"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": f"step (hours since {options.get('start_date')})",
            "entity_id": "nameOrig",
            "amount": "amount",
            "is_fraud": "isFraud",
        }
