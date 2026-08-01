"""SAML-D: a synthetic anti-money-laundering transaction monitoring dataset.

https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import DatasetAdapter, register
from fraud_benchmark.datasets.files import require_file
from fraud_benchmark.sources import KaggleDataset

TRANSACTIONS_FILE = "SAML-D.csv"

#: Date is always YYYY-MM-DD and Time always HH:MM:SS in the source (verified across
#: real rows). Pinning the format makes a malformed or changed upstream file fail
#: loudly instead of being silently reinterpreted.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@register
class SamlDAdapter(DatasetAdapter):
    name = "saml_d"
    source = KaggleDataset(
        "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
    )
    data_license = "CC BY-NC-SA 4.0"
    commercial_use = False
    caveats = (
        "Synthetic AML data. The label is Is_laundering, and 'fraud' here means a "
        "laundering typology rather than card fraud.",
        "Amounts span 13 currencies (Payment_currency, Received_currency) and are NOT "
        "converted; cross-row amount comparison is not meaningful.",
        "entity_id is the sending account. Laundering is a multi-party phenomenon, so "
        "the receiving account (Receiver_account) matters too and is passed through.",
        "Laundering_type records the typology for both laundering and normal rows "
        "(e.g. Smurfing, Normal_Fan_Out) and is retained.",
        "Licence is CC BY-NC-SA 4.0: NonCommercial and ShareAlike.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))

        event_time = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format=TIMESTAMP_FORMAT,
        )
        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", df["Sender_account"].astype("string"))
        df.insert(2, "amount", df["Amount"].astype("float64"))
        df.insert(3, "is_fraud", df["Is_laundering"].astype(bool))
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "Date + Time",
            "entity_id": "Sender_account",
            "amount": "Amount",
            "is_fraud": "Is_laundering",
        }
