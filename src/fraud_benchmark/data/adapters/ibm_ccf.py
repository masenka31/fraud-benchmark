"""IBM CCF (Altman): a large synthetic credit-card transaction log.

https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.data.adapters.base import DatasetAdapter
from fraud_benchmark.data.adapters.base import register
from fraud_benchmark.data.adapters.files import require_file
from fraud_benchmark.data.sources import KaggleDataset

TRANSACTIONS_FILE = "credit_card_transactions-ibm_v2.csv"
CARDS_FILE = "sd254_cards.csv"
USERS_FILE = "sd254_users.csv"

#: Columns in the joined tables that are money strings such as "$24295".
MONEY_COLUMNS = (
    "Credit Limit",
    "Per Capita Income - Zipcode",
    "Yearly Income - Person",
    "Total Debt",
    "Amount",
)


def _parse_money(series: pd.Series) -> pd.Series:
    """Parse $-prefixed money strings such as "$134.09" or "$-25.00" to float64."""
    return series.astype(str).str.replace("$", "", regex=False).astype("float64")


@register
class IbmCcfAdapter(DatasetAdapter):
    name = "ibm_ccf"
    source = KaggleDataset("ealtman2019/credit-card-transactions")
    source_label_column = "Is Fraud?"
    # Two upstream sources disagree, so the more specific one wins: the dataset
    # description body states Apache-2.0, while Kaggle's licence field reads
    # CC BY 4.0. Both permit commercial use, so `commercial_use` is unaffected
    # either way. Recorded as a caveat below so the card carries the ambiguity
    # rather than hiding it. See docs/dataset-licenses.md.
    data_license = "Apache-2.0"
    commercial_use = True
    caveats = (
        "Licence provenance is ambiguous: the dataset description body states "
        "Apache-2.0, Kaggle's licence field reads CC BY 4.0. Both allow commercial "
        "use. Verify upstream before relying on either for a publication.",
        "Fully synthetic. The bundled cardholder details — names, addresses, card "
        "numbers, CVVs — are fabricated and do not describe real people.",
        "All 11 card columns and all 18 user columns are left-joined onto every "
        "transaction, so static attributes repeat across a user's rows.",
        "sd254_users.csv has no identifier column; it is joined positionally, its row "
        "index being the User id.",
        "Money columns in the joined tables ($-prefixed strings) are parsed to float.",
        "Timestamps have minute granularity; the source has no seconds.",
        "entity_key defaults to 'user'. A user may hold several cards, so set it to "
        "'card' to key on the individual card instead.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        entity_key = options.get("entity_key", "user")
        if entity_key not in ("user", "card"):
            raise ValueError(f"ibm_ccf entity_key must be 'user' or 'card', got {entity_key!r}")

        df = pd.read_csv(require_file(raw_dir, TRANSACTIONS_FILE))
        cards = pd.read_csv(require_file(raw_dir, CARDS_FILE))
        users = pd.read_csv(require_file(raw_dir, USERS_FILE))

        # sd254_users.csv carries no id; its row order is the User id.
        users = users.copy()
        users.insert(0, "User", range(len(users)))

        df = df.merge(
            cards, how="left", left_on=["User", "Card"], right_on=["User", "CARD INDEX"]
        ).drop(columns=["CARD INDEX"])
        df = df.merge(users, how="left", on="User")

        for column in MONEY_COLUMNS:
            if column in df.columns:
                df[column] = _parse_money(df[column])

        event_time = pd.to_datetime(
            df["Year"].astype(str)
            + "-"
            + df["Month"].astype(str).str.zfill(2)
            + "-"
            + df["Day"].astype(str).str.zfill(2)
            + " "
            + df["Time"].astype(str).str.zfill(5),
            format="%Y-%m-%d %H:%M",
        )

        if entity_key == "user":
            entity = df["User"].astype("string")
        else:
            entity = (df["User"].astype(str) + "-" + df["Card"].astype(str)).astype("string")

        amount = df["Amount"]
        is_fraud = df["Is Fraud?"].astype(str).str.strip().eq("Yes")

        df.insert(0, "event_time", event_time)
        df.insert(1, "entity_id", entity)
        df.insert(2, "amount", amount)
        df.insert(3, "is_fraud", is_fraud)
        return df

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        entity_key = options.get("entity_key", "user")
        return {
            "event_time": "Year + Month + Day + Time",
            "entity_id": "User" if entity_key == "user" else "User + Card",
            "amount": "Amount",
            "is_fraud": "Is Fraud?",
        }
