"""Sparkov (Shenoy): simulated credit-card transactions with real timestamps.

https://www.kaggle.com/datasets/kartik2112/fraud-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_benchmark.data.adapters.base import DatasetAdapter, register
from fraud_benchmark.data.adapters.files import require_file
from fraud_benchmark.data.sources import KaggleDataset
from fraud_benchmark.data.splitting import SPLIT_NAMES, boundary_at

#: The bundle ships a pre-made temporal split. Both halves are labelled and do not
#: overlap (train ends 2020-06-21 12:13:37, test starts 2020-06-21 12:14:25).
TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"
SOURCE_FILES = (TRAIN_FILE, TEST_FILE)

#: Fraction of the upstream train file held back as validation.
DEFAULT_VAL_FRACTION = 0.1


@register
class SparkovAdapter(DatasetAdapter):
    name = "sparkov"
    source = KaggleDataset("kartik2112/fraud-detection")
    # The source column is already named is_fraud and is cast in place, so there is
    # nothing for the pipeline to drop.
    source_label_column = "is_fraud"
    data_license = "CC0 1.0"
    commercial_use = True
    caveats = (
        "Simulated with Sparkov/Faker. Customer names, addresses, jobs and dates of "
        "birth are fabricated, not real people.",
        "This dataset does NOT use the project's global temporal split. The upstream "
        "test file is preserved as the test split so results stay comparable with "
        "published work, and validation is the last 10% of the upstream train file "
        "(configurable via datasets.sparkov.val_fraction). Splits are therefore "
        "roughly 63/7/30 rather than 80/10/10.",
        "The source file each row came from is kept in 'source_file'.",
        "The per-file row index column ('Unnamed: 0') is dropped as meaningless after "
        "concatenation.",
    )

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        parts = []
        for filename in SOURCE_FILES:
            part = pd.read_csv(require_file(raw_dir, filename))
            part["source_file"] = filename
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

        # The source already has a column called is_fraud, so cast it in place
        # rather than inserting a second one.
        df["is_fraud"] = df["is_fraud"].astype(bool)
        df.insert(0, "event_time", pd.to_datetime(df["trans_date_trans_time"]))
        df.insert(1, "entity_id", df["cc_num"].astype("string"))
        df.insert(2, "amount", df["amt"].astype("float64"))
        return df.sort_values("event_time", kind="stable").reset_index(drop=True)

    def custom_splits(
        self, df: pd.DataFrame, options: dict[str, Any]
    ) -> pd.Series:
        """Preserve the upstream test set; carve validation from the train tail.

        The two source files are consecutive in time, so train -> val -> test remains
        strictly ordered.
        """
        val_fraction = float(options.get("val_fraction", DEFAULT_VAL_FRACTION))
        is_test = df["source_file"] == TEST_FILE
        train_part = df.loc[~is_test]
        cut = boundary_at(train_part, 1.0 - val_fraction)

        labels = np.where(
            is_test,
            "test",
            np.where(df["event_time"] <= cut, "train", "val"),
        )
        return pd.Series(
            pd.Categorical(labels, categories=SPLIT_NAMES),
            index=df.index,
            name="split",
        )

    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        return {
            "event_time": "trans_date_trans_time",
            "entity_id": "cc_num",
            "amount": "amt",
            "is_fraud": "is_fraud",
        }


@register
class SparkovSlowAdapter(SparkovAdapter):
    """Sparkov again, differing only in reporting delay.

    Row-identical to `sparkov` on every column but `reported_at`, so a model can
    be compared across delay regimes on the same data -- the same design as the
    two IBM CCF subsamples.

    Why it exists: at the card-fraud default (7-day median) Sparkov's 487-day
    train window leaves 97.8% of train labels known at the cutoff, so the delay
    barely registers. The slow regime censors 8.9%, enough for a delay-aware
    method to have something to work with. Its delay is set in
    configs/default.yaml, not here; see docs/label-delay.md for the measured
    figures across every dataset.
    """

    name = "sparkov_slow"
    #: Reads the raw files already downloaded for sparkov.
    raw_name = "sparkov"
    caveats = SparkovAdapter.caveats + (
        "Row-identical to 'sparkov' apart from reported_at. It exists to give the "
        "label-delay axis a second, harsher regime on identical data.",
        "Its delay is deliberately slower than the card-fraud default: the median "
        "(15 days) is still plausible for a cardholder noticing on a statement, but "
        "the tail is stretched well past a realistic chargeback window to make the "
        "censoring measurable. Treat it as a stress test, not as a realistic "
        "reporting regime.",
    )
