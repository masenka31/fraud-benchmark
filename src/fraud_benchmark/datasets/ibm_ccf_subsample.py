"""IBM CCF cropped to a recent, fully-labelled window.

Two measured artifacts in the full dataset motivate this.

Labelling stops on 2019-10-27 while transactions continue to 2020-02-28, so the
final 645,180 rows carry no fraud at all. Cropping on the last transaction would
put the entire test split inside that dead zone — 0 frauds in test, measured.
The right edge is therefore the last labelled fraud, found in the data rather
than hardcoded.

And the full span is 10,649 days, against which a realistic reporting delay
censors nothing: 100.0% of train labels are known at the train cutoff. Cropping
to 2016-01-01 onward gives 6.57M rows and 8,412 frauds, where the same delay
censors a real 3%.

The two registered variants below are row-identical and differ only in the delay
distribution configured for them, so a model can be compared across delay
regimes on the same data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.datasets.base import register
from fraud_benchmark.datasets.ibm_ccf import IbmCcfAdapter

#: Artifacts documented in docs/verification-notes.md, recorded on every card.
SUBSAMPLE_CAVEATS = (
    "Cropped to [start_date, last labelled fraud]. IBM CCF's fraud labelling "
    "stops on 2019-10-27 while transactions run to 2020-02-28; those final "
    "645,180 rows are dropped because they are unlabelled, not fraud-free.",
    "2017 carries 255 frauds against roughly 3,000 in each neighbouring year — "
    "very likely a second labelling artifact. It falls inside the train split, "
    "so it thins training data without affecting validation or test.",
    "Merchant State is close to a label here: in this window 'Italy' is 76.8% "
    "fraud and accounts for 55.7% of all frauds. Do not hand it to a model as a "
    "raw feature without understanding that.",
)


class IbmCcfSubsampleAdapter(IbmCcfAdapter):
    """Shared crop. Not registered: the concrete variants below are.

    Inherits the two table joins, the money-string parsing and the entity_key
    logic from IbmCcfAdapter; only the window is new.
    """

    #: Both variants read the raw files already downloaded for ibm_ccf.
    raw_name = "ibm_ccf"
    caveats = IbmCcfAdapter.caveats + SUBSAMPLE_CAVEATS

    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        if "start_date" not in options:
            raise ValueError(
                f"{self.name} requires a 'start_date' option giving the left edge "
                "of the window"
            )
        df = super().to_canonical(raw_dir, options)

        # Found in the data, not hardcoded: the point is to end where labelling
        # ends, whatever date that turns out to be.
        last_fraud = df.loc[df["is_fraud"], "event_time"].max()
        if pd.isna(last_fraud):
            raise ValueError(
                f"{self.name}: the source has no fraudulent rows, so the window "
                "has no right edge"
            )
        start = pd.Timestamp(options["start_date"])
        if start > last_fraud:
            raise ValueError(
                f"{self.name}: start_date {start.date()} is after the last "
                f"labelled fraud {last_fraud.date()}, leaving an empty window"
            )

        window = (df["event_time"] >= start) & (df["event_time"] <= last_fraud)
        return df.loc[window].reset_index(drop=True)


@register
class IbmCcfSubsampleFastAdapter(IbmCcfSubsampleAdapter):
    """Median 7 / mean 30 day reporting delay. Parameters in configs/default.yaml."""

    name = "ibm_ccf_subsample_fast"


@register
class IbmCcfSubsampleSlowAdapter(IbmCcfSubsampleAdapter):
    """Median 15 / mean 60 day reporting delay. Parameters in configs/default.yaml."""

    name = "ibm_ccf_subsample_slow"
