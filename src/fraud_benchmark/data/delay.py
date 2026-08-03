"""Synthetic label-availability delay.

Each campaign is discovered once, some time after its last fraudulent transaction,
and every transaction in it is labelled at that moment. This module samples that
delay and turns it into a `reported_at` timestamp.

Lognormal rather than Poisson: Poisson models a count of events with variance tied
to its mean, whereas reporting delay is a continuous, strongly right-skewed waiting
time — most frauds surface within days, a minority take months.

The delay is calibrated per domain, not per dataset, and whatever censoring results
is reported rather than tuned. docs/label-delay.md has the shipped parameters and the
measured censoring for all eight datasets, plus which of them the delay bites on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPORTED_AT_COLUMN = "reported_at"


@dataclass(frozen=True)
class DelayParams:
    """Parameters of the lognormal reporting delay, in days.

    `median_days` is the distribution's median, i.e. exp(mu); `sigma` is the
    log-scale spread; `max_delay_days` truncates the draw when set.
    """

    median_days: float
    sigma: float
    seed: int
    max_delay_days: float | None = None

    def __post_init__(self) -> None:
        if self.median_days <= 0:
            raise ValueError(f"median_days must be positive, got {self.median_days}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.max_delay_days is not None and self.max_delay_days <= 0:
            raise ValueError(
                f"max_delay_days must be positive, got {self.max_delay_days}"
            )


def assign_reported_at(df: pd.DataFrame, params: DelayParams) -> pd.Series:
    """reported_at per row as datetime64[us], NaT where `campaign_id` is null.

    Requires `campaign_id` (from `campaigns.assign_campaigns`) and `event_time`;
    `is_fraud` is never read, since a null campaign id already means "not
    reportable". One delay is drawn per campaign and added to that campaign's last
    event_time, so a campaign is never reported before it has finished.

    Raises ValueError if a draw is too large to represent as a microsecond offset.
    """
    reported = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[us]", name=REPORTED_AT_COLUMN
    )
    labelled = df["campaign_id"].notna()
    if not labelled.any():
        return reported

    # groupby sorts by key, so the draws line up with campaign ids regardless of
    # row order — which is what makes a given seed reproducible.
    campaign_end = (
        df.loc[labelled].groupby("campaign_id", observed=True)["event_time"].max()
    )

    rng = np.random.default_rng(params.seed)
    days = rng.lognormal(
        mean=math.log(params.median_days), sigma=params.sigma, size=len(campaign_end)
    )
    if params.max_delay_days is not None:
        days = np.minimum(days, params.max_delay_days)

    # Offsets are built in microseconds to match the [us] resolution used
    # throughout: pd.to_timedelta(..., unit="D") promotes to [ns], whose range is
    # only ~1677-2262 and overflows on heavy-tailed draws.
    offset_micros = days * 86_400_000_000.0
    if (
        not np.isfinite(offset_micros).all()
        or offset_micros.max() >= float(np.iinfo("int64").max)
    ):
        raise ValueError(
            "sampled reporting delay exceeds the representable range; set "
            f"max_delay_days to bound it (largest draw was {days.max():,.0f} days)"
        )
    per_campaign = campaign_end + offset_micros.astype("int64").astype("timedelta64[us]")
    mapped = df.loc[labelled, "campaign_id"].map(per_campaign.to_dict())
    reported.loc[labelled] = pd.to_datetime(mapped).astype("datetime64[us]")
    return reported
