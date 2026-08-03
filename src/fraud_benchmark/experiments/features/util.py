"""Primitives the per-dataset feature modules are built from.

Nothing here declares a feature. `EntityHistory` answers questions about an
entity's own past, the free functions do arithmetic, and `write_features` enforces
the parquet contract. Which windows over which columns is always decided in the
dataset module, so that reading that one file tells you the whole feature set.

Every aggregate is strictly past-only. `EntityHistory` sorts once by
(entity, event_time), computes with left-closed windows or shifted expanding
statistics, and scatters results back into the caller's row order -- so a feature
can never see the row it describes. Included windows made every measurement in
this project look better rather than noisier, which is exactly the failure mode
nothing downstream would have caught.

Aggregates are computed over the full timeline rather than per split. Per-split
computation gives every split a cold-start artifact at its left edge, where
entities look new only because the window was truncated there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.experiments.columns import assert_no_excluded

#: Where feature parquets are written, and where the model side reads them.
FEATURE_DIR = Path("data/features")

#: Carried by every feature parquet, in this order, ahead of the features.
#: `reported_at` is NaT on non-frauds; `split` comes from the preparation that
#: produced the frame and is a default, not the authority (see `experiments.splits`).
KEY_COLUMNS = ("entity_id", "event_time", "reported_at", "is_fraud", "split")

#: Keys only some datasets carry. Listed here rather than passed around so that
#: `feature_columns` gives the right answer on any parquet without being told which
#: dataset it came from. `reported_at_slow` is Sparkov's second delay regime.
OPTIONAL_KEY_COLUMNS = ("reported_at_slow",)

#: Prefix for a column naming an absolute place or a specific counterparty --
#: kept, but droppable as a group by a prefix test.
ARTIFACT_PREFIX = "artifact_"

#: Label -> pandas offset alias. The labels name the output columns and stay
#: lowercase; the offsets must use "D", since lowercase "d" is deprecated.
WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7D", "30d": "30D"}


def _offset(window: str) -> str:
    """Resolve a window label such as '7d' to the pandas offset alias."""
    try:
        return WINDOWS[window]
    except KeyError:
        raise ValueError(
            f"unknown window {window!r}; known windows are {sorted(WINDOWS)}"
        ) from None


class EntityHistory:
    """Past-only aggregates over one entity's own transaction history.

    Constructed once per (entity key, frame) and reused for every window, because
    the sort it needs is the expensive part: IBM CCF is 24.4M rows and SAML-D 9.5M.
    A dataset with a second meaningful key -- SAML-D's receiving account, where
    fan-in is the phenomenon -- builds a second instance on that key.

    Every method returns a float64 numpy array in the *input* frame's row order,
    so a dataset module reads as a list of plain assignments.
    """

    def __init__(self, entity: pd.Series, event_time: pd.Series) -> None:
        if len(entity) != len(event_time):
            raise ValueError("entity and event_time must have the same length")
        self._n = len(entity)

        work = pd.DataFrame(
            {
                "entity": entity.to_numpy(),
                "event_time": pd.to_datetime(event_time).to_numpy(),
                "_pos": np.arange(self._n),
            }
        )
        # mergesort is stable, so rows sharing an (entity, time) tie keep their
        # input order. Coarse timestamps make those ties common: IBM CCF has
        # minute resolution and no seconds at all.
        self._work = work.sort_values(["entity", "event_time"], kind="mergesort")
        self._pos = self._work["_pos"].to_numpy()
        self._time = self._work["event_time"].to_numpy()
        self._group_key = self._work["entity"]
        self._indexed = self._work.set_index("event_time")

    def __len__(self) -> int:
        return self._n

    def _scatter(self, values) -> np.ndarray:
        """Sorted-order values back into input order."""
        out = np.empty(self._n, dtype="float64")
        out[self._pos] = np.asarray(values, dtype="float64")
        return out

    def _sorted(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Caller-order values into this history's sorted order."""
        return np.asarray(values, dtype="float64")[self._pos]

    def _rolled(self, values: np.ndarray, window: str):
        frame = self._indexed.assign(_v=values)
        return frame.groupby("entity", observed=True)["_v"].rolling(
            _offset(window), closed="left"
        )

    # --- counts and sums over a trailing window -------------------------------

    def rolling_count(self, window: str) -> np.ndarray:
        """Transactions by this entity in the trailing `window`, excluding this one.

        An empty left-closed window counts as NaN rather than 0, so a row with no
        prior history would otherwise carry NaN into every count column.
        """
        ones = np.ones(self._n, dtype="float64")
        counts = self._rolled(ones, window).count().to_numpy()
        return self._scatter(np.nan_to_num(counts))

    def rolling_sum(self, values: pd.Series | np.ndarray, window: str) -> np.ndarray:
        """Sum of `values` over this entity's trailing `window`. 0.0 with no history."""
        rolled = self._rolled(self._sorted(values), window).sum().to_numpy()
        return self._scatter(np.nan_to_num(rolled))

    def rolling_mean(self, values: pd.Series | np.ndarray, window: str) -> np.ndarray:
        """Mean of `values` over the trailing `window`. NaN with no history."""
        return self._scatter(self._rolled(self._sorted(values), window).mean().to_numpy())

    def rolling_zscore(
        self, values: pd.Series | np.ndarray, window: str
    ) -> np.ndarray:
        """`values` against the mean and sd of this entity's trailing `window`.

        Zero rather than NaN when the history is empty or flat: "no evidence of
        deviation" is the honest reading, and it keeps the column dense.
        """
        ordered = self._sorted(values)
        rolled = self._rolled(ordered, window)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (ordered - rolled.mean().to_numpy()) / rolled.std().to_numpy()
        z[~np.isfinite(z)] = 0.0
        return self._scatter(z)

    # --- expanding statistics over everything before this row -----------------

    def ordinal(self) -> np.ndarray:
        """How many transactions this entity made before this one. 0 on the first."""
        counts = self._work.groupby("entity", observed=True).cumcount().to_numpy()
        return self._scatter(counts.astype("float64"))

    def prior_mean(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Mean of `values` over this entity's entire past. 0.0 on the first row."""
        ordered = self._sorted(values)
        grouped = pd.Series(ordered).groupby(
            self._group_key.to_numpy(), observed=True
        )
        prior_sum = grouped.cumsum().to_numpy() - ordered
        prior_count = self._work.groupby("entity", observed=True).cumcount().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = prior_sum / prior_count
        mean[~np.isfinite(mean)] = 0.0
        return self._scatter(mean)

    def prior_max(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Maximum of `values` over this entity's entire past. NaN on the first row."""
        grouped = pd.Series(self._sorted(values)).groupby(
            self._group_key.to_numpy(), observed=True
        )
        return self._scatter(grouped.cummax().groupby(
            self._group_key.to_numpy(), observed=True
        ).shift(1).to_numpy())

    def prior_distinct(self, values: pd.Series) -> np.ndarray:
        """Distinct values of `values` this entity saw before this row.

        The cumulative sum of first-occurrence flags, shifted -- so "counterparties
        this account has ever paid" costs one groupby rather than a nunique.
        """
        firsts = self._first_flags(values)
        cumulative = pd.Series(firsts).groupby(
            self._group_key.to_numpy(), observed=True
        ).cumsum().to_numpy()
        return self._scatter(cumulative - firsts)

    # --- context: has this entity seen this value before, and when -------------

    def _keyed(self, values: pd.Series) -> pd.DataFrame:
        """This history's sorted frame with `values` attached as a grouping key."""
        keys = values.astype("string").fillna("~na").to_numpy()[self._pos]
        return self._work.assign(_key=keys)

    def _first_flags(self, values: pd.Series) -> np.ndarray:
        """Sorted-order 1.0 on an entity's first transaction with this value."""
        keyed = self._keyed(values)
        counted = keyed.groupby(["entity", "_key"], observed=True).cumcount()
        return (counted == 0).to_numpy().astype("float64")

    def first_occurrence(self, values: pd.Series) -> np.ndarray:
        """1.0 the first time this entity sees this value, 0.0 after.

        "First transaction this card has ever made in this country" is the fraud
        pattern in words, and it is *relative*, so unlike the country itself it
        carries no location identity and can survive a regime shift.
        """
        return self._scatter(self._first_flags(values))

    def gap_seconds(self) -> np.ndarray:
        """Seconds since this entity's previous transaction. NaN on the first."""
        gaps = (
            self._work.groupby("entity", observed=True)["event_time"]
            .diff()
            .dt.total_seconds()
            .to_numpy()
        )
        return self._scatter(gaps)

    def gap_since_same(self, values: pd.Series) -> np.ndarray:
        """Seconds since this entity's previous transaction sharing this value.

        NaN when there is none -- which is exactly where `first_occurrence` is 1.
        Kept NaN rather than filled so the dataset module decides the sentinel.
        """
        keyed = self._keyed(values)
        gaps = (
            keyed.groupby(["entity", "_key"], observed=True)["event_time"]
            .diff()
            .dt.total_seconds()
            .to_numpy()
        )
        return self._scatter(gaps)

    def rolling_distinct(self, values: pd.Series, window: str) -> np.ndarray:
        """Distinct values of `values` in this entity's trailing `window`.

        Fan-out over a bounded window, which is what separates smurfing from a
        merchant that simply has many customers -- `prior_distinct` grows forever
        and so cannot express "twelve new counterparties this week".

        A trailing distinct count is a genuine sliding-window problem: whether a
        row is the first of its value inside the window depends on where the
        window starts, so it is not a rolling sum of any per-row flag. This walks
        each entity once with a running multiset, which costs a Python-level loop
        over the rows -- roughly half a minute on SAML-D's 9.5M. Call it for the
        windows that earn it, not for all four.

        Rows sharing a timestamp are handled as one block, all reading the multiset
        before any of them joins it. That is what the window `[t - W, t)` means, and
        it is what the pandas-backed methods here do -- and the ties are not an edge
        case: IBM CCF has minute resolution and no seconds at all.
        """
        offset = pd.Timedelta(_offset(window))
        keys = values.astype("string").fillna("~na").to_numpy()[self._pos]
        times = self._time
        groups = self._group_key.to_numpy()

        out = np.zeros(self._n, dtype="float64")
        counts: dict[str, int] = {}
        start = 0  # left edge of the window, an index into the sorted arrays
        block = 0
        while block < self._n:
            if block == 0 or groups[block] != groups[block - 1]:
                counts = {}
                start = block

            end = block + 1
            while (
                end < self._n
                and groups[end] == groups[block]
                and times[end] == times[block]
            ):
                end += 1

            # Rows in [start, block) are this entity's strict past. Retire the ones
            # that have aged out, then let the whole tied block read the multiset.
            cutoff = times[block] - offset
            while start < block and times[start] < cutoff:
                key = keys[start]
                if counts[key] == 1:
                    del counts[key]
                else:
                    counts[key] -= 1
                start += 1

            out[block:end] = len(counts)
            for k in range(block, end):
                counts[keys[k]] = counts.get(keys[k], 0) + 1
            block = end
        return self._scatter(out)


# --- arithmetic ------------------------------------------------------------


def safe_ratio(numerator, denominator) -> np.ndarray:
    """`numerator / denominator`, with 0.0 wherever that is not finite.

    A ratio of two windows separates "busy card" from "suddenly busy card", which
    neither raw count can. Zero is the right value when the denominator is empty:
    there is no burst to report.
    """
    num = np.asarray(pd.to_numeric(numerator, errors="coerce"), dtype="float64")
    den = np.asarray(pd.to_numeric(denominator, errors="coerce"), dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = num / den
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio


def signed_log1p(values) -> np.ndarray:
    """`sign(x) * log1p(|x|)`. Compresses a heavy tail while keeping refunds negative."""
    x = np.asarray(pd.to_numeric(values, errors="coerce"), dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between two coordinate pairs."""
    radius = 6371.0088
    p1, p2 = np.radians(np.asarray(lat1, dtype="float64")), np.radians(
        np.asarray(lat2, dtype="float64")
    )
    dlat = p2 - p1
    dlon = np.radians(np.asarray(lon2, dtype="float64")) - np.radians(
        np.asarray(lon1, dtype="float64")
    )
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def days_between(later, earlier) -> np.ndarray:
    """Whole and fractional days from `earlier` to `later`. NaN if either is missing."""
    delta = pd.to_datetime(later) - pd.to_datetime(earlier)
    return (delta.dt.total_seconds() / 86400.0).to_numpy(dtype="float64")


def clock_features(event_time: pd.Series) -> pd.DataFrame:
    """Cyclical and relative clock parts. Never the absolute date.

    Hour, weekday, day-of-month and month generalise forward and are what a real
    detector uses. The absolute date does not: the splits are temporal, so a tree
    given a year can isolate the split boundary as a threshold. `hour_sin`/
    `hour_cos` give a model without a notion of wraparound the fact that 23:00 and
    01:00 are adjacent.
    """
    t = pd.to_datetime(event_time)
    hour = t.dt.hour.to_numpy(dtype="float64")
    angle = 2 * np.pi * hour / 24.0
    weekday = t.dt.weekday.to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "hour": hour,
            "minute": t.dt.minute.to_numpy(dtype="float64"),
            "weekday": weekday,
            "day": t.dt.day.to_numpy(dtype="float64"),
            "month": t.dt.month.to_numpy(dtype="float64"),
            "is_weekend": (weekday >= 5).astype("float64"),
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
        },
        index=t.index,
    )


def amount_shape(amount: pd.Series) -> pd.DataFrame:
    """The amount itself and the shape of the number.

    Roundness and micro-amounts are behavioural, not magnitude: the IBM audit
    measured $0.01-$0.09 at 11-35x the base rate, which is card testing rather
    than an expensive purchase. `amount_log1p` is signed so a refund stays negative.
    """
    raw = pd.to_numeric(amount, errors="coerce")
    absolute = raw.abs().fillna(0.0)
    cents = (absolute * 100).round().astype("int64") % 100
    return pd.DataFrame(
        {
            "amount": raw.to_numpy(dtype="float64"),
            "amount_log1p": signed_log1p(raw),
            "amount_is_refund": (raw.fillna(0.0) < 0).to_numpy().astype("float64"),
            "amount_cents": cents.to_numpy(dtype="float64"),
            "amount_is_round_10": ((absolute % 10 == 0) & (absolute > 0))
            .to_numpy()
            .astype("float64"),
            "amount_is_round_100": ((absolute % 100 == 0) & (absolute > 0))
            .to_numpy()
            .astype("float64"),
            "amount_is_micro": ((absolute > 0) & (absolute < 0.10))
            .to_numpy()
            .astype("float64"),
        },
        index=raw.index,
    )


def parse_money(series: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5. Some sources keep money as strings."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


# --- the parquet contract -----------------------------------------------------


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column of a feature parquet that is a feature, in stored order."""
    keys = {*KEY_COLUMNS, *OPTIONAL_KEY_COLUMNS}
    return [c for c in df.columns if c not in keys]


def artifact_columns(df: pd.DataFrame) -> list[str]:
    """The features naming an absolute place or a specific counterparty."""
    return [c for c in feature_columns(df) if c.startswith(ARTIFACT_PREFIX)]


class FeatureContractError(AssertionError):
    """Raised when a frame does not satisfy the feature-parquet contract."""


def write_features(
    dataset: str,
    keys: pd.DataFrame,
    features: pd.DataFrame,
    *,
    features_dir: Path | str = FEATURE_DIR,
    extra_keys: tuple[str, ...] = (),
) -> Path:
    """Check the contract, then write `data/features/<dataset>.parquet`.

    `keys` supplies KEY_COLUMNS (plus `extra_keys`, which is how Sparkov carries a
    second `reported_at_slow` regime). `features` supplies everything else. The
    two are checked for overlap, for a surviving label column, and for infinities,
    then numeric features are cast to float32 and object/string ones to `category`.

    The `assert_no_excluded` call is the point of the split: a raw column that is
    really a label -- SAML-D's `Laundering_type`, a source's own detector output --
    fails here rather than quietly becoming the best feature in the model.
    """
    unknown = sorted(set(extra_keys) - set(OPTIONAL_KEY_COLUMNS))
    if unknown:
        raise FeatureContractError(
            f"{dataset}: {unknown} is not a known optional key; add it to "
            f"OPTIONAL_KEY_COLUMNS or pass it as a feature, or `feature_columns` will "
            "report it as a feature"
        )
    expected_keys = (*KEY_COLUMNS, *extra_keys)
    missing = [c for c in expected_keys if c not in keys.columns]
    if missing:
        raise FeatureContractError(f"{dataset}: keys are missing {missing}")
    stray = [c for c in keys.columns if c not in expected_keys]
    if stray:
        raise FeatureContractError(
            f"{dataset}: keys carries non-key column(s) {stray}; pass them as features"
        )
    if len(keys) != len(features):
        raise FeatureContractError(
            f"{dataset}: {len(keys)} key rows against {len(features)} feature rows"
        )

    overlap = sorted(set(keys.columns) & set(features.columns))
    if overlap:
        raise FeatureContractError(f"{dataset}: {overlap} is both a key and a feature")
    assert_no_excluded(list(features.columns))

    out = features.copy()
    for column in out.columns:
        values = out[column]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
            if np.isinf(numeric).any():
                raise FeatureContractError(
                    f"{dataset}: feature {column!r} contains infinities"
                )
            out[column] = numeric.astype("float32")
        else:
            # Unfitted on purpose: the levels are named, not coded or capped.
            # `experiments.encoding` does that, fitted on the chosen train split.
            out[column] = values.astype("string").fillna("~na").astype("category")

    frame = pd.concat(
        [keys[list(expected_keys)].reset_index(drop=True), out.reset_index(drop=True)],
        axis=1,
    )

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    destination = features_dir / f"{dataset}.parquet"
    frame.to_parquet(destination, index=False)
    return destination
