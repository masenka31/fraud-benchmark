"""Categorical encoding and scaling, fitted on train only.

Fitting on anything but train is lookahead: the val and test rows would have
contributed to the category vocabulary and to the scaling statistics. Values that
appear only in val or test go to a dedicated bucket rather than silently onto a
real level, which would otherwise make an unseen merchant indistinguishable from
a specific known one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNSEEN = '__unseen__'
_NULL = '__null__'

#: Vocabulary cap for `CappedOrdinalEncoder`: keep levels up to this share of
#: cumulative train frequency, and at most this many.
RARE_COVERAGE = 0.99
MAX_LEVELS = 256


class CappedOrdinalEncoder:
    """Integer codes for categorical columns, with a rare-value bucket.

    `Encoder` above keeps every level it saw in train, which is right for a linear
    reference but not for a model that pays per level. IBM CCF's
    `artifact_merchant_name` runs to ~100,000 values: one-hot expanding that is
    129 GiB dense, and a tree spends its splits enumerating merchants. Capping at
    99% cumulative train frequency and at most 256 levels keeps the head and pools
    the tail, which is the treatment the retired IBM pipeline used and the reason
    its numbers were reachable at all.

    Fitted on train rows only. A value appearing first in val or test therefore
    lands in the rare bucket, which is also where a genuinely rare train value
    goes -- the two are indistinguishable to a deployed model, so they should be
    indistinguishable here.
    """

    def __init__(self, coverage: float = RARE_COVERAGE, max_levels: int = MAX_LEVELS) -> None:
        self.coverage = coverage
        self.max_levels = max_levels
        self.vocabularies_: dict[str, list[str]] = {}

    def fit(self, train: pd.DataFrame, columns: list[str]) -> CappedOrdinalEncoder:
        for column in columns:
            values = train[column].astype('string').fillna(_NULL)
            counts = values.value_counts(normalize=True)
            # shift(1) so the level that crosses the threshold is itself kept:
            # the test is on the coverage *before* adding it.
            keep = counts.cumsum().shift(1).fillna(0.0) < self.coverage
            self.vocabularies_[column] = list(counts[keep].index[: self.max_levels])
        return self

    def cardinality(self, column: str) -> int:
        """Number of distinct codes, the rare bucket included."""
        return len(self.vocabularies_[column]) + 1

    def transform_column(self, values: pd.Series, column: str) -> np.ndarray:
        """Codes for one column. The rare bucket is the last code, never a real level."""
        vocabulary = self.vocabularies_[column]
        lookup = {value: index for index, value in enumerate(vocabulary)}
        coded = values.astype('string').fillna(_NULL).map(lookup)
        return coded.fillna(len(vocabulary)).to_numpy(dtype='int32')


class Encoder:
    """Ordinal category codes plus optional standardisation."""

    def __init__(self) -> None:
        self.categories_: dict[str, list[str]] = {}
        self.numeric_: list[str] = []
        self.scale_: bool = False
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(
        self,
        train: pd.DataFrame,
        categorical: list[str] | None = None,
        numeric: list[str] | None = None,
        scale: bool = False,
    ) -> Encoder:
        self.scale_ = scale
        self.numeric_ = list(numeric or [])
        for col in categorical or []:
            values = train[col].astype('string').fillna(_NULL)
            # UNSEEN last, so its code cannot collide with a real category.
            self.categories_[col] = sorted(set(values.tolist())) + [UNSEEN]
        if scale:
            # Scale the ENCODED matrix, not just the originally-numeric columns.
            # Ordinal codes are numbers too, and on IBM CCF `Merchant Name` runs
            # to ~100,000 categories -- feeding lbfgs raw codes of that magnitude
            # alongside amounts in the tens does not converge, and a
            # non-converged linear reference is worse than none.
            encoded = self._encode(train)
            for col in encoded.columns:
                values = pd.to_numeric(encoded[col], errors='coerce')
                mean = float(values.mean())
                std = float(values.std())
                self.mean_[col] = 0.0 if not np.isfinite(mean) else mean
                # A constant column has no spread; dividing by 1 leaves it at 0.
                self.std_[col] = std if np.isfinite(std) and std > 0 else 1.0
        return self

    def code_of(self, column: str, value: str) -> int:
        return self.categories_[column].index(value)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorical columns to ordinal codes. No scaling."""
        out = df.copy()
        for col, categories in self.categories_.items():
            lookup = {c: i for i, c in enumerate(categories)}
            unseen = lookup[UNSEEN]
            values = out[col].astype('string').fillna(_NULL)
            out[col] = values.map(lookup).fillna(unseen).astype('int32')
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._encode(df)
        if self.scale_:
            for col in self.mean_:
                values = pd.to_numeric(out[col], errors='coerce').fillna(self.mean_[col])
                out[col] = (values - self.mean_[col]) / self.std_[col]
        return out
