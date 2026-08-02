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

UNSEEN = "__unseen__"
_NULL = "__null__"


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
    ) -> "Encoder":
        self.scale_ = scale
        self.numeric_ = list(numeric or [])
        for col in categorical or []:
            values = train[col].astype("string").fillna(_NULL)
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
                values = pd.to_numeric(encoded[col], errors="coerce")
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
            values = out[col].astype("string").fillna(_NULL)
            out[col] = values.map(lookup).fillna(unseen).astype("int32")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._encode(df)
        if self.scale_:
            for col in self.mean_:
                values = pd.to_numeric(out[col], errors="coerce").fillna(self.mean_[col])
                out[col] = (values - self.mean_[col]) / self.std_[col]
        return out
