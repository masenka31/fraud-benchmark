"""The dataset adapter interface and its registry.

An adapter's only job is turning one dataset's raw files into a canonical frame.
It knows nothing about splitting, label delay, or output formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.sources import Source


class UnknownDatasetError(KeyError):
    """Raised when a dataset name has no registered adapter."""


class DatasetAdapter(ABC):
    """Base class for per-dataset adapters."""

    #: Short identifier used on the CLI and as the output directory name.
    name: str
    #: Where the raw files come from.
    source: Source
    #: The upstream data licence, recorded in the dataset card. This is the
    #: dataset's own terms, which are separate from this repository's MIT
    #: licence. See docs/dataset-licenses.md.
    data_license: str = "unknown"
    #: False when the upstream licence forbids commercial use (e.g. CC BY-NC-SA).
    #: Drives `prepare --all --exclude-noncommercial`.
    commercial_use: bool = True
    #: Human-readable warnings recorded in the dataset card.
    caveats: tuple[str, ...] = ()

    @abstractmethod
    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        """Read raw files from `raw_dir` and return a canonical frame.

        The result must contain event_time, entity_id, amount, and is_fraud with the
        dtypes in schema.REQUIRED_DTYPES, plus any source columns to pass through.
        """

    @abstractmethod
    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        """Map each canonical column to the source column(s) it came from."""

    def custom_splits(
        self, df: pd.DataFrame, options: dict[str, Any]
    ) -> pd.Series | None:
        """Optionally supply this dataset's own split labels.

        Return None — the default — to accept the shared temporal split. Override
        only when the source dictates the split, e.g. an upstream test set that must
        be preserved for comparability with published results.
        """
        return None


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Class decorator adding an adapter to the registry."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{cls.__name__} must set a non-empty string 'name' class attribute "
            "before it can be registered"
        )
    if name in _REGISTRY:
        raise ValueError(f"dataset {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """Return an adapter instance for `name`."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise UnknownDatasetError(f"unknown dataset {name!r}; available: {available}")
    return _REGISTRY[name]()


def list_datasets() -> list[str]:
    """All registered dataset names, sorted."""
    return sorted(_REGISTRY)
