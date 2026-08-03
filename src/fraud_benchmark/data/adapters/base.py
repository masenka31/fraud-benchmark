"""The dataset adapter interface and its registry.

An adapter's only job is turning one dataset's raw files into a canonical frame.
It knows nothing about splitting, label delay, or output formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.data.sources import Source


class UnknownDatasetError(KeyError):
    """Raised when a dataset name has no registered adapter."""


def require_start_date(
    options: dict[str, Any], dataset: str, column: str
) -> pd.Timestamp:
    """The configured anchor for a dataset whose time column is a relative offset.

    Three datasets ship an offset rather than a date -- PaySim's hourly `step`,
    BankSim's daily `step`, IEEE-CIS's `TransactionDT` in seconds -- and none of
    them can produce a plausible `event_time` without being told where zero is.
    Defaulting the anchor would silently invent absolute dates, so it is required
    and the error names the option to set.
    """
    start_date = options.get("start_date")
    if not start_date:
        raise ValueError(
            f"{dataset} requires a 'start_date' option to anchor its relative "
            f"{column!r} column; set datasets.{dataset}.start_date in the config"
        )
    return pd.Timestamp(start_date)


class DatasetAdapter(ABC):
    """Base class for per-dataset adapters."""

    #: Short identifier used on the CLI and as the output directory name.
    name: str
    #: Directory under data/raw to fetch into. Defaults to `name`, resolved by
    #: `register`. A variant that reuses another dataset's raw files sets this to
    #: that dataset's name, so the download is shared rather than duplicated.
    raw_name: str = ""
    #: Where the raw files come from.
    source: Source
    #: The upstream data licence, recorded in the dataset card. This is the
    #: dataset's own terms, which are separate from this repository's MIT
    #: licence. See docs/dataset-licenses.md.
    data_license: str = "unknown"
    #: False when the upstream licence forbids commercial use (e.g. CC BY-NC-SA).
    #: Drives `prepare --all --exclude-noncommercial`.
    commercial_use: bool = True
    #: The source column `is_fraud` was derived from. The pipeline drops it, so the
    #: canonical frame carries exactly one binary label and no consumer has to
    #: remember to exclude a second one. Required: `register` refuses an adapter
    #: without it, because forgetting it is how a model gets handed its own answer.
    #:
    #: Two adapters are special and both are handled by the drop rule rather than
    #: by an exception here: sparkov's source column is already named `is_fraud`,
    #: so there is nothing to drop, and amaretto's is multi-class and therefore
    #: also listed in `label_descriptive_columns`, which keeps it.
    source_label_column: str = ""
    #: Columns that describe the label rather than the transaction, kept because
    #: they carry what `is_fraud` loses -- amaretto's five FATF classes, saml_d's
    #: laundering typology. They stay in the frame and out of every model:
    #: `experiments.columns` builds its drop-list from this declaration.
    label_descriptive_columns: tuple[str, ...] = ()
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

    def auxiliary_frames(
        self, raw_dir: Path, options: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """Extra frames to write beside the canonical one, keyed by file stem.

        Default: none. Override for data that belongs with the dataset but is not
        canonical — e.g. an unlabelled competition test set, which has no is_fraud
        column and so cannot be validated or split.
        """
        return {}


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Class decorator adding an adapter to the registry."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{cls.__name__} must set a non-empty string 'name' class attribute "
            "before it can be registered"
        )
    if not getattr(cls, "source_label_column", ""):
        raise ValueError(
            f"{cls.__name__} must set 'source_label_column' to the source column "
            "its is_fraud was derived from, so the pipeline can drop it. Without "
            "it the raw label reaches the output and a model can read its own "
            "answer. Use 'is_fraud' if the source column is already named that."
        )
    if not getattr(cls, "raw_name", ""):
        cls.raw_name = name
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
