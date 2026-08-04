"""The dataset adapter interface and its registry.

An adapter's only job is turning one dataset's raw files into a canonical frame.
It knows nothing about splitting, label delay, or output formats.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_benchmark.data.sources import Source


class UnknownDatasetError(KeyError):
    """Raised when a dataset name has no registered adapter."""


def require_start_date(options: dict[str, Any], dataset: str, column: str) -> pd.Timestamp:
    """The timestamp `column`'s zero point, from `datasets.<dataset>.start_date`.

    Required rather than defaulted: a dataset shipping an offset instead of a date
    cannot place it in absolute time without being told, and a default would invent
    dates silently. Raises ValueError naming the option when it is unset.
    """
    start_date = options.get('start_date')
    if not start_date:
        raise ValueError(
            f"{dataset} requires a 'start_date' option to anchor its relative "
            f'{column!r} column; set datasets.{dataset}.start_date in the config'
        )
    return pd.Timestamp(start_date)


class DatasetAdapter(ABC):
    """Base class for per-dataset adapters."""

    #: Short identifier used on the CLI and as the output directory name.
    name: str
    #: Directory under data/raw to fetch into. Defaults to `name`, resolved by
    #: `register`. A variant that reuses another dataset's raw files sets this to
    #: that dataset's name, so the download is shared rather than duplicated.
    raw_name: str = ''
    #: Where the raw files come from.
    source: Source
    #: The upstream data licence, recorded in the dataset card. This is the
    #: dataset's own terms, which are separate from this repository's MIT
    #: licence. See docs/dataset-licenses.md.
    data_license: str = 'unknown'
    #: False when the upstream licence forbids commercial use (e.g. CC BY-NC-SA).
    #: Drives `prepare --all --exclude-noncommercial`.
    commercial_use: bool = True
    #: The source column `is_fraud` was derived from. `register` requires it, and
    #: the pipeline drops it so the frame carries exactly one binary label; set it
    #: to "is_fraud" when the source column already has that name. See
    #: `pipeline._drop_source_label` for the two cases that keep the column.
    source_label_column: str = ''
    #: Columns describing the label rather than the transaction, e.g. a laundering
    #: typology that `is_fraud` reduces to a bool. They stay in the frame, and
    #: `experiments.columns` builds its model drop-list from this declaration.
    label_descriptive_columns: tuple[str, ...] = ()
    #: Human-readable warnings recorded in the dataset card.
    caveats: tuple[str, ...] = ()

    @abstractmethod
    def to_canonical(self, raw_dir: Path, options: dict[str, Any]) -> pd.DataFrame:
        """Read raw files from `raw_dir` and return a canonical frame.

        Must contain event_time, entity_id, amount and is_fraud with the dtypes in
        `schema.REQUIRED_DTYPES`; source columns to pass through may follow.
        """

    @abstractmethod
    def column_mapping(self, options: dict[str, Any]) -> dict[str, str]:
        """Canonical column name -> the source column(s) or expression behind it."""

    def custom_splits(self, df: pd.DataFrame, options: dict[str, Any]) -> pd.Series | None:
        """This dataset's own split labels, or None to accept the temporal split.

        Override only when the source dictates the split, e.g. an upstream test set
        preserved for comparability with published results.
        """
        return None

    def auxiliary_frames(self, raw_dir: Path, options: dict[str, Any]) -> dict[str, pd.DataFrame]:
        """Extra frames to write beside the canonical one, keyed by output file stem.

        Override for data that belongs with the dataset but cannot be validated or
        split, such as an unlabelled competition test set.
        """
        return {}


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Class decorator registering an adapter under its `name`.

    Defaults `raw_name` to `name`. Raises ValueError if `name` or
    `source_label_column` is unset, or if the name is already registered.
    """
    name = getattr(cls, 'name', None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{cls.__name__} must set a non-empty string 'name' class attribute "
            'before it can be registered'
        )
    if not getattr(cls, 'source_label_column', ''):
        raise ValueError(
            f"{cls.__name__} must set 'source_label_column' to the source column "
            'its is_fraud was derived from, so the pipeline can drop it. Without '
            'it the raw label reaches the output and a model can read its own '
            "answer. Use 'is_fraud' if the source column is already named that."
        )
    if not getattr(cls, 'raw_name', ''):
        cls.raw_name = name
    if name in _REGISTRY:
        raise ValueError(f'dataset {name!r} is already registered')
    _REGISTRY[name] = cls
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """A fresh adapter instance for `name`. Raises UnknownDatasetError if unknown."""
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY)) or '(none)'
        raise UnknownDatasetError(f'unknown dataset {name!r}; available: {available}')
    return _REGISTRY[name]()


def list_datasets() -> list[str]:
    """All registered dataset names, sorted."""
    return sorted(_REGISTRY)
