"""Configuration loading. Defaults live in configs/default.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    processed_dir: Path
    split_ratios: tuple[float, float, float]
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_dataset(self, name: str) -> dict[str, Any]:
        """Options for one dataset, or an empty dict if it has none."""
        return self.datasets.get(name, {})


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_ratios(ratios: Any) -> tuple[float, float, float]:
    if not isinstance(ratios, list) or len(ratios) != 3:
        raise ConfigError(f"split.ratios must be a list of three numbers, got {ratios!r}")
    values = tuple(float(r) for r in ratios)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ConfigError(f"split.ratios must sum to 1, got {sum(values)}")
    if any(r <= 0 for r in values):
        raise ConfigError(f"split.ratios must all be positive, got {values}")
    return values  # type: ignore[return-value]


def load_config(path: Path | str | None = None) -> Config:
    """Load the default config, overlaying `path` on top of it when given."""
    with open(DEFAULT_CONFIG_PATH) as handle:
        data = yaml.safe_load(handle) or {}

    if path is not None:
        with open(path) as handle:
            data = _deep_merge(data, yaml.safe_load(handle) or {})

    paths = data.get("paths", {})
    return Config(
        raw_dir=Path(paths["raw"]),
        processed_dir=Path(paths["processed"]),
        split_ratios=_validate_ratios(data.get("split", {}).get("ratios")),
        datasets=data.get("datasets") or {},
    )
