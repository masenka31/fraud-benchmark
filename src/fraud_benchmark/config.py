"""Configuration loading. Defaults live in configs/default.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud_benchmark.delay import DelayParams

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    processed_dir: Path
    split_ratios: tuple[float, float, float]
    delay: DelayParams
    campaign_gap: pd.Timedelta
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_dataset(self, name: str) -> dict[str, Any]:
        """Options for one dataset, or an empty dict if it has none."""
        return self.datasets.get(name, {})

    def campaign_gap_for(self, name: str) -> pd.Timedelta:
        """The campaign gap for one dataset, honouring a per-dataset override.

        The right gap is not the same everywhere: Amaretto's anomalies are three
        orders of magnitude burstier than card fraud.
        """
        override = self.for_dataset(name).get("campaign_gap")
        return _parse_gap(override) if override is not None else self.campaign_gap

    def delay_for(self, name: str) -> DelayParams:
        """The delay parameters for one dataset, honouring a partial override.

        A dataset's `delay:` block overrides only the keys it names; the rest
        inherit the global block, so changing the global seed still moves every
        dataset. One distribution does not fit every dataset: PaySim's whole span
        is 30 days, while IBM CCF's is 10,649.
        """
        override = self.for_dataset(name).get("delay")
        if override is None:
            return self.delay
        return _merge_delay(self.delay, override, name)


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


def _parse_gap(value) -> pd.Timedelta:
    """Parse a duration like '1d', '1h', '30min' into a Timedelta."""
    try:
        gap = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"campaign gap {value!r} is not a valid duration: {exc}"
        ) from exc
    if gap is pd.NaT:
        raise ConfigError(f"campaign gap {value!r} is not a valid duration")
    if gap < pd.Timedelta(0):
        raise ConfigError(f"campaign gap must not be negative, got {value!r}")
    return gap


def _build_delay(data: dict) -> DelayParams:
    delay = data.get("delay") or {}
    try:
        return DelayParams(
            median_days=float(delay.get("median_days", 7.0)),
            sigma=float(delay.get("sigma", 1.0)),
            seed=int(delay.get("seed", 0)),
            max_delay_days=(
                float(delay["max_delay_days"])
                if delay.get("max_delay_days") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid delay settings: {exc}") from exc


_DELAY_FIELDS = {f.name for f in fields(DelayParams)}


def _merge_delay(base: DelayParams, override: Any, name: str) -> DelayParams:
    """Overlay a per-dataset `delay:` block onto the global one."""
    if not isinstance(override, dict):
        raise ConfigError(f"datasets.{name}.delay must be a mapping, got {override!r}")
    unknown = sorted(set(override) - _DELAY_FIELDS)
    if unknown:
        # Silently ignoring a typo would leave the dataset on the global default
        # while the config claims otherwise.
        raise ConfigError(
            f"unknown delay setting(s) for {name}: {', '.join(unknown)}; "
            f"valid keys are {', '.join(sorted(_DELAY_FIELDS))}"
        )
    try:
        coerced = {
            key: (
                int(value)
                if key == "seed"
                else None
                if value is None
                else float(value)
            )
            for key, value in override.items()
        }
        # replace() re-runs DelayParams.__post_init__, so an override gets exactly
        # the validation the global block gets.
        return replace(base, **coerced)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid delay settings for {name}: {exc}") from exc


def _load_yaml(path: Path | str) -> dict:
    """Read a YAML file, turning any read/parse failure into a ConfigError."""
    try:
        with open(path) as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def _require_paths(data: dict) -> dict:
    paths = data.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError(f"config must define a 'paths' mapping, got {paths!r}")
    for key in ("raw", "processed"):
        if not paths.get(key):
            raise ConfigError(f"config is missing 'paths.{key}'")
    return paths


def load_config(path: Path | str | None = None) -> Config:
    """Load the default config, overlaying `path` on top of it when given."""
    data = _load_yaml(DEFAULT_CONFIG_PATH)

    if path is not None:
        data = _deep_merge(data, _load_yaml(path))

    paths = _require_paths(data)
    config = Config(
        raw_dir=Path(paths["raw"]),
        processed_dir=Path(paths["processed"]),
        split_ratios=_validate_ratios(data.get("split", {}).get("ratios")),
        delay=_build_delay(data),
        campaign_gap=_parse_gap((data.get("campaign") or {}).get("gap", "1d")),
        datasets=data.get("datasets") or {},
    )
    # Resolve every override now: a bad one that only raised when its dataset was
    # prepared would let `prepare --all` die halfway, after writing other datasets.
    for name in config.datasets:
        config.delay_for(name)
    return config
