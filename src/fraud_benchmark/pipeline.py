"""Stage orchestration: fetch, canonicalize, validate, split, write."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import fraud_benchmark.datasets  # noqa: F401  (registers all adapters)
from fraud_benchmark.config import Config
from fraud_benchmark.datasets.base import get_adapter
from fraud_benchmark.schema import order_columns, validate_canonical
from fraud_benchmark.splitting import assign_splits, split_boundaries
from fraud_benchmark.sources import fetch


def prepare(name: str, config: Config, *, force: bool = False) -> Path:
    """Run the full pipeline for one dataset. Returns its output directory."""
    adapter = get_adapter(name)
    options = config.for_dataset(name)

    raw_dir = fetch(adapter.source, config.raw_dir / name, force=force)

    df = adapter.to_canonical(raw_dir, options)
    validate_canonical(df)

    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)
    df["split"] = assign_splits(df, config.split_ratios)
    df = order_columns(df)

    card = _build_card(name, adapter, options, df, config)
    return _write_atomically(config.processed_dir / name, df, card)


def _build_card(name, adapter, options, df, config) -> dict:
    bounds = split_boundaries(df, config.split_ratios)
    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": _describe_source(adapter.source),
        # The dataset's own terms, not this repo's licence. Recorded here so the
        # terms travel with the output. See docs/dataset-licenses.md.
        "data_license": adapter.data_license,
        "commercial_use": adapter.commercial_use,
        "n_rows": int(len(df)),
        "n_fraud": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "n_entities": int(df["entity_id"].nunique()),
        "time_range": {
            "start": df["event_time"].min().isoformat(),
            "end": df["event_time"].max().isoformat(),
        },
        "split": {
            "strategy": "temporal, cut on timestamp values",
            "ratios": list(config.split_ratios),
            "train_end": bounds["train_end"].isoformat(),
            "val_end": bounds["val_end"].isoformat(),
            "counts": {
                str(k): int(v) for k, v in df["split"].value_counts().items()
            },
        },
        "column_mapping": adapter.column_mapping(options),
        "options": options,
        "caveats": list(adapter.caveats),
    }


def _describe_source(source) -> dict:
    described = asdict(source) if is_dataclass(source) else {"repr": repr(source)}
    described["type"] = type(source).__name__
    described["url"] = getattr(source, "url", None)
    return described


def _write_atomically(dest: Path, df: pd.DataFrame, card: dict) -> Path:
    """Write into a staging directory, then swap it into place.

    The previous output is renamed aside rather than deleted, so an interrupted
    swap leaves it recoverable instead of destroying it. `os.rename` onto a
    non-empty directory fails, which is why the old output must be moved out of
    the way before the replace rather than replaced directly.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"{dest.name}.tmp{os.getpid()}"
    previous = dest.parent / f"{dest.name}.old{os.getpid()}"
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir()

    try:
        df.to_parquet(staging / "data.parquet", index=False)
        (staging / "dataset_card.json").write_text(
            json.dumps(card, indent=2, default=str) + "\n"
        )

        try:
            os.rename(dest, previous)
        except FileNotFoundError:
            # No previous output, or a concurrent run moved it first. Either is fine.
            pass
        os.replace(staging, dest)
    except BaseException:
        # The swap failed after the old output was moved aside; put it back.
        if previous.exists() and not dest.exists():
            os.rename(previous, dest)
        raise
    finally:
        for path in (staging, previous):
            if path.exists():
                shutil.rmtree(path)

    return dest
