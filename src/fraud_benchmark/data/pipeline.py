"""Stage orchestration: fetch, canonicalize, validate, split, write."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd

import fraud_benchmark.data.adapters  # noqa: F401  (registers all adapters)
from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.campaigns import assign_campaigns
from fraud_benchmark.data.campaigns import campaign_sizes
from fraud_benchmark.data.config import Config
from fraud_benchmark.data.delay import DelayParams
from fraud_benchmark.data.delay import assign_reported_at
from fraud_benchmark.data.schema import order_columns
from fraud_benchmark.data.schema import validate_canonical
from fraud_benchmark.data.sources import fetch
from fraud_benchmark.data.splitting import assign_splits
from fraud_benchmark.data.splitting import split_boundaries


def download(name: str, config: Config, *, force: bool = False) -> Path:
    """Fetch one dataset's raw files and stop. Returns the directory they landed in.

    Stage 1 on its own, so a slow download can be done once and separately from the
    preparation that reads it. `prepare` calls this rather than duplicating it, and
    `fetch` skips a directory that already has files -- so downloading first and
    preparing after costs nothing, and neither does the reverse.
    """
    adapter = get_adapter(name)
    return fetch(adapter.source, config.raw_dir / adapter.raw_name, force=force)


def _drop_source_label(df: pd.DataFrame, adapter) -> str | None:
    """The adapter's raw label column, or None if it must stay in the frame.

    It stays in two cases: a source column already named `is_fraud` is the
    canonical label itself, and one listed in `label_descriptive_columns` carries a
    typology `is_fraud` reduces to a bool (`experiments.columns` hides those from
    models instead).

    Raises ValueError if the adapter names a column its frame does not contain.
    """
    column = adapter.source_label_column
    if column == 'is_fraud' or column in adapter.label_descriptive_columns:
        return None
    if column not in df.columns:
        # An adapter bug, not a dataset quirk: staying silent would let the raw
        # label ride along under a name nobody is watching.
        raise ValueError(
            f'{adapter.name} declares source_label_column {column!r}, which is not '
            f'in its canonical frame; columns are: {sorted(df.columns)}'
        )
    return column


def prepare(name: str, config: Config, *, force: bool = False) -> Path:
    """Fetch, canonicalize, split, delay and write one dataset.

    Returns the output directory: <processed_dir>/<name>.
    """
    adapter = get_adapter(name)
    options = config.for_dataset(name)

    raw_dir = download(name, config, force=force)

    df = adapter.to_canonical(raw_dir, options)
    dropped_label = _drop_source_label(df, adapter)
    if dropped_label is not None:
        df = df.drop(columns=[dropped_label])
    validate_canonical(df)

    df = df.sort_values('event_time', kind='stable').reset_index(drop=True)
    supplied = adapter.custom_splits(df, options)
    df['split'] = assign_splits(df, config.split_ratios) if supplied is None else supplied

    gap = config.campaign_gap_for(name)
    delay = config.delay_for(name)
    df['campaign_id'] = assign_campaigns(df, gap=gap)
    df['reported_at'] = assign_reported_at(df, delay)

    # Re-validate: the adapter's output was checked earlier, but the delay stage
    # is the one that can produce an impossible reported_at, and nothing should
    # reach disk unchecked.
    validate_canonical(df)
    df = order_columns(df)

    aux = adapter.auxiliary_frames(raw_dir, options)
    card = _describe_dataset(
        name,
        adapter,
        options,
        df,
        config,
        custom_split=supplied is not None,
        gap=gap,
        delay=delay,
        dropped_label=dropped_label,
    )
    card['auxiliary'] = {key: int(len(frame)) for key, frame in aux.items()}
    return _write_atomically(config.processed_dir / name, df, card, aux)


def _describe_dataset(
    name,
    adapter,
    options,
    df,
    config,
    *,
    custom_split: bool,
    gap: pd.Timedelta,
    delay: DelayParams,
    dropped_label: str | None,
) -> dict:
    """Build the dataset card: provenance, licence, shape, split and delay."""
    return {
        'name': name,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source': _describe_source(adapter.source),
        # The dataset's own terms, not this repo's licence. Recorded here so the
        # terms travel with the output. See docs/dataset-licenses.md.
        'data_license': adapter.data_license,
        'commercial_use': adapter.commercial_use,
        'n_rows': int(len(df)),
        'n_fraud': int(df['is_fraud'].sum()),
        'fraud_rate': float(df['is_fraud'].mean()),
        'n_entities': int(df['entity_id'].nunique()),
        'time_range': {
            'start': df['event_time'].min().isoformat(),
            'end': df['event_time'].max().isoformat(),
        },
        'split': _describe_split(adapter, df, config, custom_split=custom_split),
        'label_delay': _describe_delay(df, delay, gap),
        'column_mapping': adapter.column_mapping(options),
        # What the label-drop stage removed. `column_mapping` still names the source
        # column as is_fraud's provenance; this says it is no longer in the output.
        'dropped_source_label': dropped_label,
        'label_descriptive_columns': list(adapter.label_descriptive_columns),
        'options': options,
        'caveats': list(adapter.caveats),
    }


def _describe_split(adapter, df, config, *, custom_split: bool) -> dict:
    """Describe how `df` was split, and how many rows landed in each split."""
    counts = {str(k): int(v) for k, v in df['split'].value_counts().items()}
    if custom_split:
        return {
            'strategy': f'supplied by the {adapter.name} adapter',
            'ratios': None,
            'counts': counts,
        }
    bounds = split_boundaries(df, config.split_ratios)
    return {
        'strategy': 'temporal, cut on timestamp values',
        'ratios': list(config.split_ratios),
        'train_end': bounds['train_end'].isoformat(),
        'val_end': bounds['val_end'].isoformat(),
        'counts': counts,
    }


def _describe_delay(df, delay: DelayParams, gap) -> dict:
    """Describe the delay settings alongside the campaign sizes and delays realised."""
    sizes = campaign_sizes(df['campaign_id'])
    fraud = df.loc[df['is_fraud']]
    delays = (fraud['reported_at'] - fraud['event_time']).dt.total_seconds() / 86_400
    return {
        'distribution': 'lognormal',
        'median_days': delay.median_days,
        'sigma': delay.sigma,
        'seed': delay.seed,
        'max_delay_days': delay.max_delay_days,
        'campaign_gap': str(gap),
        'n_campaigns': int(sizes.size),
        'largest_campaign': int(sizes.max()) if sizes.size else 0,
        'median_campaign_size': float(sizes.median()) if sizes.size else 0.0,
        'observed_median_delay_days': float(delays.median()) if len(delays) else 0.0,
        # Truncation pulls the realised mean below nominal, so record what
        # actually happened rather than only what was configured.
        'observed_mean_delay_days': float(delays.mean()) if len(delays) else 0.0,
    }


def _describe_source(source) -> dict:
    """Flatten a Source into card fields, whatever its concrete type."""
    described = asdict(source) if is_dataclass(source) else {'repr': repr(source)}
    described['type'] = type(source).__name__
    described['url'] = getattr(source, 'url', None)
    return described


def _validate_auxiliary_keys(aux: dict[str, pd.DataFrame]) -> None:
    """Require every auxiliary key to be a bare filename.

    A key like "../escape" would write outside the staging directory, bypassing the
    atomic swap and its rollback.
    """
    for key in aux:
        if not key or key != Path(key).name or key in ('.', '..'):
            raise ValueError(
                f'auxiliary frame key {key!r} is not a valid filename; keys must not '
                'contain path separators or refer to parent directories'
            )


def _write_atomically(
    dest: Path,
    df: pd.DataFrame,
    card: dict,
    aux: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Write `df`, `card` and `aux` to `dest`, replacing it atomically. Returns `dest`.

    Any previous output is renamed aside rather than deleted, so a failed swap can
    put it back; `os.rename` onto a non-empty directory would otherwise fail.
    """
    _validate_auxiliary_keys(aux or {})
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f'{dest.name}.tmp{os.getpid()}'
    previous = dest.parent / f'{dest.name}.old{os.getpid()}'
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir()

    try:
        df.to_parquet(staging / 'data.parquet', index=False)
        for key, frame in (aux or {}).items():
            frame.to_parquet(staging / f'{key}.parquet', index=False)
        (staging / 'dataset_card.json').write_text(json.dumps(card, indent=2, default=str) + '\n')

        # No previous output, or a concurrent run moved it first. Either is fine.
        with contextlib.suppress(FileNotFoundError):
            Path(dest).rename(previous)
        Path(staging).replace(dest)
    except BaseException:
        # The swap failed after the old output was moved aside; put it back.
        if previous.exists() and not dest.exists():
            Path(previous).rename(dest)
        raise
    finally:
        for path in (staging, previous):
            if path.exists():
                shutil.rmtree(path)

    return dest
