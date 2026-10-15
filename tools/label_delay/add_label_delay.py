#!/usr/bin/env python3
"""Add a synthetic label-delay timestamp to a table of transactions.

No public fraud dataset records *when* a fraud became known, yet a detector trained
at some cutoff only knows the frauds reported by then. This script synthesises that
timestamp:

1. Frauds are grouped into **campaigns**: frauds on one entity, each within
   `--campaign-gap` of the previous one. An investigation discovers them together.
2. One delay per campaign is drawn from a lognormal in days and added to the
   campaign's **last** transaction, so a campaign is never reported before it ends.
3. Every fraud in the campaign gets that `reported_at`; other rows get none.

Every delay produced here is synthetic. The presets are calibrated to be plausible
for a domain, not fitted to any observed reporting behaviour.

Requires numpy, pandas and, for Parquet, pyarrow. Standalone copy of
`fraud_benchmark.data.campaigns` and `fraud_benchmark.data.delay` from
https://github.com/masenka31/fraud-benchmark; for the same inputs, seed and
parameters it produces identical columns.

Example:
    python add_label_delay.py transactions.parquet \\
        --entity card_id --time timestamp --fraud is_fraud \\
        --preset card --regime slow:15,1.665,365 --cutoff 2020-06-01
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '1.0.0'


@dataclass(frozen=True)
class Regime:
    """One lognormal reporting delay, in days, written to its own column.

    `median_days` is the distribution's median, i.e. exp(mu); `sigma` is the
    log-scale spread; `max_delay_days` truncates the draw when set.
    """

    name: str
    median_days: float
    sigma: float
    max_delay_days: float | None = None

    def __post_init__(self) -> None:
        if self.median_days <= 0:
            raise ValueError(f'median_days must be positive, got {self.median_days}')
        if self.sigma <= 0:
            raise ValueError(f'sigma must be positive, got {self.sigma}')
        if self.max_delay_days is not None and self.max_delay_days <= 0:
            raise ValueError(f'max_delay_days must be positive, got {self.max_delay_days}')

    @property
    def column(self) -> str:
        return f'reported_at_{self.name}' if self.name else 'reported_at'


#: card: a cardholder disputes within the chargeback window; median 7 days, p95 ~36.
#: card-slow: a stress regime with a fat tail; median 15 days, nominal mean 60,
#:     capped at a year. Its harshness comes from sigma rather than the median.
#: aml: suspicious-activity investigations run on a ~30-day clock.
PRESETS = {
    'card': Regime('', 7.0, 1.0),
    'card-slow': Regime('', 15.0, 1.665, 365.0),
    'aml': Regime('', 30.0, 1.0),
}


# --- the method: identical to fraud_benchmark.data.{campaigns,delay} ------------


def assign_campaigns(df: pd.DataFrame, gap: pd.Timedelta) -> pd.Series:
    """Campaign id per fraud row, <NA> elsewhere, aligned to `df`'s index.

    Requires `is_fraud`, `entity_id` and `event_time`. Consecutive frauds on one
    entity share a campaign while at most `gap` apart. Ids are Int64, dense from
    0, in no meaningful order.
    """
    if gap < pd.Timedelta(0):
        raise ValueError(f'gap must not be negative, got {gap}')

    ids = pd.Series(pd.NA, index=df.index, dtype='Int64', name='campaign_id')
    frauds = df.loc[df['is_fraud'], ['entity_id', 'event_time']]
    if frauds.empty:
        return ids

    ordered = frauds.sort_values(['entity_id', 'event_time'], kind='stable')
    entity = ordered['entity_id']
    elapsed = ordered['event_time'].diff()

    # Comparing `string` dtype yields <NA> on the first row rather than True, and
    # cumsum on a pyarrow bool mask raises on pandas 3, hence the numpy round-trip.
    is_campaign_start = (entity.ne(entity.shift()) | elapsed.gt(gap)).fillna(True)
    campaign_number = np.cumsum(is_campaign_start.to_numpy(dtype='bool')) - 1
    ids.loc[ordered.index] = pd.array(campaign_number, dtype='Int64')
    return ids


def assign_reported_at(df: pd.DataFrame, regime: Regime, seed: int) -> pd.Series:
    """reported_at per row as datetime64[us], NaT where `campaign_id` is null.

    Requires `campaign_id` and `event_time`. One delay is drawn per campaign and
    added to that campaign's last event_time.
    """
    reported = pd.Series(pd.NaT, index=df.index, dtype='datetime64[us]', name=regime.column)
    labelled = df['campaign_id'].notna()
    if not labelled.any():
        return reported

    # groupby sorts by key, so the draws line up with campaign ids regardless of
    # row order, which is what makes a given seed reproducible.
    campaign_end = df.loc[labelled].groupby('campaign_id', observed=True)['event_time'].max()

    rng = np.random.default_rng(seed)
    days = rng.lognormal(
        mean=math.log(regime.median_days), sigma=regime.sigma, size=len(campaign_end)
    )
    if regime.max_delay_days is not None:
        days = np.minimum(days, regime.max_delay_days)

    # Microsecond offsets: a [ns] timedelta overflows on heavy-tailed draws.
    offset_micros = days * 86_400_000_000.0
    if not np.isfinite(offset_micros).all() or offset_micros.max() >= float(np.iinfo('int64').max):
        raise ValueError(
            'sampled reporting delay exceeds the representable range; set a cap to '
            f'bound it (largest draw was {days.max():,.0f} days)'
        )
    per_campaign = campaign_end + offset_micros.astype('int64').astype('timedelta64[us]')
    mapped = df.loc[labelled, 'campaign_id'].map(per_campaign.to_dict())
    reported.loc[labelled] = pd.to_datetime(mapped).astype('datetime64[us]')
    return reported


# --- command line ----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Add synthetic campaign_id and reported_at columns to a '
        'transaction table (Parquet or CSV).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input', type=Path, help='.parquet/.pq or .csv file')
    parser.add_argument(
        '-o', '--output', type=Path, help='default: <input stem>.label_delay<suffix>'
    )
    parser.add_argument('--version', action='version', version=VERSION)

    cols = parser.add_argument_group('input columns')
    cols.add_argument('--entity', required=True, help='card, user or account column')
    cols.add_argument('--time', required=True, help='transaction time column')
    cols.add_argument('--fraud', required=True, help='fraud label column (0/1 or bool)')
    cols.add_argument(
        '--time-origin',
        help='start date for a numeric time column (e.g. seconds since an unstated origin)',
    )
    cols.add_argument(
        '--time-unit', default='s', help='unit of a numeric time column: s, min, h, D, ...'
    )

    delay = parser.add_argument_group('delay (written to reported_at)')
    delay.add_argument('--preset', choices=sorted(PRESETS), default='card')
    delay.add_argument('--median-days', type=float, help='overrides the preset')
    delay.add_argument('--sigma', type=float, help='overrides the preset')
    delay.add_argument(
        '--max-delay-days', type=float, help='overrides the preset; 0 removes its cap'
    )
    delay.add_argument(
        '--regime',
        action='append',
        default=[],
        metavar='NAME:MEDIAN,SIGMA[,CAP]',
        help='an extra regime written to reported_at_NAME; repeatable',
    )
    delay.add_argument('--campaign-gap', default='1D', help='max gap inside one campaign')
    delay.add_argument('--seed', type=int, default=0)

    parser.add_argument(
        '--cutoff', help='training cutoff; reports the share of frauds up to it still unreported'
    )
    args = parser.parse_args(argv)
    try:
        args.campaign_gap = pd.Timedelta(args.campaign_gap)
        args.cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
        args.regimes = [base_regime(args), *(parse_regime(s) for s in args.regime)]
    except ValueError as exc:
        parser.error(str(exc))
    # pd.Timedelta('NaT') parses without error, and comparing against it is always
    # False, which would silently make every entity a single campaign.
    if args.campaign_gap is pd.NaT or args.campaign_gap < pd.Timedelta(0):
        parser.error(f'--campaign-gap must be a non-negative duration, got {args.campaign_gap}')
    names = [r.name for r in args.regimes]
    if len(set(names)) != len(names):
        parser.error(f'regime names must be distinct, got {names[1:]}')

    args.output = args.output or args.input.with_name(
        f'{args.input.stem}.label_delay{args.input.suffix}'
    )
    args.sidecar = args.output.with_name(f'{args.output.name}.json')
    for path, role in ((args.input, 'input'), (args.output, 'output')):
        if path.suffix.lower() not in FORMATS:
            parser.error(f'unsupported {role} format {path.suffix!r}; use .parquet, .pq or .csv')
    if args.input.resolve() in {args.output.resolve(), args.sidecar.resolve()}:
        parser.error('the output would overwrite the input; pass a different -o')
    return args


def base_regime(args: argparse.Namespace) -> Regime:
    """The preset with any explicit --median-days/--sigma/--max-delay-days applied."""
    preset = PRESETS[args.preset]
    cap = preset.max_delay_days
    if args.max_delay_days is not None:
        cap = args.max_delay_days or None
    return Regime(
        '',
        args.median_days if args.median_days is not None else preset.median_days,
        args.sigma if args.sigma is not None else preset.sigma,
        cap,
    )


def parse_regime(spec: str) -> Regime:
    """`slow:15,1.665,365` -> Regime('slow', 15.0, 1.665, 365.0)."""
    match = re.fullmatch(r'(\w+):([^,]+),([^,]+)(?:,([^,]+))?', spec)
    if not match:
        raise ValueError(f'--regime {spec!r} is not NAME:MEDIAN,SIGMA[,CAP]')
    name, median, sigma, cap = match.groups()
    return Regime(name, float(median), float(sigma), float(cap) if cap else None)


FORMATS = {'.parquet', '.pq', '.csv'}


def read_table(path: Path, entity: str) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        # Read entity ids as literal text: as numbers, '0007' and '7' would become one
        # entity, and with the default NA parsing an entity named 'NA' or 'null' would
        # be missing. Either way CSV and Parquet copies of one table would disagree.
        return pd.read_csv(path, converters={entity: str})
    return pd.read_parquet(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == '.csv':
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path)


def parse_time(time: pd.Series, args: argparse.Namespace) -> tuple[pd.Series, bool]:
    """Transaction times as naive datetime64[us], and whether the input was tz-aware.

    Timezone-aware input is converted to UTC for the computation, so that the
    campaign gap and the delay are measured in elapsed time across DST changes.
    """
    numeric = pd.api.types.is_numeric_dtype(time)
    if numeric and args.time_origin is None:
        raise SystemExit(f'error: {args.time!r} is numeric; pass --time-origin')
    if not numeric and args.time_origin is not None:
        raise SystemExit('error: --time-origin applies only to a numeric time column')
    try:
        if numeric:
            time = pd.Timestamp(args.time_origin) + pd.to_timedelta(time, unit=args.time_unit)
        elif not pd.api.types.is_datetime64_any_dtype(time):
            time = parse_time_strings(time)
    except (ValueError, TypeError, OverflowError) as exc:
        raise SystemExit(f'error: cannot read {args.time!r} as times: {exc}') from exc
    aware = time.dt.tz is not None
    if aware:
        time = time.dt.tz_convert('UTC').dt.tz_localize(None)
    return time.astype('datetime64[us]'), aware


def parse_time_strings(time: pd.Series) -> pd.Series:
    """Datetimes from text, falling back to UTC for strings with mixed offsets.

    Mixed offsets (e.g. either side of a DST change) raise on pandas 3 but come
    back as an object column of Timestamps on pandas 2, so both are checked.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            parsed = pd.to_datetime(time)
    except ValueError:
        parsed = None
    if parsed is None or not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(time, utc=True)
    return parsed


FRAUD_TEXT = {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}


def parse_fraud(fraud: pd.Series, column: str) -> pd.Series:
    """The fraud label as bool, from 0/1, booleans or their text forms."""
    values = fraud
    if not (pd.api.types.is_bool_dtype(fraud) or pd.api.types.is_numeric_dtype(fraud)):
        values = fraud.astype(str).str.strip().str.lower().map(FRAUD_TEXT)
    bad = values.isna() | ~values.isin([0, 1])
    if bad.any():
        raise SystemExit(
            f'error: {column!r} must hold only 0/1, true/false or booleans; '
            f'found {sorted(map(str, fraud[bad].unique()))[:5]}'
        )
    return values.astype(bool)


def canonicalize(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    """entity_id/event_time/is_fraud in the dtypes the method expects, on a RangeIndex,
    and whether the input times were timezone-aware.

    Entities are compared as strings, which fixes the campaign numbering and hence
    which delay each campaign draws for a given seed.
    """
    missing = [c for c in (args.entity, args.time, args.fraud) if c not in df.columns]
    if missing:
        raise SystemExit(f'error: input has no column(s) {missing}')
    for c in (args.entity, args.time, args.fraud):
        if df[c].isna().any():
            raise SystemExit(f'error: column {c!r} has missing values')

    time, aware = parse_time(df[args.time], args)
    canon = pd.DataFrame(
        {
            'entity_id': df[args.entity].astype(str).astype('string').to_numpy(),
            'event_time': time.to_numpy(),
            'is_fraud': parse_fraud(df[args.fraud], args.fraud).to_numpy(),
        }
    )
    return canon, aware


def utc_cutoff(cutoff: pd.Timestamp | None, aware: bool) -> pd.Timestamp | None:
    """The cutoff on the naive-UTC clock the computation runs on.

    A cutoff without a timezone against timezone-aware data (or the reverse) is
    refused rather than guessed, since a guess shifts it by the UTC offset.
    """
    if cutoff is None:
        return None
    if aware != (cutoff.tz is not None):
        need = 'with' if aware else 'without'
        raise SystemExit(f'error: the time column is {"" if aware else "not "}timezone-aware; '
                         f'pass --cutoff {need} a timezone')  # fmt: skip
    return cutoff.tz_convert('UTC').tz_localize(None) if aware else cutoff


def describe(canon: pd.DataFrame, cutoff: pd.Timestamp | None, regime: Regime, aware: bool) -> dict:
    """Realised delay of one regime and how many frauds it leaves unknown."""
    fraud = canon[canon['is_fraud']]
    delays = (fraud[regime.column] - fraud['event_time']).dt.total_seconds() / 86_400
    out = {
        'column': regime.column,
        **{k: v for k, v in asdict(regime).items() if k != 'name'},
        'observed_median_delay_days': float(delays.median()) if len(delays) else 0.0,
        'observed_mean_delay_days': float(delays.mean()) if len(delays) else 0.0,
    }
    cutoffs = [cutoff] if cutoff is not None else canon['event_time'].quantile([0.5, 0.7, 0.8])
    censoring = []
    for c in cutoffs:
        known = fraud[fraud['event_time'] <= c]
        unreported = int((known[regime.column] > c).sum())
        censoring.append({
            'cutoff': str(pd.Timestamp(c).tz_localize('UTC') if aware else pd.Timestamp(c)),
            'frauds_up_to_cutoff': len(known),
            'unreported_at_cutoff': unreported,
            'share': unreported / len(known) if len(known) else 0.0,
        })  # fmt: skip
    if cutoff is not None:
        out['censored_at_cutoff'] = censoring[0]
    else:
        out['censored_at_time_quantiles'] = dict(zip(['p50', 'p70', 'p80'], censoring))
    return out


def report(card: dict) -> None:
    print(
        f'{card["rows"]:,} rows, {card["frauds"]:,} frauds in {card["n_campaigns"]:,} '
        f'campaigns (largest {card["largest_campaign"]:,}, gap {card["campaign_gap"]})'
    )
    for r in card['regimes']:
        cap = f', cap {r["max_delay_days"]:g}d' if r['max_delay_days'] else ''
        print(
            f'{r["column"]}: median {r["median_days"]:g}d, sigma {r["sigma"]:g}{cap}; '
            f'realised median {r["observed_median_delay_days"]:.1f}d, '
            f'mean {r["observed_mean_delay_days"]:.1f}d'
        )
        rows = [r['censored_at_cutoff']] if 'censored_at_cutoff' in r else []
        rows += r.get('censored_at_time_quantiles', {}).values()
        for c in rows:
            print(
                f'  unreported at {c["cutoff"]}: {c["unreported_at_cutoff"]:,} of '
                f'{c["frauds_up_to_cutoff"]:,} frauds ({100 * c["share"]:.1f}%)'
            )
    print(f'wrote {card["output"]} and its .json record')


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f'error: input file {str(args.input)!r} does not exist')
    if not args.output.parent.is_dir():
        raise SystemExit(f'error: output directory {str(args.output.parent)!r} does not exist')
    df = read_table(args.input, args.entity)
    new_columns = ['campaign_id', *(r.column for r in args.regimes)]
    clash = [c for c in new_columns if c in df.columns]
    if clash:
        raise SystemExit(f'error: input already has column(s) {clash}; drop or rename them first')

    canon, aware = canonicalize(df, args)
    cutoff = utc_cutoff(args.cutoff, aware)
    canon['campaign_id'] = assign_campaigns(canon, args.campaign_gap)
    try:
        for regime in args.regimes:
            canon[regime.column] = assign_reported_at(canon, regime, args.seed)
    except ValueError as exc:
        raise SystemExit(f'error: {exc}') from exc

    out = df.copy()
    out['campaign_id'] = canon['campaign_id'].array
    for regime in args.regimes:
        reported = canon[regime.column]
        # Timezone-aware input gets a UTC-aware reported_at, so it compares
        # directly with the input's own time column and cutoffs.
        out[regime.column] = (reported.dt.tz_localize('UTC') if aware else reported).array
    write_table(out, args.output)

    sizes = canon['campaign_id'].dropna().value_counts()
    card = {
        'script': 'add_label_delay.py',
        'version': VERSION,
        'note': 'reported_at is synthetic; it is not observed reporting behaviour',
        'input': str(args.input),
        'output': str(args.output),
        'columns': {'entity': args.entity, 'time': args.time, 'fraud': args.fraud},
        'time_origin': args.time_origin,
        'time_unit': args.time_unit if args.time_origin else None,
        'reported_at_timezone': 'UTC' if aware else None,
        'preset': args.preset,
        'seed': args.seed,
        'campaign_gap': str(args.campaign_gap),
        'rows': len(canon),
        'frauds': int(canon['is_fraud'].sum()),
        'n_campaigns': int(sizes.size),
        'largest_campaign': int(sizes.max()) if sizes.size else 0,
        'median_campaign_size': float(sizes.median()) if sizes.size else 0.0,
        'regimes': [describe(canon, cutoff, r, aware) for r in args.regimes],
    }
    args.sidecar.write_text(json.dumps(card, indent=2) + '\n')
    report(card)


if __name__ == '__main__':
    sys.exit(main())
