#!/usr/bin/env python
"""Stage 3: a prepared dataset → the feature parquet its models read.

    python scripts/features.py --dataset sparkov     # -> data/features/sparkov.parquet
    python scripts/features.py --all                 # all three, largest first

Only the three experimental datasets have a feature module -- `ibm_ccf`, `saml_d`,
`sparkov` -- so `--all` means those, not the eight registered ones. `sparkov_slow` is
not among them: its only distinct column is `reported_at`, which sparkov's module reads
directly and carries as `reported_at_slow`, so one parquet holds both regimes.

Each module names every feature it produces, and reading that one file is meant to be
the whole answer to what its model sees. This script only chooses which of them to run;
`python -m fraud_benchmark.experiments.features.sparkov` remains equivalent for one
dataset, and is what the sbatch jobs call.

Requires stage 2 to have run: a missing `data/processed/<name>/data.parquet` fails that
dataset alone and the others continue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraud_benchmark.data.selection import STAGE_ERRORS
from fraud_benchmark.experiments.features import DATASETS
from fraud_benchmark.experiments.features import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', choices=DATASETS, help='dataset name')
    parser.add_argument('--all', action='store_true', help='build every experimental dataset')
    parser.add_argument('--processed-dir', type=Path, default=None)
    parser.add_argument('--features-dir', type=Path, default=None)
    args = parser.parse_args(argv)

    if bool(args.dataset) == bool(args.all):
        parser.error('give exactly one of: --dataset, or --all')

    # Forwarded rather than re-declared, so a module stays the authority on its own
    # defaults.
    forwarded: list[str] = []
    if args.processed_dir is not None:
        forwarded += ['--processed-dir', str(args.processed_dir)]
    if args.features_dir is not None:
        forwarded += ['--features-dir', str(args.features_dir)]

    names = list(DATASETS) if args.all else [args.dataset]
    failures: list[str] = []
    for name in names:
        try:
            build(name, forwarded)
        except STAGE_ERRORS as exc:
            # Same rule the data stages follow: one dataset that has not been prepared
            # must not stop the ones that have.
            print(f'{name}: FAILED {exc}', file=sys.stderr)
            failures.append(name)

    if failures:
        print(f'\n{len(failures)} of {len(names)} failed: {", ".join(failures)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
