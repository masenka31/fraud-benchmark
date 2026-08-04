#!/usr/bin/env python
"""Stage 2: raw files → one prepared dataset per name, on the shared schema.

    python scripts/prepare.py --dataset sparkov           # one
    python scripts/prepare.py --all                       # every registered dataset
    python scripts/prepare.py --all --exclude-noncommercial
    python scripts/prepare.py --dataset ibm_ccf --force    # re-fetch, then re-prepare

Downloads whatever stage 1 has not already fetched, then canonicalizes to the shared
schema, joins the auxiliary tables, assigns the temporal split, groups frauds into
campaigns and draws each campaign's `reported_at`. Output is
`data/processed/<name>/data.parquet` plus the `dataset_card.json` recording provenance,
counts, split boundaries, delay parameters and caveats.

One dataset failing does not stop the others, so `--all` over eight sources is worth
running even when one of them is unavailable.

`fraud-benchmark prepare` is the same stage over the same function, for someone who
wants prepared data rather than the study.
"""

from __future__ import annotations

import argparse

from fraud_benchmark.data.config import load_config
from fraud_benchmark.data.pipeline import prepare
from fraud_benchmark.data.selection import add_selection_arguments
from fraud_benchmark.data.selection import dataset_names
from fraud_benchmark.data.selection import require_one_selection
from fraud_benchmark.data.selection import run_over


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_selection_arguments(parser, "prepare")
    args = parser.parse_args(argv)
    require_one_selection(parser, args)

    config = load_config(args.config)
    return run_over(
        dataset_names(args.dataset, args.all),
        lambda name: prepare(name, config, force=args.force),
        exclude_noncommercial=args.exclude_noncommercial,
    )


if __name__ == "__main__":
    raise SystemExit(main())
