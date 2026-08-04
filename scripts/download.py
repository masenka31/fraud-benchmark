#!/usr/bin/env python
"""Stage 1: fetch a dataset's raw files and stop.

    python scripts/download.py --dataset sparkov          # one
    python scripts/download.py --all                      # every registered dataset
    python scripts/download.py --all --exclude-noncommercial
    python scripts/download.py --dataset ibm_ccf --force  # re-fetch a cached copy

Everything lands in `data/raw/<name>/` and stays there. `scripts/prepare.py` finds it
and skips the fetch, so a slow download can be done once, separately, and re-run
freely -- an already-populated directory is left alone unless `--force`.

Kaggle credentials are required, and IEEE-CIS additionally needs its competition rules
accepted once in a browser; see docs/kaggle-setup.md. A dataset that cannot be fetched
fails on its own and the rest continue.

`fraud-benchmark download` is the same stage over the same function, for someone who
wants prepared data rather than the study.
"""

from __future__ import annotations

import argparse

from fraud_benchmark.data.config import load_config
from fraud_benchmark.data.pipeline import download
from fraud_benchmark.data.selection import (
    add_selection_arguments,
    dataset_names,
    require_one_selection,
    run_over,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_selection_arguments(parser, "download")
    args = parser.parse_args(argv)
    require_one_selection(parser, args)

    config = load_config(args.config)
    return run_over(
        dataset_names(args.dataset, args.all),
        lambda name: download(name, config, force=args.force),
        verb="fetched",
        exclude_noncommercial=args.exclude_noncommercial,
    )


if __name__ == "__main__":
    raise SystemExit(main())
