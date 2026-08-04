#!/usr/bin/env python
"""Show what the study measured, from `results/experiments/`.

    python scripts/summarize.py                  # print to the terminal
    python scripts/summarize.py --write          # also write results/experiments.md
    python scripts/summarize.py --json           # the flat table, for a notebook

Argparse and an output path. Every decision about what the numbers mean -- the noise
threshold, the baseline each cell is compared against, what counts as *not run* --
lives in `fraud_benchmark.experiments.summary`, so this file has nothing to disagree
with the figures about.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_benchmark.experiments.summary import RESULTS_DIR
from fraud_benchmark.experiments.summary import SUMMARY
from fraud_benchmark.experiments.summary import flatten
from fraud_benchmark.experiments.summary import load
from fraud_benchmark.experiments.summary import print_table
from fraud_benchmark.experiments.summary import render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY)
    parser.add_argument("--write", action="store_true", help="write the markdown summary as well")
    parser.add_argument("--json", action="store_true", help="print the flat table as JSON and stop")
    args = parser.parse_args(argv)

    rows = flatten(load(args.results_dir))
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print_table(rows)
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(rows))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
