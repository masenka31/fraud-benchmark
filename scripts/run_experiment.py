#!/usr/bin/env python
"""Run one experiment: a dataset, a model, and the four axes.

Argparse and an output path. Everything importable lives in
`fraud_benchmark.experiments.experiment`, so this file has no logic to disagree with
a future runner about.

    # the defaults: xgboost, no history, true labels, artifacts dropped
    python scripts/run_experiment.py --dataset sparkov

    # history of 10, and the label-delay contrast the delay regimes exist for
    python scripts/run_experiment.py --dataset sparkov --history 10 --label-delay off
    python scripts/run_experiment.py --dataset sparkov --history 10 --label-delay on
    python scripts/run_experiment.py --dataset sparkov --history 10 --label-delay slow

    # the MLP on IBM CCF, artifacts included for the contrast
    python scripts/run_experiment.py --dataset ibm_ccf --model mlp \\
        --history 10 --artifacts keep

Every run uses every row. There is no subsampling: a model fitted on part of a dataset
is not comparable to one fitted on all of it, so a gap between two runs would measure
how much data each saw rather than what the axis changed.

Each run appends one JSON line to `--out` and prints a readable summary. Requires
`data/features/<dataset>.parquet`; build it with
`python -m fraud_benchmark.experiments.features.<dataset>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fraud_benchmark.experiments.estimators import MODELS
from fraud_benchmark.experiments.experiment import (
    ARTIFACT_MODES,
    DATASETS,
    DEFAULT_RESULTS,
    LABEL_DELAYS,
    SPLITS,
    ExperimentConfig,
    ExperimentError,
    append_record,
    describe,
    run,
)
from fraud_benchmark.experiments.features.util import FEATURE_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="See docs/experiments.md for what each axis measures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=MODELS,
        help="default: xgboost -- indifferent to the ordinal codes, handles NaN "
        "natively, needs no standardisation",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        metavar="N",
        help="lags of the dataset's HISTORY_COLUMNS concatenated onto each row "
        "(default: 0, no history)",
    )
    parser.add_argument(
        "--label-delay",
        default="off",
        choices=LABEL_DELAYS,
        help="off: true labels. on: train labels known at the train cutoff. "
        "slow: the same against reported_at_slow (sparkov only). "
        "Val and test always use true labels",
    )
    parser.add_argument(
        "--artifacts",
        default="drop",
        choices=ARTIFACT_MODES,
        help="drop (default) excludes the artifact_ columns. On IBM CCF, keeping "
        "them is what produces a score measuring the generator's geography",
    )
    parser.add_argument("--split", default="standard", choices=SPLITS)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        metavar="N",
        help="default: 0 1 2. Reported as mean +/- population sd",
    )
    parser.add_argument(
        "--history-columns",
        default="default",
        choices=("default", "all"),
        help="default: the dataset module's HISTORY_COLUMNS. 'all' lags every "
        "feature, which on IBM CCF at 10 lags is ~88 GB",
    )
    parser.add_argument("--features-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"JSONL to append one record to (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare the matrix and report its shape, then stop without fitting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        history=args.history,
        label_delay=args.label_delay,
        artifacts=args.artifacts,
        split=args.split,
        seeds=tuple(args.seeds),
        history_columns=args.history_columns,
    )

    try:
        config.validate()
    except ExperimentError as error:
        # A rejected combination is a usage error, not a crash: exit 2 like argparse.
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"{config.dataset} / {config.model} / history={config.history} / "
        f"delay={config.label_delay} / artifacts={config.artifacts} / "
        f"split={config.split} / seeds={list(config.seeds)}",
        flush=True,
    )

    try:
        if args.dry_run:
            from fraud_benchmark.experiments.experiment import prepare

            prepared = prepare(config, args.features_dir)
            for note in prepared.notes:
                print(f"  {note}")
            print(
                f"  matrix {prepared.x.shape[0]:,} x {prepared.x.shape[1]} "
                f"({prepared.x.nbytes / 1e9:.1f} GB), "
                f"{len(prepared.categorical)} coded categorical column(s), "
                f"one-hot width {sum(prepared.cardinalities)}"
            )
            print("  dry run: nothing fitted")
            return 0

        record = run(config, args.features_dir)
    except ExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    destination = append_record(record, args.out)
    print(describe(record), flush=True)
    print(f"  appended to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
