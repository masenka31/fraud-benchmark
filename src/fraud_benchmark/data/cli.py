"""The `fraud-benchmark` console script: the data pipeline, and nothing after it.

Four commands over `data/`, for someone who wants prepared datasets rather than the
study built on them: `list`, `download`, `prepare`, `info`. Every one of them needs
only the base dependencies.

The experiment side has no command here on purpose. It needs the `dev` extras
(scikit-learn, xgboost, torch), so a subcommand for it would make `fraud-benchmark
list` fail on a base install -- and `scripts/` is its surface. Stages 1 to 3 are
reachable from both this script and `scripts/`, which is deliberate: the two doors call
the same functions, and `data/selection.py` answers what `--all` means for both.
"""

from __future__ import annotations

import argparse
import json
import sys

import fraud_benchmark.data.adapters  # noqa: F401  (registers all adapters)
from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.adapters.base import list_datasets
from fraud_benchmark.data.config import load_config
from fraud_benchmark.data.pipeline import download
from fraud_benchmark.data.pipeline import prepare
from fraud_benchmark.data.selection import add_selection_arguments
from fraud_benchmark.data.selection import dataset_names
from fraud_benchmark.data.selection import require_one_selection
from fraud_benchmark.data.selection import run_over


def build_parser() -> argparse.ArgumentParser:
    """The `fraud-benchmark` parser: `list`, `download`, `prepare` and `info`."""
    parser = argparse.ArgumentParser(
        prog='fraud-benchmark',
        description='Prepare fraud and AML benchmark datasets.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('list', help='list available datasets')

    download_parser = subparsers.add_parser('download', help="fetch a dataset's raw files and stop")
    add_selection_arguments(download_parser, 'download', positional=True)

    prepare_parser = subparsers.add_parser('prepare', help='download and process a dataset')
    add_selection_arguments(prepare_parser, 'prepare', positional=True)

    info_parser = subparsers.add_parser('info', help="print a prepared dataset's card")
    info_parser.add_argument('dataset')
    info_parser.add_argument('--config', help='path to a config file')

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one command line to its handler. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'list':
        return _cmd_list()
    if args.command in ('download', 'prepare'):
        require_one_selection(parser, args)
        return _cmd_download(args) if args.command == 'download' else _cmd_prepare(args)
    return _cmd_info(args)


def _cmd_list() -> int:
    """Print one line per registered dataset: name, licence, URL."""
    for name in list_datasets():
        adapter = get_adapter(name)
        flag = '' if adapter.commercial_use else '  [noncommercial]'
        url = getattr(adapter.source, 'url', '')
        print(f'{name:12s} {adapter.data_license:18s} {url}{flag}')
    return 0


def _cmd_download(args) -> int:
    """Fetch one dataset's raw files or every dataset's. Returns 1 if any failed."""
    config = load_config(args.config)
    return run_over(
        dataset_names(args.dataset, args.all),
        lambda name: download(name, config, force=args.force),
        verb='fetched',
        exclude_noncommercial=args.exclude_noncommercial,
    )


def _cmd_prepare(args) -> int:
    """Prepare one dataset or all of them. Returns 1 if any dataset failed."""
    config = load_config(args.config)
    return run_over(
        dataset_names(args.dataset, args.all),
        lambda name: prepare(name, config, force=args.force),
        exclude_noncommercial=args.exclude_noncommercial,
    )


def _cmd_info(args) -> int:
    """Print a prepared dataset's card. Returns 1 if it has not been prepared."""
    config = load_config(args.config)
    card = config.processed_dir / args.dataset / 'dataset_card.json'
    if not card.exists():
        print(
            f'{args.dataset} has not been prepared; run: fraud-benchmark prepare {args.dataset}',
            file=sys.stderr,
        )
        return 1
    print(json.dumps(json.loads(card.read_text()), indent=2))
    return 0


if __name__ == '__main__':
    # `python -m fraud_benchmark.data.cli` would otherwise import and exit silently,
    # which reads as a broken install rather than a missing entry point.
    sys.exit(main())
