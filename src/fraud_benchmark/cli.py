"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

import fraud_benchmark.datasets  # noqa: F401  (registers all adapters)
from fraud_benchmark.config import load_config
from fraud_benchmark.datasets.base import UnknownDatasetError, get_adapter, list_datasets
from fraud_benchmark.pipeline import prepare
from fraud_benchmark.sources import FetchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-benchmark",
        description="Prepare fraud and AML benchmark datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available datasets")

    prepare_parser = subparsers.add_parser("prepare", help="download and process a dataset")
    prepare_parser.add_argument("dataset", nargs="?", help="dataset name")
    prepare_parser.add_argument("--all", action="store_true", help="prepare every dataset")
    prepare_parser.add_argument("--config", help="path to a config file")
    prepare_parser.add_argument(
        "--force", action="store_true", help="re-download even if raw files exist"
    )
    prepare_parser.add_argument(
        "--exclude-noncommercial", action="store_true",
        help="skip datasets whose licence forbids commercial use",
    )

    info_parser = subparsers.add_parser("info", help="print a prepared dataset's card")
    info_parser.add_argument("dataset")
    info_parser.add_argument("--config", help="path to a config file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "prepare":
        # Validated here rather than with a mutually exclusive group: argparse
        # handles an optional positional inside such a group unreliably.
        if bool(args.dataset) == bool(args.all):
            parser.error("give exactly one of: a dataset name, or --all")
        return _cmd_prepare(args)
    return _cmd_info(args)


def _cmd_list() -> int:
    for name in list_datasets():
        adapter = get_adapter(name)
        flag = "" if adapter.commercial_use else "  [noncommercial]"
        url = getattr(adapter.source, "url", "")
        print(f"{name:12s} {adapter.data_license:18s} {url}{flag}")
    return 0


def _cmd_prepare(args) -> int:
    config = load_config(args.config)
    names = list_datasets() if args.all else [args.dataset]

    failures: list[str] = []
    for name in names:
        try:
            adapter = get_adapter(name)
        except UnknownDatasetError as exc:
            # UnknownDatasetError subclasses KeyError, whose __str__ adds repr
            # quotes. Print the raw message instead.
            print(exc.args[0], file=sys.stderr)
            return 1

        if args.exclude_noncommercial and not adapter.commercial_use:
            print(f"{name}: skipped, licence forbids commercial use "
                  f"({getattr(adapter, 'data_license', 'unknown')})")
            continue

        try:
            out = prepare(name, config, force=args.force)
        except (FetchError, ValueError, FileNotFoundError) as exc:
            # Keep going: one unavailable dataset must not block the rest.
            print(f"{name}: FAILED {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"{name}: wrote {out}")

    if failures:
        print(f"\n{len(failures)} of {len(names)} failed: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    return 0


def _cmd_info(args) -> int:
    config = load_config(args.config)
    card = config.processed_dir / args.dataset / "dataset_card.json"
    if not card.exists():
        print(
            f"{args.dataset} has not been prepared; run: "
            f"fraud-benchmark prepare {args.dataset}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(json.loads(card.read_text()), indent=2))
    return 0
