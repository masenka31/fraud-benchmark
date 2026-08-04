"""Which datasets a stage runs over, and what happens when one of them fails.

Two front doors reach every data stage -- the `fraud-benchmark` console script for
someone who wants prepared data, and `scripts/download.py` / `scripts/prepare.py` for
someone working on the study -- and both need the same answers: what `--all` means,
which licences `--exclude-noncommercial` drops, and whether one unavailable dataset
stops the other seven. Answering that twice is how the two surfaces drift, so it is
answered here and each door only supplies the work to do.

One dataset failing must never block the rest: `--all` over eight sources, one of which
needs competition rules accepted in a browser, would otherwise be all-or-nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path

from fraud_benchmark.data.adapters.base import UnknownDatasetError
from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.adapters.base import list_datasets
from fraud_benchmark.data.sources import FetchError

#: What a stage is allowed to fail with per dataset: the source is unreachable, the
#: raw files are not what the adapter expected, or they were never downloaded. Anything
#: else is a bug in this repository and should reach the user as a traceback.
STAGE_ERRORS = (FetchError, ValueError, FileNotFoundError)


def dataset_names(dataset: str | None, every: bool) -> list[str]:
    """The datasets named by a `<name>` / `--all` pair, in registry order."""
    return list_datasets() if every else [dataset]


def add_selection_arguments(parser, verb: str, *, positional: bool = False) -> None:
    """The flags every per-dataset stage takes, on either surface.

    The console script names its dataset positionally (`fraud-benchmark prepare
    sparkov`) and the scripts name it with a flag (`--dataset sparkov`, as
    `run_experiment.py` already does); that is the only difference between the two, and
    the rest -- `--all`, `--config`, `--force`, `--exclude-noncommercial` -- is shared
    so no stage can drift from another about what they mean.
    """
    if positional:
        parser.add_argument('dataset', nargs='?', help='dataset name')
    else:
        parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('--all', action='store_true', help=f'{verb} every dataset')
    parser.add_argument('--config', help='path to a config file')
    parser.add_argument('--force', action='store_true', help='re-download even if raw files exist')
    parser.add_argument(
        '--exclude-noncommercial',
        action='store_true',
        help='skip datasets whose licence forbids commercial use',
    )


def require_one_selection(parser, args) -> None:
    """Exit unless exactly one of a dataset name or `--all` was given.

    Checked here rather than with a mutually exclusive group: argparse handles an
    optional positional inside such a group unreliably.
    """
    if bool(args.dataset) == bool(args.all):
        parser.error('give exactly one of: a dataset name, or --all')


def run_over(
    names: Iterable[str],
    action: Callable[[str], Path | str],
    *,
    verb: str = 'wrote',
    exclude_noncommercial: bool = False,
) -> int:
    """Run `action` per dataset, reporting each. Returns a process exit code.

    `action` takes the dataset name and returns what it produced, printed after
    `verb`. Raising anything in `STAGE_ERRORS` fails that dataset alone.
    """
    names = list(names)
    failures: list[str] = []

    for name in names:
        try:
            adapter = get_adapter(name)
        except UnknownDatasetError as exc:
            # UnknownDatasetError subclasses KeyError, whose __str__ adds repr
            # quotes. Print the raw message instead.
            print(exc.args[0], file=sys.stderr)
            return 1

        if exclude_noncommercial and not adapter.commercial_use:
            print(f'{name}: skipped, licence forbids commercial use ({adapter.data_license})')
            continue

        try:
            produced = action(name)
        except STAGE_ERRORS as exc:
            # Keep going: one unavailable dataset must not block the rest.
            print(f'{name}: FAILED {exc}', file=sys.stderr)
            failures.append(name)
            continue
        print(f'{name}: {verb} {produced}')

    if failures:
        print(f'\n{len(failures)} of {len(names)} failed: {", ".join(failures)}', file=sys.stderr)
        return 1
    return 0
