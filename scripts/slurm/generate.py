#!/usr/bin/env python
"""Write every sbatch file this project runs into `scripts/slurm/jobs/`.

    python scripts/slurm/generate.py             # -> scripts/slurm/jobs/
    scripts/slurm/jobs/submit_all.sh             # submits the study from cold

Argparse and an output directory. What each job asks the cluster for, and the
dependency chaining that lets the study be submitted before any feature parquet
exists, live in `fraud_benchmark.experiments.slurm`.

Add a job by declaring it in `grid.py` or that module's feature table, never by
hand-writing a file into `jobs/` -- the directory is gitignored, so a hand-edited job
is gone on the next clone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_benchmark.experiments.grid import CELLS
from fraud_benchmark.experiments.slurm import FEATURE_JOBS, write_all

#: This runner's own checkout, which is the one the generated jobs should cd into --
#: `scripts/slurm/generate.py`, so two parents up.
REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO, help="checkout jobs run in")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <repo>/scripts/slurm/jobs"
    )
    args = parser.parse_args(argv)

    target = args.out or args.repo / "scripts" / "slurm" / "jobs"
    write_all(target, args.repo)
    (args.repo / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (args.repo / "results" / "experiments").mkdir(parents=True, exist_ok=True)

    total = len(FEATURE_JOBS) + len(CELLS)
    print(
        f"wrote {total} sbatch files to {target} "
        f"({len(FEATURE_JOBS)} feature builds, {len(CELLS)} experiments)"
    )
    print(f"submit the study with: {target / 'submit_all.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
