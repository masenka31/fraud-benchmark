#!/usr/bin/env python
"""Emit the sbatch files for the feature builds, plus a script that submits them.

Generated rather than hand-written so `scripts/slurm/jobs/` can stay gitignored
without losing anything: a hand-edited job is gone on the next clone.

One job per experimental dataset, each running that dataset's module in
`fraud_benchmark.experiments.features`. They are independent -- no dependencies,
nothing to sequence -- so `submit_all.sh` fires all three at once.

Cluster facts this encodes: `cpu` is 48-core / 384 GB nodes with a 1-day limit,
`cpulong` allows 3 days, the default account is `smidlva1`, and there is no module
system, so jobs call the venv's Python by absolute path.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "bin" / "python"
ACCOUNT = "smidlva1"

#: (dataset, partition, memory, walltime). Every figure is measured rather than
#: estimated -- from the builds of 2026-08-03, on the full prepared datasets:
#:
#:   dataset   rows    features  runtime  peak RSS
#:   ibm_ccf   24.4M   82        843s     67.1 GB
#:   saml_d     9.5M   63        906s     20.2 GB
#:   sparkov    1.85M  50         78s      3.7 GB
#:
#: The requests below carry roughly 2.5x the measured memory and 7x the runtime, which
#: is headroom for a wider feature set rather than a guess at the current one. saml_d
#: costs more wall-clock than the dataset 2.5x its size because four of its features
#: are `rolling_distinct`, a Python-level walk over the sorted rows.
#:
#: All three fit the short partition. An earlier version put ibm_ccf on `cpulong`
#: against a 1-day limit, on the assumption that 24.4M rows with the card and user
#: tables joined on would not finish; it finishes in 14 minutes, and `cpulong` only
#: bought a slower queue.
#:
#: `sparkov_slow` has no job of its own: its only distinct column is `reported_at`,
#: which sparkov's module reads directly and carries as `reported_at_slow`.
FEATURE_JOBS = (
    ("ibm_ccf", "cpu", "160G", "02:00:00"),
    ("saml_d", "cpu", "64G", "02:00:00"),
    ("sparkov", "cpu", "32G", "00:30:00"),
)

DATASETS = tuple(dataset for dataset, *_ in FEATURE_JOBS)


def render_feature_job(dataset: str, partition: str, memory: str, walltime: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=feat_{dataset}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/feat_{dataset}_%j.out
#SBATCH --error={REPO}/results/logs/feat_{dataset}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
{PYTHON} -m fraud_benchmark.experiments.features.{dataset}
"""


def write_all(directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {directory}", ""]
    for dataset, partition, memory, walltime in FEATURE_JOBS:
        (directory / f"feat_{dataset}.sbatch").write_text(
            render_feature_job(dataset, partition, memory, walltime)
        )
        lines.append(f"sbatch feat_{dataset}.sbatch")
    lines.append("")

    submit = directory / "submit_all.sh"
    submit.write_text("\n".join(lines))
    submit.chmod(0o755)


if __name__ == "__main__":
    target = REPO / "scripts" / "slurm" / "jobs"
    write_all(target)
    (REPO / "results" / "logs").mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(FEATURE_JOBS)} sbatch files to {target}")
