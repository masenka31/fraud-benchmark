#!/usr/bin/env python
"""Emit the sbatch files for the leakage ablation, plus a submit script.

Cluster facts this encodes: `cpu` is 48-core / 384 GB nodes with a 1-day limit,
`cpulong` allows 3 days, the default account is `smidlva1`, and there is no
module system -- jobs call the venv's Python by absolute path.

Jobs are small and numerous on purpose. The grid is 18 model jobs that run in
parallel across the cluster, so wall-clock is bounded by the slowest single job
rather than by their sum.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "bin" / "python"
ACCOUNT = "smidlva1"

DATASETS = (
    "ibm_ccf",
    "saml_d",
    "sparkov",
    "sparkov_slow",
)

# ibm_ccf runs oracle only: it has 3 censored train labels out of 24,924, so the
# censored regime is identical to oracle and would cost roughly eight hours of
# the most expensive fits in the study to confirm arithmetic.
CELLS = tuple(
    (dataset, feature_set, regime)
    for dataset in DATASETS
    for feature_set in ("leaky", "clean")
    for regime in (("oracle",) if dataset == "ibm_ccf" else ("oracle", "censored"))
)

FEATURE_JOBS = DATASETS

_BIG = {"ibm_ccf"}


def _resources(dataset: str, stage: str) -> tuple[str, str]:
    if dataset in _BIG:
        return "cpulong", "24:00:00" if stage == "cell" else "12:00:00"
    return "cpu", "12:00:00" if stage == "cell" else "08:00:00"


def render_feature_job(dataset: str) -> str:
    partition, walltime = _resources(dataset, "features")
    return f"""#!/bin/bash
#SBATCH --job-name=feat_{dataset}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/feat_{dataset}_%j.out
#SBATCH --error={REPO}/results/logs/feat_{dataset}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
{PYTHON} -m fraud_benchmark.ablation.build_features {dataset}
"""


def render_cell_job(dataset: str, feature_set: str, regime: str) -> str:
    partition, walltime = _resources(dataset, "cell")
    name = f"{dataset}_{feature_set}_{regime}"
    # The trivial rule is fixed, so it is emitted once per dataset -- from the
    # leaky/oracle cell, which every dataset has.
    rule = " --include-rule" if (feature_set, regime) == ("leaky", "oracle") else ""
    # One results file per job. Eighteen jobs appending to a single file would
    # race: a concurrent append is only atomic below PIPE_BUF, and an ibm_ccf
    # record carrying a 45-name feature list plus both split scores approaches
    # that. summarize() reads the directory.
    results = f"{REPO}/results/runs/{name}.jsonl"
    return f"""#!/bin/bash
#SBATCH --job-name=abl_{name}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/abl_{name}_%j.out
#SBATCH --error={REPO}/results/logs/abl_{name}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
{PYTHON} -m fraud_benchmark.ablation.cell {dataset} {feature_set} {regime}{rule} --results {results}
"""


def write_all(directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {directory}", ""]
    for dataset in FEATURE_JOBS:
        (directory / f"feat_{dataset}.sbatch").write_text(render_feature_job(dataset))
        lines.append(f'feat_{dataset}=$(sbatch --parsable feat_{dataset}.sbatch)')
    lines.append("")
    for dataset, feature_set, regime in CELLS:
        name = f"{dataset}_{feature_set}_{regime}"
        (directory / f"abl_{name}.sbatch").write_text(
            render_cell_job(dataset, feature_set, regime)
        )
        lines.append(
            f'sbatch --dependency=afterok:$feat_{dataset} abl_{name}.sbatch'
        )
    lines.append("")
    submit = directory / "submit_all.sh"
    submit.write_text("\n".join(lines))
    submit.chmod(0o755)


if __name__ == "__main__":
    target = REPO / "scripts" / "slurm" / "jobs"
    write_all(target)
    (REPO / "results" / "logs").mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(FEATURE_JOBS) + len(CELLS)} sbatch files to {target}")
