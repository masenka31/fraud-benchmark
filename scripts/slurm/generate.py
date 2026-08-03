#!/usr/bin/env python
"""Emit every sbatch file this project runs, plus a script that submits them.

Generated rather than hand-written so `scripts/slurm/jobs/` can stay gitignored
without losing anything: a hand-edited job is gone on the next clone.

Two families, and the dependency between them is the reason they are generated
together:

  * **feature builds** -- one per experimental dataset, running that dataset's module
    in `fraud_benchmark.experiments.features`. Independent of each other.
  * **experiments** -- the grid in `fraud_benchmark.experiments.grid`, each cell one
    `scripts/run_experiment.py` invocation. Every cell reads a feature parquet, so
    `submit_all.sh` chains each one behind its dataset's build with
    `--dependency=afterok` and the whole study can be submitted from cold.

Each experiment writes its own JSONL under `results/experiments/`. One shared file
would race: a concurrent append is only atomic below PIPE_BUF, and a record carrying
a 200-name feature list and three seeds is well past that. `scripts/summarize.py`
reads the directory.

Cluster facts this encodes: `cpu` is 48-core / 384 GB nodes with a 1-day limit,
`cpulong` allows 3 days, the default account is `smidlva1`, and there is no module
system, so jobs call the venv's Python by absolute path.
"""

from __future__ import annotations

from pathlib import Path

from fraud_benchmark.experiments.grid import CELLS, estimate_cost

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


#: Where each experiment appends its single JSON record.
RESULTS_DIR = REPO / "results" / "experiments"


def cell_arguments(cell) -> list[str]:
    """The CLI flags for one grid cell. Only non-defaults are passed, so the command
    line reads as the difference from the baseline."""
    config = cell.config
    arguments = ["--dataset", config.dataset, "--model", config.model]
    if config.history:
        arguments += ["--history", str(config.history)]
    if config.label_delay != "off":
        arguments += ["--label-delay", config.label_delay]
    if config.artifacts != "drop":
        arguments += ["--artifacts", config.artifacts]
    if config.split != "standard":
        arguments += ["--split", config.split]
    if config.max_rows is not None:
        arguments += ["--max-rows", str(config.max_rows)]
    arguments += ["--seeds", *(str(s) for s in config.seeds)]
    return arguments


def render_experiment_job(cell) -> str:
    partition, memory, walltime = estimate_cost(cell)
    arguments = " ".join(cell_arguments(cell))
    return f"""#!/bin/bash
#SBATCH --job-name=exp_{cell.name}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/exp_{cell.name}_%j.out
#SBATCH --error={REPO}/results/logs/exp_{cell.name}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
{PYTHON} scripts/run_experiment.py {arguments} \\
  --out {RESULTS_DIR}/{cell.name}.jsonl
"""


def write_all(directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        "# Generated by scripts/slurm/generate.py -- do not edit.",
        "set -euo pipefail",
        f"cd {directory}",
        "",
        "# One feature build per dataset. Every experiment waits on its own dataset.",
    ]
    for dataset, partition, memory, walltime in FEATURE_JOBS:
        (directory / f"feat_{dataset}.sbatch").write_text(
            render_feature_job(dataset, partition, memory, walltime)
        )
        lines.append(f"feat_{dataset}=$(sbatch --parsable feat_{dataset}.sbatch)")
    lines.append("")

    group = None
    for cell in CELLS:
        (directory / f"exp_{cell.name}.sbatch").write_text(render_experiment_job(cell))
        if cell.group != group:
            group = cell.group
            lines.append(f"# {group}")
        lines.append(
            f"sbatch --dependency=afterok:$feat_{cell.config.dataset} "
            f"exp_{cell.name}.sbatch"
        )
    lines.append("")

    submit = directory / "submit_all.sh"
    submit.write_text("\n".join(lines))
    submit.chmod(0o755)


if __name__ == "__main__":
    target = REPO / "scripts" / "slurm" / "jobs"
    write_all(target)
    (REPO / "results" / "logs").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(FEATURE_JOBS) + len(CELLS)
    print(
        f"wrote {total} sbatch files to {target} "
        f"({len(FEATURE_JOBS)} feature builds, {len(CELLS)} experiments)"
    )
    print(f"submit the study with: {target / 'submit_all.sh'}")
