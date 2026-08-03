#!/usr/bin/env python
"""Emit every sbatch file this project runs, plus a submit script for the ablation.

Two families, both generated here so that `scripts/slurm/jobs/` can stay
gitignored without losing anything:

  * the ablation grid -- 4 feature jobs and 14 cell jobs, from
    `fraud_benchmark.experiments.ablation.grid`, wired together by submit_all.sh;
  * the one-off experiments in `scripts/` -- declared in EXPERIMENTS below. These
    are submitted by hand, one at a time, and each expects
    `data/features/ibm_ccf.parquet` to exist already.

Cluster facts this encodes: `cpu` is 48-core / 384 GB nodes with a 1-day limit,
`cpulong` allows 3 days, the default account is `smidlva1`, and there is no
module system -- jobs call the venv's Python by absolute path.

Jobs are small and numerous on purpose. The grid is 14 model jobs that run in
parallel across the cluster, so wall-clock is bounded by the slowest single job
rather than by their sum.
"""

from __future__ import annotations

from pathlib import Path

# The grid itself lives in the package, so summarize.py can report which cells
# have no results yet. Re-exported here because this module is its historical home.
from fraud_benchmark.experiments.ablation.grid import CELLS, DATASETS  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "bin" / "python"
ACCOUNT = "smidlva1"

FEATURE_JOBS = DATASETS

_BIG = {"ibm_ccf"}

#: The one-off experiments: (job name, script, arguments, results file, memory,
#: walltime). All run on IBM CCF, all on the `cpulong` partition.
#:
#: Memory is the reason these are declared rather than shared: the flattened
#: sequence window is a 15-17 GiB float32 block, and the v2 variant adds 26
#: columns on top, which is why it asks for 280G where the tabular jobs need 128G.
#:
#: `--encoding ordinal` is passed EXPLICITLY on the two ordinal MLP jobs. The
#: script's default is `onehot`, so omitting the flag -- as the original
#: hand-written jobs did, before the flag existed -- would write one-hot results
#: into a file named for the ordinal run.
EXPERIMENTS = (
    ("italy_holdout", "italy_holdout.py", (), "italy_holdout.jsonl", "128G", "24:00:00"),
    (
        "geo_dilution", "geo_dilution.py", ("--dataset", "ibm_ccf"),
        "geo_dilution.jsonl", "128G", "12:00:00",
    ),
    (
        "ibm_featv2", "ibm_features_v2.py", (),
        "ibm_features_v2.jsonl", "250G", "48:00:00",
    ),
    (
        "seq_standard", "seq_window.py", ("--split", "standard"),
        "seq_window_standard.jsonl", "250G", "24:00:00",
    ),
    (
        "seq_italy", "seq_window.py", ("--split", "italy_holdout"),
        "seq_window_italy_holdout.jsonl", "250G", "24:00:00",
    ),
    (
        "mlp_standard", "seq_mlp.py", ("--split", "standard", "--encoding", "ordinal"),
        "seq_mlp_standard.jsonl", "250G", "48:00:00",
    ),
    (
        "mlp_italy", "seq_mlp.py",
        ("--split", "italy_holdout", "--encoding", "ordinal"),
        "seq_mlp_italy_holdout.jsonl", "250G", "48:00:00",
    ),
    (
        "mlpoh_standard", "seq_mlp.py", ("--split", "standard", "--encoding", "onehot"),
        "seq_mlp_onehot_standard.jsonl", "250G", "48:00:00",
    ),
    (
        "mlpoh_italy", "seq_mlp.py",
        ("--split", "italy_holdout", "--encoding", "onehot"),
        "seq_mlp_onehot_italy_holdout.jsonl", "250G", "48:00:00",
    ),
    (
        "mlpv2_standard", "seq_mlp_v2.py", ("--split", "standard"),
        "seq_mlp_v2_standard.jsonl", "280G", "48:00:00",
    ),
    (
        "mlpv2_italy", "seq_mlp_v2.py", ("--split", "italy_holdout"),
        "seq_mlp_v2_italy_holdout.jsonl", "280G", "48:00:00",
    ),
)


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
{PYTHON} -m fraud_benchmark.experiments.build_features {dataset}
"""


def render_cell_job(dataset: str, feature_set: str, regime: str) -> str:
    partition, walltime = _resources(dataset, "cell")
    name = f"{dataset}_{feature_set}_{regime}"
    # The trivial rule is fixed, so it is emitted once per dataset -- from the
    # leaky/oracle cell, which every dataset has.
    rule = " --include-rule" if (feature_set, regime) == ("leaky", "oracle") else ""
    # One results file per job. Fourteen jobs appending to a single file would
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
{PYTHON} -m fraud_benchmark.experiments.ablation.cell {dataset} {feature_set} {regime}{rule} --results {results}
"""


def render_experiment_job(
    name: str, script: str, arguments: tuple[str, ...], results: str,
    memory: str, walltime: str,
) -> str:
    """One of the `scripts/` experiments. Submitted by hand, not by submit_all.sh."""
    extra = (" \\\n  " + " ".join(arguments)) if arguments else ""
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={ACCOUNT}
#SBATCH --partition=cpulong
#SBATCH --cpus-per-task=4
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={REPO}/results/logs/{name}_%j.out
#SBATCH --error={REPO}/results/logs/{name}_%j.err

set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
{PYTHON} scripts/{script}{extra} \\
  --out {REPO}/results/{results}
"""


def write_all(directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for name, script, arguments, results, memory, walltime in EXPERIMENTS:
        (directory / f"{name}.sbatch").write_text(
            render_experiment_job(name, script, arguments, results, memory, walltime)
        )

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
    total = len(FEATURE_JOBS) + len(CELLS) + len(EXPERIMENTS)
    print(f"wrote {total} sbatch files to {target}")
