"""The generated sbatch files.

`scripts/slurm/jobs/` is gitignored, so every job must come out of the generator.
A hand-written one survives locally and is gone on the next clone.
"""

import pytest

from fraud_benchmark.experiments.features import DATASETS as WITH_A_MODULE
from fraud_benchmark.experiments.grid import CELLS
from fraud_benchmark.experiments.slurm import (
    DATASETS,
    FEATURE_JOBS,
    cell_arguments,
    render_experiment_job,
    render_feature_job,
    write_all,
)


def test_there_is_one_job_per_experimental_dataset():
    """The job table and the feature package must name the same datasets: a job for a
    dataset with no module fails at submission, and a module with no job never runs."""
    assert set(DATASETS) == set(WITH_A_MODULE)


def test_sparkov_slow_has_no_job_of_its_own():
    """Its only distinct column is reported_at, which sparkov's module reads directly."""
    assert "sparkov_slow" not in DATASETS


def test_every_job_runs_its_dataset_module():
    for dataset, partition, memory, walltime in FEATURE_JOBS:
        script = render_feature_job(dataset, partition, memory, walltime)
        assert f"fraud_benchmark.experiments.features.{dataset}" in script


def test_jobs_invoke_the_venv_python_by_absolute_path():
    """There is no module system on this cluster."""
    assert "/.venv/bin/python" in render_feature_job("sparkov", "cpu", "64G", "08:00:00")


def test_every_job_fits_the_short_partition():
    """Measured: the widest build is 14 minutes, well inside the 1-day cpu limit.
    `cpulong` would only buy a slower queue."""
    assert {partition for _, partition, *_ in FEATURE_JOBS} == {"cpu"}


def test_memory_is_ordered_by_dataset_size():
    """ibm_ccf peaked at 67.1 GB, saml_d at 20.2, sparkov at 3.7."""
    memory = {dataset: int(mem.rstrip("G")) for dataset, _, mem, _ in FEATURE_JOBS}
    assert memory["ibm_ccf"] > memory["saml_d"] > memory["sparkov"]
    # Headroom over the measured peaks, not a guess at them.
    assert memory["ibm_ccf"] >= 2 * 67
    assert memory["saml_d"] >= 2 * 21
    assert memory["sparkov"] >= 2 * 4


def test_a_job_declares_its_resources():
    script = render_feature_job("ibm_ccf", "cpu", "160G", "02:00:00")
    assert "--cpus-per-task=4" in script
    assert "--mem=160G" in script
    assert "--time=02:00:00" in script
    assert "--partition=cpu\n" in script


@pytest.mark.parametrize("dataset,partition,memory,walltime", FEATURE_JOBS)
def test_a_job_fails_loudly_rather_than_writing_a_partial_result(
    dataset, partition, memory, walltime
):
    assert "set -euo pipefail" in render_feature_job(dataset, partition, memory, walltime)


def test_write_all_emits_both_families_and_a_submit_script(tmp_path):
    write_all(tmp_path)
    emitted = {path.name for path in tmp_path.glob("*.sbatch")}
    assert emitted == {f"feat_{dataset}.sbatch" for dataset in DATASETS} | {
        f"exp_{cell.name}.sbatch" for cell in CELLS
    }

    submit = (tmp_path / "submit_all.sh").read_text()
    for dataset in DATASETS:
        assert f"feat_{dataset}=$(sbatch --parsable feat_{dataset}.sbatch)" in submit


def test_the_feature_builds_do_not_wait_on_each_other(tmp_path):
    write_all(tmp_path)
    submit = (tmp_path / "submit_all.sh").read_text()
    for line in submit.splitlines():
        if line.startswith("feat_"):
            assert "--dependency" not in line


def test_every_experiment_waits_on_its_own_dataset(tmp_path):
    """A cell reads a feature parquet, so the study must be submittable from cold."""
    write_all(tmp_path)
    submit = (tmp_path / "submit_all.sh").read_text()
    for cell in CELLS:
        assert (
            f"sbatch --dependency=afterok:$feat_{cell.config.dataset} "
            f"exp_{cell.name}.sbatch" in submit
        ), cell.name


def test_each_experiment_writes_its_own_results_file(tmp_path):
    """One shared file would race: a concurrent append is only atomic below PIPE_BUF."""
    destinations = set()
    for cell in CELLS:
        script = render_experiment_job(cell)
        assert f"results/experiments/{cell.name}.jsonl" in script
        destinations.add(f"{cell.name}.jsonl")
    assert len(destinations) == len(CELLS)


def test_a_cell_command_line_passes_only_what_differs_from_the_default():
    """So the command reads as the difference from the baseline."""
    baseline = [c for c in CELLS if c.name == "sparkov_xgboost"][0]
    arguments = cell_arguments(baseline)
    for flag in ("--history", "--label-delay", "--artifacts", "--split", "--max-rows"):
        assert flag not in arguments

    slow = [c for c in CELLS if c.config.label_delay == "slow"][0]
    assert "--label-delay" in cell_arguments(slow)


def test_no_job_can_subsample():
    """Every run uses every row, and there is no flag left that could change that."""
    for cell in CELLS:
        assert "--max-rows" not in cell_arguments(cell), cell.name
        assert "--max-rows" not in render_experiment_job(cell), cell.name


def test_every_experiment_job_declares_its_seeds():
    for cell in CELLS:
        assert "--seeds 0 1 2" in render_experiment_job(cell)
