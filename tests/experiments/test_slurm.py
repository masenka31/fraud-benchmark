"""The generated sbatch files.

`scripts/slurm/jobs/` is gitignored, so every job must come out of the generator.
A hand-written one survives locally and is gone on the next clone.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "slurm"))

from generate import DATASETS, FEATURE_JOBS, render_feature_job, write_all


def test_there_is_one_job_per_experimental_dataset():
    assert set(DATASETS) == {"ibm_ccf", "saml_d", "sparkov"}


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


def test_the_ibm_ccf_job_uses_the_long_partition():
    """24.4M rows with the card and user tables joined on will not finish inside
    the 1-day cpu limit reliably."""
    sizing = {dataset: partition for dataset, partition, *_ in FEATURE_JOBS}
    assert sizing["ibm_ccf"] == "cpulong"
    assert sizing["sparkov"] == "cpu"


def test_a_job_declares_its_resources():
    script = render_feature_job("ibm_ccf", "cpulong", "250G", "24:00:00")
    assert "--cpus-per-task=4" in script
    assert "--mem=250G" in script
    assert "--time=24:00:00" in script
    assert "--partition=cpulong" in script


@pytest.mark.parametrize("dataset,partition,memory,walltime", FEATURE_JOBS)
def test_a_job_fails_loudly_rather_than_writing_a_partial_result(
    dataset, partition, memory, walltime
):
    assert "set -euo pipefail" in render_feature_job(dataset, partition, memory, walltime)


def test_write_all_emits_every_job_and_a_submit_script(tmp_path):
    write_all(tmp_path)
    emitted = {path.name for path in tmp_path.glob("*.sbatch")}
    assert emitted == {f"feat_{dataset}.sbatch" for dataset in DATASETS}

    submit = (tmp_path / "submit_all.sh").read_text()
    for dataset in DATASETS:
        assert f"sbatch feat_{dataset}.sbatch" in submit


def test_the_submit_script_declares_no_dependencies():
    """The three builds are independent, so nothing waits on anything."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        write_all(Path(directory))
        assert "--dependency" not in (Path(directory) / "submit_all.sh").read_text()
