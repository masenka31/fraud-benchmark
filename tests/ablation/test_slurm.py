import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "slurm"))

from generate import CELLS, FEATURE_JOBS, render_cell_job, render_feature_job, write_all


def test_ibm_ccf_has_no_censored_cell():
    """3 censored labels of 24,924 -- the contrast is arithmetically null."""
    assert ("ibm_ccf", "leaky", "censored") not in CELLS
    assert ("ibm_ccf", "clean", "censored") not in CELLS


def test_ibm_ccf_keeps_both_oracle_cells():
    assert ("ibm_ccf", "leaky", "oracle") in CELLS
    assert ("ibm_ccf", "clean", "oracle") in CELLS


def test_there_are_eighteen_cells():
    assert len(CELLS) == 18


def test_every_other_dataset_has_four_cells():
    for dataset in ["ibm_ccf_subsample_fast", "ibm_ccf_subsample_slow", "saml_d", "sparkov"]:
        assert len([c for c in CELLS if c[0] == dataset]) == 4


def test_a_cell_job_declares_cpu_and_memory():
    script = render_cell_job("sparkov", "leaky", "oracle")
    assert "--cpus-per-task=4" in script
    assert "--mem=128G" in script


def test_the_ibm_ccf_jobs_use_the_long_partition():
    """24.4M rows will not finish inside the 1-day cpu limit reliably."""
    assert "--partition=cpulong" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--partition=cpu\n" in render_cell_job("sparkov", "leaky", "oracle")


def test_jobs_invoke_the_venv_python_by_absolute_path():
    """There is no module system on this cluster."""
    assert "/.venv/bin/python" in render_cell_job("sparkov", "leaky", "oracle")


def test_only_the_leaky_oracle_cell_requests_the_rule():
    """The rule is fixed, so it is evaluated once per dataset."""
    assert "--include-rule" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--include-rule" not in render_cell_job("ibm_ccf", "clean", "oracle")


def test_feature_jobs_exist_for_every_dataset():
    assert set(FEATURE_JOBS) == {
        "ibm_ccf",
        "ibm_ccf_subsample_fast",
        "ibm_ccf_subsample_slow",
        "saml_d",
        "sparkov",
    }


def test_write_all_emits_a_submit_script_with_dependencies(tmp_path):
    write_all(tmp_path)
    submit = (tmp_path / "submit_all.sh").read_text()
    assert "--dependency=afterok" in submit
    assert len(list(tmp_path.glob("*.sbatch"))) == 23


def test_feature_job_writes_to_the_features_dir():
    assert "build_features" in render_feature_job("sparkov")
