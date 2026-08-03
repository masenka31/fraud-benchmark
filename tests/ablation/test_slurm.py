import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "slurm"))

from generate import (
    CELLS,
    EXPERIMENTS,
    FEATURE_JOBS,
    render_cell_job,
    render_experiment_job,
    render_feature_job,
    write_all,
)


def test_ibm_ccf_has_no_censored_cell():
    """3 censored labels of 24,924 -- the contrast is arithmetically null."""
    assert ("ibm_ccf", "leaky", "censored") not in CELLS
    assert ("ibm_ccf", "clean", "censored") not in CELLS


def test_ibm_ccf_keeps_both_oracle_cells():
    assert ("ibm_ccf", "leaky", "oracle") in CELLS
    assert ("ibm_ccf", "clean", "oracle") in CELLS


def test_the_cell_count_matches_the_dataset_grid():
    """4 datasets: ibm_ccf contributes 2 (oracle only), the rest 4 each."""
    assert len(CELLS) == 14


def test_every_other_dataset_has_four_cells():
    for dataset in ["saml_d", "sparkov", "sparkov_slow"]:
        assert len([c for c in CELLS if c[0] == dataset]) == 4


def test_a_cell_job_declares_cpu_and_memory():
    script = render_cell_job("sparkov", "leaky", "oracle")
    assert "--cpus-per-task=4" in script
    assert "--mem=128G" in script


def test_the_ibm_ccf_jobs_use_the_long_partition():
    """24.4M rows will not finish inside the 1-day cpu limit reliably."""
    assert "--partition=cpulong" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--partition=cpu\n" in render_cell_job("sparkov", "leaky", "oracle")


def test_the_retired_subsamples_are_gone():
    """Their fast/slow delay contrast measured nothing: 0.975 vs 0.975 at a
    0.001 seed noise floor. sparkov_slow replaces that role."""
    assert not any("subsample" in c[0] for c in CELLS)
    assert any(c[0] == "sparkov_slow" for c in CELLS)


def test_jobs_invoke_the_venv_python_by_absolute_path():
    """There is no module system on this cluster."""
    assert "/.venv/bin/python" in render_cell_job("sparkov", "leaky", "oracle")


def test_only_the_leaky_oracle_cell_requests_the_rule():
    """The rule is fixed, so it is evaluated once per dataset."""
    assert "--include-rule" in render_cell_job("ibm_ccf", "leaky", "oracle")
    assert "--include-rule" not in render_cell_job("ibm_ccf", "clean", "oracle")


def test_feature_jobs_exist_for_every_dataset():
    assert set(FEATURE_JOBS) == {"ibm_ccf", "saml_d", "sparkov", "sparkov_slow"}


def test_write_all_emits_a_submit_script_with_dependencies(tmp_path):
    write_all(tmp_path)
    submit = (tmp_path / "submit_all.sh").read_text()
    assert "--dependency=afterok" in submit
    # 4 feature + 14 cell + 11 experiment
    assert len(list(tmp_path.glob("*.sbatch"))) == 29


def test_feature_job_writes_to_the_features_dir():
    assert "build_features" in render_feature_job("sparkov")


def test_every_experiment_job_is_generated_not_hand_written(tmp_path):
    """scripts/slurm/jobs/ is gitignored, so a hand-written job is lost on clone."""
    write_all(tmp_path)
    for name, *_ in EXPERIMENTS:
        assert (tmp_path / f"{name}.sbatch").exists()


def test_the_ordinal_mlp_jobs_pass_their_encoding_explicitly():
    """seq_mlp.py defaults to --encoding onehot.

    Omitting the flag -- as the original hand-written jobs did, before it existed
    -- writes one-hot results into a file named for the ordinal run.
    """
    # seq_mlp_v2.py is excluded: it has no --encoding flag, being one-hot only.
    configurable = [e for e in EXPERIMENTS if e[1] == "seq_mlp.py"]
    assert len(configurable) == 4
    for name, _script, arguments, results, *_ in configurable:
        assert "--encoding" in arguments, f"{name} relies on the default encoding"
        encoding = arguments[arguments.index("--encoding") + 1]
        # Only the one-hot runs carry the encoding in their filename; the ordinal
        # ones predate the flag and keep their original names.
        assert (encoding == "onehot") == ("onehot" in results), (
            f"{name} runs {encoding} but writes to {results}"
        )


def test_no_experiment_job_names_a_retired_dataset():
    for entry in EXPERIMENTS:
        assert not any("subsample" in argument for argument in entry[2])


def test_an_experiment_job_declares_its_own_memory():
    """The sequence-window jobs need 250-280G; the tabular ones need 128G."""
    script = render_experiment_job(
        "seq_standard", "seq_window.py", ("--split", "standard"),
        "seq_window_standard.jsonl", "250G", "24:00:00",
    )
    assert "--mem=250G" in script
    assert "scripts/seq_window.py" in script
    assert "--split standard" in script
