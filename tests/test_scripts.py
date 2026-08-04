"""The `scripts/` surface: four stages, and its overlap with the console script.

Stages 1 to 3 are reachable two ways -- `fraud-benchmark download|prepare` for someone
who wants prepared data, and `scripts/*.py` for someone working on the study. That is
deliberate, and the risk it carries is drift: two front doors that gradually come to
mean different things. The parity tests below are the guard, and they assert *identity*
of the function each door calls rather than similarity of what each one does.

Importing a script by path is what these tests need and no others should: the point
here is the runner itself, not any decision inside it. Every decision lives in the
installed package, which is what the last test in the first block checks -- a runner
that defines a helper is one a later test will be tempted to reach in and import.
"""

import importlib.util
from pathlib import Path

import pytest

from fraud_benchmark.data import cli, pipeline, selection
from fraud_benchmark.experiments import features

SCRIPTS = Path(__file__).parent.parent / "scripts"


def load(name: str):
    """One script, imported from its path under `scripts/`."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(
        f"paths:\n"
        f"  raw: {tmp_path / 'raw'}\n"
        f"  processed: {tmp_path / 'processed'}\n"
        f"split:\n"
        f"  ratios: [0.6, 0.2, 0.2]\n"
    )
    return path


# --- the four stages exist, and each is a runner ------------------------------


STAGES = ("download", "prepare", "features", "run_experiment")


@pytest.mark.parametrize("name", STAGES)
def test_every_stage_has_a_script(name):
    assert (SCRIPTS / f"{name}.py").exists()


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_script_returns_an_exit_code_rather_than_calling_exit(name):
    """So `main` is callable from a test and from another script."""
    module = load(name)
    assert callable(module.main)


@pytest.mark.parametrize("name", STAGES + ("summarize", "slurm/generate"))
def test_a_runner_defines_nothing_but_its_command_line(name):
    """A runner that grows a helper is a runner a test has to reach into `scripts/` to
    exercise, which is how this repository ended up with two sys.path hacks. Counting
    lines would only measure the docstring; what matters is that no name defined here
    is worth importing.

    `build_parser` is allowed: it is the command line, and a test that wants to assert
    a flag exists should not have to run the stage to find out.
    """
    body = (SCRIPTS / f"{name}.py").read_text()
    assert "sys.path" not in body
    defined = {
        line.split("(")[0].removeprefix("def ").removeprefix("class ")
        for line in body.splitlines()
        if line.startswith(("def ", "class "))
    }
    assert defined <= {"main", "build_parser"}, f"{name} defines {defined - {'main'}}"


# --- stages 1-3 are one function behind two doors -----------------------------


def test_download_is_one_function():
    assert load("download").download is pipeline.download
    assert cli.download is pipeline.download


def test_prepare_is_one_function():
    assert load("prepare").prepare is pipeline.prepare
    assert cli.prepare is pipeline.prepare


def test_both_doors_agree_on_what_all_means():
    """`--all` and `--exclude-noncommercial` are answered in one place for both."""
    for name in ("download", "prepare"):
        module = load(name)
        assert module.dataset_names is selection.dataset_names
        assert module.run_over is selection.run_over
        assert module.add_selection_arguments is selection.add_selection_arguments
    assert cli.dataset_names is selection.dataset_names
    assert cli.run_over is selection.run_over


def test_the_experiment_has_no_console_command():
    """It needs the dev extras (sklearn, xgboost, torch), so a subcommand for it would
    make `fraud-benchmark list` fail on a base install."""
    commands = cli.build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"list", "download", "prepare", "info"}


# --- stage 1 fetches and stops -------------------------------------------------


@pytest.fixture
def fake_fetch(monkeypatch):
    fetched = []

    def _fetch(source, dest, *, force=False):
        fetched.append((dest.name, force))
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr("fraud_benchmark.data.pipeline.fetch", _fetch)
    return fetched


def test_download_fetches_one_dataset_and_does_not_prepare(
    tmp_path, config_file, fake_fetch, capsys
):
    assert load("download").main(["--dataset", "paysim", "--config", str(config_file)]) == 0
    assert fake_fetch == [("paysim", False)]
    assert not (tmp_path / "processed").exists()
    assert "fetched" in capsys.readouterr().out


def test_download_all_fetches_every_registered_dataset(config_file, fake_fetch):
    assert load("download").main(["--all", "--config", str(config_file)]) == 0
    assert len(fake_fetch) == 8


def test_download_forwards_force(config_file, fake_fetch):
    load("download").main(["--dataset", "paysim", "--force", "--config", str(config_file)])
    assert fake_fetch == [("paysim", True)]


def test_download_skips_noncommercial_when_asked(config_file, fake_fetch):
    assert load("download").main(
        ["--all", "--exclude-noncommercial", "--config", str(config_file)]
    ) == 0
    names = {name for name, _ in fake_fetch}
    assert "banksim" not in names and "saml_d" not in names
    assert "paysim" in names


@pytest.mark.parametrize("name", ["download", "prepare"])
def test_a_data_stage_requires_exactly_one_target(name, config_file):
    module = load(name)
    with pytest.raises(SystemExit):
        module.main(["--config", str(config_file)])
    with pytest.raises(SystemExit):
        module.main(["--dataset", "paysim", "--all", "--config", str(config_file)])


# --- stage 2 runs the pipeline -------------------------------------------------


def test_prepare_writes_a_prepared_dataset(tmp_path, config_file, monkeypatch):
    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setattr(
        "fraud_benchmark.data.pipeline.fetch",
        lambda source, dest, *, force=False: fixtures / dest.name,
    )
    assert load("prepare").main(["--dataset", "paysim", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


# --- stage 3 covers the experimental datasets only ----------------------------


def test_features_all_means_the_three_with_a_module():
    assert set(features.DATASETS) == {"ibm_ccf", "saml_d", "sparkov"}


def test_sparkov_slow_has_no_feature_module():
    """Its only distinct column is reported_at, carried by sparkov's parquet."""
    assert "sparkov_slow" not in features.DATASETS


def test_features_rejects_a_dataset_with_no_module():
    with pytest.raises(SystemExit):
        load("features").main(["--dataset", "paysim"])
    with pytest.raises(ValueError, match="no feature module"):
        features.build("paysim")


def test_features_requires_exactly_one_target():
    with pytest.raises(SystemExit):
        load("features").main([])
    with pytest.raises(SystemExit):
        load("features").main(["--dataset", "sparkov", "--all"])


def test_features_builds_each_dataset_and_forwards_the_directories(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(
        "fraud_benchmark.experiments.features.build",
        lambda name, argv=None: built.append((name, tuple(argv or ()))),
    )
    module = load("features")
    monkeypatch.setattr(module, "build", features.build)

    assert module.main(["--all", "--features-dir", str(tmp_path)]) == 0
    assert [name for name, _ in built] == list(features.DATASETS)
    assert built[0][1] == ("--features-dir", str(tmp_path))


def test_features_continues_past_a_dataset_that_was_never_prepared(monkeypatch, capsys):
    def flaky(name, argv=None):
        if name == "ibm_ccf":
            raise FileNotFoundError("data/processed/ibm_ccf/data.parquet")
        return Path(f"data/features/{name}.parquet")

    module = load("features")
    monkeypatch.setattr(module, "build", flaky)

    assert module.main(["--all"]) == 1
    captured = capsys.readouterr()
    assert "ibm_ccf: FAILED" in captured.err
    assert "1 of 3 failed" in captured.err
