import json
from pathlib import Path

import pytest

from fraud_benchmark.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "paysim"


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


@pytest.fixture
def no_download(monkeypatch):
    monkeypatch.setattr(
        "fraud_benchmark.pipeline.fetch", lambda source, dest, force=False: FIXTURE
    )


def test_list_prints_registered_datasets(capsys):
    assert main(["list"]) == 0
    assert "paysim" in capsys.readouterr().out


def test_prepare_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "paysim", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


def test_prepare_all_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "--all", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


def test_prepare_requires_a_target(config_file):
    with pytest.raises(SystemExit):
        main(["prepare", "--config", str(config_file)])


def test_prepare_rejects_both_target_forms(config_file):
    with pytest.raises(SystemExit):
        main(["prepare", "paysim", "--all", "--config", str(config_file)])


def test_info_prints_card(tmp_path, config_file, no_download, capsys):
    main(["prepare", "paysim", "--config", str(config_file)])
    capsys.readouterr()
    assert main(["info", "paysim", "--config", str(config_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["name"] == "paysim"


def test_info_on_unprepared_dataset_returns_error(config_file, capsys):
    assert main(["info", "paysim", "--config", str(config_file)]) == 1
    assert "not been prepared" in capsys.readouterr().err


def test_unknown_dataset_returns_error(config_file, capsys):
    assert main(["prepare", "nope", "--config", str(config_file)]) == 1
    err = capsys.readouterr().err
    assert "unknown dataset" in err
    # KeyError-style repr quoting must not leak into user-facing output.
    assert not err.startswith('"')
