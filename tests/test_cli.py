import json
from pathlib import Path

import pytest

from fraud_benchmark.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(
        f"paths:\n"
        f"  raw: {tmp_path / 'raw'}\n"
        f"  processed: {tmp_path / 'processed'}\n"
        f"split:\n"
        f"  ratios: [0.6, 0.2, 0.2]\n"
        # The subsample variants crop to [start_date, last labelled fraud]. The
        # real default is 2016-01-01, but the shared ibm_ccf fixture's last fraud
        # is 2011-01-01, which would leave an empty window here.
        f"datasets:\n"
        f"  ibm_ccf_subsample_fast:\n"
        f"    start_date: '2002-01-01'\n"
        f"  ibm_ccf_subsample_slow:\n"
        f"    start_date: '2002-01-01'\n"
    )
    return path


@pytest.fixture
def no_download(monkeypatch):
    """Serve each dataset its own fixture directory instead of downloading.

    Keyed by dataset name: prepare() calls fetch with dest = raw_dir / <name>,
    so a single hardcoded directory would hand one adapter another's files.
    """

    def fake_fetch(source, dest, *, force=False):
        return FIXTURES / dest.name

    monkeypatch.setattr("fraud_benchmark.pipeline.fetch", fake_fetch)


def test_list_prints_registered_datasets(capsys):
    assert main(["list"]) == 0
    assert "paysim" in capsys.readouterr().out


def test_prepare_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "paysim", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()


def test_prepare_all_creates_output(tmp_path, config_file, no_download):
    assert main(["prepare", "--all", "--config", str(config_file)]) == 0
    assert (tmp_path / "processed" / "paysim" / "data.parquet").exists()
    assert (tmp_path / "processed" / "banksim" / "data.parquet").exists()


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


def test_list_marks_noncommercial_datasets(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    paysim_line = [ln for ln in out.splitlines() if ln.startswith("paysim")][0]
    banksim_line = [ln for ln in out.splitlines() if ln.startswith("banksim")][0]
    # paysim is commercially usable; banksim is CC BY-NC-SA and must be flagged.
    assert "noncommercial" not in paysim_line.lower()
    assert "noncommercial" in banksim_line.lower()
    assert "CC BY-NC-SA 4.0" in banksim_line


def test_prepare_all_continues_past_a_failure(tmp_path, config_file, monkeypatch, capsys):
    """A failing dataset must not prevent the others from being prepared."""
    from fraud_benchmark.sources import FetchError

    calls = []

    def flaky(name, config, *, force=False):
        calls.append(name)
        if name == "always_fails":
            raise FetchError("simulated network failure")
        return config.processed_dir / name

    class _Fake:
        def __init__(self, name):
            self.name = name
            self.commercial_use = True
            self.data_license = "CC0 1.0"
            self.source = None

    monkeypatch.setattr("fraud_benchmark.cli.prepare", flaky)
    monkeypatch.setattr(
        "fraud_benchmark.cli.list_datasets", lambda: ["always_fails", "paysim"]
    )
    # _cmd_prepare resolves the adapter before preparing, so this must be stubbed
    # too — otherwise "always_fails" raises UnknownDatasetError and the loop exits
    # before either dataset is attempted.
    monkeypatch.setattr("fraud_benchmark.cli.get_adapter", _Fake)

    assert main(["prepare", "--all", "--config", str(config_file)]) == 1
    # Both were attempted, not just the first.
    assert calls == ["always_fails", "paysim"]
    captured = capsys.readouterr()
    assert "always_fails" in captured.err
    assert "paysim" in captured.out


def test_exclude_noncommercial_skips_those_datasets(tmp_path, config_file, monkeypatch, capsys):
    prepared = []

    def record(name, config, *, force=False):
        prepared.append(name)
        return config.processed_dir / name

    class _Fake:
        def __init__(self, name, commercial):
            self.name = name
            self.commercial_use = commercial
            self.data_license = "CC BY-NC-SA 4.0"
            self.source = None

    monkeypatch.setattr("fraud_benchmark.cli.prepare", record)
    monkeypatch.setattr("fraud_benchmark.cli.list_datasets", lambda: ["open_one", "nc_one"])
    monkeypatch.setattr(
        "fraud_benchmark.cli.get_adapter",
        lambda n: _Fake(n, commercial=(n == "open_one")),
    )

    assert main(["prepare", "--all", "--exclude-noncommercial",
                 "--config", str(config_file)]) == 0
    assert prepared == ["open_one"]
    assert "nc_one" in capsys.readouterr().out
