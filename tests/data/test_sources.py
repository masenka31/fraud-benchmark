from pathlib import Path

import pytest
import requests
from kagglehub.exceptions import CredentialError
from kagglehub.exceptions import KaggleApiHTTPError

from fraud_benchmark.data.sources import FetchError
from fraud_benchmark.data.sources import KaggleCompetition
from fraud_benchmark.data.sources import KaggleDataset
from fraud_benchmark.data.sources import fetch


def test_kaggle_dataset_download_is_called_with_output_dir(tmp_path, monkeypatch):
    calls = {}

    def fake_download(handle, *, force_download=False, output_dir=None):
        calls.update(handle=handle, force=force_download, out=output_dir)
        (tmp_path / "dest").mkdir(parents=True, exist_ok=True)
        (tmp_path / "dest" / "data.csv").write_text("a,b\n1,2\n")
        return str(tmp_path / "dest")

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fake_download)

    result = fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")

    assert result == tmp_path / "dest"
    assert calls["handle"] == "ealaxi/paysim1"
    assert calls["out"] == str(tmp_path / "dest")


def test_fetch_skips_download_when_already_present(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "data.csv").write_text("a,b\n1,2\n")

    def fail(*args, **kwargs):
        raise AssertionError("should not download when cached")

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fail)

    assert fetch(KaggleDataset("ealaxi/paysim1"), dest) == dest


def test_force_redownloads_even_when_present(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "data.csv").write_text("a,b\n1,2\n")
    called = []

    def fake_download(handle, *, force_download=False, output_dir=None):
        called.append(force_download)
        # A real download always (re)writes files into output_dir; forcing now
        # clears dest first, so the fake must replicate that or find nothing.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "data.csv").write_text("a,b\n1,2\n")
        return str(dest)

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fake_download)

    fetch(KaggleDataset("ealaxi/paysim1"), dest, force=True)
    assert called == [True]


def test_credential_error_produces_actionable_message(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        raise CredentialError("no creds")

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fake_download)

    with pytest.raises(FetchError, match="docs/kaggle-setup.md"):
        fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")


def test_competition_403_explains_rule_acceptance(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        response = requests.Response()
        response.status_code = 403
        raise KaggleApiHTTPError("forbidden", response=response)

    monkeypatch.setattr(
        "fraud_benchmark.data.sources.kagglehub.competition_download", fake_download
    )

    with pytest.raises(FetchError, match="accept.*rules"):
        fetch(KaggleCompetition("ieee-fraud-detection"), tmp_path / "dest")


def test_competition_403_message_includes_the_rules_url(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        response = requests.Response()
        response.status_code = 403
        raise KaggleApiHTTPError("forbidden", response=response)

    monkeypatch.setattr(
        "fraud_benchmark.data.sources.kagglehub.competition_download", fake_download
    )

    with pytest.raises(FetchError, match="ieee-fraud-detection/rules"):
        fetch(KaggleCompetition("ieee-fraud-detection"), tmp_path / "dest")


def test_empty_download_directory_is_an_error(tmp_path, monkeypatch):
    def fake_download(handle, **kwargs):
        (tmp_path / "dest").mkdir(parents=True, exist_ok=True)
        return str(tmp_path / "dest")

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fake_download)

    with pytest.raises(FetchError, match="no files"):
        fetch(KaggleDataset("ealaxi/paysim1"), tmp_path / "dest")


def test_force_clears_stale_files(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old_v1.csv").write_text("stale\n")

    def fake_download(handle, *, force_download=False, output_dir=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "new_v2.csv").write_text("fresh\n")
        return output_dir

    monkeypatch.setattr("fraud_benchmark.data.sources.kagglehub.dataset_download", fake_download)

    fetch(KaggleDataset("ealaxi/paysim1"), dest, force=True)

    names = sorted(p.name for p in dest.iterdir())
    assert names == ["new_v2.csv"], f"stale files survived a forced re-fetch: {names}"


def test_git_repo_exposes_its_url():
    from fraud_benchmark.data.sources import GitRepo

    repo = GitRepo("https://github.com/necst/amaretto_dataset")
    assert repo.url == "https://github.com/necst/amaretto_dataset"
    assert repo.ref == "main"


def test_git_clone_is_invoked_with_the_ref(tmp_path, monkeypatch):
    from fraud_benchmark.data.sources import GitRepo

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("cloned\n")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr("fraud_benchmark.data.sources.subprocess.run", fake_run)

    fetch(GitRepo("https://example.com/repo", ref="v1"), tmp_path / "dest")

    assert "clone" in calls["cmd"]
    assert "v1" in calls["cmd"]
    assert "https://example.com/repo" in calls["cmd"]


def test_git_clone_failure_is_a_fetch_error(tmp_path, monkeypatch):
    from fraud_benchmark.data.sources import GitRepo

    def fake_run(cmd, **kwargs):
        class _Result:
            returncode = 128
            stderr = "fatal: repository not found"

        return _Result()

    monkeypatch.setattr("fraud_benchmark.data.sources.subprocess.run", fake_run)

    with pytest.raises(FetchError, match="repository not found"):
        fetch(GitRepo("https://example.com/nope"), tmp_path / "dest")
