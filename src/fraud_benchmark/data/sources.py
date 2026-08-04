"""Fetching raw dataset files, with errors a human can act on."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import kagglehub
from kagglehub.exceptions import CredentialError
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.exceptions import NotFoundError
from kagglehub.exceptions import UnauthenticatedError

SETUP_HINT = (
    "Kaggle credentials are missing or invalid. "
    "Follow docs/kaggle-setup.md to create ~/.kaggle/access_token "
    "or set KAGGLE_API_TOKEN."
)


class FetchError(RuntimeError):
    """Raised when raw data could not be fetched."""


@dataclass(frozen=True)
class KaggleDataset:
    """A regular Kaggle dataset, e.g. 'ealaxi/paysim1'."""

    handle: str

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/datasets/{self.handle}"


@dataclass(frozen=True)
class KaggleCompetition:
    """A Kaggle competition, e.g. 'ieee-fraud-detection'."""

    handle: str

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/c/{self.handle}"

    @property
    def rules_url(self) -> str:
        return f"https://www.kaggle.com/c/{self.handle}/rules"


@dataclass(frozen=True)
class GitRepo:
    """A dataset distributed as a git repository rather than via Kaggle."""

    url: str
    ref: str = "main"


Source = KaggleDataset | KaggleCompetition | GitRepo


def _has_files(directory: Path) -> bool:
    """Whether `directory` exists and contains at least one file, recursively."""
    return directory.is_dir() and any(p.is_file() for p in directory.rglob("*"))


def fetch(source: Source, dest: Path, *, force: bool = False) -> Path:
    """Download `source` into `dest`, returning `dest`.

    A `dest` that already holds files is left alone unless `force` is set, in which
    case it is removed and re-downloaded. Raises FetchError on any download failure,
    or if the download produced no files.
    """
    if _has_files(dest) and not force:
        return dest

    if force and dest.exists():
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(source, KaggleDataset):
            kagglehub.dataset_download(source.handle, force_download=force, output_dir=str(dest))
        elif isinstance(source, KaggleCompetition):
            kagglehub.competition_download(
                source.handle, force_download=force, output_dir=str(dest)
            )
        elif isinstance(source, GitRepo):
            _git_clone(source, dest)
        else:
            raise FetchError(f"unsupported source type: {type(source).__name__}")
    except (CredentialError, UnauthenticatedError) as exc:
        raise FetchError(f"{SETUP_HINT}\nOriginal error: {exc}") from exc
    except KaggleApiHTTPError as exc:
        raise FetchError(_http_error_message(source, exc)) from exc
    except NotFoundError as exc:
        raise FetchError(f"{source.url} was not found on Kaggle: {exc}") from exc

    if not _has_files(dest):
        raise FetchError(f"download of {source.url} produced no files in {dest}")

    return dest


def _git_clone(source: GitRepo, dest: Path) -> None:
    """Shallow-clone `source` at its ref into `dest`, which must be empty."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", source.ref, source.url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FetchError(
            f"git clone of {source.url} (ref {source.ref}) failed: {result.stderr.strip()}"
        )


def _http_error_message(source: Source, exc: KaggleApiHTTPError) -> str:
    """Turn a Kaggle HTTP failure into a message naming the next action to take."""
    status = getattr(exc.response, "status_code", None)

    if status == 403 and isinstance(source, KaggleCompetition):
        return (
            f"Kaggle returned 403 for competition {source.handle!r}. "
            "This usually means your credentials are fine but you have not yet "
            "accepted the competition rules. Open the page below in a browser while "
            "signed in and click 'I Understand and Accept', then retry:\n"
            f"  {source.rules_url}"
        )
    if status in (401, 403):
        return f"Kaggle returned {status} for {source.url}.\n{SETUP_HINT}"
    return f"Kaggle request for {source.url} failed with status {status}: {exc}"
