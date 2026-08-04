"""Locating raw files inside a downloaded dataset directory.

Kaggle bundles vary: some ship one CSV, others ship several plus auxiliary tables.
Adapters that need a specific file name it explicitly rather than globbing.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _all_files(raw_dir: Path) -> list[Path]:
    """Every file under `raw_dir`, recursively, in path order."""
    return sorted(p for p in raw_dir.rglob("*") if p.is_file())


def require_file(raw_dir: Path, name: str) -> Path:
    """The file called `name` under `raw_dir`, searched recursively.

    Raises FileNotFoundError listing what was present instead.
    """
    for path in _all_files(raw_dir):
        if path.name == name:
            return path
    present = ", ".join(p.name for p in _all_files(raw_dir)) or "(directory is empty)"
    raise FileNotFoundError(f"expected a file named {name!r} under {raw_dir}; found: {present}")


def find_single_csv(raw_dir: Path) -> Path:
    """The single CSV under `raw_dir`. Raises FileNotFoundError unless there is one.

    For datasets whose one CSV has an unstable name; prefer `require_file` otherwise.
    """
    matches = [p for p in _all_files(raw_dir) if p.suffix.lower() == ".csv"]
    if not matches:
        raise FileNotFoundError(f"no CSV file found under {raw_dir}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(f"expected exactly one CSV under {raw_dir}, found: {names}")
    return matches[0]


def require_split_zip_member(raw_dir: Path, part_glob: str, member: str, cache_dir: Path) -> Path:
    """Extract `member` from the fragments matching `part_glob`, cached in `cache_dir`.

    Fragments are numbered (`x.zip.001`, `x.zip.002`, ...) and concatenating them in
    numeric order reproduces the archive. Returns the extracted path, re-using it on
    later calls. Raises FileNotFoundError if the parts are absent, non-numeric, or
    not a complete 1..N sequence.
    """
    extracted = cache_dir / member
    if extracted.exists():
        return extracted

    parts = sorted(raw_dir.rglob(part_glob))
    if not parts:
        raise FileNotFoundError(f"no archive parts matching {part_glob!r} under {raw_dir}")

    suffixes = [path.suffix.lstrip(".") for path in parts]
    if not all(s.isdigit() for s in suffixes):
        raise FileNotFoundError(
            f"archive parts under {raw_dir} do not all end in a numeric suffix: "
            f"{', '.join(p.name for p in parts)}"
        )
    numbers = sorted(int(s) for s in suffixes)
    if numbers != list(range(1, len(numbers) + 1)):
        raise FileNotFoundError(
            f"archive parts under {raw_dir} are not a complete 1..N sequence "
            f"(found {numbers[0]}..{numbers[-1]}, {len(numbers)} parts); "
            "some may be missing"
        )
    # Order by the NUMERIC suffix, so unpadded names like .1/.2/.10 cannot be
    # concatenated in lexicographic order and silently corrupt the archive.
    parts = sorted(parts, key=lambda p: int(p.suffix.lstrip(".")))

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "_reassembled.zip"
    try:
        with open(archive, "wb") as combined:
            for part in parts:
                with open(part, "rb") as fragment:
                    shutil.copyfileobj(fragment, combined)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(member, cache_dir)
    finally:
        # The reassembled archive is a large temporary; never leave it behind.
        archive.unlink(missing_ok=True)

    return extracted
