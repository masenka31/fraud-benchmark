"""Locating raw files inside a downloaded dataset directory.

Kaggle bundles vary: some ship one CSV, others ship several plus auxiliary tables.
Adapters that need a specific file name it explicitly rather than globbing.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _all_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.is_file())


def require_file(raw_dir: Path, name: str) -> Path:
    """Return the file called `name` under `raw_dir`, searched recursively."""
    for path in _all_files(raw_dir):
        if path.name == name:
            return path
    present = ", ".join(p.name for p in _all_files(raw_dir)) or "(directory is empty)"
    raise FileNotFoundError(
        f"expected a file named {name!r} under {raw_dir}; found: {present}"
    )


def find_single_csv(raw_dir: Path) -> Path:
    """Return the one CSV under `raw_dir`, erroring if there is not exactly one.

    Only for datasets that genuinely ship a single CSV whose name is unstable.
    Prefer `require_file` when the name is known.
    """
    matches = [p for p in _all_files(raw_dir) if p.suffix.lower() == ".csv"]
    if not matches:
        raise FileNotFoundError(f"no CSV file found under {raw_dir}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"expected exactly one CSV under {raw_dir}, found: {names}"
        )
    return matches[0]


def require_split_zip_member(
    raw_dir: Path, part_glob: str, member: str, cache_dir: Path
) -> Path:
    """Reassemble a multi-part zip under `raw_dir` and extract one member.

    Some datasets ship as numbered fragments (`x.zip.001`, `x.zip.002`, ...) because
    of file-size limits. Concatenating them in name order reproduces the original
    archive. The extracted member is cached in `cache_dir`, so the cost is paid once.
    """
    extracted = cache_dir / member
    if extracted.exists():
        return extracted

    parts = sorted(raw_dir.rglob(part_glob))
    if not parts:
        raise FileNotFoundError(
            f"no archive parts matching {part_glob!r} under {raw_dir}"
        )

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
