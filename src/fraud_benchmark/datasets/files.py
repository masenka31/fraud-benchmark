"""Locating raw files inside a downloaded dataset directory.

Kaggle bundles vary: some ship one CSV, others ship several plus auxiliary tables.
Adapters that need a specific file name it explicitly rather than globbing.
"""

from __future__ import annotations

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
