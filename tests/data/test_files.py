import zipfile
from pathlib import Path

import pytest

from fraud_benchmark.data.adapters.files import find_single_csv, require_file


def test_require_file_finds_a_named_file(tmp_path):
    (tmp_path / "wanted.csv").write_text("a\n1\n")
    (tmp_path / "other.csv").write_text("a\n1\n")
    assert require_file(tmp_path, "wanted.csv").name == "wanted.csv"


def test_require_file_searches_subdirectories(tmp_path):
    nested = tmp_path / "inner"
    nested.mkdir()
    (nested / "wanted.csv").write_text("a\n1\n")
    assert require_file(tmp_path, "wanted.csv") == nested / "wanted.csv"


def test_require_file_error_lists_what_is_present(tmp_path):
    (tmp_path / "actual.csv").write_text("a\n1\n")
    with pytest.raises(FileNotFoundError, match="actual.csv"):
        require_file(tmp_path, "missing.csv")


def test_require_file_error_names_the_file_it_wanted(tmp_path):
    (tmp_path / "actual.csv").write_text("a\n1\n")
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        require_file(tmp_path, "missing.csv")


def test_find_single_csv_still_works(tmp_path):
    (tmp_path / "only.csv").write_text("a\n1\n")
    assert find_single_csv(tmp_path).name == "only.csv"


def test_find_single_csv_rejects_two(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.csv").write_text("x\n1\n")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_single_csv(tmp_path)


def _make_split_zip(tmp_path, member_name, member_bytes, part_size):
    """Build a real zip, then chop it into numbered parts like the Amaretto release."""
    whole = tmp_path / "whole.zip"
    with zipfile.ZipFile(whole, "w") as zf:
        zf.writestr(member_name, member_bytes)
    data = whole.read_bytes()
    whole.unlink()
    parts_dir = tmp_path / "raw" / "Data"
    parts_dir.mkdir(parents=True)
    index = 1
    for offset in range(0, len(data), part_size):
        (parts_dir / f"archive.zip.{index:03d}").write_bytes(data[offset : offset + part_size])
        index += 1
    return tmp_path / "raw"


def test_split_zip_member_is_reassembled_and_extracted(tmp_path):
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"a,b\n1,2\n" * 500, part_size=97)
    out = require_split_zip_member(raw, "archive.zip.*", "data.csv", tmp_path / "cache")
    assert out.read_bytes() == b"a,b\n1,2\n" * 500


def test_split_zip_extraction_is_cached(tmp_path):
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"x\n" * 100, part_size=64)
    cache = tmp_path / "cache"
    first = require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    # Removing the parts must not break a second call: the result is cached.
    for part in (raw / "Data").iterdir():
        part.unlink()
    second = require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    assert first == second
    assert second.exists()


def test_split_zip_leaves_no_reassembled_archive(tmp_path):
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"y\n" * 100, part_size=64)
    cache = tmp_path / "cache"
    require_split_zip_member(raw, "archive.zip.*", "data.csv", cache)
    assert not list(cache.glob("*.zip"))


def test_missing_split_parts_is_an_error(tmp_path):
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    empty = tmp_path / "raw"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="archive.zip"):
        require_split_zip_member(empty, "archive.zip.*", "data.csv", tmp_path / "cache")


def test_unpadded_parts_are_ordered_numerically(tmp_path):
    """Lexicographic order would put .10 before .2 and corrupt the archive."""
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    payload = b"z\n" * 4000
    whole = tmp_path / "whole.zip"
    with zipfile.ZipFile(whole, "w") as zf:
        zf.writestr("data.csv", payload)
    data = whole.read_bytes()
    whole.unlink()

    parts_dir = tmp_path / "raw" / "Data"
    parts_dir.mkdir(parents=True)
    size = len(data) // 12 + 1
    for index, offset in enumerate(range(0, len(data), size), start=1):
        # Deliberately UNPADDED suffixes.
        (parts_dir / f"archive.zip.{index}").write_bytes(data[offset : offset + size])

    out = require_split_zip_member(
        tmp_path / "raw", "archive.zip.*", "data.csv", tmp_path / "cache"
    )
    assert out.read_bytes() == payload


def test_a_missing_part_is_reported_clearly(tmp_path):
    from fraud_benchmark.data.adapters.files import require_split_zip_member

    raw = _make_split_zip(tmp_path, "data.csv", b"q\n" * 500, part_size=97)
    parts = sorted((raw / "Data").iterdir())
    parts[2].unlink()  # punch a hole in the sequence

    with pytest.raises(FileNotFoundError, match="complete 1..N sequence"):
        require_split_zip_member(raw, "archive.zip.*", "data.csv", tmp_path / "cache")
