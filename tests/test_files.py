import pytest

from fraud_benchmark.datasets.files import find_single_csv, require_file


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
