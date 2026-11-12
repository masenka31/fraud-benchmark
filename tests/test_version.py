"""The package, its metadata and its citation file state one version."""

import tomllib
from pathlib import Path

import yaml

import fraud_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_package_pyproject_and_citation_versions_agree():
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    citation = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))['version']
    assert fraud_benchmark.__version__ == pyproject == citation
