# fraud-benchmark

Downloads public fraud and AML transaction datasets, normalizes them to a shared schema,
assigns temporal train/validation/test splits, and attaches a synthetic label-availability
("fraud reported at") timestamp to fraudulent transactions.

See `docs/superpowers/specs/2026-08-01-fraud-benchmark-design.md` for the full design.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Kaggle credentials

See `docs/kaggle-setup.md`. Required before any dataset can be downloaded.

## Usage

    fraud-benchmark list
    fraud-benchmark prepare paysim
    fraud-benchmark info paysim

## Tests

    pytest
