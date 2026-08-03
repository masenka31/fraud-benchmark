"""Everything that happens after a dataset is prepared.

Features, encoding, models, metrics, and the splits derived from a prepared frame.
Reads `data/processed/` and `data/features/`; never writes to `data/processed/`.

Imports from `fraud_benchmark.data` freely. The reverse is forbidden: preparation
knows nothing about features or models.
"""
