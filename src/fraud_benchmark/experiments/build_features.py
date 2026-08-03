"""Stage 1: build velocity features once per dataset and cache them.

The 14 stage-2 jobs read this cache, as do the one-off experiments in `scripts/`.
Rebuilding 24M rows of aggregates inside each of them would be the obvious way to
waste the day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_benchmark.experiments.features import add_velocity_features

DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_FEATURES = Path("data/features")

# The merchant-like column each dataset uses for `merchant_novelty`.
MERCHANT_COLUMNS: dict[str, str | None] = {
    "ibm_ccf": "Merchant Name",
    "saml_d": "Receiver_account",
    "sparkov": "merchant",
    "sparkov_slow": "merchant",
}


def build(
    dataset: str,
    processed_dir: Path = DEFAULT_PROCESSED,
    features_dir: Path = DEFAULT_FEATURES,
) -> Path:
    """Read the processed parquet, append velocity features, write the cache."""
    source = Path(processed_dir) / dataset / "data.parquet"
    df = pd.read_parquet(source)
    before = len(df)

    df = add_velocity_features(df, merchant_col=MERCHANT_COLUMNS[dataset])
    if len(df) != before:
        raise AssertionError(f"{dataset}: feature build changed the row count")

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    destination = features_dir / f"{dataset}.parquet"
    df.to_parquet(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation features for one dataset")
    parser.add_argument("dataset")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args()
    destination = build(args.dataset, args.processed_dir, args.features_dir)
    print(f"{args.dataset}: wrote {destination}")


if __name__ == "__main__":
    main()
