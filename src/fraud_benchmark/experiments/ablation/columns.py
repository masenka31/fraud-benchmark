"""Which columns the leakage ablation removes.

The study's own condition, not a correctness invariant: LEAKY_COLUMNS is present in
the `leaky` arm and absent in the `clean` one, and the gap between them is the
result. What no model may see under any condition lives in `experiments/columns.py`,
and is re-exported here so a caller reading one cell's column list finds both in
one place.
"""

from __future__ import annotations

import pandas as pd

from fraud_benchmark.experiments.columns import (  # noqa: F401  (re-exported)
    ABSOLUTE_TIME_COLUMNS,
    ALWAYS_EXCLUDED,
    ExcludedColumnError,
    assert_no_excluded,
)

FEATURE_SETS = ("leaky", "clean")

# From docs/verification-notes.md, "## Known leakage". `Merchant Name` is added
# beyond the tuple the audit suggested: a merchant id encodes its own location,
# so retaining it would reintroduce the geography artifact under another name.
_IBM_CCF_LEAKY = (
    "Merchant State",
    "Merchant City",
    "Zip",
    "MCC",
    "Errors?",
    "Merchant Name",
)

LEAKY_COLUMNS: dict[str, tuple[str, ...]] = {
    "ibm_ccf": _IBM_CCF_LEAKY,
    # Account identifiers only. Sender_bank_location and Receiver_bank_location
    # are kept: the audit found their effects directional and plausible for money
    # laundering -- the phenomenon, not an artifact.
    "saml_d": ("Sender_account", "Receiver_account", "entity_id"),
    # The audit found zero flagged values. Sparkov is the negative control: its
    # two conditions are identical by construction, so the gap between them
    # measures this harness's own noise floor.
    "sparkov": (),
    # Row-identical to sparkov, so it carries the same (empty) leaky set.
    "sparkov_slow": (),
}


def feature_columns(df: pd.DataFrame, dataset: str, feature_set: str) -> list[str]:
    """The columns a model may see for one cell, in stable order."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    leaky = LEAKY_COLUMNS[dataset]

    dropped = set(ALWAYS_EXCLUDED) | set(ABSOLUTE_TIME_COLUMNS)
    if feature_set == "clean":
        dropped |= set(leaky)

    columns = [c for c in df.columns if c not in dropped]
    assert_no_excluded(columns)
    return columns
