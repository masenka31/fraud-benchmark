"""Which columns a model may see.

Two separate sets, deliberately not merged. ALWAYS_EXCLUDED is the label and
anything derived from it -- leaving one of these in invalidates every number in
the study. LEAKY_COLUMNS is the ablation: the generation artifacts the audit
measured, present in the `leaky` condition and absent in the `clean` one.
"""

from __future__ import annotations

import pandas as pd

FEATURE_SETS = ("leaky", "clean")

# The label, anything derived from it, row identifiers, and split provenance.
# `source_file` is here because Sparkov's splits come from separate upstream
# files, so it predicts the split perfectly. `Laundering_type` describes the
# label and is non-null only for laundering rows.
ALWAYS_EXCLUDED = frozenset(
    {
        "is_fraud",
        "reported_at",
        "campaign_id",
        "split",
        "Is Fraud?",
        "Is_laundering",
        "Laundering_type",
        "trans_num",
        "source_file",
    }
)

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


# Columns that restate event_time and therefore encode split membership. The
# splits are temporal, so an absolute clock lets a tree isolate the split
# boundary as a threshold -- sparkov's `unix_time` separates them exactly
# (train max 1367492969 < val min 1367493022). Dropping event_time while
# keeping these would be self-defeating, and a dtype check does not catch them:
# `trans_date_trans_time` is stored as a string and `unix_time` as an int.
#
# Cyclical and relative parts are KEPT -- Month, Day, time-of-day -- because
# they are what a real detector uses and they generalise forward. Customer
# attributes that happen to be dates (Birth Year, Acct Open Date) are kept too:
# they describe the cardholder, not when the transaction happened.
ABSOLUTE_TIME_COLUMNS = frozenset(
    {
        "trans_date_trans_time",  # sparkov: event_time as a string
        "unix_time",              # sparkov: event_time as an int
        "Date",                   # saml_d: the absolute date
        "Year",                   # ibm_ccf: with Month and Day, reconstructs it
    }
)


class ExcludedColumnError(AssertionError):
    """Raised when a column that must never reach a model is about to."""


def assert_no_excluded(columns: list[str]) -> None:
    """Raise if any always-excluded column is in `columns`.

    A tripwire for callers that assemble a column list themselves rather than
    taking `feature_columns` output verbatim -- which is every consumer that
    adds, renames, or re-orders columns downstream. Calling it on
    `feature_columns` output cannot fail today; it is there so that a future
    change to the filter is caught by a test rather than by a wrong result.
    """
    leaked = sorted(set(ALWAYS_EXCLUDED) & set(columns))
    if leaked:
        raise ExcludedColumnError(
            f"columns that must never reach a model are present: {leaked}"
        )


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
