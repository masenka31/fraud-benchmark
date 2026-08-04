"""Which columns no model may ever see, whatever the experiment.

General, not study-specific. This list is a correctness invariant: leaving one of
these in invalidates every number produced. A column that is merely *suspect* --
IBM CCF's merchant geography, say -- does not belong here, because excluding it is
an experimental condition rather than a rule; `experiments/features/` prefixes
those `artifact_` instead, so a study can include or drop the group by name.

The label part is derived from the adapter registry rather than restated here.
Before that, each source's raw label column was named in this file as well as in
the adapter that read it, and nothing checked the two agreed. The pipeline now
drops those columns outright, so the only label-ish columns left to exclude are
the ones deliberately kept for the typology they carry, and the adapters declare
them.
"""

from __future__ import annotations

import fraud_benchmark.data.adapters  # noqa: F401  (registers all adapters)
from fraud_benchmark.data.adapters.base import get_adapter
from fraud_benchmark.data.adapters.base import list_datasets

# Produced by the pipeline: the label, its availability time, the campaign it was
# discovered with, and which split the row landed in.
_PIPELINE_COLUMNS = frozenset({'is_fraud', 'reported_at', 'campaign_id', 'split'})

# Not labels, and not derived from one, but not inputs a model would have either.
# `isFlaggedFraud` is PaySim's own detector output. `trans_num` is a row id.
# `source_file` is here because Sparkov's splits come from separate upstream files,
# so it predicts the split perfectly.
_NOT_MODEL_INPUTS = frozenset({'isFlaggedFraud', 'trans_num', 'source_file'})


def build_always_excluded() -> frozenset[str]:
    """Compute the drop-list from the adapter registry.

    A function rather than only a constant so a test can prove the derivation is
    live: registering an adapter changes what this returns. `ALWAYS_EXCLUDED`
    below is its value at import time, which is what every caller uses.
    """
    label_descriptive = frozenset(
        column for name in list_datasets() for column in get_adapter(name).label_descriptive_columns
    )
    return _PIPELINE_COLUMNS | _NOT_MODEL_INPUTS | label_descriptive


#: Every column that must never reach a model. Read from the registry, so a newly
#: added dataset is covered the moment its adapter declares its label-descriptive
#: columns rather than when someone remembers to edit this file.
ALWAYS_EXCLUDED = build_always_excluded()


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
        'trans_date_trans_time',  # sparkov: event_time as a string
        'unix_time',  # sparkov: event_time as an int
        'Date',  # saml_d: the absolute date
        'Year',  # ibm_ccf: with Month and Day, reconstructs it
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
        raise ExcludedColumnError(f'columns that must never reach a model are present: {leaked}')
