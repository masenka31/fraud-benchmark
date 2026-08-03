"""The ablation grid: which (dataset, feature set, label regime) cells exist.

One definition, two consumers. `scripts/slurm/generate.py` turns it into sbatch
files; `summarize.py` uses it to name the cells that have no results yet. Before
this was shared, a retired dataset kept appearing in the summary and a newly
added one silently did not -- the summary could not distinguish "no records
because the job has not run" from "no records because the cell does not exist".
"""

from __future__ import annotations

#: Datasets in the study. The other four registered datasets are out of scope:
#: the audit found no leakage to ablate in them (see docs/verification-notes.md,
#: "## Known leakage").
DATASETS = (
    "ibm_ccf",
    "saml_d",
    "sparkov",
    "sparkov_slow",
)

FEATURE_SETS = ("leaky", "clean")

#: ibm_ccf runs oracle only: it has 3 censored train labels out of 24,924, so the
#: censored regime is identical to oracle and would cost roughly eight hours of
#: the most expensive fits in the study to confirm arithmetic.
ORACLE_ONLY = frozenset({"ibm_ccf"})


def regimes_for(dataset: str) -> tuple[str, ...]:
    """The label regimes worth running for one dataset."""
    return ("oracle",) if dataset in ORACLE_ONLY else ("oracle", "censored")


CELLS = tuple(
    (dataset, feature_set, regime)
    for dataset in DATASETS
    for feature_set in FEATURE_SETS
    for regime in regimes_for(dataset)
)
