"""Per-dataset feature extraction: one module per experimental dataset.

Three datasets are in scope, and each has exactly one module that names every
feature it produces:

    ibm_ccf.py   ->  data/features/ibm_ccf.parquet
    sparkov.py   ->  data/features/sparkov.parquet
    saml_d.py    ->  data/features/saml_d.parquet

Reading one of those modules is meant to be the whole answer to "what does this
dataset's model see?". Shared machinery lives in `util.py`, but no feature is
*declared* there -- a module that inherited half its columns from somewhere else
would defeat the point. The primitives are windowing and arithmetic; the choice of
which windows over which columns is always local.

The four other registered datasets (paysim, banksim, ieee_cis, amaretto) are
prepared and documented by `fraud_benchmark.data` but are not run experimentally,
so they have no module here.


## The parquet contract

A feature parquet carries KEY_COLUMNS plus feature columns, and nothing else. No
raw source column survives under its original name: every column is either a key
or something this package deliberately computed.

Numeric features are float32. Categorical features stay as pandas `category` --
*unfitted*, meaning the engineering is done (an MCC is grouped, a location is
resolved) but no vocabulary is capped, no ordinal code is assigned and nothing is
scaled. Those steps belong to `experiments.encoding`, fitted on whichever train
split the experiment chose. Baking them in here would silently tie the parquet to
one split, and there are two in use -- the standard temporal 80/10/10 and the
Italy holdout, which cuts in a different place on purpose.

`split` is carried for convenience, copied from the preparation that produced the
frame. `experiments.splits` recomputes it; treat the stored value as a default,
not as the authority.

A NaN in a feature column always means *this history does not exist* -- no previous
transaction, no previous visit to this merchant -- and never means missing data.
The distinction matters when choosing a sentinel: `seconds_since_prev_txn` is NaN
on an entity's first row because no gap exists, and filling it with 0 would claim
the opposite of the truth. Counts and sums are the exception and are 0 rather than
NaN, because an empty window genuinely contains nothing; a mean over an empty
window stays NaN, because it is undefined. Imputation is left to
`experiments.encoding`, which fits it on the chosen train split.


## The artifact_ prefix

Columns naming an absolute place or a specific counterparty are prefixed
`artifact_`. They are real signal and they are also how a model memorises
"Italy = fraud" instead of learning fraud, so they are kept and made removable by
name: a downstream column filter drops the group with a prefix test rather than a
per-dataset list. Each module's docstring says which of its columns are in the
group and what the audit measured about them.


## Label delay

Sparkov's parquet carries two report timestamps, `reported_at` (the 7-day
card-fraud default) and `reported_at_slow` (the 15-day fat-tailed regime, taken
from the `sparkov_slow` preparation, which is row-identical apart from that
column). The three label regimes an experiment can run are therefore: no delay
(ignore both, labels known immediately), delay (`reported_at`), and slow delay
(`reported_at_slow`) -- a choice made at training time rather than a choice of
input file. ibm_ccf and saml_d carry only `reported_at`.
"""

from importlib import import_module
from pathlib import Path

from fraud_benchmark.experiments.features.util import ARTIFACT_PREFIX
from fraud_benchmark.experiments.features.util import FEATURE_DIR
from fraud_benchmark.experiments.features.util import KEY_COLUMNS
from fraud_benchmark.experiments.features.util import artifact_columns
from fraud_benchmark.experiments.features.util import feature_columns
from fraud_benchmark.experiments.features.util import write_features

#: The datasets with a module here, in descending build cost. `scripts/features.py
#: --all` walks this, and `experiments.slurm` sizes one job per entry -- so a fourth
#: dataset becomes buildable by adding its module and its name, in one place each.
DATASETS = ("ibm_ccf", "saml_d", "sparkov")


def build(dataset: str, argv: list[str] | None = None) -> Path:
    """Run one dataset's feature module, as `python -m ...features.<dataset>` would.

    Imported on demand rather than at package import: the three modules are heavy, and
    `experiments.columns` imports this package to read the key columns.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"{dataset!r} has no feature module; the experimental datasets are: "
            f"{', '.join(DATASETS)}"
        )
    module = import_module(f"fraud_benchmark.experiments.features.{dataset}")
    return module.main(argv)


__all__ = [
    "DATASETS",
    "FEATURE_DIR",
    "KEY_COLUMNS",
    "ARTIFACT_PREFIX",
    "artifact_columns",
    "build",
    "feature_columns",
    "write_features",
]
