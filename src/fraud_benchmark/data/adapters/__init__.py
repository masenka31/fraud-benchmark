"""Per-dataset adapters: one dataset's raw files -> a canonical frame.

Importing this package registers every adapter. An adapter knows nothing about
splitting, label delay, or output formats -- see `base.DatasetAdapter`.
"""

from fraud_benchmark.data.adapters import (  # noqa: F401
    amaretto,
    banksim,
    ibm_ccf,
    ieee_cis,
    paysim,
    saml_d,
    sparkov,
)

