"""Per-dataset adapters: one dataset's raw files -> a canonical frame.

Importing this package registers every adapter. An adapter knows nothing about label
delay or output formats, and nothing about splitting either -- with one deliberate
exception: a dataset whose upstream split must be preserved supplies its own through
`custom_splits`, as sparkov does. See `base.DatasetAdapter`.
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
