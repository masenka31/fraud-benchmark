"""Per-dataset adapters: one dataset's raw files -> a canonical frame.

Importing this package registers every adapter. An adapter knows nothing about label
delay or output formats, and nothing about splitting either -- with one deliberate
exception: a dataset whose upstream split must be preserved supplies its own through
`custom_splits`, as sparkov does. See `base.DatasetAdapter`.
"""

from fraud_benchmark.data.adapters import amaretto  # noqa: F401
from fraud_benchmark.data.adapters import banksim  # noqa: F401
from fraud_benchmark.data.adapters import ibm_ccf  # noqa: F401
from fraud_benchmark.data.adapters import ieee_cis  # noqa: F401
from fraud_benchmark.data.adapters import paysim  # noqa: F401
from fraud_benchmark.data.adapters import saml_d  # noqa: F401
from fraud_benchmark.data.adapters import sparkov  # noqa: F401
