"""Fraud and AML benchmark datasets, and the experiments run on them.

Two halves, and the boundary between them is the point:

* `data` prepares datasets -- download, canonicalize, validate, split, group,
  label-delay, write. It is the whole of dataset handling.
* `experiments` is everything after that -- per-dataset feature extraction, then
  encoding, models and metrics. Three datasets are run experimentally: ibm_ccf,
  saml_d and sparkov, the last under three label-delay regimes.

`experiments` imports `data`. The reverse never happens: preparation knows nothing
about features or models.
"""

__version__ = "0.1.0"
