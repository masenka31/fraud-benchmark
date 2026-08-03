"""Everything that happens after a dataset is prepared.

Reads `data/processed/`, writes `data/features/`, and never writes to
`data/processed/`. Imports from `fraud_benchmark.data` freely; the reverse is
forbidden, because preparation knows nothing about features or models.

Two stages, and the boundary between them is deliberate:

`features/` turns a prepared frame into a feature parquet, one module per
experimental dataset -- `ibm_ccf`, `sparkov`, `saml_d`. Everything it produces is
*unfitted*: engineered, but with no vocabulary capped, no ordinal code assigned and
nothing scaled, so a parquet is valid under any split.

`encoding`, `splits`, `models`, `mlp` and `metrics` are the model side, and they
are where the split is chosen and the fitting happens. Fitting inside a feature
build would tie the parquet to one split, and there are two in use -- the standard
temporal 80/10/10 and the Italy holdout, which cuts elsewhere on purpose.

Four of the seven registered datasets (paysim, banksim, ieee_cis, amaretto) are
prepared and documented by `fraud_benchmark.data` but not run experimentally, so
nothing here mentions them.
"""
