"""Dataset preparation: download, canonicalize, validate, split, group, delay, write.

Everything a prepared dataset needs and nothing more. The stages run in the order
`pipeline.prepare` calls them; `censoring` is the exception, used at training time
rather than at preparation time, and lives here so the label delay's semantics and
the only correct way to consume them stay together.

Nothing in this package may import from `fraud_benchmark.experiments`. The
dependency runs one way: experiments read prepared datasets, and preparation knows
nothing about features or models.
"""
