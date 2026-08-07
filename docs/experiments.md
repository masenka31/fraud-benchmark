# Paper experiments

This public repository records two supplementary experiments from the paper. Both use
full experimental populations, preserve temporal preprocessing semantics, select any
thresholds on validation data only, and report average precision as the primary metric.
The exact result locations are indexed in [`results/paper.md`](../results/paper.md).

Before running either experiment, build the corresponding feature parquet:

```bash
.venv/bin/python scripts/features.py --dataset ibm_ccf
.venv/bin/python scripts/features.py --dataset sparkov
```

## IBM CCF: pre-Italy temporal versus IID splits

The IBM comparison uses every transaction strictly before the first Italy fraud at
`2017-11-19 12:06:00`, yielding one shared population of 20,403,999 rows. It compares:

- `pre_italy`: chronological 80/10/10;
- `pre_italy_iid_rows`: deterministic transaction-IID 80/10/10;
- `pre_italy_iid_customers`: deterministic customer-disjoint IID splitting, balanced
  against the 80/10/10 transaction-count targets.

All three regimes use the same rows, source features, causal engineered features,
XGBoost settings, and model seeds 0--4. Only split assignment changes. Label-derived
features use training labels only, are computed chronologically, and score a training row
before incorporating that row's label.

```bash
.venv/bin/python scripts/ibm_split_protocol.py
```

The runner writes three JSON records under `results/paper/` and the readable table
[`results/ibm_split_protocol.md`](../results/ibm_split_protocol.md). Use `--render-only`
to regenerate the Markdown from the recorded JSON without refitting.

## Sparkov: synthetic training-label delay

The Sparkov comparison uses one `data/features/sparkov.parquet` population under three
training-label regimes:

- `off`: oracle/true training labels;
- `on`: synthetic default reporting timestamps from `reported_at`;
- `slow`: synthetic fat-tailed stress timestamps from `reported_at_slow`.

All regimes use the same rows, split, features, XGBoost settings, and model seeds 0--4.
A fraud reported after the training cutoff remains in training with provisional label
`0`; it is not dropped. Validation and test are always scored against true `is_fraud`
labels. The slow regime is a stress test, not an estimate of real reporting behavior.

```bash
.venv/bin/python scripts/sparkov_delay_protocol.py
```

The runner writes machine-readable records under `results/paper/` and the readable table
[`results/sparkov_label_delay.md`](../results/sparkov_label_delay.md).
Use `--render-only` to regenerate the Markdown without refitting.

## Shared experiment implementation

The public code under `src/fraud_benchmark/experiments/` contains the feature builders,
split functions, causal encoders, model preparation, and fixed XGBoost fitting required by the
two protocols. It fits preprocessing on training rows only and keeps evaluation labels
uncensored.
