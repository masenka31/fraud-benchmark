# Paper experiment records

Two supplementary experiments are reported for the paper. These are the only dedicated
paper protocols in this repository; exploratory ROC/PR and subsampling work is private.

## IBM CCF: pre-Italy temporal versus IID splits

The experiment uses the same 20,403,999 transactions strictly before the first Italy
fraud (`2017-11-19 12:06:00`) in all three regimes. It compares a temporal split,
transaction-IID split, and customer-disjoint IID split with five fixed XGBoost seeds.
Only split assignment changes. Label-derived features are chronological and use training
labels only.

- Runner: `scripts/ibm_split_protocol.py`
- Readable table: `results/ibm_split_protocol.md`
- Machine-readable records: `results/paper/ibm_pre_italy_*.json`

## Sparkov: synthetic training-label delay

The experiment uses the same Sparkov rows, split, features, model configuration, and five
XGBoost seeds under no delay, the default synthetic delay, and the synthetic slow stress
regime. Only training-label availability changes. Validation and test use true labels.

- Runner: `scripts/sparkov_delay_protocol.py`
- Readable table: `results/sparkov_label_delay.md`
- Machine-readable records: `results/paper/sparkov_delay_*.json`
