# Paper experiment records

The original XGBoost protocols and their LSTM sequence extensions are recorded here.
Exploratory ROC/PR and subsampling work is private.

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

## IBM CCF: 30-transaction endpoint sequence extension

Each entity contributes one target per complete, disjoint 30-transaction chunk:
the final transaction. The temporal, chunk-IID, and customer-IID splits share the
same 679,372 endpoints and 77 label-free feature names. XGBoost receives the
endpoint's feature row; the LSTM receives all 30 rows in its chunk. Both score only
the endpoint. These reduced-target AP values are not comparable to the original
full-row IBM protocol above. The temporal test set contains only 36 fraud targets,
so its AP is sensitive to a few predictions.

- Runner: `scripts/lstm_protocol.py --dataset ibm`
- Readable table: `results/ibm_lstm_protocol.md`
- Machine-readable records: `results/paper/ibm_lstm_*.json` and
  `results/paper/ibm_xgboost_*.json`

The 10-transaction repeat uses the same model settings and feature names but
predicts 2,039,680 different chunk endpoints. It is indexed separately:

- Runner: `scripts/lstm_protocol.py --dataset ibm --window-length 10`
- Readable table: `results/ibm_lstm_protocol_len10.md`
- Machine-readable records: `results/paper/ibm_lstm_*_len10.json` and
  `results/paper/ibm_xgboost_*_len10.json`

## Sparkov: LSTM synthetic training-label delay extension

The LSTM scores every Sparkov transaction using its current features and at most
29 strictly earlier-time transactions from the same entity. The no-delay,
default-delay, and slow-delay regimes use the same rows, split, inputs, and five
seeds. Only synthetic training-label availability changes; evaluation uses true
labels.

- Runner: `scripts/lstm_protocol.py --dataset sparkov`
- Readable table: `results/sparkov_lstm_delay.md`
- Machine-readable records: `results/paper/sparkov_lstm_*.json`

The 10-transaction repeat keeps the same target rows and split:

- Runner: `scripts/lstm_protocol.py --dataset sparkov --window-length 10`
- Readable table: `results/sparkov_lstm_delay_len10.md`
- Machine-readable records: `results/paper/sparkov_lstm_*_len10.json`

The raw-ish input ablation uses 21 current-event/static features and no
entity-history statistics, at both maximum sequence lengths. It keeps the
same target rows, split, delay regimes, architecture, and seeds as the
corresponding full-feature LSTM cells:

- Runner: `scripts/lstm_protocol.py --dataset sparkov --feature-set rawish`,
  with `--window-length 10` or `--window-length 30`
- Readable tables: `results/sparkov_lstm_delay_rawish_len10.md` and
  `results/sparkov_lstm_delay_rawish_len30.md`
- Machine-readable records: `results/paper/sparkov_lstm_*_rawish_len10.json` and
  `results/paper/sparkov_lstm_*_rawish_len30.json`
