# Experiment records

`results/paper/` holds only the records behind tables the paper prints.
`results/archive/` holds everything else: earlier versions of paper tables, sequence
extensions, and diagnostics. Exploratory ROC/PR and subsampling work is private.

The runners write each record to its folder by default; `scripts/lstm_protocol.py`
decides per cell, so rerunning an archived cell never touches `results/paper/`.

## Paper

### IBM CCF: temporal versus IID splits (LSTM)

The paper's split table. The pre-Italy rows (20,403,999 transactions strictly before
the first Italy fraud at `2017-11-19 12:06:00`) are split chronologically 68/16/16,
with cutoffs chosen after inspecting labels so validation and test have similar fraud
rates. Each entity contributes complete 10-transaction windows every five transactions
(4,078,583 endpoints), and the LSTM scores each window's final transaction. The
temporal, window-IID, and customer-IID cells share those endpoints; window IID can
share context rows across partitions. The LSTM uses the `sqrt_weight_slow` preset,
selected from five presets on temporal validation AP.

- Runner: `scripts/lstm_protocol.py --dataset ibm --regime pre_italy_68_16_16 --window-length 10 --window-stride 5 --model lstm --lstm-config sqrt_weight_slow`,
  repeated with `pre_italy_iid_chunks_68_16_16` and `pre_italy_iid_customers_68_16_16`
- Readable table: `results/paper/ibm_lstm_split_comparison.md`
- Machine-readable records: `results/paper/ibm_lstm_pre_italy_*68_16_16_len10_stride5_cfg-sqrt_weight_slow.json`

### Sparkov: synthetic training-label delay (XGBoost)

The experiment uses the same Sparkov rows, split, features, model configuration, and five
XGBoost seeds under no delay, the default synthetic delay, and the synthetic slow stress
regime. Only training-label availability changes. Validation and test use true labels.

- Runner: `scripts/sparkov_delay_protocol.py`
- Readable table: `results/paper/sparkov_label_delay.md`
- Machine-readable records: `results/paper/sparkov_delay_*.json`

## Archive

### IBM CCF: full-row XGBoost split comparison — `archive/ibm_xgboost_full_rows/`

The earlier version of the paper's split table. Every pre-Italy transaction is
scored under chronological 80/10/10, transaction-IID, and customer-disjoint IID
splits with five fixed XGBoost seeds. Label-derived features are chronological and
use training labels only.

- Runner: `scripts/ibm_split_protocol.py`
- Readable table: `results/archive/ibm_xgboost_full_rows/ibm_split_protocol.md`
- Machine-readable records: `results/archive/ibm_xgboost_full_rows/ibm_pre_italy_*.json`

### IBM CCF: sequence endpoint cells — `archive/ibm_sequence/`

LSTM cells and endpoint-matched XGBoost comparators that are not in the paper.
XGBoost receives the endpoint's feature row; the LSTM receives the whole window. These
reduced-target AP values are not comparable to the full-row protocol.

- **80/10/10, disjoint 30-transaction chunks** (679,372 endpoints; 36 temporal test
  frauds): `scripts/lstm_protocol.py --dataset ibm`; table `ibm_lstm_protocol.md`;
  records `ibm_{lstm,xgboost}_pre_italy*.json` without a length suffix.
- **80/10/10, disjoint 10-transaction chunks** (2,039,680 endpoints):
  `--window-length 10`; table `ibm_lstm_protocol_len10.md`; records `*_len10.json`.
- **68/16/16, length 10, stride 5, `default` preset:** LSTM and XGBoost, `--regime
  pre_italy_68_16_16 --window-length 10 --window-stride 5` and the two IID regimes;
  table `ibm_lstm_pre_italy_68_16_16_len10_stride5.md`; records
  `*68_16_16_len10_stride5.json`.
- **80/10/10 with `sqrt_weight_slow`:** the selected preset rerun at both lengths, as
  a check that the split conclusion does not depend on the 68/16/16 cutoffs; records
  `ibm_lstm_pre_italy*_cfg-sqrt_weight_slow.json`, no readable table. Results are in
  [`docs/experiments.md`](../docs/experiments.md).

### LSTM presets — `archive/lstm_presets/`

The three presets rejected during selection (`sqrt_weight`, `full_weight_slow`,
`sqrt_weight_slow_ln`) on the 68/16/16 temporal cell. They remain runnable with
`--lstm-config`; records only, no readable table.

### Sparkov: LSTM synthetic training-label delay — `archive/sparkov_lstm/`

The LSTM scores every Sparkov transaction using its current features and up to 29
(or 9) strictly earlier-time transactions from the same entity. The no-delay,
default-delay, and slow-delay regimes use the same rows, split, inputs, and five
seeds; only synthetic training-label availability changes.

- Length 30: `scripts/lstm_protocol.py --dataset sparkov`; table `sparkov_lstm_delay.md`;
  records `sparkov_lstm_{off,on,slow}.json`.
- Length 10: `--window-length 10`; table `sparkov_lstm_delay_len10.md`; records `*_len10.json`.
- Raw-ish ablation (21 current-event/static features, no entity-history statistics):
  `--feature-set rawish --window-length 10` or `30`; tables
  `sparkov_lstm_delay_rawish_len{10,30}.md`; records `*_rawish_len{10,30}.json`.
