# Paper experiments

Two experiments back the paper's Tables 3 and 4. Both train a one-layer LSTM, report
average precision (AP) as the primary metric, and fit all preprocessing on the training
split only. Generated tables and records are in [`results/`](../results/README.md).

Build the inputs first. The Sparkov features also need `sparkov_slow`, which supplies the
slow-delay timestamps:

```bash
for dataset in ibm_ccf sparkov sparkov_slow; do
  .venv/bin/python scripts/download.py --dataset $dataset
  .venv/bin/python scripts/prepare.py --dataset $dataset
done
.venv/bin/python scripts/features.py --dataset ibm_ccf
.venv/bin/python scripts/features.py --dataset sparkov
```

## IBM CCF: temporal versus iid splits

Suggests that an iid split hides concept drift that a temporal split reveals.

- **Data:** IBM CCF transactions strictly before the first Italy fraud
  (`2017-11-19 12:06`), which removes the Italy artifact: 20,403,999 transactions.
- **Windows:** each user's transactions (across all their cards) in time order, cut into
  complete windows of 10 transactions every 5 transactions (4,078,583 windows). A window
  is labeled by its last transaction.
- **Splits**, over the same windows:
  - *temporal*: chronological 68/16/16 of the transactions (cutoffs `2014-01-17 13:39` and
    `2015-12-25 02:37`); each window goes to the partition of its last transaction;
  - *window iid*: windows assigned independently, exactly 68/16/16;
  - *customer iid*: all windows of a user kept together, 68/16/16 by window count.

  The iid assignments use one fixed seed, separate from the model seeds.
- **Model:** a one-layer LSTM (64 units) over the 10 rows of 77 label-free features,
  using the `sqrt_weight_slow` preset in [`lstm.py`](../src/fraud_benchmark/experiments/lstm.py):
  fraud weight √(negatives/positives), AdamW (learning rate 3e-4, weight decay 1e-4) with
  one-epoch warmup and cosine decay, at most 15 epochs with patience 4, checkpoint chosen
  by validation AP.
- **Seeds:** five (0–4).

```bash
.venv/bin/pip install -e ".[lstm]"   # PyTorch matching your GPU; CPU works but is slow
for regime in pre_italy_68_16_16 pre_italy_iid_chunks_68_16_16 pre_italy_iid_customers_68_16_16; do
  .venv/bin/python scripts/lstm_protocol.py --dataset ibm --regime $regime \
    --window-length 10 --window-stride 5 --model lstm --lstm-config sqrt_weight_slow --device cuda
done
```

| | Temporal | Window iid | Customer iid |
|---|---:|---:|---:|
| Validation AP | 0.8279 ± 0.0308 | 0.7777 ± 0.0131 | 0.7659 ± 0.0116 |
| Test AP | 0.6301 ± 0.0670 | 0.7709 ± 0.0092 | 0.7761 ± 0.0102 |
| Test − validation AP | −0.1978 ± 0.0362 | −0.0068 ± 0.0065 | +0.0102 ± 0.0049 |

Caveats:

- The 68/16/16 ratio was chosen after looking at the data, so that validation and test
  have similar fraud rates (0.121% and 0.117%).
- The LSTM preset was chosen from five candidates on temporal validation AP.
- Overlapping windows let window-iid share context transactions across partitions.
- The last fraud in this population is on 2016-12-22, so about 48% of the test-period
  transactions (2016-12-22 to 2017-11-19) contain no fraud.
- The three test sets have slightly different fraud rates, so the test − validation row
  is a cleaner comparison than test AP alone.

## Sparkov: synthetic label delay

Shows how delayed fraud labels at training time affect a detector.

- **Data:** `data/features/sparkov.parquet`, all rows, no subsampling. The runner recomputes a
  chronological 80/10/10 split (1,481,915 / 185,239 / 185,240 rows; training ends
  `2020-08-25 01:31`). This is not the dataset's own split described in the README, so the
  censoring figures there do not apply here.
- **Targets:** every transaction, scored with up to 9 earlier transactions of the same card.
- **Regimes**, differing only in which training labels are available at the cutoff:
  - `off`: true labels;
  - `on`: synthetic default delay (`reported_at`);
  - `slow`: synthetic fat-tailed stress delay (`reported_at_slow`).

  A fraud reported after the cutoff stays in training as a negative: 132 (1.6%) of
  the 8,302 training frauds under `on`, 509 (6.1%) under `slow`. Validation and test always use
  the true labels. The delays are synthetic, and `slow` is a stress test.
- **Model:** a one-layer LSTM (64 units) over 46 label-free features, using the `fixed15`
  preset in [`lstm.py`](../src/fraud_benchmark/experiments/lstm.py): fraud weight
  negatives/positives from the available training labels, AdamW (learning rate 1e-3, weight
  decay 1e-4), exactly 15 epochs without early stopping, final weights scored. Validation
  labels never choose a checkpoint. This preset differs from the IBM experiment's.
- **Seeds:** ten (0–9). Seeds 0–4 ran as one job per regime and seeds 5–9 as one job per
  seed; the records are kept as run, and the table merges them.

The recorded runs used the CPU. GPU runs are not bit-identical to them.

```bash
.venv/bin/pip install -e ".[lstm]"
# seeds 0–4: one record per regime
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --window-length 10 \
  --lstm-config fixed15 --device cpu
# seeds 5–9: one run per seed, each in its own folder
for seed in 5 6 7 8 9; do
  .venv/bin/python scripts/lstm_protocol.py --dataset sparkov --window-length 10 \
    --lstm-config fixed15 --device cpu --seeds $seed \
    --results-dir results/paper/sparkov_lstm_delay/seed$seed
done
# the table, from all ten seeds
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --window-length 10 \
  --lstm-config fixed15 --render-only
```

| | No delay | Default delay | Slow delay |
|---|---:|---:|---:|
| Median delay | – | 7.3 days | 13.9 days |
| Mean delay | – | 11.3 days | 41.1 days |
| Hidden train frauds | 0 | 132 (1.6%) | 509 (6.1%) |
| Validation AP | 0.9789 ± 0.0031 | 0.9734 ± 0.0067 | 0.9562 ± 0.0132 |
| Test AP | 0.9772 ± 0.0065 | 0.9621 ± 0.0164 | 0.9376 ± 0.0260 |
| Δ test AP | – | −0.0151 (0.0041) | −0.0396 (0.0090) |
| Seeds with lower test AP than no delay | – | 10/10 | 10/10 |

AP entries are mean ± population standard deviation over the ten seeds. Δ is the mean
within-seed difference in test AP to the no-delay run, with its standard error in
parentheses. Delays are realised transaction-to-report delays over all Sparkov frauds.

To regenerate the IBM table from the saved records without training, add `--render-only`
to any one of the three IBM commands above (it needs the same flags).
Other experiments from the project are in [`archive.md`](archive.md).
