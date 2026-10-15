# Paper experiments

Two experiments back the paper's tables. Both report average precision (AP) as the
primary metric, fit all preprocessing on the training split only, and use five fixed
seeds (0–4). Generated tables and records are in [`results/`](../results/README.md).

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
- **Regimes**, differing only in which training labels are available at the cutoff:
  - `off`: true labels;
  - `on`: synthetic default delay (`reported_at`);
  - `slow`: synthetic fat-tailed stress delay (`reported_at_slow`).

  A fraud reported after the cutoff stays in training as a negative: 132 (1.6%) of
  the 8,302 training frauds under `on`, 509 (6.1%) under `slow`. Validation and test always use
  the true labels. The delays are synthetic, and `slow` is a stress test.
- **Model:** XGBoost (300 trees, depth 6, learning rate 0.1) on 46 features, with
  `scale_pos_weight` set from the available training labels. See
  [`models.py`](../src/fraud_benchmark/experiments/models.py) and
  [`features/sparkov.py`](../src/fraud_benchmark/experiments/features/sparkov.py).

```bash
.venv/bin/python scripts/sparkov_delay_protocol.py
```

| Training labels | Hidden train frauds | Validation AP | Test AP | Δ test AP |
|---|---:|---:|---:|---:|
| No delay | 0 | 0.9774 ± 0.0004 | 0.9734 ± 0.0013 | +0.0000 |
| Default delay | 132 | 0.9713 ± 0.0011 | 0.9504 ± 0.0029 | −0.0230 |
| Slow delay | 509 | 0.9569 ± 0.0020 | 0.9388 ± 0.0031 | −0.0346 |

To regenerate a table from the saved records without training, add `--render-only` to the
Sparkov command, or to any one of the three IBM commands above (it needs the same flags).
Other experiments from the project are in [`archive.md`](archive.md).
