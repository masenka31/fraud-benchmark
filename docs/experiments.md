# Paper experiments

Three experiments in this repository back the paper's Figure 2 and Tables 3 and 4; Table 5
is [described at the end](#ibm-ccf-tabformer-fraud-task-table-5-described-only). All report average
precision (AP) as the primary metric and fit all preprocessing on the training split only.
Figure 2 uses XGBoost and also reports ROC-AUC, to show what it hides; Tables 3 and 4 use a
one-layer LSTM. Generated tables, figures and records are in
[`results/`](../results/README.md).

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

## IBM CCF: ROC-AUC versus average precision under downsampling

Shows that ROC-AUC can make two models look interchangeable when average precision
separates them clearly.

- **Data:** the pre-Italy IBM CCF transactions and the temporal 68/16/16 split of the next
  section, one row per transaction (13,874,719 / 3,264,640 / 3,264,640 rows).
- **Models:** two XGBoost models (300 trees, depth 6, learning rate 0.1) with the same
  features, hyperparameters and model seed (0). `scale_pos_weight` is negatives/positives of
  the rows each model is fitted on.
  - *Model X*: all 17,170 training frauds plus as many random training negatives (1:1,
    sample seed 1729), so it sees fewer than 0.2% of the legitimate training rows;
  - *Model Y*: all training rows.
- **Features:** 90 columns. Unlike the repository's standard preparation, they keep the
  `artifact_*` columns and add merchant-state rarity and chronological fraud rates for five
  artifact categories. Every count and rate is fitted on training rows only, and a
  training row sees only earlier training labels.

```bash
.venv/bin/python scripts/ibm_downsampling_curves.py
```

It peaks at about 20 GB of memory and takes about 12 minutes on 4 CPU cores. It writes the
record and the figure (PDF and PNG) to `results/paper/ibm_downsampling_roc_pr.*`.

| | Model X (1:1) | Model Y (all rows) |
|---|---:|---:|
| Validation AP | 0.7805 | 0.8812 |
| Validation ROC-AUC | 0.9980 | 0.9984 |
| Test AP | 0.6716 | 0.8083 |
| Test ROC-AUC | 0.9973 | 0.9979 |

Caveats:

- One model seed and one subsample, so there is no spread across seeds.
- Only the training set is subsampled. Validation and test are always complete, with true
  labels.

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
  Earlier means a strictly earlier timestamp, so transactions at the same instant are
  excluded. Shorter histories are padded after the target, and the score is read at the
  target, so padding has no effect.
- **Regimes**, differing only in which training labels are available at the cutoff:
  - `off`: true labels;
  - `on`: synthetic default delay (`reported_at`): median 7 days, σ 1, no cap;
  - `slow`: synthetic fat-tailed stress delay (`reported_at_slow`): median 15 days,
    σ 1.665, capped at 365 days. Uncapped, its nominal mean would be 60 days; with the
    cap, the realised mean is 41.1 days.

  Frauds on one card form a campaign while each follows the previous within one day; one
  lognormal delay is drawn per campaign, from its last fraud.

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

## IBM CCF: TabFormer fraud task (Table 5, described only)

Shows that the window labeling convention and the split decide which method looks better.
The code for this experiment is not part of this repository; it is available from the
authors on request. It runs on the upstream
[TabFormer](https://github.com/IBM/TabFormer) code and its software stack (Python 3.7,
PyTorch 1.6, transformers 3.2), not on this package.

- **Task:** the TabFormer fraud-detection task (Padhi et al., 2021) on IBM CCF, rebuilt
  from the released code because the paper states neither the labeling rule nor the split.
- **Windows:** each user's transactions in file order, cut into complete windows of 10
  with stride 10, as upstream builds them: 2,437,789 windows.
- **Labels:**
  - *any-fraud* (the original rule, recovered from the released code): a window is fraud
    if any of its transactions is;
  - *window-end*: the label of the window's last transaction, closer to scoring each
    transaction.

  Every run is scored against the label it was trained on.
- **Splits:**
  - *seeded*: upstream's random 60/20/20 window split with its hard-coded seed
    (20200706), which reproduces the original partition exactly;
  - *temporal*: windows ending before the first Italy fraud (`2017-11-19 12:06`),
    split 60/20/20 by window end time.
- **Models:** an LSTM head on either the raw tokenized fields or frozen TabBERT
  representations. TabBERT is pretrained with upstream's script for one epoch. Training
  upsamples fraud windows to about 1:1 per epoch with a weighted sampler and uses Adam for
  at most 4 epochs; the epoch is chosen by validation F1 at threshold 0.5 (the original
  paper's convention), with patience 2. Three seeds (0–2).

| | Raw features | TabBERT |
|---|---:|---:|
| Published F1, any-fraud label | 0.83 | 0.86 |
| Reproduced F1, any-fraud label | 0.836 ± 0.008 | 0.780 ± 0.016 |
| Test AP, any-fraud label (seeded) | 0.845 ± 0.003 | 0.823 ± 0.005 |
| Test AP, window-end label (seeded) | 0.763 ± 0.013 | 0.801 ± 0.007 |
| Test AP, window-end label (temporal) | 0.409 ± 0.124 | 0.709 ± 0.009 |

Entries are mean ± standard deviation over the three seeds, as in the paper.

Caveats:

- The published count of 29,342 fraud samples cannot be reproduced: the released pipeline
  yields 11,493 any-fraud windows, and no natural construction gives 29,342.
- TabBERT stays below its published F1, most likely because the pretraining budget is
  unstated and ours (one epoch) is smaller.
- With stride 10, window-end labeling scores only every tenth transaction, so it is a
  diagnostic rather than the protocol's per-transaction evaluation.
- TabBERT's self-supervised pretraining may have included transactions from the
  validation and test periods (without labels), so the temporal TabBERT row may not be
  strictly chronological.

To regenerate the IBM table from the saved records without training, add `--render-only`
to any one of the three IBM commands above (it needs the same flags).
Other experiments from the project are in [`archive.md`](archive.md).
