# fraud-benchmark

Downloads public fraud and AML transaction datasets, normalizes them to a shared schema,
assigns temporal train/validation/test splits, and attaches a synthetic label-availability
("fraud reported at") timestamp to fraudulent transactions — so that an experiment can
model the labels a detector would actually have had at training time, rather than the ones
that exist in hindsight.

No data is redistributed. Everything is fetched at run time with your own credentials.

**Label delay for your own data.** The synthetic reporting delay is also available as
one standalone script, which needs only numpy and pandas. It adds `campaign_id` and
`reported_at` to any Parquet or CSV file of transactions:
**[`tools/label_delay/`](tools/label_delay/README.md)**.

**Documentation index: [`docs/README.md`](docs/README.md).**

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Kaggle credentials are required before anything can be downloaded, and IEEE-CIS
additionally needs its competition rules accepted once in a browser —
**[`docs/kaggle-setup.md`](docs/kaggle-setup.md)**.

```bash
fraud-benchmark list                # registered datasets and their licences
fraud-benchmark download paysim     # fetch the raw files and stop
fraud-benchmark prepare paysim      # normalize, split, delay, write (downloads if needed)
fraud-benchmark info paysim         # print the prepared dataset's card
```

Output lands in `data/processed/<name>/` as `data.parquet` plus a `dataset_card.json`
recording provenance, counts, split strategy, delay parameters and caveats.

`download` exists so a slow fetch can be done once and separately; `prepare` finds the
raw files already there and skips it. Both take `--all`, and one unavailable dataset
fails on its own without stopping the rest.

Every preparation stage is also a script under `scripts/`; the two paper experiments
have dedicated reproducible runners described below.

## Datasets

Eight registered: seven distinct sources plus `sparkov_slow`, which is `sparkov` under a
second delay regime. 73 million transactions in total. All but IEEE-CIS are synthetic;
IEEE-CIS is the only real-world data.

### Recommended

We recommend using the following datasets for sequential fraud (and AML) detection.
The datasets contain entity features, have a realistic time-span, and the artifacts
of the data do not disqualify them from benchmark purposes.

| dataset | rows | fraud rate | span | licence |
|---|---:|---:|---:|---|
| [`sparkov`](docs/datasets/sparkov.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`sparkov_slow`](docs/datasets/sparkov_slow.md) | 1,852,394 | 0.521% | 730d | CC0 1.0 |
| [`saml_d`](docs/datasets/saml_d.md) | 9,504,852 | 0.104% | 320d | CC BY-NC-SA 4.0 |
| [`ibm_ccf`](docs/datasets/ibm_ccf.md) | 24,386,900 | 0.122% | 10,649d | Apache-2.0 ⚠ |

There is a known artifact in the IBM CCF dataset: nearly all fraudulent transactions in the
validation and test periods have `Merchant State = Italy`.

### Use with caution

The following datasets are available, but are recommended to use with caution.

| dataset | rows | fraud rate | span | licence |
|---|---:|---:|---:|---|
| [`paysim`](docs/datasets/paysim.md) | 6,362,620 | 0.129% | 30d | CC BY-SA 4.0 |
| [`banksim`](docs/datasets/banksim.md) | 594,643 | 1.211% | 179d | CC BY-NC-SA 4.0 |
| [`ieee_cis`](docs/datasets/ieee_cis.md) | 590,540 | 3.499% | 181d | competition rules |
| [`amaretto`](docs/datasets/amaretto.md) | 29,704,090 | 0.274% | 83d | MIT |

⚠ **Read a dataset's page before using it.** Each page gives the numbers,
the schema mapping, and the disclaimers: **[`docs/datasets/`](docs/datasets/README.md)**.

## Time splits

Temporal, 80/10/10 by default, and **cut on timestamp values rather than row positions** —
datasets with coarse time resolution (PaySim's hours, BankSim's days) have heavy ties, and
slicing through a tied block would leak same-instant information across the boundary. A
consequence: the realised ratios drift slightly from 80/10/10 when ties are large.

`sparkov` and `sparkov_slow` are the exception. They keep their upstream test file as the
test split so results stay comparable with published work, with validation carved from the
train tail — roughly 63/7/30.

Every split boundary and count is recorded in the dataset card.

## Label delay

Real detectors learn from labels that arrive late. To model that, fraudulent transactions
are grouped into **campaigns** — one entity, each fraud within a configurable gap of the
last — and one delay is drawn per campaign from a lognormal in days, measured from the
campaign's **last** transaction. Every fraud in the campaign gets that same `reported_at`.

The number that matters is how many **train** frauds are still unreported at the train
cutoff. Those rows are not missing from training data; they sit in it looking legitimate.

| dataset | delay median | train labels censored |
|---|---:|---:|
| `saml_d` | 30d (AML) | 20.0% |
| `amaretto` | 7d | 17.5% |
| `paysim` | 1d | 10.1% |
| `sparkov_slow` | 15d, σ 1.665 | 8.9% |
| `banksim` | 7d | 8.1% |
| `ieee_cis` | 7d | 7.1% |
| `sparkov` | 7d | 2.2% |
| `ibm_ccf` | 7d | 0.0% |

`ibm_ccf` is effectively undelayed — 7 days against a 9,629-day train window censors 3
labels of 24,924 — so it cannot exercise a delay-aware method. `sparkov` and
`sparkov_slow` are the one paired contrast: identical rows, only `reported_at` differs.

The delay is calibrated per **domain** and whatever censoring results is reported rather
than tuned. Every `reported_at` in this repository is synthetic; no public dataset here
ships a real one. Full picture, including the two datasets whose spans cannot carry a
realistic clock: **[`docs/label-delay.md`](docs/label-delay.md)**.

To add the same synthetic delay to your own data, use
[`tools/label_delay/add_label_delay.py`](tools/label_delay/README.md). It is one
standalone file that reads Parquet or CSV and needs only numpy and pandas.

## Paper experiments

Three experiments back the paper's Figure 2 and Tables 3 and 4:

- **IBM CCF, ROC-AUC vs average precision:** XGBoost trained on a 1:1 downsample and on all
  training rows look interchangeable by ROC-AUC (0.9973 vs 0.9979) but not by average
  precision (0.672 vs 0.808).
- **IBM CCF, temporal vs iid splits:** an LSTM over 10-transaction windows loses about 0.20
  average precision from validation to test under a temporal split, but almost none under
  iid splits, which suggests that iid splits hide concept drift.
- **Sparkov, synthetic label delay:** an LSTM trained with delayed fraud labels loses test
  AP as the delay grows (−0.015 default, −0.040 slow, lower for every one of ten seeds).

```bash
for dataset in sparkov sparkov_slow; do
  .venv/bin/python scripts/download.py --dataset $dataset
  .venv/bin/python scripts/prepare.py --dataset $dataset
done
.venv/bin/python scripts/features.py --dataset sparkov
.venv/bin/pip install -e ".[lstm]"
.venv/bin/python scripts/lstm_protocol.py --dataset sparkov --window-length 10 \
  --lstm-config fixed15 --device cpu   # seeds 0–4
```

The two LSTM experiments need the optional `lstm` dependencies, and the IBM one needs a
GPU in practice. The full commands, including Sparkov seeds 5–9, are in
[`docs/experiments.md`](docs/experiments.md), along with the exact setup and caveats of
all three experiments. Tables and records are in [`results/`](results/README.md).
Table 5, the TabFormer reproduction, runs on the upstream TabFormer code rather than this
package; its protocol is described in the same document, and its code is available from
the authors on request.

## Tests

```bash
pytest
```

Network tests are deselected by default. Run them with `pytest -m network` — they perform
real Kaggle downloads.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{maskova2026standardized,
  author    = {Ma\v{s}kov\'{a}, Michaela and \v{S}m\'{\i}dl, V\'{a}clav},
  title     = {A Standardized Training and Evaluation Protocol for Sequential Fraud Detection},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance},
  series    = {ICAIF '26},
  year      = {2026},
  location  = {Milan, Italy},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  numpages  = {9},
  isbn      = {979-8-4007-3087-0},
  doi       = {10.1145/3858063.3858860},
  url       = {https://doi.org/10.1145/3858063.3858860}
}
```

## Licence

The code in this repository is MIT licensed (see `LICENSE`).

**The datasets are not.** Each carries its own terms, which you accept directly with the
upstream provider when you download it. Two sources are **NonCommercial** (BankSim,
SAML-D), three are **ShareAlike** (PaySim, BankSim, SAML-D), and IEEE-CIS is governed by
Kaggle competition rules rather than an open licence. `prepare --all
--exclude-noncommercial` skips the restricted ones. Full table:
**[`docs/dataset-licenses.md`](docs/dataset-licenses.md)**.
