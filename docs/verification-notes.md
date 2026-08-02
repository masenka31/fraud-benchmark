# Verification notes

Observations from preparing all seven datasets against the real cached data on 2026-08-01.
These are findings worth remembering, not action items with owners. Anything requiring code
changes was fixed at the time or is called out as open.

## Measured figures — all seven

| dataset | rows | fraud rate | entities | split |
|---|---:|---:|---:|---|
| paysim | 6,362,620 | 0.1291% | 6,353,307 | 80.4 / 10.1 / 9.5 |
| banksim | 594,643 | 1.2108% | 4,112 | 80.0 / 10.6 / 9.4 |
| sparkov | 1,852,394 | 0.5210% | 999 | 63.0 / 7.0 / 30.0 |
| saml_d | 9,504,852 | 0.1039% | 292,715 | 80.0 / 10.0 / 10.0 |
| ibm_ccf | 24,386,900 | 0.1220% | 2,000 | 80.0 / 10.0 / 10.0 |
| ieee_cis | 590,540 | 3.4990% | 290,492 | 80.0 / 10.0 / 10.0 |
| amaretto | 29,704,090 | 0.2736% | 400 | 80.0 / 10.0 / 10.0 |

Total: 73 million transactions. Every dataset: **0 null entity_ids, 0 timestamps straddling
a split boundary, no train/val or val/test leakage.**

Sparkov's 63/7/30 is deliberate — it preserves the upstream test file so results stay
comparable with published work.

IEEE-CIS additionally writes `unlabelled_test.parquet`: 506,691 rows, no `isFraud`,
spanning 2018-07-02 to 2018-12-31 — cleanly after the labelled period.

## Discrepancies against upstream documentation

- **Amaretto's README undercounts its own anomalies.** It states 81,262; the file contains
  **81,268** (`Anomaly > 0`), a difference of 6. Class breakdown over all 29.7M rows:
  1: 5,884 · 2: 2,576 · 3: 12,088 · 4: 30,640 · 5: 30,080. The README's 0.27% rounds our
  measured 0.2736%.
- **BankSim has 4,112 distinct customers, not 4,098** as an earlier note claimed. Verified
  the raw file has 4,112 both before and after quote-stripping, so nothing is being merged.
- **SAML-D's true fraud rate is 0.1039%**, not the 0.118% that a sample-based estimate
  suggested.

## Campaign groupability — the input to the label-delay design

"% grouped" is the share of fraud ROWS whose entity has more than one fraud.

| dataset | frauds | fraud entities | entities >1 | % grouped | max/entity | median group |
|---|---:|---:|---:|---:|---:|---:|
| paysim | 8,213 | 8,213 | 0 | 0.0% | 1 | – |
| banksim | 7,200 | 1,483 | 1,057 | 94.1% | 144 | 3 |
| sparkov | 9,651 | 976 | 976 | 100.0% | 19 | 10 |
| saml_d | 9,873 | 4,950 | 1,026 | 60.3% | 37 | 5 |
| ibm_ccf | 29,757 | 1,343 | 1,313 | 99.9% | 113 | 19 |
| ieee_cis | 20,663 | 13,320 | 2,205 | 46.2% | 72 | 3 |
| amaretto | 81,268 | **21** | 21 | 100.0% | **36,673** | 756 |

Two datasets sit at the extremes, and both matter:

**PaySim has no campaigns at all.** Every fraud has a unique `nameOrig`. Grouping by
`nameDest` barely helps (44 destinations with 2 frauds). Its fraud is 4,097 TRANSFER +
4,116 CASH_OUT; matching on `(amount, step)` finds 4,272 pairs, but the TRANSFER's
`nameDest` never equals the CASH_OUT's `nameOrig` — 0 of 4,272, a known generation artifact.
So the two legs of one fraud are not linkable by account.

**Amaretto is the opposite and is the real constraint on the design.** All 81,268 anomalies
belong to just **21 of its 400 clients**, and each anomalous client's anomalies span
**76–83 days** — essentially the dataset's whole period. Amaretto's anomalies are injected
as client-level *behaviours* (the five FATF typologies), not as bursts. An entity-only
campaign rule would therefore collapse up to 36,673 transactions into a single reported
timestamp, which is obviously wrong.

**Conclusion for the label-delay work: the campaign rule cannot be entity-only.** It must be
entity *plus* a time-gap threshold, and the threshold's effect varies enormously by dataset —
from no-op (PaySim) to decisive (Amaretto). Whatever threshold is chosen, its resulting
campaign-size distribution should be measured per dataset rather than assumed.

## Label delay as shipped — measured on all seven

Produced by `fraud-benchmark prepare --all` at the shipped defaults: lognormal delay with
`median_days: 7.0`, `sigma: 1.0`, `seed: 0`, no truncation; campaign gap `1d` globally with
Amaretto overridden to `1h`.

### Invariants

Every dataset: no fraud reported before it happened, no fraud left unreported, no
`reported_at` or `campaign_id` on a non-fraud row.

| dataset | frauds | early | null | stray | delay p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| paysim | 8,213 | 0 | 0 | 0 | 6.87d | 36.33d | 181.84d |
| banksim | 7,200 | 0 | 0 | 0 | 7.50d | 35.17d | 181.84d |
| sparkov | 9,651 | 0 | 0 | 0 | 7.29d | 31.07d | 152.03d |
| saml_d | 9,873 | 0 | 0 | 0 | 7.06d | 36.21d | 181.84d |
| ibm_ccf | 29,757 | 0 | 0 | 0 | 7.29d | 36.52d | 227.64d |
| ieee_cis | 20,663 | 0 | 0 | 0 | 7.01d | 36.29d | 361.93d |
| amaretto | 81,268 | 0 | 0 | 0 | 6.85d | 34.52d | 150.40d |

Per-row median delay lands on the configured 7 days everywhere, and mean-above-median holds:
the p95 near 36 days is the lognormal tail the choice was made for.

### Campaign sizes at the shipped gaps

| dataset | gap | campaigns | size p50 | p95 | max | singletons |
|---|---|---:|---:|---:|---:|---:|
| paysim | 1d | 8,213 | 1.0 | 1.0 | 1 | 100.0% |
| banksim | 1d | 5,551 | 1.0 | 3.0 | 35 | 86.5% |
| sparkov | 1d | 1,005 | 10.0 | 15.0 | 19 | 1.9% |
| saml_d | 1d | 7,887 | 1.0 | 3.0 | 10 | 85.1% |
| ibm_ccf | 1d | 9,769 | 2.0 | 9.0 | 20 | 30.6% |
| ieee_cis | 1d | 16,170 | 1.0 | 3.0 | 18 | 85.3% |
| amaretto | 1h | 1,852 | 8.0 | 181.4 | 1,145 | 27.9% |

**These reproduce the offline threshold sweep exactly**, dataset for dataset, which is the
useful result: the shipped `assign_campaigns` behaves as the gap was chosen to make it.

**Recommendation: keep `1d` global and `1h` for Amaretto.** Amaretto's largest campaign is
1,145 rows against the 36,673 an entity-only rule would have produced, and its 27.9%
singletons show the gap is fragmenting rather than fusing. No card dataset is collapsed —
Sparkov's genuine multi-fraud runs survive intact at a median of 10, and no maximum
elsewhere exceeds 35. PaySim stays all-singletons, which is correct for it.

Amaretto's p95 of 181 rows against a median of 8 is the expected residue of its five
typologies differing by two orders of magnitude in burstiness; a per-class gap remains
possible later if that turns out to matter.

### Label availability at the train cutoff

The point of the exercise: of the frauds in the train split, how many would a model training
at the end of that window actually have labels for?

| dataset | train frauds | known at cutoff | train window |
|---|---:|---:|---:|
| paysim | 3,963 | **47.3%** | 14d |
| banksim | 5,920 | 91.9% | 147d |
| sparkov | 6,674 | 97.8% | 487d |
| saml_d | 7,730 | 94.3% | 255d |
| ibm_ccf | 24,924 | 100.0% (3 unknown) | 9,629d |
| ieee_cis | 16,599 | 92.9% | 140d |
| amaretto | 65,580 | 82.5% | 65d |

High but not total is exactly the intended effect: frauds late in the train window are still
unreported at the cutoff, and any experiment that trains on them is using the future.

**Two datasets sit at the extremes, and both are properties of their time spans, not bugs.**

- **PaySim loses more than half its train labels.** Its entire span is 30 simulated days, so
  a 14-day train window against a 7-day median delay censors 2,090 of 3,963 train frauds.
  This is the most severe delay regime in the suite. It is defensible as-is — PaySim is a
  simulation and its clock is arbitrary — but anyone benchmarking on PaySim should know the
  delay is huge relative to the data, and a per-dataset `median_days` override is the lever
  if a milder regime is wanted.
- **IBM CCF is effectively undelayed**, 3 unknown of 24,924. Its 10,649-day span makes a
  7-day delay negligible. Realistic, and it means IBM CCF will not exercise delay-aware
  evaluation much.

## Open, non-blocking

- **IEEE-CIS emits `PerformanceWarning: DataFrame is highly fragmented`** during preparation.
  It comes from `df.insert` on a 394-column frame and is cosmetic, but it is noisy on every
  run. Building the canonical columns with a single `pd.concat(axis=1)` would silence it.
- **IBM CCF uses no `category` dtype** for its roughly ten low-cardinality string columns
  repeated across 24.4M rows (`Use Chip`, `Card Brand`, `Card Type`, `Gender`, …). Peak
  memory is tens of GB where it need not be. Fine on this machine, wasteful elsewhere.
- **Amaretto's extraction cache is 3.6 GB** at `data/raw/amaretto/_extracted/`, retained so
  repeat runs skip reassembly. It is gitignored. Deleting it costs one reassembly.
- Preparing all seven from cached raw data takes **7m47s** end to end.
