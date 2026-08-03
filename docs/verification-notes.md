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
| saml_d | 7,730 | 94.3% → **80.0%** | 255d |
| ibm_ccf | 24,924 | 100.0% (3 unknown) | 9,629d |
| ieee_cis | 16,599 | 92.9% | 140d |
| amaretto | 65,580 | 82.5% | 65d |

High but not total is exactly the intended effect: frauds late in the train window are still
unreported at the cutoff, and any experiment that trains on them is using the future.

These figures are all under the single global 7-day delay. `paysim` and `saml_d` were later
given per-dataset overrides — see *Per-dataset delay overrides* and *Delay calibration by
domain* below for the current values.

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

## Per-dataset delay overrides, and two IBM CCF subsamples (2026-08-02)

Two datasets were degenerate under one global distribution, so `delay:` is now overridable
per dataset. Designed in a local working spec (`docs/superpowers/`, not tracked).

### PaySim: median 1 day

Its whole span is 30 simulated days, so a 7-day median censored more than half its train
labels. `sigma` and `seed` still inherit.

| | before | after |
|---|---:|---:|
| known at train cutoff | 47.3% | **89.9%** |
| delay p50 | 6.87d | 0.96d |
| delay p95 | 36.33d | 5.23d |

`early=0 null=0 stray=0`. This makes PaySim usable; it does not make a 30-day clock
realistic.

### `ibm_ccf_subsample_fast` and `ibm_ccf_subsample_slow`

Both crop IBM CCF to `[2016-01-01, 2019-10-27 14:54]` — the right edge is the last
labelled fraud, found in the data. **Verified row-identical: equal on all 50 non-delay
columns across 6,569,157 rows, differing only in `reported_at`.** That property is what
makes the two comparable.

| | rows | frauds | train / val / test | campaigns |
|---|---:|---:|---|---:|
| both | 6,569,157 | 8,412 | 6,441 / 989 / 982 | 3,848 (p50 2, max 16) |

| | median | nominal mean | sigma | cap | realised p50 | realised mean | p95 | known @ cutoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fast | 7d | 30d | 1.706 | 365d | 6.8d | 25.3d | 117.2d | **97.0%** |
| slow | 15d | 60d | 1.665 | 730d | 14.2d | 51.3d | 234.8d | **94.8%** |

`early=0 null=0 stray=0` for both. Against the full `ibm_ccf`'s 100.0%, both now exercise
delay-aware evaluation, and slow roughly doubles fast's censoring.

Two figures that look wrong but are not:

- **Realised means sit ~15% below nominal** (25.3 against 30, 51.3 against 60). The caps
  bite about 1% of campaigns. Untruncated, these distributions sample delays out to 1,813
  and 3,400 days — years past the end of the data. Cards record realised as well as
  configured values, via `observed_mean_delay_days`.
- **Max row-level delay slightly exceeds the cap** (366.9d against 365, 731.9d against
  730). The cap applies to the campaign's single draw, measured from its *last*
  transaction; a row earlier in the campaign carries the campaign's span on top of it.

### Why 2016-01-01 rather than a relative window

Measured, all cropped at the last fraud:

| left crop | rows | frauds | test frauds | known @ cutoff |
|---|---:|---:|---:|---:|
| 2015-01-01 | 8.27M | 11,693 | 1,256 | 98.2% |
| **2016-01-01** | 6.57M | 8,412 | 982 | 97.0% |
| 2016-10-27 (3y) | 5.17M | 5,327 | 779 | 94.5% |
| 2017-01-01 | 4.86M | 4,833 | 761 | 95.2% |

A three-year window starts in late October 2016 and discards most of a heavy fraud year:
+27% rows buys +58% frauds. Cropping on the last *transaction* instead of the last
labelled fraud yields **0 frauds in test** — the four dead months swallow the whole split.

### Artifacts found in IBM CCF while doing this

Three, each found incidentally, during the IBM CCF artifact audit.

1. **The generator stops emitting fraud 2019-10-27** while transactions run to 2020-02-28 — 645,180 rows,
   zero frauds. Handled by the crop.
2. **2017 has 255 frauds** against 3,579 in 2016 and 2,491 in 2018, a 14× dip. Falls
   inside the subsample train split; not otherwise handled.
3. **`Merchant State` is close to an oracle.** In the subsample window `Italy` covers
   6,099 rows at **76.8% fraud** and accounts for **55.7% of all frauds**; `Algeria` is
   96.2% fraud over 654 rows in the full data. Recorded in the adapters' caveats. A model
   handed this column raw will score well and have learned nothing.

All three are superseded by the audit below, which ran on 2026-08-02.

## Known leakage (2026-08-02)

The IBM CCF artifact audit, run
against all seven processed datasets plus `ibm_ccf_subsample_fast` (the slow variant is
row-identical outside `reported_at`, so it is not swept separately).

**Method.** For every column of every dataset, the per-value fraud rate against the
dataset's base rate. Excluded: the label and everything derived from it — `is_fraud`,
`reported_at`, `campaign_id`, `split`, and each source's own raw label column (`Is Fraud?`,
`fraud`, `isFraud`, `Is_laundering`, `Laundering_type`, `Anomaly`). A value is **flagged**
when it covers ≥100 rows, carries ≥20 frauds, runs at ≥10× the base rate, and has a Poisson
upper-tail p < 1e-9. The last two conditions earn their keep: without them SAML-D's `Time`
column returns 627 "oracles" that are pure small-number noise — 86,400 distinct
second-of-day values, ~130 rows each, 3 frauds apiece.

Ranked worst to cleanest:

| dataset | flagged values | worst artifact | severity |
|---|---:|---|---|
| ibm_ccf | 180 | after 2016 fraud is almost entirely foreign-merchant | **evaluation split is a single categorical value** |
| ibm_ccf_subsample_* | 85 | same, concentrated | **val and test frauds are 100% `Italy`** |
| paysim | 6 | fraud is flat in time, traffic is not | 320 steps are 100% fraud |
| amaretto | 20 | 3 clients; round amounts | one whole anomaly class is a one-liner |
| banksim | 76 | `category`, `merchant` | injected into named merchants |
| saml_d | 73 | receiver geography | mild, plausible |
| ieee_cis | 470 | narrow slices of `V*`/`C*` | not an artifact — see below |
| sparkov | **0** | — | clean |

### IBM CCF — the fraud generator changes regime in 2017, and the split inherits it

This is the significant finding and it is worse than the `Italy` note above suggested.
Fraud counts by year and merchant geography, full data (`ONLINE` = `Merchant City` is
literally `ONLINE`; `US` = a two-letter state code; `foreign` = a country name):

| year | ONLINE | US | foreign |
|---|---:|---:|---:|
| 2015 | 2,777 | 504 | 0 |
| 2016 | 3,073 | 506 | 0 |
| 2017 | 0 | 0 | 255 |
| 2018 | 128 | 23 | 2,340 |
| 2019 | 0 | 0 | 2,087 |

The **last online fraud is 2018-12-04** and the **last US fraud 2018-12-03**; the last
labelled fraud of any kind is 2019-10-27. The **first `Italy` fraud is 2017-11-19**, and
`Italy` had 2,985 transactions and zero frauds across the preceding 22 years.

Because the splits are temporal, the whole of val and nearly all of test fall inside the
foreign-only regime:

| dataset | split | frauds | from `Italy` | online |
|---|---|---:|---:|---:|
| ibm_ccf | train | 24,924 | 0 (0.0%) | 18,221 (73.1%) |
| ibm_ccf | val | 2,115 | 2,115 (**100%**) | 0 |
| ibm_ccf | test | 2,718 | 2,567 (94.4%) | 128 (4.7%) |
| subsample | train | 6,441 | 2,711 (42.1%) | 3,201 (49.7%) |
| subsample | val | 989 | 989 (**100%**) | 0 |
| subsample | test | 982 | 982 (**100%**) | 0 |

So `Merchant State == "Italy"`, a one-line rule with no fitting of any kind, scores:

| dataset | split | TP | FP | FN | precision | recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ibm_ccf | val | 2,115 | 399 | 0 | 0.841 | 1.000 | **0.914** |
| ibm_ccf | test | 2,567 | 608 | 151 | 0.809 | 0.944 | **0.871** |
| subsample | val | 989 | 248 | 0 | 0.800 | 1.000 | **0.889** |
| subsample | test | 982 | 188 | 0 | 0.839 | 1.000 | **0.913** |

Any baseline on IBM CCF or its subsamples is competing against that number, and train
teaches a fraud population — three-quarters online in the full data — that no longer exists
by validation time. This is not a feature to drop and move on from; it is what the
evaluation labels *are*.

**Foreign is not the artifact; ten countries are.** Of 171 country values, **161 have zero
frauds** across 92,091 rows. The ten that do, full data:

| country | rows | frauds | rate | share of all frauds |
|---|---:|---:|---:|---:|
| Tuvalu | 59 | 59 | **100.0%** | 0.2% |
| Algeria | 654 | 629 | 96.2% | 2.1% |
| Haiti | 446 | 375 | 84.1% | 1.3% |
| Fiji | 40 | 32 | 80.0% | 0.1% |
| Nigeria | 233 | 143 | 61.4% | 0.5% |
| Turkey | 472 | 257 | 54.4% | 0.9% |
| Italy | 8,730 | 4,682 | 53.6% | 15.7% |
| Japan | 3,955 | 49 | 1.2% | 0.2% |
| Mexico | 47,152 | 272 | 0.6% | 0.9% |
| India | 3,482 | 5 | 0.1% | 0.0% |

`Merchant City` mirrors this exactly (Rome, Algiers, Port au Prince, Istanbul, Abuja) and
is therefore leaky in the same way. Note `Rome` also occurs in NY, GA, OH and PA — 5,035 US
rows carrying a single fraud — so it is the pair, not the city name, that predicts.

**`MCC` is a second, independent artifact.** Not explained by geography: MCC 5732 has 843
frauds at 6.7% (55× base) with **none** in Italy; MCC 4411 is 50.0% fraud over 634 rows.
24 MCC values are flagged.

**A US-side location oracle exists too.** `Merchant City == "Strasburg"`, OH, zip 44680:
1,852 rows, 322 frauds, 17.4%. Inside it, whole merchant×MCC groups are pure — 56/56,
53/54, 33/35. `Zip == 44680` is flagged in its own right (889 rows, 36.0%).

**`Errors?` — the a priori suspicion was wrong, but there is leakage of a different shape.**
Measured on non-Italy rows of the subsample window (base 0.057%):

| value | rows | frauds | rate | lift |
|---|---:|---:|---:|---:|
| Bad CVV | 3,214 | 47 | 1.46% | **25.7×** |
| Bad Expiration | 3,216 | 26 | 0.81% | **14.2×** |
| Bad Card Number | 4,112 | 19 | 0.46% | 8.1× |
| Bad PIN | 15,578 | 10 | 0.06% | 1.1× |
| Insufficient Balance | 63,839 | 49 | 0.08% | 1.4× |
| Technical Glitch | 13,025 | 8 | 0.06% | 1.1× |

The two the audit note suspected — `Bad PIN` and `Insufficient Balance` — are not enriched
at all. The enriched trio is card-not-present detail failures, which is the card-testing
signature: a consequence of the same attempt, not an independent predictor. Real leakage,
but narrow — 92 frauds in the window.

**Entity concentration is *not* an artifact here**, in direct contrast to Amaretto. In the
subsample, 977 of 1,565 users carry at least one fraud (62.4%); the top 100 hold 28.0% of
frauds; the worst single user runs at 1.82% fraud. `entity_id` is nothing like a label.

**`Amount` is shifted, not separated.** Fraud mean $100.28 against $42.81 legitimate, with
fully overlapping distributions; only the top decile is enriched (4.1× overall, 4.4× with
Italy removed) and every other decile sits *below* base. Two narrow exceptions: micro-amounts
$0.01–$0.09 run at 11–35× base (~120 frauds, card testing), and 509 individual values clear
the flag threshold but between them cover only 1,825 frauds (6%). No separate generator.

**`Use Chip` confirmed as signal, not artifact.** Online 0.392% against chip 0.098%, 3.1×
and directionally right. Interacted with Italy it is severe (in the subsample window,
chip-in-Italy 77.1% and swipe-in-Italy 74.0% fraud), but that is the geography artifact,
not the channel.

**The fraud-per-year swing is not a 2017 story.** 2017's 255 frauds against ~3,000 in each
neighbour was already noted, but **2011 is worse — 55 frauds against 3,835 in 2010 and 1,333
in 2012, a 70× dip.** Yearly fraud rate over the labelled span ranges from 0.0035% to
0.303%, an 87× spread. The generator's fraud intensity is not stationary at any scale.

### PaySim — fraud is uniform in time, legitimate traffic is not

`step` is the simulated hour. Fraud is dead flat across the 24-hour cycle (~340 per
hour-of-day slot, every slot) while legitimate volume swings 500×. The result:

| hour of day | rows | frauds | rate | lift |
|---:|---:|---:|---:|---:|
| 03 | 2,007 | 326 | **16.2%** | 126× |
| 04 | 1,241 | 274 | **22.1%** | 171× |
| 05 | 1,641 | 366 | **22.3%** | 173× |
| 12 | 483,418 | 339 | 0.07% | 0.5× |
| 19 | 647,814 | 342 | 0.05% | 0.4× |

Correlation between log step volume and step fraud rate: **−0.925**. **320 of 743 steps
contain nothing but fraud** — 3,620 rows, **44.1% of all PaySim frauds**, identified
perfectly by the timestamp alone. It holds in every split: 34.6% of train frauds, 30.9% of
val, 55.9% of test. Any feature that touches hour-of-day carries this.

Also: `type` is a hard gate. `CASH_IN`, `DEBIT` and `PAYMENT` — 3,592,211 rows, 56.5% of
the dataset — contain **exactly zero** frauds. And `oldbalanceOrg == 10,000,000.0` is 142
rows, all 142 fraud, the simulator's balance cap showing through.

### Amaretto — one of the five typologies is a one-liner

Entity concentration was already known (21 of 400 clients). Its sharp end: **three clients
hold 83% of all anomalies** — Client_066 45.1%, Client_212 22.9%, Client_126 15.0%.

New: **`amount` an exact multiple of 1,000 is 2,817 rows, 91.4% anomalous**, and those rows
are **all 2,576 of Anomaly class 2 — the entire class, with no members outside the rule.**
The values are $8,000–$13,000; $13,000 is 463 rows at 100%. On test the rule scores 0.948
precision. One of the five FATF typologies is recoverable exactly by `amount % 1000 == 0`.

### BankSim — fraud is injected into named merchants and categories

| column | value | rows | frauds | rate | lift | share of frauds |
|---|---|---:|---:|---:|---:|---:|
| category | es_leisure | 499 | 474 | 95.0% | 78× | 6.6% |
| category | es_travel | 728 | 578 | 79.4% | 66× | 8.0% |
| category | es_sportsandtoys | 4,002 | 1,982 | 49.5% | 41× | 27.5% |
| merchant | M1294758098 | 191 | 184 | 96.3% | 80× | 2.6% |
| merchant | M980657600 | 1,769 | 1,472 | 83.2% | 69× | 20.4% |
| merchant | M732195782 | 608 | 518 | 85.2% | 70× | 7.2% |

16 merchants and 6 categories are flagged. The six categories together are 9,871 rows at 41.7% fraud and
carry 57.1% of all frauds; as a rule they score 0.463 precision / 0.557 recall on test — leaky, but nowhere
near IBM CCF's F1 0.91, because BankSim's fraud is spread across the categories rather
than confined to one value. 27 customers are also flagged (max 2.0% of frauds each).

### SAML-D — mild and plausible

Nothing oracle-like: **no value anywhere reaches a 50% fraud rate.** The strongest effects
are geographic and directional in a way money laundering actually is — `Receiver_bank_location`
Nigeria 6.3×, Morocco 6.3×, Albania 5.8×; `Payment_type` Cash Deposit 6.0×, Cash Withdrawal
4.3×. The 73 flagged values are all account ids (25 receiver, 24 sender, plus their
`entity_id` duplicates), none holding more than 0.4% of frauds — recurring mule accounts,
which is the phenomenon, not an artifact.

### IEEE-CIS — the flags are a property of the feature set, not the data

470 values flagged across 140 columns, but **every one lands in an engineered column** —
`V*` (114 columns), `C*`, `D*`, `id_*` — and **no flagged value covers as many as 20,000
rows**. The largest is `V242 == 2.0`: 5,218 rows, 35.1% fraud, 8.9% of frauds. The raw
columns are unremarkable at a 3.5% base rate: `ProductCD == C` 3.3×, `M4 == M2` 3.3×,
`card6 == credit` 1.9×, `P_emaildomain == outlook.com` 2.7×.

Vesta's `V`/`C`/`D` blocks are undocumented aggregates built by a fraud vendor, so narrow
high-fraud slices are exactly what they should look like. Treat as feature-set structure,
not generation leakage. It is the only one of the seven that is real-world data.

### Sparkov — clean

**Zero flagged values.** The strongest effect anywhere in the dataset is `job == "TEFL
teacher"` at 8.1× over 760 rows, and `job`, `city`, `state` and `dob` are per-customer
constants across 999 customers of ~1,850 rows each, so those are customer effects seen
through a demographic column. `category` tops out at `shopping_net` 3.1× and `grocery_pos`
2.4× — the categories real card fraud does favour. Sparkov is the only generated dataset
in the suite whose labels are not recoverable from a single column value.

### What this changes

Nothing in the pipeline; this audit measured and recorded, as scoped. Two things follow
for whoever runs the benchmark:

1. **IBM CCF and both subsamples should not carry headline results without the caveat**
   that their val/test labels are ~100% one categorical value, and that train's fraud
   population is a different one. The subsample caveat in the dataset cards understates
   this — it reports `Italy` at 55.7% of all frauds but not that it is 100% of val and
   test frauds. Worth strengthening.
2. **The `leaky_columns` question is now answerable.** A per-dataset tuple in the dataset
   card would be well defined and short: `ibm_ccf` → `Merchant State`, `Merchant City`,
   `Zip`, `MCC`, `Errors?`; `banksim` → `category`, `merchant`; `paysim` → `step`,
   `oldbalanceOrg`; `amaretto` → `amount` (round values), `Originator`. Recording them is
   cheap and leaves the drop decision with the experimenter. Not implemented — it is a
   change to the adapters, which this audit was not scoped to make.

## Delay calibration by domain (2026-08-02)

The censoring rate is not a tunable target. It is the quotient of the delay distribution
and the train window length, so equal delays produce very different percentages on
differently-shaped datasets. Under the single global 7-day median, Sparkov censored 2.2%
and Amaretto 17.5% — but both were censoring the same tail of their window:

| dataset | train window | censored | censoring band back from cutoff |
|---|---:|---:|---:|
| sparkov | 487d | 2.2% | 45d |
| amaretto | 65d | 17.5% | 64d (98% of the window) |

Forcing a uniform 5% across the suite would need roughly a 13-day median on Sparkov and a
1.5-day one on Amaretto — asserting that money laundering is reported 9× faster than card
fraud, which inverts reality. The delay is calibrated per **domain** instead, and whatever
censoring results is reported rather than tuned.

**Two regimes.**

- **Card fraud** — `banksim`, `sparkov`, `ieee_cis`, `ibm_ccf` and its subsamples. The
  cardholder notices on a statement or an alert and disputes inside the 60–120 day
  chargeback window. A 1–2 week median with a long right tail is the right shape; the
  7-day / sigma 1.0 default (p95 36d) stays.
- **AML** — `saml_d`, `amaretto`. An alert opens an investigation and the SAR clock starts
  only at detection, which itself lags the transaction by weeks. Realistic median 30–60
  days, i.e. 4–8× the card default.

Simulated censoring of train frauds at candidate medians, sigma 1.0, seeded:

| dataset | span | 7d | 15d | 30d | 60d |
|---|---:|---:|---:|---:|---:|
| sparkov | 730d | 2.3% | 4.5% | 9.0% | 18.7% |
| ieee_cis | 181d | 7.3% | 15.4% | 30.7% | 50.7% |
| banksim | 179d | 7.7% | 16.9% | 30.1% | 49.2% |
| saml_d | 320d | 5.6% | 10.8% | **19.2%** | 34.7% |
| amaretto | 83d | 18.2% | 33.6% | 53.4% | 74.0% |

### SAML-D moved to a 30-day median

Its 320-day span absorbs the AML regime comfortably. Measured after re-preparing:

| | before | after |
|---|---:|---:|
| known at train cutoff | 94.3% | **80.0%** |
| delay p50 | 6.9d | 29.6d |
| delay mean | — | 49.2d |
| delay p95 | — | 154.1d |

`early=0 null=0 stray=0`. The realised 20.0% censoring matches the 19.2% simulation.

Uncapped, the tail samples out to 779 days against a 320-day span, so a small number of
train frauds are never revealed within the data at all. That is not unrealistic — not every
suspicious transaction is ever filed — but unlike the IBM CCF subsamples there is no
`max_delay_days` here. Left as is; noted in case it matters later.

### Amaretto deliberately keeps the card-fraud default

Amaretto is AML too, but its whole span is 83 days and its train window 65. A realistic
30-day median censors 53.4% of train frauds and even 15 days censors 33.6% — degenerate
either way, the same trap PaySim hit from the other direction. There is no setting that is
both domain-realistic and non-degenerate, because the dataset is too short to carry its own
domain's delay. It keeps 7 days, which is **deliberately too fast for AML**, and the
mismatch is documented rather than repaired. Anyone reading an Amaretto delay result should
know its reporting lag is roughly a quarter of what the domain would produce.

### The whole delay layer is synthetic

None of the seven sources ships a real reporting timestamp. Every `reported_at` in this
repository is drawn from a distribution chosen on domain reasoning, not measured from
data. The domain medians above are estimates of how card disputes and SAR workflows
actually run — defensible, but not empirical. This bounds what any delay-aware result here
can claim: it can compare methods under a stated censoring regime, and cannot establish
what the real regime is.

## The IBM CCF subsamples are retired (2026-08-03)

`ibm_ccf_subsample_fast` and `ibm_ccf_subsample_slow` existed to give the label-delay axis
something to bite on: the full dataset's 9,629-day train window censors 3 of 24,924 train
frauds, so delay has no effect there. The subsamples cropped to 2016-01-01 and censored 3.0%
and 5.2% respectively.

**Measured, they do not work.** The leakage ablation scored them identically:

| variant | leaky, oracle | leaky, censored | clean, oracle | clean, censored | seed sd |
|---|---:|---:|---:|---:|---:|
| fast | 0.975 | 0.974 | 0.014 | 0.015 | 0.001 |
| slow | 0.975 | 0.974 | 0.014 | 0.014 | 0.001 |

The two regimes differ by 142 train frauds out of 6,441, and that difference moves no number
past the third decimal against a 0.001 seed noise floor. This confirms an earlier quick probe
that reached the same verdict with a cruder model.

They also inherited the geography artifact in its worst form — their val and test frauds are
100% `Merchant State == "Italy"` — so they were unusable for the leakage axis too.

**Replaced by `sparkov_slow`**, which carries the same design (row-identical apart from
`reported_at`) on a dataset where the effect is real: 8.9% censored against Sparkov's 2.2%,
verified identical on all 28 non-delay columns.

The delay axis is now Sparkov (2.2%, realistic card fraud), `sparkov_slow` (8.9%, an explicit
stress test) and SAML-D (20.0%, realistic AML). IBM CCF carries no delay condition.

### The early-cutoff split was not kept either

A split whose train half ends one second before the first Italy fraud (2009-09-12 to
2017-11-19, exactly 80/10/10) was built and measured — `scripts/italy_holdout.py`, test
average precision 0.0247 ± 0.0004. It is not registered as a dataset because **the standard
temporal split already has the property it was built to guarantee**: the 80% cut falls on
2017-05-14 and the first Italy fraud is 2017-11-19, so train contains zero Italy frauds
either way. The early-cutoff variant only makes the boundary exact, at the cost of a
non-standard construction and 6.2M fewer training rows.

Its value was diagnostic, and it is recorded here rather than shipped: it showed that the
model's dominant feature is `Use Chip` (37.8% of gain), and that the channel distribution of
fraud inverts across the regime boundary — train fraud is 85.5% online, val fraud 90.4%
chip-present with zero online. The model's most important feature points the wrong way, which
is why no amount of feature engineering bridges the two regimes.

## Correction: IBM CCF is fully labelled (2026-08-03)

Earlier notes in this file and in the audit plan said IBM CCF's final 645,180 rows were
"unlabelled, not fraud-free". **That is wrong, and the wording has been fixed.**

`Is Fraud?` has **zero nulls across all 24,386,900 rows**; the only values are 'No' and
'Yes'. Every row is labelled. What actually happens is that the *generator* stops emitting
fraud:

| | |
|---|---|
| data span | 1991-01-02 .. 2020-02-28 (29.2 years) |
| first fraud | 1996-07-05 |
| last fraud | 2019-10-27 |
| calendar years with >=1 fraud | 24 (1996-2019) |
| years with zero frauds | 1991-1995, 2020 |
| rows before the first fraud | 63,739 (2,011 days) |
| rows after the last fraud | 645,180 (124 days) |

The reason to crop the tail is that it holds no positives to detect, not that its labels are
missing. At the 0.122% base rate, 645,180 rows would be expected to carry roughly 787 frauds,
so observing zero is not chance — but "stopped generating" and "stopped labelling" are
different claims and only the first is supported by the data.

The same correction applies to the 11 fraud-free gaps of three months or more: those are
genuinely fraud-free stretches, not gaps in labelling. It matters for how the dataset is
described in a paper — a reader told "unlabelled" will assume missing values they could
filter on, and there are none.

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
