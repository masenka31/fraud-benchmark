# Amaretto: dataset nuances

Everything below was measured directly from the full 29,704,090-row release on 2026-08-01
using this repository's processed output, unless explicitly attributed to the upstream
README or paper. Where the upstream documentation and the data disagree, both are given.

Source: <https://github.com/necst/amaretto_dataset> · Licence: **MIT** · Paper:
Labanca, Primerano, Markland-Montgomery, Polino, Carminati, Zanero, *Amaretto: An Active
Learning Framework for Money Laundering Detection*, IEEE Access, 2022,
doi:10.1109/ACCESS.2022.3167699.

---

## 1. Summary — what is safe to claim

| Claim | Status |
|---|---|
| 29.7M transactions, 400 clients, ~12 weeks | **Confirmed.** 29,704,090 rows, 400 clients, 83 calendar days (11.9 weeks) over 60 trading days. |
| Per-client sequences are dense | **Confirmed, strongly.** Median 44,442 transactions per client; ~741 per client per trading day. |
| Released for active learning, not sequential modelling | **Confirmed** by the source paper's framing. |
| Anomalies are "deviations from a client's own history" | **Needs qualifying — see §5.** For 75% of anomalies the amount is statistically indistinguishable from the client's own distribution; for the other 25% the signature is an *absolute* constant, not a client-relative deviation. |
| A plausible, underused candidate for sequential modelling | **Supported, and for a stronger reason than usually given** — see §9. |

---

## 2. Access and packaging

Amaretto is the only dataset in this benchmark not distributed via Kaggle, and the only one
requiring archive reassembly:

- Distributed as a git repository, not a data portal. No credentials or licence acceptance.
- The data sits in `Data/` as **34 split-zip fragments** (`amaretto_dataset_anon.zip.001` …
  `.034`), 1.2 GB total, because of file-size limits. Plain byte concatenation in numeric
  order reproduces a valid archive containing a single member,
  `amaretto_dataset_anon.csv`, of 3,577,592,424 bytes.
- The fragments are zero-padded, so lexicographic and numeric ordering coincide. Tooling
  that assumes unpadded suffixes would silently misorder them; this repository sorts
  numerically and verifies the fragments form a complete `1..N` run.
- **MIT licence**, which makes it the most permissively licensed dataset in the suite —
  notable given that BankSim, SAML-D and IEEE-CIS all carry non-commercial or
  competition-rules restrictions.

## 3. Scale and shape

| Property | Value |
|---|---|
| Transactions | 29,704,090 |
| Clients (`Originator`) | 400 |
| Date range | 2019-01-01 00:00:03 → 2019-03-25 23:59:49 |
| Calendar span | 83 days (11.9 weeks) |
| Distinct trading days | 60 (weekends excluded) |
| Transactions per client | min 29,258 · median 44,442 · max 390,050 |
| Density | ~741 transactions per client per trading day |
| Anomalies | 81,268 (0.2736%) |

Columns: `Transaction ID, Originator, Originator_ID, EntryDate, InputOutput, Market,
Product ISIN, Product Type, Product Class, Normalized Amount, Currency, Anomaly` — twelve
fields, which is narrow next to IEEE-CIS's 394 or IBM CCF's 15-plus-joins.

Field notes:

- **`Originator_ID` is the constant string `_XID` in all 29.7M rows.** It carries no
  information and should be dropped by any consumer.
- **`Product ISIN` is near-unique** (1,999,972 distinct values in a 2M-row sample), so it
  behaves as an identifier rather than a category. Treating it as a categorical feature
  will produce a per-row indicator.
- `Currency` has only 2 values, `Market` 4, `Product Type` 17, `Product Class` 4,
  `InputOutput` 2 (`Buy`/`Sell`).
- Amounts are already described upstream as *normalised*; no currency conversion is
  meaningful or attempted.

**Normal activity is uniformly distributed across trading hours.** Each hour from 07:00 to
19:00 carries 7.18–7.20% of normal transactions — a flatness no real market exhibits, and a
reminder that this is a simulation built from aggregate parameters rather than a resampled
real tape.

## 4. Label structure: `Anomaly` is not binary

`Anomaly` takes **six values**: `0` for normal plus classes `1`–`5` corresponding to the
five FATF-derived typologies described upstream. Any consumer treating it as a boolean
silently discards the typology.

| class | count | share of anomalies |
|---|---:|---:|
| 1 | 5,884 | 7.2% |
| 2 | 2,576 | 3.2% |
| 3 | 12,088 | 14.9% |
| 4 | 30,640 | 37.7% |
| 5 | 30,080 | 37.0% |
| **total** | **81,268** | |

**The upstream README undercounts its own anomalies.** It states 81,262; the file contains
81,268, a difference of 6. The README's "0.27%" is a rounding of the measured 0.2736%.

## 5. What each class actually looks like — and the qualification this forces

Measured signatures, against a normal-row baseline:

| class | median amount | %Sell | % exact multiple of 1000 | % outside 07–19h | modal hour |
|---|---:|---:|---:|---:|---:|
| normal | 32,746 | 48.0% | 0.0008% | 28.1% | uniform |
| 1 | 798 | 47.4% | 0% | 29.4% | uniform |
| 2 | 11,000 | **100%** | **100%** | 29.3% | uniform |
| 3 | 49,306 | **100%** | 0% | **100%** | **02:00** |
| 4 | 32,301 | **100%** | 0% | 28.8% | uniform |
| 5 | 32,635 | 48.2% | 0% | 25.7% | uniform |

Now the decisive test. For every anomaly, where does its amount sit within **its own
client's** distribution of normal amounts? If anomalies were client-relative deviations,
they should cluster in the extreme tails.

| class | p05 | p25 | **p50** | p75 | p95 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| 1 | 18.0 | 20.3 | **20.5** | 20.6 | 20.8 | pinned to a fixed low band |
| 2 | 23.8 | 24.7 | **27.0** | 27.9 | 29.5 | pinned to a fixed low band |
| 3 | 52.2 | 61.2 | **67.0** | 74.7 | 86.1 | mid-to-upper, not extreme |
| 4 | 5.0 | 24.2 | **49.6** | 74.6 | 94.9 | **uniform — indistinguishable** |
| 5 | 4.8 | 24.5 | **49.9** | 75.1 | 95.0 | **uniform — indistinguishable** |

(50 = the client's own median; 0 or 100 = an extreme for that client.)

**This is the finding that matters for a paper.** Classes 4 and 5 — **60,720 anomalies,
75% of the total** — have amounts spread perfectly uniformly across their own client's
distribution. In the amount dimension they are not deviations from that client's history at
all; they are drawn from it.

Conversely, classes 1, 2 and 3 are defined by **absolute constants**, not client-relative
ones:

- **Class 2 is trivially separable by a single global rule.** `amount mod 1000 == 0`
  achieves **100% recall** on class 2 with just **241 false positives across 29.6M normal
  rows** — 91.4% precision from one deterministic predicate, no model and no client history.
- **Class 1** occupies a fixed absolute band, 420.65 to 1,177.44, regardless of client.
- **Class 3 fires only between 02:00 and 04:00**, where normal activity is 0.12% per hour.
  The hour alone is not sufficient (100% recall but 5.4% precision, since 213,643 normal
  rows also fall in 00–05h), but it is an extremely strong absolute feature.

So the accurate statement is: **Amaretto's injected anomalies are behaviourally rather than
statistically defined, and the definition is absolute for three classes and
sequence-dependent for two — not client-relative in the amount dimension for any of them.**

Reassuringly, the dataset is *not* trivially solvable overall: a combined hand-written rule
(`round-1k OR (hour < 06 AND amount > 40k)`) catches only **16.1%** of anomalies at 12.6%
precision. The remaining 75% genuinely require modelling.

## 6. Client concentration, and a leakage hazard

**All 81,268 anomalies belong to just 21 of the 400 clients.** The other 379 are entirely
clean for the whole period.

Within those 21, the share of a client's *own* transactions that are anomalous varies
enormously:

| client | transactions | anomalies | own rate |
|---|---:|---:|---:|
| Client_066 | 66,525 | 36,673 | 55.13% |
| Client_212 | 48,378 | 18,567 | 38.38% |
| Client_126 | 42,338 | 12,184 | 28.78% |
| Client_335 | 46,280 | 1,248 | 2.70% |
| Client_378 | 46,322 | 1,236 | 2.67% |
| … | | | median **2.13%** |

Two consequences:

1. **`Originator` is close to a label.** A model given raw client identity can learn "these
   21 clients" and score well without learning anything about transactions. Any benchmark
   using Amaretto should either exclude the client identifier from the feature set or
   partition by client, and should say which. This repository stores it as `entity_id`
   because the canonical schema requires an entity, **not** as an endorsement of using it
   as a feature.
2. **Anomalous clients are not identifiable by volume.** Median transaction count is 45,456
   for anomalous clients versus 44,340 for clean ones — essentially identical. So the
   leakage is via identity, not via an aggregate that a model could legitimately learn.

Note also the extreme imbalance *within* the anomalous group: Client_066 alone accounts for
45% of all anomalies in the dataset. Per-client evaluation will be dominated by three
clients unless explicitly reweighted.

## 7. Temporal structure: anomalies are violently bursty

A first-to-last span statistic is misleading here. Each anomalous client's anomalies span
76–83 days — essentially the whole period — which suggests a diffuse, always-on pattern. The
inter-arrival distribution says the opposite:

| percentile | gap between consecutive anomalies on the same client |
|---|---|
| p50 | **0.7 minutes** |
| p75 | 1.8 minutes |
| p90 | 5.3 minutes |
| p95 | 15.3 minutes |
| p99 | 7.5 hours |
| max | 10.4 days |

60.7% of consecutive anomaly pairs occur within **one minute**; 97.7% within one hour. The
anomalies are tight bursts scattered across the full period, not a continuous drift.

Burstiness differs by class by two orders of magnitude:

| class | median consecutive gap | p95 |
|---|---:|---:|
| 4 | 0.6 min | 3.1 min |
| 5 | 0.6 min | 3.1 min |
| 3 | 2.2 min | 28.4 min |
| 1 | 8.8 min | 30.4 h |
| 2 | 28.3 min | 96.3 h |

This directly determines any episode- or campaign-level grouping. Grouping anomalies into
episodes by "same client, consecutive, within gap *g*":

| gap | episodes | median size | max size |
|---|---:|---:|---:|
| 1 min | 31,940 | 1 | 150 |
| 10 min | 5,423 | 2 | 533 |
| **1 hour** | **1,852** | **8** | **1,145** |
| 6 hours | 918 | 6 | 3,870 |
| 1 day | 490 | 15 | 3,870 |
| 7 days | 27 | 657 | **36,673** |

At a one-week gap the dataset collapses to 27 episodes and Client_066's entire anomalous
history becomes a single event. **One hour is the defensible operating point**, and it is
what this repository uses for Amaretto while every other dataset uses one day. No single
threshold serves all five classes: one hour preserves classes 3, 4 and 5 intact while
fragmenting the slower classes 1 and 2.

## 8. Split behaviour

Under this repository's global temporal 80/10/10 split (cut on timestamp values, so no
timestamp straddles a boundary):

| split | rows | anomalies | rate |
|---|---:|---:|---:|
| train | 23,763,282 | 65,580 | 0.2760% |
| val | 2,970,404 | 8,713 | 0.2933% |
| test | 2,970,404 | 6,975 | 0.2348% |

All five classes appear in all three splits, so no typology is stranded:

| split | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|
| train | 4,716 | 2,032 | 9,573 | 25,306 | 23,953 |
| val | 558 | 269 | 1,340 | 3,026 | 3,520 |
| test | 610 | 275 | 1,175 | 2,308 | 2,607 |

The mild decline in test-split anomaly rate (0.235% vs 0.276% train) is worth reporting but
is not a stratification failure.

## 9. Implications for sequential modelling

The case for Amaretto as a sequential-modelling benchmark is **stronger than the usual
framing**, but rests on a different mechanism than "deviation from client history":

- The per-client sequences are genuinely dense — ~741 transactions per client per trading
  day, three orders of magnitude denser than SAML-D's ~12 transactions per sender account.
  There is real history to condition on.
- **75% of anomalies (classes 4 and 5) are invisible to any point-wise amount model**, since
  their amounts are drawn uniformly from the client's own distribution. Their only signals
  are transaction *direction* (class 4 is 100% Sell against a 48% baseline) and *temporal
  clustering* (0.6-minute median inter-arrival). These are precisely the signals a
  sequential model can exploit and a tabular row-wise model cannot.
- That makes Amaretto an unusually clean discriminator between point-wise and sequential
  approaches: a per-row classifier should saturate near the ~25% of anomalies with absolute
  signatures, and any gain beyond that must come from sequence structure.

Two caveats to state alongside any result:

- **Exclude or control for client identity** (§6), otherwise the 21-client concentration
  dominates.
- **Report class-wise performance.** Aggregate metrics are dominated by classes 4 and 5
  (75%) and can hide the fact that class 2 is solvable by one arithmetic predicate.

## 10. Suggested wording

The current paragraph reads:

> The underlying per-client sequences are dense and the injected anomalies are defined as
> deviations from a client's own history, making the dataset itself a plausible, underused
> candidate for sequential modeling despite the original usecase.

A version consistent with the measurements:

> The underlying per-client sequences are dense — a median of roughly 44,000 transactions
> per client across 60 trading days — and the injected anomalies are behavioural rather than
> amount-based: for the two largest typologies, comprising 75% of anomalies, transaction
> amounts are drawn uniformly from the client's own distribution, so the only discriminative
> signals are order direction and sub-minute temporal clustering. This makes the dataset a
> plausible, underused candidate for sequential modeling despite its original active-learning
> framing, with the caveats that anomalies are confined to 21 of the 400 clients — making the
> client identifier close to a label — and that one typology is separable by a single
> arithmetic predicate on the transaction amount.

## 11. Reproducing these figures

Every number above comes from `data/processed/amaretto/data.parquet`, produced by:

```bash
fraud-benchmark prepare amaretto
```

Preparation reassembles the 34 fragments, extracts the 3.58 GB CSV (cached under
`data/raw/amaretto/_extracted/` so repeat runs skip it), and writes the canonical frame with
`is_fraud = Anomaly > 0` and the `Anomaly` class retained. See `docs/verification-notes.md`
for the equivalent figures across all seven datasets.
