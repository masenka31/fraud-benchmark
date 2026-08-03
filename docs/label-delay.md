# Label delay, as shipped

**What it is.** No public fraud dataset ships a real "when was this reported"
timestamp, so the pipeline synthesises one. Fraudulent transactions are grouped
into *campaigns* (one entity, each fraud within `campaign_gap` of the last); one
delay is drawn per campaign from a lognormal in days, measured from the campaign's
**last** transaction; every fraud in the campaign gets that same `reported_at`. A
campaign is therefore never reported before it finishes, and non-fraud rows never
carry a `reported_at` at all.

**Why it matters.** A model training at the end of its train window does not know
about frauds that have not been reported yet. Those rows are not missing — they sit
in the training data *looking legitimate*. `reported_at` is what lets an experiment
model that honestly. The column to compare against is the train cutoff, not the
row's own timestamp.

**Everything here is synthetic by construction.** The distribution is calibrated to
be plausible per domain, not fitted to observed reporting behaviour, because no
observed reporting behaviour ships with these datasets. The per-domain reasoning is
in the comments in `configs/default.yaml`, next to the parameters themselves.

## Per dataset, at the shipped config

Measured from `data/processed/*/` on 2026-08-03 — the current
`configs/default.yaml`, not the historical global default. "Censored" is the share
of **train** frauds whose `reported_at` falls after the train cutoff: the labels a
model training at that moment would get wrong.

| dataset | domain | median | sigma | cap | train window | **censored** | campaigns | observed p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `saml_d` | AML | 30d | 1.0 | — | 255d | **20.0%** | 7,887 | 29.6d |
| `amaretto` | AML ⚠ | 7d | 1.0 | — | 65d | **17.5%** | 1,852 | 6.9d |
| `paysim` | card ⚠ | 1d | 1.0 | — | 14d | **10.1%** | 8,213 | 1.0d |
| `sparkov_slow` | card, stress | 15d | 1.665 | 365d | 487d | **8.9%** | 1,005 | 13.9d |
| `banksim` | card | 7d | 1.0 | — | 147d | **8.1%** | 5,551 | 7.5d |
| `ieee_cis` | card | 7d | 1.0 | — | 140d | **7.1%** | 16,170 | 7.0d |
| `sparkov` | card | 7d | 1.0 | — | 487d | **2.2%** | 1,005 | 7.3d |
| `ibm_ccf` | card | 7d | 1.0 | — | 9,629d | **0.0%** | 9,769 | 7.3d |

Campaign gap is `1d` everywhere except `amaretto`, which uses `1h` — its anomalies
have a 0.7-minute median inter-arrival, three orders of magnitude burstier than card
fraud, and a 1-day gap would fuse them into a few huge episodes.

To regenerate this table after a config change:

```bash
.venv/bin/python - <<'PY'
import json, pandas as pd
from pathlib import Path
for d in sorted(Path("data/processed").iterdir()):
    ld = json.loads((d / "dataset_card.json").read_text())["label_delay"]
    df = pd.read_parquet(d / "data.parquet",
                         columns=["event_time", "is_fraud", "split", "reported_at"])
    tr = df[df["split"] == "train"]
    frauds = tr[tr["is_fraud"]]
    censored = 100 * (1 - (frauds["reported_at"] <= tr["event_time"].max()).mean())
    print(f"{d.name:14s} median={ld['median_days']:>5} sigma={ld['sigma']:<6} "
          f"censored={censored:5.1f}%  campaigns={ld['n_campaigns']:>6}")
PY
```

## Which datasets the delay actually bites on

**Use for delay-aware work:** `saml_d` (20.0%), `amaretto` (17.5%), `paysim`
(10.1%), `sparkov_slow` (8.9%), `banksim` (8.1%), `ieee_cis` (7.1%).

**Do not:** `ibm_ccf` is effectively undelayed — 3 unknown labels out of 24,924,
because a 7-day delay against a 9,629-day train window is nothing. That is realistic
rather than broken, but it means IBM CCF cannot exercise a delay-aware method.
`sparkov` at 2.2% is nearly as weak, which is exactly why `sparkov_slow` exists.

**The one paired contrast:** `sparkov` and `sparkov_slow` are **row-identical on
every column but `reported_at`** and share one raw download, so a method can be run
twice on the same data under 2.2% and 8.9% censoring and the difference is
attributable to the delay alone. Nothing else in the suite offers that. Two earlier
attempts at such a pair, `ibm_ccf_subsample_fast` / `_slow`, were retired when their
regimes measured identically (0.975 against 0.975 at a 0.001 seed noise floor).

Note that the harsher regime comes mostly from **sigma, not the median**: at median
15d with sigma 1.0 Sparkov censors ~4.5%; the shipped 1.665 sigma nearly doubles that
to 8.9% by fattening the tail. Raising sigma is the lever that censors more without
claiming an implausible typical reporting time.

## Two datasets whose clocks cannot be made realistic

Both are documented rather than repaired, because the limit is the span of the data.

- **`amaretto` is AML but keeps the 7-day card default.** A realistic 30-day AML
  median censors 53.4% of its train frauds and 15 days censors 33.6% — degenerate
  either way against an 83-day span. The shipped 7 days is deliberately too fast for
  money laundering and is the only setting the span will carry.
- **`paysim` spans 30 simulated days**, so its train window is 14. At the global
  7-day median it censored 47.3% of train labels; the 1-day override brings that to
  10.1%. This makes PaySim usable; it does not make a 30-day clock realistic.

## Changing it

Global block plus per-dataset overrides in `configs/default.yaml`. An override names
only the keys it changes; the rest inherit, so moving the global `seed` still moves
every dataset:

```yaml
delay:
  median_days: 7.0
  sigma: 1.0
  seed: 0
  max_delay_days: null

datasets:
  saml_d:
    delay:
      median_days: 30.0     # sigma and seed inherit
```

An unknown key inside a `delay:` block — or inside a `datasets.<name>` block — is a
`ConfigError` at load time, not a silent fallback to the default. Every override is
resolved when the config loads, so a bad one cannot let `prepare --all` die halfway
through after writing other datasets.

The realised delay is recorded per dataset in `dataset_card.json` under
`label_delay`, including `observed_mean_delay_days` — truncation pulls the realised
mean below nominal, so the card reports what happened rather than only what was
configured.
