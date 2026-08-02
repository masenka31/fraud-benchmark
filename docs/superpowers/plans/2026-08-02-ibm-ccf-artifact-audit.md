# IBM CCF Artifact Audit — Note

**Status:** done, 2026-08-02. Results in the `## Known leakage` section of
`docs/verification-notes.md`. The one open follow-on is whether adapters should record a
`leaky_columns` tuple in the dataset card — see that section's closing note.

**Goal:** find out how many of IBM CCF's fraud labels are explained by generation
artifacts rather than by anything a detector should learn, and record what we find.

**Why now:** three artifacts have already turned up incidentally, each while looking for
something else. That rate of accidental discovery says the dataset has not been audited
and probably holds more. The subsample variants make this urgent: they concentrate the
window where the worst artifact lives.

---

## Already confirmed

Measured on `data/processed/ibm_ccf/data.parquet`, 24,386,900 rows, 29,757 frauds.

**1. Labelling stops on 2019-10-27** while transactions run to 2020-02-28. The final
645,180 rows carry zero frauds. Handled: the subsamples crop there. Already in
`docs/verification-notes.md`.

**2. 2017 is a near-empty fraud year** — 255 frauds against 3,579 in 2016 and 2,491 in
2018, a 14× dip. Not handled; it sits inside the subsample train split.

**3. `Merchant State` is close to an oracle.** In the 2016-01-01 window:

| value | rows | frauds | fraud rate |
|---|---:|---:|---:|
| Italy | 6,099 | 4,682 | **76.8%** |
| Algeria (full data) | 654 | 629 | **96.2%** |

`Italy` alone accounts for **55.7% of all frauds in the window**. A single categorical
value. Any model given this column raw will score brilliantly and have learned nothing.
Recorded in the subsample adapter's caveats; **not** otherwise handled.

---

## To investigate

- [x] **Sweep every categorical column for near-oracle values.** For each of
      `Merchant State`, `Merchant City`, `Merchant Name`, `MCC`, `Zip`, `Use Chip`,
      `Card Brand`, `Card Type`, `Errors?`: per-value fraud rate against the base rate,
      flagging any value with a rate above ~10× base and a non-trivial row count. This
      one query probably finds most of what is left.
- [x] **`Errors?` specifically.** "Bad PIN", "Insufficient Balance" and friends may be
      consequences of fraud rather than predictors of it — leakage in the causal sense
      even when the rate looks plausible.
- [x] **Foreign-merchant encoding.** `Merchant State` mixes US state codes with country
      names. Check whether *every* non-US value is fraud-enriched, i.e. whether the real
      artifact is "foreign" rather than "Italy".
- [x] **Entity concentration.** How many users/cards carry the frauds, and is `entity_id`
      as close to a label as it is on Amaretto (where 21 of 400 clients hold everything)?
- [x] **Amount.** Whether fraudulent amounts occupy a distinguishable range or come from
      a separate generator.
- [x] **Online vs chip.** Measured: 0.392% fraud for `Online Transaction` against 0.098%
      for chip. Only 4× and directionally realistic, so probably signal rather than
      artifact — confirm and move on.
- [x] **Does the same sweep find anything on the other six datasets?** Run it across all
      of them. Sparkov and BankSim are also generated and deserve the same scepticism.

## Deliverable

A `## Known leakage` section in `docs/verification-notes.md`, per dataset, listing any
column-value pair that is close to a label, with its rate and coverage.

If the sweep turns up much, the follow-on question is whether adapters should mark such
columns rather than merely document them — a `leaky_columns` tuple recorded in the
dataset card, leaving the decision to drop them with whoever runs the benchmark. Decide
that after seeing the results, not before.

## Not in scope

Fixing anything. This audit measures and records. IBM CCF is a synthetic dataset with
synthetic artifacts; the useful output is an honest description of what its labels
actually encode, not a repaired version of it.
