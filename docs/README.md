# Documentation

Start with the [repository README](../README.md) for what this project is and how to run
it. Everything else lives here.

## Setup

| doc | answers |
|---|---|
| [`kaggle-setup.md`](kaggle-setup.md) | Credentials — where `kagglehub` looks, which token is which, and how to accept the IEEE-CIS competition rules. Read before the first download. |
| [`dataset-licenses.md`](dataset-licenses.md) | What each dataset's licence permits, which are NonCommercial or ShareAlike, and what that means in practice. |

## The datasets

| doc | answers |
|---|---|
| [`datasets/`](datasets/README.md) | **One page per dataset**, indexed by how recoverable the labels are from a single column. Statistics, schema mapping, and the artifacts and disclaimers each one carries. Read the page for a dataset before using it. |
| [`datasets/amaretto-nuances.md`](datasets/amaretto-nuances.md) | Deep dive on the one dataset that needs it: five anomaly classes, client concentration, burst structure. |
| [`label-delay.md`](label-delay.md) | How `reported_at` is constructed, the shipped parameters and measured censoring for all eight datasets, which datasets the delay actually bites on, and how to change it. |

## The code

| doc | answers |
|---|---|
| [`architecture.md`](architecture.md) | Repository layout: the preparation stages and modules in `data/`, the per-dataset feature modules and the model stack in `experiments/`, what `scripts/` is for, where results go, and the conventions to respect before changing anything. |

## Findings

| doc | answers |
|---|---|
| [`verification-notes.md`](verification-notes.md) | The measurement log — everything checked against real data, dated. Measured statistics for all seven sources, discrepancies against upstream documentation, the **leakage audit** per dataset, campaign and delay calibration, and what was retired and why. The longest document here and the primary source for most claims elsewhere. |
| [`experiments.md`](experiments.md) | The IBM CCF follow-up experiments: every result with its seed spread, what closed the 0.041-vs-0.764 gap and what did not, and the traps in comparing across splits. |
| [`../results/summary.md`](../results/summary.md) | The leakage ablation table. The grid that produced it has since been retired; the numbers stand as a record. |

## Not in version control

`docs/superpowers/` holds the working specs and implementation plans used while building
this. They are kept local and gitignored; some notes here still refer to them by name.

`docs/dataset-characteristics-table.tex` is a LaTeX table of the dataset statistics, kept
for reuse in a write-up.
