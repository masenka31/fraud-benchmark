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
| [`label-delay.md`](label-delay.md) | How `reported_at` is constructed, the shipped parameters and measured censoring for all eight datasets, which datasets the delay actually bites on, and how to change it. |

## The code

| doc | answers |
|---|---|
| [`architecture.md`](architecture.md) | Repository layout: dataset preparation, per-dataset feature modules, the two paper protocols, public scripts and results, and the conventions to respect before changing anything. |

## Findings

| doc | answers |
|---|---|
| [`experiments.md`](experiments.md) | Exact protocols for the paper's pre-Italy IBM split comparison and synthetic Sparkov label-delay comparison. |
| [`../results/paper.md`](../results/paper.md) | Index of the two paper experiments, their runners, readable tables, and machine-readable records. |
