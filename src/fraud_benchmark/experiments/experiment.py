"""One experiment: a dataset, a model, and four axes, fitted and scored.

`scripts/run_experiment.py` is the CLI over this; everything decided here is
deliberate and recorded in the result, so a JSONL line is enough to say what was
run.

## The four axes

    dataset       ibm_ccf | saml_d | sparkov          which feature parquet
    model         xgboost | mlp | logistic            estimators/
    history       0, or n lags concatenated on        experiments.history
    label delay   off | on | slow                     data.censoring

plus two that exist because the parquets are built to allow them:

    artifacts     drop | keep                         the artifact_ column group
    split         standard | italy_holdout            experiments.splits

None of these is a property of the parquet, which is why the parquet is built once
and read by every combination.

**Every run uses every row.** There is no subsampling and no row limit: a model fitted
on part of a dataset is not comparable to one fitted on all of it, so a gap between two
runs would measure how much data each saw rather than what the axis changed. An earlier
version cropped the MLP to 6M rows to save memory it turned out not to need, and the
baseline group -- whose entire purpose is comparing models on one dataset -- silently
became a comparison of training-set sizes.

The one exclusion that remains is not a crop: `_drop_unlabelled_tail` removes rows a
dataset never labelled, which cannot be scored at all. See its docstring.

## What "label delay" does

`off` trains on `is_fraud`. `on` trains on the labels a detector would have had at
the train cutoff: `data.censoring.censored_labels` at the last train timestamp, so
a fraud reported after it stays in the training data *labelled 0* rather than being
dropped -- at that moment it is indistinguishable from a legitimate transaction,
and dropping it would presume knowing which rows to distrust. `slow` is the same
against `reported_at_slow`, which only Sparkov carries.

Validation and test always use true labels. They measure what happened, not what
was known; censoring them would make the score a measure of the delay rather than
of the model. The consequence to keep in mind is that a censored run has *fewer
positive training examples than its own test set implies*, and that is the effect
being measured.

## What is fitted, and on what

Only train rows: the categorical vocabularies (`CappedOrdinalEncoder`) and, for the
MLP, the standardisation statistics. The parquets are deliberately unfitted so this
is the only place it happens, and it happens after the split is chosen -- which is
why `--split` can be changed without rebuilding anything.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_benchmark.data.censoring import censored_labels
from fraud_benchmark.experiments.encoding import CappedOrdinalEncoder
from fraud_benchmark.experiments.features import ibm_ccf, saml_d, sparkov
from fraud_benchmark.experiments.features.util import (
    ARTIFACT_PREFIX,
    FEATURE_DIR,
    KEY_COLUMNS,
    OPTIONAL_KEY_COLUMNS,
    feature_columns,
)
from fraud_benchmark.experiments.history import lag_columns, lag_matrix
from fraud_benchmark.experiments.splits import italy_holdout_split, standard_split

#: The dataset modules, for their HISTORY_COLUMNS. Keyed by the name on the CLI.
DATASET_MODULES = {
    ibm_ccf.DATASET: ibm_ccf,
    saml_d.DATASET: saml_d,
    sparkov.DATASET: sparkov,
}
DATASETS = tuple(DATASET_MODULES)

LABEL_DELAYS = ("off", "on", "slow")
ARTIFACT_MODES = ("drop", "keep")
SPLITS = ("standard", "italy_holdout")

#: `slow` needs a second report timestamp, and only Sparkov has one.
SLOW_DELAY_COLUMN = "reported_at_slow"

#: The Italy holdout's boundaries are IBM CCF timestamps, so it means nothing
#: elsewhere. See experiments/splits.py.
ITALY_HOLDOUT_DATASET = "ibm_ccf"

DEFAULT_RESULTS = Path("results/experiments.jsonl")


class ExperimentError(ValueError):
    """Raised when a requested combination cannot be run as asked."""


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model: str
    history: int = 0
    label_delay: str = "off"
    artifacts: str = "drop"
    split: str = "standard"
    seeds: tuple[int, ...] = (0, 1, 2)
    history_columns: str = "default"

    def validate(self) -> None:
        """Reject a combination that cannot mean what it says."""
        if self.dataset not in DATASETS:
            raise ExperimentError(
                f"unknown dataset {self.dataset!r}; known: {', '.join(DATASETS)}"
            )
        if self.label_delay not in LABEL_DELAYS:
            raise ExperimentError(
                f"label_delay must be one of {LABEL_DELAYS}, got {self.label_delay!r}"
            )
        if self.artifacts not in ARTIFACT_MODES:
            raise ExperimentError(
                f"artifacts must be one of {ARTIFACT_MODES}, got {self.artifacts!r}"
            )
        if self.split not in SPLITS:
            raise ExperimentError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.history < 0:
            raise ExperimentError(f"history must not be negative, got {self.history}")
        if self.label_delay == "slow" and self.dataset != sparkov.DATASET:
            raise ExperimentError(
                f"label_delay 'slow' needs {SLOW_DELAY_COLUMN}, which only "
                f"{sparkov.DATASET} carries; {self.dataset} has one delay regime"
            )
        if self.split == "italy_holdout" and self.dataset != ITALY_HOLDOUT_DATASET:
            raise ExperimentError(
                "split 'italy_holdout' is defined by IBM CCF's own timestamps and "
                f"means nothing on {self.dataset}"
            )


@dataclass
class Prepared:
    """Everything an estimator needs, and nothing about how to fit it."""

    x: np.ndarray                 #: float32, (rows, features). Codes for categoricals.
    y: np.ndarray                 #: int, train rows possibly censored; val/test true.
    y_true: np.ndarray            #: int, always the true label. For reporting only.
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    continuous: np.ndarray        #: column indices of the continuous features
    categorical: np.ndarray       #: column indices of the coded categoricals
    cardinalities: list[int]      #: distinct codes per categorical, in that order
    names: list[str]              #: column names, in matrix order
    censored_train_labels: int    #: train frauds hidden by the delay regime
    rule_scores: np.ndarray | None = None  #: the trivial rule, where one exists
    notes: list[str] = field(default_factory=list)

    def split_rows(self, name: str) -> np.ndarray:
        return {"train": self.train_rows, "val": self.val_rows, "test": self.test_rows}[
            name
        ]


def _apply_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Place the split boundaries, returning the frame in its new row order."""
    if split == "standard":
        return standard_split(df)
    return italy_holdout_split(df)


def _resolve_history_columns(
    dataset: str, mode: str, available: list[str]
) -> list[str]:
    """Which columns get lagged copies."""
    if mode == "all":
        return list(available)
    declared = list(DATASET_MODULES[dataset].HISTORY_COLUMNS)
    missing = [c for c in declared if c not in available]
    if missing:
        # Reachable by dropping the artifact group, or by editing HISTORY_COLUMNS
        # without rebuilding. Either way, silently lagging fewer columns than the
        # module declares would make two runs incomparable under one name.
        raise ExperimentError(
            f"{dataset}: HISTORY_COLUMNS names {missing}, absent from the feature "
            "set being used; adjust the module or the --artifacts setting"
        )
    return declared


#: Datasets whose final stretch carries no labels at all, and so cannot be scored.
#: IBM CCF stops generating fraud on 2019-10-27 but keeps producing transactions until
#: 2020-02-28: 645,180 rows, 2.6% of the dataset, four months, zero frauds. They are
#: not negatives that happen to be clean -- they are rows the generator never labelled.
UNLABELLED_TAIL_DATASETS = frozenset({"ibm_ccf"})


def _drop_unlabelled_tail(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str]:
    """Cut everything after a dataset's last labelled fraud. Returns (frame, note).

    `experiments.splits.italy_holdout_split` already did this for its own boundary and
    `LAST_LABELLED_FRAUD` names the timestamp; `standard_split` did not, so the two
    disagreed about whether those rows exist. Keeping them dilutes a full-data test
    split with 645,180 unscorable rows, and breaks a cropped one outright: the last 10%
    of the last 6M rows lies entirely inside the tail, so test held zero frauds and
    average precision was undefined. That is how this was found -- the three IBM CCF
    MLP cells failed on it.

    Applied before the split rather than inside it, because it is a statement about
    which rows the dataset labelled and not about where a boundary goes.
    """
    if dataset not in UNLABELLED_TAIL_DATASETS or "is_fraud" not in df.columns:
        return df, ""
    frauds = df.loc[df["is_fraud"].astype(bool), "event_time"]
    if frauds.empty:
        return df, ""
    last_labelled = frauds.max()
    keep = df["event_time"] <= last_labelled
    dropped = int((~keep).sum())
    if not dropped:
        return df, ""
    return (
        df.loc[keep],
        f"dropped {dropped:,} row(s) after the last labelled fraud "
        f"({last_labelled:%Y-%m-%d}); they carry no labels and cannot be scored",
    )


def prepare(config: ExperimentConfig, features_dir: Path | str = FEATURE_DIR) -> Prepared:
    """Read the parquet and build the design matrix, labels and split indices."""
    config.validate()
    notes: list[str] = []

    source = Path(features_dir) / f"{config.dataset}.parquet"
    if not source.exists():
        raise ExperimentError(
            f"{source} does not exist; build it with "
            f"`python -m fraud_benchmark.experiments.features.{config.dataset}`"
        )
    df = pd.read_parquet(source)
    df, tail_note = _drop_unlabelled_tail(df, config.dataset)
    if tail_note:
        notes.append(tail_note)

    df = _apply_split(df, config.split).reset_index(drop=True)

    # The trivial rule is read before any column is dropped, so it is available as a
    # reference regardless of the artifact setting.
    rule_scores = _trivial_rule_scores(df, config.dataset)

    columns = feature_columns(df)
    if config.artifacts == "drop":
        dropped = [c for c in columns if c.startswith(ARTIFACT_PREFIX)]
        columns = [c for c in columns if c not in dropped]
        notes.append(f"dropped {len(dropped)} artifact_ column(s)")

    history_columns = _resolve_history_columns(
        config.dataset, config.history_columns, columns
    )

    categorical_names = [
        c for c in columns if isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    train_mask = (df["split"] == "train").to_numpy()
    encoder = CappedOrdinalEncoder().fit(df.loc[train_mask], categorical_names)

    # One block per column, in `columns` order, so `names` and the matrix agree.
    base = np.empty((len(df), len(columns)), dtype="float32")
    for index, column in enumerate(columns):
        if column in categorical_names:
            base[:, index] = encoder.transform_column(df[column], column)
        else:
            base[:, index] = pd.to_numeric(df[column], errors="coerce").to_numpy(
                dtype="float32"
            )

    if config.history == 0:
        x, names = base, list(columns)
    else:
        # The target row keeps the *full* feature set; only the lags are narrow. So
        # lag the subset, then discard its target block, which `base` already holds.
        narrow = base[:, [columns.index(c) for c in history_columns]]
        lagged = lag_matrix(narrow, df["entity_id"], df["event_time"], config.history)
        width = len(history_columns)
        x = np.hstack([base, lagged[:, width:]])
        names = list(columns) + lag_columns(history_columns, config.history)[width:]
        notes.append(
            f"history {config.history} lag(s) of {width} column(s) "
            f"-> {len(names)} features"
        )

    categorical_indices = [i for i, name in enumerate(names)
                           if name.split("_lag")[0] in categorical_names]
    continuous_indices = [i for i in range(len(names)) if i not in set(categorical_indices)]
    cardinalities = [
        encoder.cardinality(names[i].split("_lag")[0]) for i in categorical_indices
    ]

    y_true = df["is_fraud"].to_numpy().astype(int)
    y, censored = _labels(df, config, train_mask, y_true)
    if censored:
        notes.append(f"{censored:,} train fraud label(s) unknown at the cutoff")

    rows = {
        name: np.flatnonzero((df["split"] == name).to_numpy())
        for name in ("train", "val", "test")
    }
    for name, index in rows.items():
        if len(index) == 0:
            raise ExperimentError(f"the {name} split is empty after preparation")

    # A split with no positives scores NaN average precision, and a NaN record reads
    # as a result rather than as a run that could not measure anything -- worse, it
    # serialises as a bare `NaN`, which is not valid JSON. Reachable with a small
    # a dataset whose frauds are not spread evenly, which is all of them.
    for name in ("val", "test"):
        positives = int(y_true[rows[name]].sum())
        if positives == 0:
            raise ExperimentError(
                f"the {name} split contains no frauds, so average precision is "
                f"undefined; at a {y_true.mean() * 100:.3f}% base rate a "
                f"{len(rows[name]):,}-row split cannot be scored"
            )
    if int(y[rows["train"]].sum()) == 0:
        raise ExperimentError(
            "no fraud is labelled in train"
            + (
                f" -- the {config.label_delay!r} delay regime hid all "
                f"{int(y_true[rows['train']].sum())} of them"
                if config.label_delay != "off"
                else ""
            )
        )

    return Prepared(
        x=x,
        y=y,
        y_true=y_true,
        train_rows=rows["train"],
        val_rows=rows["val"],
        test_rows=rows["test"],
        continuous=np.asarray(continuous_indices, dtype="int64"),
        categorical=np.asarray(categorical_indices, dtype="int64"),
        cardinalities=cardinalities,
        names=names,
        censored_train_labels=censored,
        rule_scores=rule_scores,
        notes=notes,
    )


def _labels(
    df: pd.DataFrame,
    config: ExperimentConfig,
    train_mask: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Training labels, censored on train rows only. Returns (labels, n hidden)."""
    if config.label_delay == "off":
        return y_true.copy(), 0

    column = "reported_at" if config.label_delay == "on" else SLOW_DELAY_COLUMN
    if column not in df.columns:
        raise ExperimentError(f"{config.dataset}: no {column!r} column to censor against")

    cutoff = df.loc[train_mask, "event_time"].max()
    # A two-column frame rather than a rename: renaming `reported_at_slow` onto
    # `reported_at` leaves two columns of that name, and `censored_labels` then reads
    # a DataFrame where it expects a Series.
    known = censored_labels(
        pd.DataFrame({"is_fraud": df["is_fraud"], "reported_at": df[column]}), cutoff
    )

    y = y_true.copy()
    y[train_mask] = known[train_mask]
    hidden = int((y_true[train_mask] == 1).sum() - (y[train_mask] == 1).sum())
    return y, hidden


def _trivial_rule_scores(df: pd.DataFrame, dataset: str) -> np.ndarray | None:
    """The one-line rule, where a dataset has one, as scores over every row.

    A ceiling rather than a floor on IBM CCF: train holds zero Italy frauds, so the
    rule imports knowledge no train-only model could have. It bounds what the
    evaluation labels encode, and reporting a model without it invites reading 0.04
    as a hard problem rather than as a mislabelled one.
    """
    from fraud_benchmark.experiments.models import TRIVIAL_RULES

    rule = TRIVIAL_RULES.get(dataset)
    if rule is None:
        return None
    column, value = rule
    if column not in df.columns:
        return None
    matches = (df[column].astype("string") == value).fillna(False)
    return matches.to_numpy().astype("float64")


def run(config: ExperimentConfig, features_dir: Path | str = FEATURE_DIR) -> dict:
    """Prepare, fit once per seed, and return one record describing the whole run."""
    from fraud_benchmark.experiments.estimators import get_estimator

    estimator = get_estimator(config.model)
    started = time.monotonic()
    prepared = prepare(config, features_dir)

    for note in prepared.notes:
        print(f"  {note}", flush=True)
    print(
        f"  matrix {prepared.x.shape[0]:,} x {prepared.x.shape[1]} "
        f"({prepared.x.nbytes / 1e9:.1f} GB), "
        f"{len(prepared.categorical)} coded categorical column(s)",
        flush=True,
    )

    seed_records = [estimator(prepared, seed) for seed in config.seeds]
    record = {
        "config": {**asdict(config), "seeds": list(config.seeds)},
        "rows": {
            "train": len(prepared.train_rows),
            "val": len(prepared.val_rows),
            "test": len(prepared.test_rows),
        },
        "n_features": len(prepared.names),
        "censored_train_labels": prepared.censored_train_labels,
        "notes": prepared.notes,
        "seeds": seed_records,
        "aggregate": _aggregate(seed_records),
        "total_seconds": round(time.monotonic() - started, 1),
        "python": platform.python_version(),
    }
    if prepared.rule_scores is not None:
        record["trivial_rule"] = _rule_record(prepared)
    return record


def _aggregate(seed_records: list[dict]) -> dict:
    """Mean and population sd per metric per split, across seeds.

    Population sd, not sample: with three seeds the sample estimate is noisy enough
    to mislead, and the quantity wanted is the spread of the runs actually made.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("val", "test"):
        out[split] = {}
        for metric in ("average_precision", "precision", "recall", "f1"):
            values = [r["scores"][split][metric] for r in seed_records]
            out[split][metric] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
            }
    return out


def _rule_record(prepared: Prepared) -> dict:
    """The trivial rule scored on the same val and test rows, for reference."""
    from fraud_benchmark.experiments.metrics import score

    return {
        split: score(
            prepared.y_true[prepared.split_rows(split)],
            prepared.rule_scores[prepared.split_rows(split)],
            0.5,
        )
        for split in ("val", "test")
    }


def append_record(record: dict, destination: Path | str = DEFAULT_RESULTS) -> Path:
    """Append one JSON line. One file per invocation is the caller's business.

    `allow_nan=False` so a NaN cannot reach disk. Python writes it as a bare `NaN`,
    which json.loads accepts and most other readers reject, so the file would parse
    here and fail in a notebook. Serialising is also the last point at which a metric
    that could not be computed is still distinguishable from one that was.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, allow_nan=False)
    with open(destination, "a") as handle:
        handle.write(line + "\n")
    return destination


def describe(record: dict) -> str:
    """A one-screen summary for a log, since a JSON line is not readable."""
    config = record["config"]
    aggregate = record["aggregate"]
    lines = [
        f"{config['dataset']} / {config['model']} / history={config['history']} / "
        f"delay={config['label_delay']} / artifacts={config['artifacts']} / "
        f"split={config['split']}",
        f"  rows train/val/test: {record['rows']['train']:,} / "
        f"{record['rows']['val']:,} / {record['rows']['test']:,}"
        f"   features: {record['n_features']}",
    ]
    if record["censored_train_labels"]:
        lines.append(
            f"  train labels hidden by the delay: {record['censored_train_labels']:,}"
        )
    for split in ("val", "test"):
        ap = aggregate[split]["average_precision"]
        f1 = aggregate[split]["f1"]
        lines.append(
            f"  {split:<5} average precision {ap['mean']:.4f} ± {ap['sd']:.4f}"
            f"   F1 {f1['mean']:.4f} ± {f1['sd']:.4f}"
        )
    if "trivial_rule" in record:
        rule = record["trivial_rule"]
        lines.append(
            f"  trivial rule (a ceiling, not a floor): "
            f"val AP {rule['val']['average_precision']:.4f}, "
            f"test AP {rule['test']['average_precision']:.4f}"
        )
    lines.append(f"  {record['total_seconds']}s total")
    return "\n".join(lines)


#: Re-exported so a caller assembling a config sees what a parquet's keys are called.
__all__ = [
    "ARTIFACT_MODES",
    "DATASETS",
    "KEY_COLUMNS",
    "LABEL_DELAYS",
    "OPTIONAL_KEY_COLUMNS",
    "SPLITS",
    "ExperimentConfig",
    "ExperimentError",
    "Prepared",
    "append_record",
    "describe",
    "prepare",
    "run",
]
