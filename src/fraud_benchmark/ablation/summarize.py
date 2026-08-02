"""results/runs.jsonl -> results/summary.md.

Seeds are averaged and their spread reported: the seed standard deviation is the
noise floor any leakage gap has to clear before it means anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS = Path("results/runs.jsonl")
DEFAULT_SUMMARY = Path("results/summary.md")


def _load(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for split in ("val", "test"):
            scores = record["scores"][split]
            rows.append(
                {
                    "dataset": record["dataset"],
                    "feature_set": record["feature_set"] or "n/a",
                    "label_regime": record["label_regime"] or "n/a",
                    "model": record["model"],
                    "split": split,
                    "average_precision": scores["average_precision"],
                    "f1": scores["f1"],
                }
            )
    return pd.DataFrame(rows)


def summarize(results_path: Path = DEFAULT_RESULTS) -> str:
    df = _load(results_path)
    if df.empty:
        return "# Leakage ablation\n\nNo results yet.\n"

    grouped = (
        df.groupby(["dataset", "split", "label_regime", "model", "feature_set"])[
            "average_precision"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    lines = [
        "# Leakage ablation",
        "",
        "Average precision. Base rates here run from 0.10% to 0.52%, so a",
        "negative-class-dominated metric would stay high for a model with no useful",
        "precision and hide exactly what this table measures.",
        "",
        "`gap` is leaky minus clean -- how much of the score the leaky columns supply.",
        "Sparkov has no leaky columns, so its gap is a pure measurement of noise and",
        "calibrates how much of every other row to believe.",
        "",
        "| dataset | split | labels | model | leaky | clean | gap | seed sd |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    keys = ["dataset", "split", "label_regime", "model"]
    for (dataset, split, regime, model), block in grouped.groupby(keys, sort=True):
        by_set = block.set_index("feature_set")
        leaky = by_set["mean"].get("leaky")
        clean = by_set["mean"].get("clean")
        spread = block["std"].max()
        gap = (leaky - clean) if (leaky is not None and clean is not None) else None

        def cell(value):
            return "-" if value is None or pd.isna(value) else f"{value:.3f}"

        if model == "trivial_rule":
            leaky_cell = cell(by_set["mean"].get("n/a"))
            clean_cell, gap_cell = "-", "-"
        else:
            leaky_cell, clean_cell, gap_cell = cell(leaky), cell(clean), cell(gap)

        lines.append(
            f"| {dataset} | {split} | {regime} | {model} | "
            f"{leaky_cell} | {clean_cell} | {gap_cell} | {cell(spread)} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise ablation results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    text = summarize(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
