"""The experiment grid as two panels: absolute scores, and what each axis changed.

Reads `results/experiments/` through `scripts/summarize.py`, so the figure and the
markdown table can never disagree about what ran.

The left panel is grouped bars of test average precision per dataset per model, which
is the only honest way to show them: the three datasets differ by two orders of
magnitude in difficulty, so a single ranked bar chart would say more about which
dataset was chosen than about any model.

The right panel is the delta against each cell's own baseline, diverging from zero.
That is the quantity the study is actually about -- whether history, a label delay, or
the generator's geography moved anything -- and it is signed, so a diverging layout is
the correct form and a zero line the correct reference.

Error bars are population sd over seeds, not a confidence interval. With three seeds
there is no distribution to be confident about; the bar shows the spread of the runs
actually made, which is what tells you whether a delta is worth reading.

Colours are the dataviz reference palette's categorical slots, unchanged, plus its
chrome inks. Both light and dark are emitted, each stepped for its own surface rather
than flipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize import RESULTS_DIR, _baseline_index, flatten, load  # noqa: E402

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#3d9970"), positive="#3d9970",
        negative="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=("#3987e5", "#d95926", "#4fae82"), positive="#4fae82",
        negative="#d95926",
    ),
}

MODELS = ("xgboost", "mlp", "logistic")
DATASETS = ("ibm_ccf", "saml_d", "sparkov")

#: A short label per non-baseline cell, for the delta panel's y-axis.
def _delta_label(row: dict) -> str:
    if row["group"] == "history":
        return f"{row['dataset']} {row['model']}  +{row['history']} lags"
    if row["group"] == "delay":
        return f"{row['dataset']} {row['model']}  delay {row['label_delay']}"
    if row["artifacts"] == "keep":
        return f"{row['dataset']} {row['model']}  artifacts kept"
    return f"{row['dataset']} {row['model']}  {row['split'].replace('_', ' ')}"


def _style(ax, colours: dict) -> None:
    ax.set_facecolor(colours["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colours["axis"])
    ax.tick_params(colors=colours["secondary"], labelsize=9)
    ax.xaxis.label.set_color(colours["secondary"])
    ax.yaxis.label.set_color(colours["secondary"])


def draw_baselines(ax, rows: list[dict], colours: dict) -> None:
    """Grouped bars: test average precision, one group per dataset."""
    baseline = [row for row in rows if row["group"] == "baseline" and row["ran"]]
    by_key = {(row["dataset"], row["model"]): row for row in baseline}

    positions = np.arange(len(DATASETS))
    width = 0.26
    for index, model in enumerate(MODELS):
        values, errors = [], []
        for dataset in DATASETS:
            row = by_key.get((dataset, model))
            values.append(row["test_ap"] if row else 0.0)
            errors.append(row["test_ap_sd"] if row else 0.0)
        ax.bar(
            positions + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=2.5,
            color=colours["series"][index],
            edgecolor="none",
            error_kw={"ecolor": colours["muted"], "elinewidth": 1},
            label=model,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("test average precision")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=colours["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Baseline: no history, true labels, artifacts dropped",
        color=colours["ink"], fontsize=10.5, loc="left", pad=10,
    )
    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper left",
        labelcolor=colours["secondary"],
    )
    legend.set_title("")


def draw_deltas(ax, rows: list[dict], colours: dict) -> None:
    """Horizontal diverging bars: each cell against its own baseline."""
    baselines = _baseline_index(rows)
    entries = []
    for row in rows:
        if row["group"] == "baseline" or not row["ran"]:
            continue
        baseline = baselines.get((row["dataset"], row["model"]))
        if baseline is None:
            continue
        entries.append((_delta_label(row), row["test_ap"] - baseline["test_ap"]))

    if not entries:
        ax.text(
            0.5, 0.5, "no comparable cells yet",
            transform=ax.transAxes, ha="center", va="center",
            color=colours["muted"], fontsize=10,
        )
        _style(ax, colours)
        return

    entries.reverse()  # highest row at the top
    labels = [label for label, _ in entries]
    values = [value for _, value in entries]
    positions = np.arange(len(entries))
    colour = [
        colours["positive"] if value >= 0 else colours["negative"] for value in values
    ]

    ax.barh(positions, values, 0.66, color=colour, edgecolor="none")
    ax.axvline(0, color=colours["axis"], linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("change in test average precision vs the same model's baseline")
    ax.xaxis.grid(True, color=colours["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "What each axis changed", color=colours["ink"], fontsize=10.5,
        loc="left", pad=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor=colours["positive"], label="better than baseline"),
            Patch(facecolor=colours["negative"], label="worse"),
        ],
        frameon=False, fontsize=8.5, loc="lower right",
        labelcolor=colours["secondary"],
    )


def figure(rows: list[dict], theme: str) -> plt.Figure:
    colours = THEMES[theme]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1, 1.35]}
    )
    fig.patch.set_facecolor(colours["surface"])

    draw_baselines(left, rows, colours)
    draw_deltas(right, rows, colours)
    for ax in (left, right):
        _style(ax, colours)

    ran = sum(row["ran"] for row in rows)
    fig.text(
        0.005, 0.015,
        f"{ran} of {len(rows)} cells. Error bars are population sd over seeds. "
        "Average precision, never ROC AUC.",
        color=colours["muted"], fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    rows = flatten(load(args.results_dir))
    if not any(row["ran"] for row in rows):
        print("no results yet; nothing to plot")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        fig = figure(rows, theme)
        out = args.out_dir / f"experiments_{theme}.png"
        fig.savefig(out, dpi=150, facecolor=THEMES[theme]["surface"])
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
