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

**A bar is greyed when it is smaller than its own noise.** Colouring by sign alone made
the figure contradict the table: SAML-D's delay cell is the longest bar in the panel and
is 1.5 sigma, its seed ranges overlapping the baseline's. The error bar is sigma of the
*difference* (the two variances add, since seeds are independent between runs), so bar
length against error-bar length is the comparison a reader should make, and the grey
saves them making it.

Left-panel error bars are population sd over seeds, not a confidence interval: with three
seeds there is no distribution to be confident about, only the spread of the runs made.
Values are printed on the bars because the datasets differ by an order of magnitude in
difficulty -- IBM CCF's bars are a few pixels tall beside SAML-D's.

Colours come from `fraud_benchmark.figures`, the palette every other figure here draws
with. Both light and dark are emitted, each stepped for its own surface rather than
flipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fraud_benchmark.experiments.summary import NOISE_SIGMA
from fraud_benchmark.experiments.summary import RESULTS_DIR
from fraud_benchmark.experiments.summary import baseline_index
from fraud_benchmark.experiments.summary import flatten
from fraud_benchmark.experiments.summary import is_measured
from fraud_benchmark.experiments.summary import load
from fraud_benchmark.experiments.summary import sigma_of_difference
from fraud_benchmark.figures import THEMES

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
        offsets, values, errors = [], [], []
        for slot, dataset in enumerate(DATASETS):
            row = by_key.get((dataset, model))
            offset = slot + (index - 1) * width
            if row is None:
                # No bar at all. A zero-height bar is indistinguishable from a
                # measured average precision of 0.0, which is a real possible result.
                ax.text(
                    offset,
                    0.012,
                    "not run",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color=colours["muted"],
                    fontsize=7.5,
                )
                continue
            offsets.append(offset)
            values.append(row["test_ap"])
            errors.append(row["test_ap_sd"])
        ax.bar(
            offsets,
            values,
            width,
            yerr=errors,
            capsize=2.5,
            color=colours["series"][index],
            edgecolor="none",
            error_kw={"ecolor": colours["muted"], "elinewidth": 1},
            label=model,
        )
        # The three datasets differ by an order of magnitude in difficulty, so IBM CCF's
        # bars are a few pixels tall next to SAML-D's. The number goes on the bar.
        for offset, value, error in zip(offsets, values, errors):
            ax.text(
                offset,
                value + error + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                color=colours["secondary"],
                fontsize=7.5,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("test average precision")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=colours["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Baseline: no history, true labels, artifacts dropped",
        color=colours["ink"],
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    legend = ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        labelcolor=colours["secondary"],
    )
    legend.set_title("")


def draw_deltas(ax, rows: list[dict], colours: dict) -> None:
    """Horizontal diverging bars: each cell against its own baseline."""
    baselines = baseline_index(rows)
    entries = []
    for row in rows:
        if row["group"] == "baseline" or not row["ran"]:
            continue
        baseline = baselines.get((row["dataset"], row["model"]))
        if baseline is None:
            continue
        difference = row["test_ap"] - baseline["test_ap"]
        sigma = sigma_of_difference(row, baseline)
        # The table's own rule, called rather than restated. Without it the figure
        # contradicts the table: saml_d's delay cell is the largest bar here and is
        # 1.5 sigma.
        entries.append((_delta_label(row), difference, sigma, is_measured(difference, sigma)))

    if not entries:
        ax.text(
            0.5,
            0.5,
            "no comparable cells yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=colours["muted"],
            fontsize=10,
        )
        _style(ax, colours)
        return

    entries.reverse()  # highest row at the top
    labels = [entry[0] for entry in entries]
    values = [entry[1] for entry in entries]
    sigmas = [entry[2] for entry in entries]
    positions = np.arange(len(entries))
    colour = [
        (colours["positive"] if value >= 0 else colours["negative"]) if real else colours["noise"]
        for _, value, _, real in entries
    ]

    ax.barh(
        positions,
        values,
        0.66,
        xerr=sigmas,
        color=colour,
        edgecolor="none",
        error_kw={"ecolor": colours["muted"], "elinewidth": 1, "capsize": 2},
    )
    ax.axvline(0, color=colours["axis"], linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("change in test average precision vs the same model's baseline")
    ax.xaxis.grid(True, color=colours["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "What each axis changed",
        color=colours["ink"],
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor=colours["positive"], label="better than baseline"),
            Patch(facecolor=colours["negative"], label="worse"),
            Patch(
                facecolor=colours["noise"],
                label=f"within noise (<{NOISE_SIGMA:.0f}σ)",
            ),
        ],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
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
        0.005,
        0.015,
        f"{ran} of {len(rows)} cells. Left: error bars are population sd over seeds. "
        f"Right: error bars are σ of the difference; a bar shorter than {NOISE_SIGMA:.0f}σ "
        "is greyed as no measured effect. Average precision, never ROC AUC.",
        color=colours["muted"],
        fontsize=8,
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
