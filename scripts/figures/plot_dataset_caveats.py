"""One figure per dataset caveat that words undersell, for the dataset pages.

Three caveats earn a figure, because each is a *shape* rather than a number:

  paysim   -- fraud is flat across the 24-hour cycle while legitimate volume
              swings 500x, so hour-of-day alone identifies 44% of frauds.
  banksim  -- fraud is injected into named categories, and the top ones run at
              40-95% against a 1.2% base rate.
  sparkov  -- the delay regimes of `sparkov` and `sparkov_slow` on identical rows.

IBM CCF's regime shift already has `plot_monthly_fraud.py`; the other datasets'
caveats are adequately stated as text in their doc pages.

Colours come from `fraud_benchmark.figures` -- the reference palette's categorical
slots 1 (blue) and 2 (orange) plus its chrome inks, the same palette every other
figure here draws with, so no slot is substituted and no revalidation is needed.
Each figure is emitted for light and dark, stepped for its own surface.

Both panels of the PaySim figure carry ONE series each and share an x-axis: a
frauds-per-hour count and a rows-per-hour count differ by three orders of
magnitude, and putting them on two y-scales would invent a correlation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_benchmark.figures import THEMES

PROCESSED = Path("data/processed")
BAR_RADIUS = 4  # px, rounded data-end away from the baseline


def _frame(theme: str, figsize, nrows=1, **kwargs):
    c = THEMES[theme]
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, **kwargs)
    fig.patch.set_facecolor(c["surface"])
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(c["surface"])
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=c["muted"], labelsize=9, length=3, width=0.8)
    return c, fig, axes


def _save(fig, out: Path, surface: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=surface)
    plt.close(fig)
    print(f"wrote {out}")


def paysim(theme: str, out: Path) -> None:
    df = pd.read_parquet(PROCESSED / "paysim/data.parquet",
                         columns=["event_time", "is_fraud"])
    h = df.groupby(df["event_time"].dt.hour).agg(
        rows=("is_fraud", "size"), frauds=("is_fraud", "sum")
    )
    rate = h["frauds"] / h["rows"] * 100

    c, fig, (top, bottom) = _frame(
        theme, (9, 5.4), nrows=2, sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.18),
    )
    # 2px surface gap between adjacent bars: width 0.82 at unit spacing.
    for ax, values, label in (
        (top, h["frauds"], "frauds per hour"),
        (bottom, h["rows"], "transactions per hour"),
    ):
        ax.bar(h.index, values, width=0.82, color=c["series"][0], zorder=3)
        ax.grid(True, axis="y", color=c["grid"], lw=0.8, zorder=0)
        ax.set_ylabel(label, color=c["secondary"], fontsize=10)
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")

    top.set_ylim(0, h["frauds"].max() * 1.35)
    top.set_title(
        "PaySim: fraud is flat around the clock, legitimate traffic is not",
        color=c["ink"], fontsize=12.5, loc="left", pad=38,
    )
    top.annotate(
        f"Frauds per hour-of-day vary {h['frauds'].min()}–{h['frauds'].max()} "
        f"({h['frauds'].max() / h['frauds'].min():.1f}x) while transactions vary "
        f"{h['rows'].min():,}–{h['rows'].max():,} ({h['rows'].max() / h['rows'].min():.0f}x).\n"
        f"The quotient runs {rate.min():.2f}% fraud at {rate.idxmin():02d}:00 up to "
        f"{rate.max():.1f}% at {rate.idxmax():02d}:00 — a {rate.max() / rate.min():.0f}x "
        "spread from the clock alone.",
        xy=(0, 1.03), xycoords="axes fraction", color=c["secondary"], fontsize=9.5,
        va="bottom", ha="left", linespacing=1.5,
    )
    # Direct labels on the two extremes only, never on every bar.
    top.annotate(f"{rate.max():.1f}% fraud", xy=(rate.idxmax(), h.loc[rate.idxmax(), "frauds"]),
                 xytext=(0, 8), textcoords="offset points", color=c["secondary"],
                 fontsize=9, ha="center")
    bottom.annotate(f"{rate.min():.2f}% fraud", xy=(rate.idxmin(), h.loc[rate.idxmin(), "rows"]),
                    xytext=(0, 8), textcoords="offset points", color=c["secondary"],
                    fontsize=9, ha="center")
    bottom.set_xlabel("hour of day (simulated)", color=c["secondary"], fontsize=10)
    bottom.set_xticks(range(0, 24, 2))
    fig.text(0.005, 0.005,
             "6,362,620 transactions, 8,213 frauds. 320 of 743 steps contain nothing "
             "but fraud, carrying 44.1% of all PaySim frauds.",
             color=c["muted"], fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.815, bottom=0.125)
    _save(fig, out, c["surface"])


def banksim(theme: str, out: Path) -> None:
    df = pd.read_parquet(PROCESSED / "banksim/data.parquet",
                         columns=["category", "is_fraud"])
    base = df["is_fraud"].mean() * 100
    g = df.groupby("category").agg(rows=("is_fraud", "size"), frauds=("is_fraud", "sum"))
    g["rate"] = g["frauds"] / g["rows"] * 100
    g = g.sort_values("rate")

    c, fig, ax = _frame(theme, (9, 5.2))
    ax.barh(range(len(g)), g["rate"], height=0.78, color=c["series"][0], zorder=3)
    ax.axvline(base, color=c["accent"], lw=2.0, zorder=4)
    # Below the shortest bars, the only region of the panel with no mark in it.
    ax.annotate(f"base rate {base:.2f}%", xy=(base, -0.95), xytext=(7, 0),
                textcoords="offset points", color=c["accent"], fontsize=9, va="center")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([n.replace("es_", "") for n in g.index], fontsize=9,
                       color=c["secondary"])
    ax.grid(True, axis="x", color=c["grid"], lw=0.8, zorder=0)
    ax.set_xlabel("fraud rate, % of the category's transactions",
                  color=c["secondary"], fontsize=10)
    ax.set_title("BankSim: fraud is injected into named categories",
                 color=c["ink"], fontsize=12.5, loc="left", pad=38)
    ax.annotate(
        "The six worst categories hold 57.1% of all frauds in 9,871 rows at 41.7% fraud.\n"
        "Three — transportation, food, contents — contain none at all. Any model that "
        "reads `category` inherits this.",
        xy=(0, 1.03), xycoords="axes fraction", color=c["secondary"], fontsize=9.5,
        va="bottom", ha="left", linespacing=1.5,
    )
    for i, (rate, rows) in enumerate(zip(g["rate"], g["rows"], strict=True)):
        if rate > 25 or rate == 0:      # selective labels: the extremes only
            ax.annotate(f"{rate:.0f}%  ({rows:,} rows)", xy=(rate, i), xytext=(6, 0),
                        textcoords="offset points", color=c["secondary"], fontsize=8.5,
                        va="center")
    ax.set_xlim(0, 118)
    fig.text(0.005, 0.005,
             "594,643 transactions, 7,200 frauds, base rate 1.21%. 16 merchants are "
             "flagged the same way.", color=c["muted"], fontsize=8)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.815, bottom=0.115)
    _save(fig, out, c["surface"])


def sparkov_delay(theme: str, out: Path) -> None:
    series = {}
    for name in ("sparkov", "sparkov_slow"):
        df = pd.read_parquet(PROCESSED / f"{name}/data.parquet",
                             columns=["event_time", "is_fraud", "reported_at"])
        frauds = df[df["is_fraud"]]
        series[name] = np.sort(
            (frauds["reported_at"] - frauds["event_time"]).dt.total_seconds().to_numpy()
            / 86_400
        )

    c, fig, ax = _frame(theme, (9, 5.0))
    colours = {"sparkov": c["series"][0], "sparkov_slow": c["accent"]}
    for name, days in series.items():
        y = np.arange(1, len(days) + 1) / len(days) * 100
        ax.plot(days, y, color=colours[name], lw=2.0, zorder=3, label=name)
        median = float(np.median(days))
        # 2px surface ring, so the marker reads where the two curves overlap.
        ax.plot([median], [50], marker="o", markersize=8, color=colours[name],
                markeredgecolor=c["surface"], markeredgewidth=2, zorder=4)
        # sparkov's label goes LEFT of its marker: to the right it would sit on
        # top of the sparkov_slow curve.
        left = name == "sparkov"
        ax.annotate(
            f"{name}\nmedian {median:.1f}d", xy=(median, 50),
            xytext=(-12 if left else 12, -30 if left else 8),
            textcoords="offset points", color=c["secondary"], fontsize=9,
            ha="right" if left else "left",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.05, 500)
    ax.set_ylim(0, 100)
    ax.grid(True, color=c["grid"], lw=0.8, zorder=0)
    ax.set_xlabel("days from the transaction to its reported_at (log scale)",
                  color=c["secondary"], fontsize=10)
    ax.set_ylabel("% of frauds reported by then", color=c["secondary"], fontsize=10)
    ax.set_title(
        "Sparkov's two delay regimes, on byte-identical rows",
        color=c["ink"], fontsize=12.5, loc="left", pad=56,
    )
    ax.annotate(
        "The only paired delay contrast in the suite: same 1,852,394 rows, same 9,651 "
        "frauds,\nsame splits — only `reported_at` differs. 2.2% of train labels are "
        "censored at the\ncutoff under `sparkov`, 8.9% under `sparkov_slow`.",
        xy=(0, 1.03), xycoords="axes fraction", color=c["secondary"], fontsize=9.5,
        va="bottom", ha="left", linespacing=1.5,
    )
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(c["secondary"])
    fig.text(0.005, 0.005,
             "Lognormal per campaign, measured from the campaign's last transaction. "
             "sparkov: median 7d, sigma 1.0.\nsparkov_slow: median 15d, sigma 1.665, "
             "capped at 365d. The cap bounds the campaign's draw, not the row's.",
             color=c["muted"], fontsize=8, linespacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.795, bottom=0.155)
    _save(fig, out, c["surface"])


FIGURES = {
    "paysim_hourly": paysim,
    "banksim_categories": banksim,
    "sparkov_delay_regimes": sparkov_delay,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figures", nargs="*", choices=list(FIGURES),
                    help="which figures to draw; default is all of them")
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    for name in args.figures or list(FIGURES):
        for theme in ("light", "dark"):
            FIGURES[name](theme, args.out_dir / f"{name}_{theme}.png")


if __name__ == "__main__":
    main()
