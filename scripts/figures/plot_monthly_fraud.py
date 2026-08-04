"""Monthly fraud rate and count across the full IBM CCF span.

Two stacked panels sharing one x-axis rather than a dual-axis chart: rate and
count are different scales, and overlaying them on two y-axes would invent a
correlation by the arbitrary alignment of the scales.

Colours come from `fraud_benchmark.figures` -- the reference palette's slot 1 for
the series and slot 2 for the Italy-regime band, plus its chrome inks. Both light
and dark are emitted, each stepped for its own surface rather than flipped.

The monthly CSV written alongside is the table-view twin of the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fraud_benchmark.figures import THEMES

ITALY_REGIME_START = pd.Timestamp("2017-11-19")
MIN_GAP_MONTHS = 3

def monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(df["event_time"].dt.to_period("M")).agg(
        rows=("is_fraud", "size"), frauds=("is_fraud", "sum")
    )
    m["rate_pct"] = m["frauds"] / m["rows"] * 100
    m.index = m.index.to_timestamp()
    return m


def zero_fraud_runs(m: pd.DataFrame, min_months: int = MIN_GAP_MONTHS):
    """Contiguous runs of fraud-free months, starting from the first fraud ever."""
    first_fraud = m.index[m["frauds"] > 0][0]
    sub = m[m.index >= first_fraud]
    runs, start, prev = [], None, None
    for when, n in sub["frauds"].items():
        if n == 0 and start is None:
            start = when
        elif n > 0 and start is not None:
            runs.append((start, prev))
            start = None
        prev = when
    if start is not None:
        runs.append((start, prev))
    month = pd.Timedelta(days=31)
    return [(a, b + month) for a, b in runs if (b - a) / month >= min_months - 1]


def draw(m: pd.DataFrame, theme: str, out: Path) -> None:
    c = THEMES[theme]
    gaps = zero_fraud_runs(m)

    fig, (ax_rate, ax_count) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.12),
    )
    fig.patch.set_facecolor(c["surface"])

    for ax in (ax_rate, ax_count):
        ax.set_facecolor(c["surface"])
        for gap_start, gap_end in gaps:
            ax.axvspan(gap_start, gap_end, color=c["muted"], alpha=0.18, lw=0, zorder=1)
        ax.axvspan(ITALY_REGIME_START, m.index[-1] + pd.Timedelta(days=31),
                   color=c["accent"], alpha=0.14, lw=0, zorder=1)
        ax.grid(True, axis="y", color=c["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=c["muted"], labelsize=9, length=3, width=0.8)

    ax_rate.plot(m.index, m["rate_pct"], color=c["series"][0], lw=2.0, zorder=3)
    ax_rate.set_ylabel("fraud rate, % of transactions", color=c["secondary"], fontsize=10)
    ax_rate.set_title(
        "IBM CCF: the fraud rate is not stationary, and the generator stops emitting fraud for months at a time",
        color=c["ink"], fontsize=12.5, loc="left", pad=26,
    )
    # Facts that would otherwise need arrows across the data.
    n_gaps = len(gaps)
    lo = m.loc[m["frauds"] > 0, "rate_pct"].min()
    hi = m["rate_pct"].max()
    ax_rate.annotate(
        f"Monthly rate spans {lo:.3f}%–{hi:.2f}% ({hi / lo:.0f}x) while volume only rises. "
        f"{n_gaps} fraud-free gaps of {MIN_GAP_MONTHS}+ months; the last runs 332 days "
        "and ends as the Italy regime begins.",
        xy=(0, 1.035), xycoords="axes fraction", color=c["secondary"], fontsize=9.5,
        va="bottom", ha="left",
    )

    ax_count.plot(m.index, m["frauds"], color=c["series"][0], lw=2.0, zorder=3)
    ax_count.set_ylabel("frauds per month", color=c["secondary"], fontsize=10)
    ax_count.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_count.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # One direct label: the extreme. Placed left of the peak so it cannot
    # collide with the legend, which sits in the genuinely empty pre-1996 area.
    peak = m["rate_pct"].idxmax()
    ax_rate.annotate(
        f"{m.loc[peak, 'rate_pct']:.2f}%  {peak:%b %Y}",
        xy=(peak, m.loc[peak, "rate_pct"]), xytext=(-58, 0),
        textcoords="offset points", color=c["secondary"], fontsize=9,
    )

    handles = [
        Line2D([], [], color=c["series"][0], lw=2.0, label="monthly fraud rate / count"),
        Patch(facecolor=c["muted"], alpha=0.18, label=f"zero-fraud gap ({MIN_GAP_MONTHS}+ months)"),
        Patch(facecolor=c["accent"], alpha=0.14, label="Italy regime (from 2017-11-19)"),
    ]
    leg = ax_rate.legend(handles=handles, loc="upper left", frameon=False,
                         fontsize=9, labelcolor=c["secondary"])
    for text in leg.get_texts():
        text.set_color(c["secondary"])

    fig.text(0.005, 0.005,
             f"{int(m['rows'].sum()):,} transactions, {int(m['frauds'].sum()):,} labelled frauds, "
             f"{m.index[0]:%Y-%m}..{m.index[-1]:%Y-%m}. Labelling stops 2019-10-27.",
             color=c["muted"], fontsize=8)

    # subplots_adjust rather than tight_layout: the axes carry annotations
    # outside their bounds, which tight_layout cannot account for.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.085)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/features/ibm_ccf.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    df = pd.read_parquet(args.features, columns=["event_time", "is_fraud"])
    m = monthly(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out_dir / "ibm_ccf_monthly_fraud.csv")
    print(f"wrote {args.out_dir / 'ibm_ccf_monthly_fraud.csv'} (table view)")
    for theme in ("light", "dark"):
        draw(m, theme, args.out_dir / f"ibm_ccf_monthly_fraud_{theme}.png")

    gaps = zero_fraud_runs(m)
    print(f"\n{len(gaps)} zero-fraud gaps of {MIN_GAP_MONTHS}+ months")
    print(f"fraud rate range: {m[m.frauds>0].rate_pct.min():.4f}% .. {m.rate_pct.max():.4f}%")


if __name__ == "__main__":
    main()
