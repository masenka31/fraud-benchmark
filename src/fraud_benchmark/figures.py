"""The one palette every figure in this repository draws with.

Shared so the figures read as one system: three scripts under `scripts/figures/` emit
into `results/figures/`, and a colour defined twice drifts. Previously
`plot_dataset_caveats.py` imported it from `plot_monthly_fraud.py` as a bare sibling
module, which only resolved because the interpreter puts a script's own directory on
the path.

The values are the dataviz reference palette's categorical slots 1-3 and its chrome
inks, unchanged -- no slot is substituted, so no contrast revalidation is needed. Light
and dark are each stepped for their own surface rather than one flipped into the other.

`accent`, `positive` and `negative` are names for slots, not new colours: a figure
saying "the reference line is the accent" reads better than one indexing the tuple, and
routing them through here keeps the count of actual inks at three.
"""

from __future__ import annotations


def _theme(
    *,
    surface: str,
    ink: str,
    secondary: str,
    muted: str,
    grid: str,
    axis: str,
    series: tuple[str, str, str],
    noise: str,
) -> dict:
    return dict(
        surface=surface,
        ink=ink,
        secondary=secondary,
        muted=muted,
        grid=grid,
        axis=axis,
        series=series,
        accent=series[1],
        positive=series[2],
        negative=series[1],
        #: A delta smaller than its own sigma is drawn in this rather than by sign --
        #: see `experiments.summary.is_measured`. Chrome, deliberately not a slot: it
        #: has to read as "no measured effect" beside two saturated bars.
        noise=noise,
    )


THEMES = {
    'light': _theme(
        surface='#fcfcfb',
        ink='#0b0b0b',
        secondary='#52514e',
        muted='#898781',
        grid='#e1e0d9',
        axis='#c3c2b7',
        series=('#2a78d6', '#eb6834', '#3d9970'),
        noise='#c3c2b7',
    ),
    'dark': _theme(
        surface='#1a1a19',
        ink='#ffffff',
        secondary='#c3c2b7',
        muted='#898781',
        grid='#2c2c2a',
        axis='#383835',
        series=('#3987e5', '#d95926', '#4fae82'),
        noise='#4a4a47',
    ),
}
