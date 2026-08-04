"""Which experiments exist, as one definition read by three consumers.

`scripts/slurm/generate.py` turns this into sbatch files, `scripts/summarize.py` uses
it to name the cells with no results yet, and a reader uses it to see the shape of the
study without running anything. Before this was shared, a retired cell kept appearing
in a summary and a new one silently did not -- the summary could not tell "no results
because the job has not run" from "no results because the cell does not exist".

## The shape

Four questions, each isolating one axis against a common baseline rather than a full
cross product. A full cross of 3 datasets x 3 models x 2 histories x 3 delays x 2
artifact modes x 2 splits is 216 cells, most of which answer nothing: what a
one-hot MLP does on IBM CCF with artifacts kept, no history and a slow delay it does
not have is not a question anyone asked.

    baseline     every dataset x every model, no history, true labels, artifacts
                 dropped. The reference every other group is read against.
    history      does concatenating the previous 10 transactions help? Same
                 everything else. The retired flattened-window experiment measured
                 lagged columns earning 13% of a tree's gain, so the expected
                 answer is "barely", and it is worth knowing per dataset.
    delay        what does training on the labels a detector would actually have
                 had cost? Sparkov carries three regimes, the other two carry one,
                 so this group is where sparkov_slow finally earns its place.
    artifacts    what does the generator's own geography supply? IBM CCF only, plus
                 the Italy holdout, which is the same question asked by moving the
                 split instead of by dropping columns.

Sizes assume the parquets exist. `estimate_cost` is what the job generator uses to
put each cell on a partition it can finish on.
"""

from __future__ import annotations

from dataclasses import dataclass

from fraud_benchmark.experiments.experiment import ExperimentConfig

#: Lags for the history group. 10 matches the retired flattened-window experiment,
#: so its 13%-of-gain finding is a direct comparison.
HISTORY_LAGS = 10

DATASETS = ("ibm_ccf", "saml_d", "sparkov")
MODELS = ("xgboost", "mlp", "logistic")


@dataclass(frozen=True)
class Cell:
    """One job: a name, the config it runs, and which question it answers."""

    group: str
    config: ExperimentConfig

    @property
    def name(self) -> str:
        """Filename-safe and unique across the grid."""
        c = self.config
        parts = [c.dataset, c.model]
        if c.history:
            parts.append(f"h{c.history}")
        if c.label_delay != "off":
            parts.append(f"delay-{c.label_delay}")
        if c.artifacts != "drop":
            parts.append("artifacts")
        if c.split != "standard":
            parts.append(c.split)
        return "_".join(parts)


def _config(dataset: str, model: str, **overrides) -> ExperimentConfig:
    """A config with the grid's shared defaults. Every cell sees every row."""
    return ExperimentConfig(dataset=dataset, model=model, seeds=(0, 1, 2), **overrides)


def build_cells() -> tuple[Cell, ...]:
    """Every cell in the grid, in the order the summary reports them."""
    cells: list[Cell] = []

    # --- baseline: the reference every other group is read against
    for dataset in DATASETS:
        for model in MODELS:
            cells.append(Cell("baseline", _config(dataset, model)))

    # --- history: the same thing with the previous 10 transactions concatenated
    for dataset in DATASETS:
        for model in MODELS:
            cells.append(
                Cell("history", _config(dataset, model, history=HISTORY_LAGS))
            )

    # --- delay: what training on the labels a detector would have had costs.
    # Trees only: the question is about labels, and adding two more models triples
    # the cost of the group without changing what it measures.
    for dataset in DATASETS:
        cells.append(Cell("delay", _config(dataset, "xgboost", label_delay="on")))
    cells.append(Cell("delay", _config("sparkov", "xgboost", label_delay="slow")))

    # --- artifacts: what the generator's geography supplies, two ways
    for model in ("xgboost", "mlp"):
        cells.append(Cell("artifacts", _config("ibm_ccf", model, artifacts="keep")))
    cells.append(
        Cell("artifacts", _config("ibm_ccf", "xgboost", split="italy_holdout"))
    )

    return tuple(cells)


CELLS = build_cells()

#: Cells whose matrix or fit is large enough to need the long partition. Measured
#: from the feature builds: ibm_ccf is 24.4M rows and saml_d 9.5M, and the MLP is
#: 15 epochs over them where a tree is one pass.
def estimate_cost(cell: Cell) -> tuple[str, str, str]:
    """(partition, memory, walltime) for one cell.

    Deliberately coarse. A cell that finishes in a tenth of its request costs nothing
    but queue position; one that runs out of either loses the whole run, and the two
    big datasets are where a bad guess is expensive.
    """
    config = cell.config
    rows = {"ibm_ccf": 24_400_000, "saml_d": 9_500_000, "sparkov": 1_900_000}[
        config.dataset
    ]

    # Feature count drives the matrix; history multiplies the lagged subset only.
    features = 90 + (config.history * 14)
    matrix_gb = rows * features * 4 / 1e9

    if config.model == "mlp":
        # Batches are gathered by index from the resident matrix, so memory is the
        # matrix plus room for the one-hot expansion per batch; time is 15 epochs.
        memory = max(32, int(matrix_gb * 6) + 16)
        hours = max(4, int(rows / 1_000_000) * 2)
    elif config.model == "xgboost":
        # hist tree method builds its own quantile sketch alongside the matrix.
        memory = max(16, int(matrix_gb * 5) + 8)
        hours = max(2, int(rows / 4_000_000) + 1)
    else:
        # Logistic standardises into a second copy of the matrix per split.
        memory = max(16, int(matrix_gb * 6) + 8)
        hours = max(2, int(rows / 6_000_000) + 1)

    hours = min(hours, 70)
    partition = "cpu" if hours <= 20 else "cpulong"
    return partition, f"{memory}G", f"{hours:02d}:00:00"


def cells_by_group() -> dict[str, tuple[Cell, ...]]:
    """The grid grouped by the question each cell answers."""
    groups: dict[str, list[Cell]] = {}
    for cell in CELLS:
        groups.setdefault(cell.group, []).append(cell)
    return {group: tuple(members) for group, members in groups.items()}


GROUPS = tuple(cells_by_group())
