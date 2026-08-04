"""The experiment grid, and the summary read off it.

The grid is shared by the job generator and the summary so that a cell cannot exist
in one and not the other. These tests are mostly about that: names are unique and
filename-safe, every cell is runnable, and a cell with no results is reported as
missing rather than dropped.
"""

import json

import pytest

from fraud_benchmark.experiments.grid import (
    CELLS,
    GROUPS,
    HISTORY_LAGS,
    build_cells,
    cells_by_group,
    estimate_cost,
)


# --- the grid ------------------------------------------------------------


def test_every_cell_name_is_unique():
    names = [cell.name for cell in CELLS]
    assert len(names) == len(set(names))


def test_every_cell_name_is_a_safe_filename():
    for cell in CELLS:
        assert cell.name.replace("_", "").replace("-", "").isalnum(), cell.name


def test_every_cell_config_validates():
    """A cell that cannot run is a grid that cannot be submitted."""
    for cell in CELLS:
        cell.config.validate()


def test_the_baseline_covers_every_dataset_and_model():
    baseline = cells_by_group()["baseline"]
    keys = {(cell.config.dataset, cell.config.model) for cell in baseline}
    assert len(keys) == len(baseline) == 9


def test_the_baseline_varies_nothing_but_dataset_and_model():
    for cell in cells_by_group()["baseline"]:
        config = cell.config
        assert config.history == 0
        assert config.label_delay == "off"
        assert config.artifacts == "drop"
        assert config.split == "standard"


def test_every_non_baseline_cell_has_a_baseline_to_compare_against():
    """A delta needs the same dataset and model with nothing varied."""
    baselines = {
        (cell.config.dataset, cell.config.model)
        for cell in cells_by_group()["baseline"]
    }
    for cell in CELLS:
        if cell.group == "baseline":
            continue
        assert (cell.config.dataset, cell.config.model) in baselines, cell.name


def test_each_group_changes_exactly_one_thing_from_the_baseline():
    for cell in CELLS:
        if cell.group == "baseline":
            continue
        config = cell.config
        varied = sum(
            [
                config.history != 0,
                config.label_delay != "off",
                config.artifacts != "drop",
                config.split != "standard",
            ]
        )
        assert varied == 1, f"{cell.name} varies {varied} axes at once"


def test_the_history_group_uses_the_lag_count_the_old_experiment_used():
    """10 makes the 13%-of-gain finding a direct comparison."""
    assert HISTORY_LAGS == 10
    for cell in cells_by_group()["history"]:
        assert cell.config.history == HISTORY_LAGS


def test_only_sparkov_carries_a_slow_delay_cell():
    slow = [c for c in CELLS if c.config.label_delay == "slow"]
    assert len(slow) == 1
    assert slow[0].config.dataset == "sparkov"


def test_every_delay_regime_is_represented():
    delays = {cell.config.label_delay for cell in CELLS}
    assert delays == {"off", "on", "slow"}


def test_the_italy_holdout_cell_is_ibm_ccf_only():
    holdout = [c for c in CELLS if c.config.split == "italy_holdout"]
    assert holdout and all(c.config.dataset == "ibm_ccf" for c in holdout)


def test_nothing_can_subsample():
    """Every run uses every row. A model fitted on part of a dataset is not comparable
    to one fitted on all of it, so the capability is gone rather than merely unused."""
    from dataclasses import fields

    assert "max_rows" not in {f.name for f in fields(CELLS[0].config)}


def test_every_cell_runs_three_seeds():
    for cell in CELLS:
        assert cell.config.seeds == (0, 1, 2)


def test_build_cells_is_deterministic():
    assert [c.name for c in build_cells()] == [c.name for c in build_cells()]


def test_the_groups_are_reported_in_a_stable_order():
    assert GROUPS == ("baseline", "history", "delay", "artifacts")


# --- cost estimates ------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_every_cell_gets_a_usable_request(cell):
    partition, memory, walltime = estimate_cost(cell)
    assert partition in ("cpu", "cpulong")
    assert memory.endswith("G") and int(memory[:-1]) >= 16
    hours, minutes, seconds = (int(p) for p in walltime.split(":"))
    assert 1 <= hours <= 70 and minutes == 0 and seconds == 0


def test_a_long_walltime_moves_to_the_long_partition():
    """The cpu partition has a 1-day limit, so nothing over it may be scheduled there."""
    for cell in CELLS:
        partition, _, walltime = estimate_cost(cell)
        if int(walltime.split(":")[0]) > 20:
            assert partition == "cpulong", cell.name


def test_history_asks_for_more_memory_than_no_history():
    flat = [c for c in CELLS if c.name == "ibm_ccf_xgboost"][0]
    lagged = [c for c in CELLS if c.name == "ibm_ccf_xgboost_h10"][0]
    assert int(estimate_cost(lagged)[1][:-1]) > int(estimate_cost(flat)[1][:-1])


def test_the_bigger_dataset_asks_for_more(monkeypatch):
    small = [c for c in CELLS if c.name == "sparkov_xgboost"][0]
    large = [c for c in CELLS if c.name == "ibm_ccf_xgboost"][0]
    assert int(estimate_cost(large)[1][:-1]) > int(estimate_cost(small)[1][:-1])


# --- the summary ---------------------------------------------------------


def summarize_module():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import summarize

    return summarize


def fake_record(cell, test_ap=0.5, sd=0.01):
    return {
        "config": {**cell.config.__dict__, "seeds": list(cell.config.seeds)},
        "rows": {"train": 1000, "val": 100, "test": 100},
        "n_features": 40,
        "censored_train_labels": 7 if cell.config.label_delay != "off" else 0,
        "notes": [],
        "seeds": [{"seed": s} for s in cell.config.seeds],
        "aggregate": {
            split: {
                metric: {"mean": test_ap, "sd": sd}
                for metric in ("average_precision", "precision", "recall", "f1")
            }
            for split in ("val", "test")
        },
        "total_seconds": 60.0,
        "python": "3.13.0",
    }


def test_an_empty_results_directory_reports_every_cell_as_not_run(tmp_path):
    summarize = summarize_module()
    rows = summarize.flatten(summarize.load(tmp_path))
    assert len(rows) == len(CELLS)
    assert not any(row["ran"] for row in rows)
    assert "not run" in summarize.render(rows)


def test_a_present_record_is_reported_with_its_score(tmp_path):
    summarize = summarize_module()
    cell = CELLS[0]
    (tmp_path / f"{cell.name}.jsonl").write_text(
        json.dumps(fake_record(cell, test_ap=0.42)) + "\n"
    )
    rows = {row["cell"]: row for row in summarize.flatten(summarize.load(tmp_path))}
    assert rows[cell.name]["ran"]
    assert rows[cell.name]["test_ap"] == 0.42


def test_the_newest_line_of_a_rerun_cell_wins(tmp_path):
    summarize = summarize_module()
    cell = CELLS[0]
    (tmp_path / f"{cell.name}.jsonl").write_text(
        json.dumps(fake_record(cell, test_ap=0.1))
        + "\n"
        + json.dumps(fake_record(cell, test_ap=0.9))
        + "\n"
    )
    rows = {row["cell"]: row for row in summarize.flatten(summarize.load(tmp_path))}
    assert rows[cell.name]["test_ap"] == 0.9


def test_a_delta_is_reported_against_the_matching_baseline(tmp_path):
    summarize = summarize_module()
    baseline = [c for c in CELLS if c.name == "sparkov_xgboost"][0]
    lagged = [c for c in CELLS if c.name == "sparkov_xgboost_h10"][0]
    (tmp_path / f"{baseline.name}.jsonl").write_text(
        json.dumps(fake_record(baseline, test_ap=0.40)) + "\n"
    )
    (tmp_path / f"{lagged.name}.jsonl").write_text(
        json.dumps(fake_record(lagged, test_ap=0.50)) + "\n"
    )
    rows = summarize.flatten(summarize.load(tmp_path))
    rendered = summarize.render(rows)
    assert "+0.1000" in rendered
    assert "1.25×" in rendered


def test_an_unrun_baseline_leaves_the_delta_unclaimed(tmp_path):
    """Better than inventing a comparison against a cell that has not run."""
    summarize = summarize_module()
    lagged = [c for c in CELLS if c.name == "sparkov_xgboost_h10"][0]
    (tmp_path / f"{lagged.name}.jsonl").write_text(
        json.dumps(fake_record(lagged)) + "\n"
    )
    rendered = summarize.render(summarize.flatten(summarize.load(tmp_path)))
    assert "no baseline" in rendered


def test_the_summary_never_names_roc_auc(tmp_path):
    summarize = summarize_module()
    rendered = summarize.render(summarize.flatten(summarize.load(tmp_path)))
    assert "ROC AUC" in rendered  # named only to say it is deliberately absent
    assert "average precision" in rendered.lower()


# --- deltas carry their own noise scale ----------------------------------


def delta_of(baseline_ap, baseline_sd, cell_ap, cell_sd) -> str:
    """`_delta` for one comparison. Asserted directly rather than through `render`,
    whose header explains the notation and so contains the phrases being looked for."""
    summarize = summarize_module()
    row = {"ran": True, "group": "history", "test_ap": cell_ap, "test_ap_sd": cell_sd}
    baseline = {"test_ap": baseline_ap, "test_ap_sd": baseline_sd}
    return summarize._delta(row, baseline)


def test_a_delta_inside_the_seed_spread_is_marked_within_noise():
    """IBM CCF's delay cell moved -0.0035 at 1.4 sigma with overlapping seed ranges.
    Printed as a bare 0.91x it reads as a 9% drop."""
    delta = delta_of(0.0377, 0.0023, 0.0342, 0.0012)
    assert "within noise" in delta
    assert "×" not in delta


def test_a_delta_well_outside_the_spread_reports_its_factor():
    """Sparkov's delay cell moved -0.0242 at 10 sigma, which is a real effect."""
    delta = delta_of(0.9733, 0.0014, 0.9491, 0.0021)
    assert "within noise" not in delta
    assert "0.98×" in delta and "10σ" in delta


def test_a_deterministic_pair_is_not_reported_in_sigmas():
    """Logistic returns identical scores per seed; a population sd of ~1e-19 once
    reported a delta as 1.6e15 sigma."""
    delta = delta_of(0.2356, 0.0, 0.2516, 0.0)
    assert delta == "+0.0160  (deterministic)"


def test_a_baseline_cell_has_no_delta_of_its_own():
    summarize = summarize_module()
    row = {"ran": True, "group": "baseline", "test_ap": 0.5, "test_ap_sd": 0.01}
    assert summarize._delta(row, row) == "—"


def test_the_noise_scale_combines_both_cells_spreads():
    """Seeds are independent between runs, so the variances add. Using one cell's sd
    alone would understate the spread of the comparison."""
    summarize = summarize_module()
    row = {"test_ap": 0.5, "test_ap_sd": 0.03}
    baseline = {"test_ap": 0.4, "test_ap_sd": 0.04}
    assert summarize._sigma(row, baseline) == pytest.approx(0.05)


def test_the_rendered_table_explains_the_notation_it_uses():
    """A reader meeting "1.4σ, within noise" in a cell needs the header to define it."""
    summarize = summarize_module()
    rendered = summarize.render(summarize.flatten({}))
    assert "within noise" in rendered
    assert "standard deviation" in rendered
