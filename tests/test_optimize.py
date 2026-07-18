import numpy as np
import pytest

from ibs_strategy.optimize import (
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    best_thresholds,
    grid_search,
    walk_forward,
)


def test_default_grids_match_notebook():
    assert DEFAULT_ENTRY_GRID.tolist() == pytest.approx(np.arange(0.01, 0.20, 0.02).tolist())
    assert DEFAULT_EXIT_GRID.tolist() == pytest.approx(np.arange(0.81, 1.00, 0.02).tolist())


def test_grid_search_covers_grid_and_sorts(random_frame):
    entry_grid = [0.1, 0.2, 0.3]
    exit_grid = [0.8, 0.9]
    results = grid_search(random_frame, entry_grid, exit_grid, objective="total_return")
    assert len(results) == 6
    assert set(zip(results["entry_threshold"], results["exit_threshold"])) == {
        (entry, exit_) for entry in entry_grid for exit_ in exit_grid
    }
    values = results["total_return"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)
    assert best_thresholds(results) == (
        results.iloc[0]["entry_threshold"],
        results.iloc[0]["exit_threshold"],
    )


def test_grid_search_by_cagr(random_frame):
    results = grid_search(random_frame, [0.1, 0.2], [0.8, 0.9], objective="cagr")
    assert "cagr" in results.columns
    values = results["cagr"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)


def test_grid_search_rejects_unknown_objective(random_frame):
    with pytest.raises(ValueError, match="objective"):
        grid_search(random_frame, [0.1], [0.9], objective="alpha")


def test_walk_forward_folds_are_purged_and_sequential(random_frame):
    wf = walk_forward(random_frame, entry_grid=[0.1, 0.2], exit_grid=[0.8, 0.9], n_folds=3, purge_days=5)
    assert len(wf.folds) == 3

    index = random_frame.index
    for fold in wf.folds:
        gap = index.get_loc(fold.test_start) - index.get_loc(fold.train_end) - 1
        assert gap >= 5
        assert fold.train_start == index[0]  # anchored (expanding) training window
    for previous, current in zip(wf.folds, wf.folds[1:]):
        assert current.test_start > previous.test_end

    first_test = int(len(random_frame) * 0.5)
    assert wf.oos_equity.index.equals(index[first_test:])
    assert wf.oos_equity.iloc[0] == pytest.approx(10_000.0)
    assert not wf.oos_equity.isna().any()

    summary = wf.summary()
    for key in (
        "sharpe",
        "total_return",
        "max_drawdown",
        "win_rate",
        "num_trades",
        "cagr",
        "exposure",
        "final_capital",
    ):
        assert key in summary

    table = wf.fold_table()
    assert table["fold"].tolist() == [1, 2, 3]


def test_walk_forward_validates_inputs(random_frame):
    with pytest.raises(ValueError, match="n_folds"):
        walk_forward(random_frame, n_folds=0)
    with pytest.raises(ValueError, match="training data"):
        walk_forward(random_frame, min_train_frac=0.05)
    with pytest.raises(ValueError, match="purge_days"):
        walk_forward(random_frame, purge_days=-1)
    with pytest.raises(ValueError, match="min_train_frac"):
        walk_forward(random_frame, min_train_frac=1.5)
