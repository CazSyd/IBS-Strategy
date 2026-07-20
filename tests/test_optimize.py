import numpy as np
import pandas as pd
import pytest

from ibs_strategy.backtest import run_backtest
from ibs_strategy.optimize import (
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    best_thresholds,
    grid_search,
    objective_surface,
    plateau_thresholds,
    walk_forward,
)


def test_default_grids_are_fine_grained():
    assert len(DEFAULT_ENTRY_GRID) == 200
    assert DEFAULT_ENTRY_GRID[0] == pytest.approx(0.001)
    assert DEFAULT_ENTRY_GRID[-1] == pytest.approx(0.2)
    assert np.diff(DEFAULT_ENTRY_GRID) == pytest.approx(np.full(199, 0.001))
    assert len(DEFAULT_EXIT_GRID) == 200
    assert DEFAULT_EXIT_GRID[0] == pytest.approx(0.8)
    assert DEFAULT_EXIT_GRID[-1] == pytest.approx(0.999)
    assert np.diff(DEFAULT_EXIT_GRID) == pytest.approx(np.full(199, 0.001))


def test_grid_search_matches_engine_exactly(random_frame, scenario_frame):
    for entry, exit_ in [(0.05, 0.85), (0.1, 0.9), (0.2, 0.8), (0.13, 0.97)]:
        row = grid_search(random_frame, [entry], [exit_]).iloc[0]
        expected = run_backtest(random_frame, entry, exit_).summary()
        for column in ("sharpe", "total_return", "cagr", "max_drawdown", "win_rate", "num_trades"):
            assert row[column] == pytest.approx(expected[column], rel=1e-9, abs=1e-12), column

    # covers NaN IBS bars and a position still open at the end of the data
    row = grid_search(scenario_frame, [0.2], [0.9], initial_capital=1_000.0).iloc[0]
    expected = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0).summary()
    for column in ("sharpe", "total_return", "cagr", "max_drawdown", "win_rate", "num_trades"):
        assert row[column] == pytest.approx(expected[column], rel=1e-9, abs=1e-12), column


def test_grid_search_matches_engine_exactly_with_cash_interest(random_frame):
    """The vectorized replay compounds idle cash the same way the loop does."""
    rate = pd.Series(0.05, index=random_frame.index)
    for entry, exit_ in [(0.05, 0.85), (0.13, 0.97)]:
        row = grid_search(random_frame, [entry], [exit_], cash_rate=rate).iloc[0]
        expected = run_backtest(random_frame, entry, exit_, cash_rate=rate).summary()
        for column in ("sharpe", "total_return", "cagr", "max_drawdown", "win_rate", "num_trades"):
            assert row[column] == pytest.approx(expected[column], rel=1e-9, abs=1e-12), column

    # interest must actually change the answer, or the test proves nothing
    plain = grid_search(random_frame, [0.05], [0.85]).iloc[0]
    earning = grid_search(random_frame, [0.05], [0.85], cash_rate=rate).iloc[0]
    assert earning["total_return"] > plain["total_return"]


def test_plateau_thresholds_ignores_an_isolated_spike():
    """A lone noise spike must lose to the centre of a broad, slightly lower plateau."""
    # index arithmetic, not float comparisons: `abs(0.87 - 0.85) <= 0.02` is
    # False in binary floating point and would skew the plateau off-centre
    rows = []
    for i in range(11):
        for j in range(11):
            on_plateau = abs(i - 5) <= 2 and abs(j - 5) <= 2  # 5x5 block of 0.30
            cagr = 0.30 if on_plateau else 0.10
            if (i, j) == (10, 10):
                cagr = 0.45  # taller, but a single cell surrounded by 0.10
            rows.append({
                "entry_threshold": round(0.10 + 0.01 * i, 2),
                "exit_threshold": round(0.80 + 0.01 * j, 2),
                "cagr": cagr,
                "sharpe": cagr,
            })
    results = pd.DataFrame(rows).sort_values("cagr", ascending=False, ignore_index=True)

    assert best_thresholds(results) == (0.20, 0.90)  # argmax takes the spike
    assert plateau_thresholds(results, "cagr", radius=0.02) == (0.15, 0.85)


def test_objective_surface_is_an_entry_by_exit_grid(random_frame):
    results = grid_search(random_frame, [0.1, 0.2], [0.8, 0.9], objective="cagr")
    surface = objective_surface(results, "cagr")
    assert surface.shape == (2, 2)
    assert surface.index.tolist() == [0.1, 0.2]
    assert surface.columns.tolist() == [0.8, 0.9]


def test_grid_search_overlapping_thresholds_fall_back(random_frame):
    # entry > exit means buy and sell signals can fire on the same bar; the
    # fast latch replay is invalid there, so the exact engine must be used
    row = grid_search(random_frame, [0.9], [0.5], objective="sharpe").iloc[0]
    expected = run_backtest(random_frame, 0.9, 0.5).summary()
    for column in ("sharpe", "total_return", "max_drawdown", "win_rate", "num_trades"):
        assert row[column] == pytest.approx(expected[column]), column


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
