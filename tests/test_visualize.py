import matplotlib.pyplot as plt

from ibs_strategy.backtest import run_backtest
from ibs_strategy.optimize import grid_search, walk_forward
from ibs_strategy.visualize import (
    plot_backtest,
    plot_heatmap,
    plot_signals,
    plot_walk_forward,
)


def test_plot_backtest_smoke(scenario_frame):
    result = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0)
    fig = plot_backtest(result, ticker="TEST")
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_signals_without_trades(scenario_frame):
    result = run_backtest(scenario_frame, entry_threshold=-1.0, exit_threshold=2.0)
    ax = plot_signals(result)
    plt.close(ax.figure)


def test_plot_equity_log_scale(scenario_frame):
    from ibs_strategy.visualize import plot_equity

    result = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0)
    ax = plot_equity(result, log=True)
    assert ax.get_yscale() == "log"
    plt.close(ax.figure)


def test_plot_heatmap_smoke(random_frame):
    results = grid_search(random_frame, [0.1, 0.2], [0.8, 0.9])
    ax = plot_heatmap(results)
    plt.close(ax.figure)


def test_plot_walk_forward_smoke(random_frame):
    wf = walk_forward(random_frame, entry_grid=[0.1, 0.2], exit_grid=[0.8, 0.9], n_folds=2, purge_days=3)
    ax = plot_walk_forward(wf)
    plt.close(ax.figure)
