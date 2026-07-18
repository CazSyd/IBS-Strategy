"""IBS (Internal Bar Strength) mean-reversion strategy toolkit.

Package port of the original Colab notebook: data loading, an event-driven
backtester, notebook-identical metrics, threshold grid search, purged
walk-forward validation, matplotlib charts, and a live signal check.
"""

from .backtest import (
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_EXIT_THRESHOLD,
    BacktestResult,
    Trade,
    run_backtest,
)
from .data import compute_ibs, flatten_columns, load_data
from .live import SignalReport, latest_signal, signal_from_frame
from .metrics import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from .optimize import (
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    OBJECTIVES,
    Fold,
    WalkForwardResult,
    best_thresholds,
    grid_search,
    walk_forward,
)
from .visualize import (
    plot_backtest,
    plot_drawdown,
    plot_equity,
    plot_heatmap,
    plot_signals,
    plot_walk_forward,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_ENTRY_THRESHOLD",
    "DEFAULT_EXIT_THRESHOLD",
    "BacktestResult",
    "Trade",
    "run_backtest",
    "compute_ibs",
    "flatten_columns",
    "load_data",
    "SignalReport",
    "latest_signal",
    "signal_from_frame",
    "TRADING_DAYS_PER_YEAR",
    "cagr",
    "drawdown_series",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "win_rate",
    "DEFAULT_ENTRY_GRID",
    "DEFAULT_EXIT_GRID",
    "OBJECTIVES",
    "Fold",
    "WalkForwardResult",
    "best_thresholds",
    "grid_search",
    "walk_forward",
    "plot_backtest",
    "plot_drawdown",
    "plot_equity",
    "plot_heatmap",
    "plot_signals",
    "plot_walk_forward",
]
