"""Threshold selection: in-sample grid search and purged walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Trade, run_backtest
from .metrics import cagr, max_drawdown, sharpe_ratio, total_return, win_rate

__all__ = [
    "DEFAULT_ENTRY_GRID",
    "DEFAULT_EXIT_GRID",
    "OBJECTIVES",
    "grid_search",
    "best_thresholds",
    "Fold",
    "WalkForwardResult",
    "walk_forward",
]

DEFAULT_ENTRY_GRID: np.ndarray = np.round(np.arange(0.01, 0.20, 0.02), 2)
DEFAULT_EXIT_GRID: np.ndarray = np.round(np.arange(0.81, 1.00, 0.02), 2)

OBJECTIVES = ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate")

_RESULT_COLUMNS = [
    "entry_threshold",
    "exit_threshold",
    "sharpe",
    "total_return",
    "cagr",
    "max_drawdown",
    "win_rate",
    "num_trades",
]

MIN_TRAIN_BARS = 30


def grid_search(
    data: pd.DataFrame,
    entry_grid: np.ndarray | None = None,
    exit_grid: np.ndarray | None = None,
    objective: str = "total_return",
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Backtest every (entry, exit) pair and rank the results, best row first.

    Rows are sorted by ``objective`` descending with Sharpe as the tiebreaker,
    mirroring the notebook's ranking. ``max_drawdown`` values are negative, so
    descending order still puts the shallowest drawdown first.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose one of {OBJECTIVES}")
    entry_grid = DEFAULT_ENTRY_GRID if entry_grid is None else np.asarray(entry_grid, dtype=float)
    exit_grid = DEFAULT_EXIT_GRID if exit_grid is None else np.asarray(exit_grid, dtype=float)

    rows = []
    for entry_threshold in entry_grid:
        for exit_threshold in exit_grid:
            result = run_backtest(data, entry_threshold, exit_threshold, initial_capital)
            summary = result.summary()
            rows.append({column: summary[column] for column in _RESULT_COLUMNS})

    results = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    tiebreaker = "sharpe" if objective != "sharpe" else "total_return"
    return results.sort_values([objective, tiebreaker], ascending=False, ignore_index=True)


def best_thresholds(results: pd.DataFrame) -> tuple[float, float]:
    """The (entry, exit) pair from the top row of a ``grid_search`` table."""
    top = results.iloc[0]
    return float(top["entry_threshold"]), float(top["exit_threshold"])


@dataclass
class Fold:
    """One walk-forward fold: thresholds fitted on train, evaluated on test."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    entry_threshold: float
    exit_threshold: float
    train_objective: float
    result: BacktestResult


@dataclass
class WalkForwardResult:
    """All folds plus the stitched out-of-sample equity curve."""

    folds: list[Fold]
    oos_equity: pd.Series
    objective: str
    initial_capital: float

    @property
    def trades(self) -> list[Trade]:
        return [trade for fold in self.folds for trade in fold.result.trades]

    def summary(self) -> dict[str, float]:
        returns = self.oos_equity.pct_change().fillna(0)
        closed = [trade for trade in self.trades if not trade.is_open]
        position = pd.concat([fold.result.data["Position"] for fold in self.folds])
        return {
            "sharpe": sharpe_ratio(returns),
            "total_return": total_return(self.oos_equity),
            "max_drawdown": max_drawdown(self.oos_equity),
            "win_rate": win_rate(self.trades),
            "num_trades": len(closed),
            "cagr": cagr(self.oos_equity),
            "exposure": float(position.mean()),
            "final_capital": float(self.oos_equity.iloc[-1]),
        }

    def fold_table(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            summary = fold.result.summary()
            rows.append(
                {
                    "fold": fold.number,
                    "train_start": fold.train_start.date(),
                    "train_end": fold.train_end.date(),
                    "test_start": fold.test_start.date(),
                    "test_end": fold.test_end.date(),
                    "entry_threshold": fold.entry_threshold,
                    "exit_threshold": fold.exit_threshold,
                    "sharpe": summary["sharpe"],
                    "total_return": summary["total_return"],
                    "cagr": summary["cagr"],
                    "max_drawdown": summary["max_drawdown"],
                    "win_rate": summary["win_rate"],
                    "num_trades": summary["num_trades"],
                }
            )
        return pd.DataFrame(rows)


def walk_forward(
    data: pd.DataFrame,
    entry_grid: np.ndarray | None = None,
    exit_grid: np.ndarray | None = None,
    n_folds: int = 5,
    min_train_frac: float = 0.5,
    purge_days: int = 5,
    objective: str = "total_return",
    initial_capital: float = 10_000.0,
) -> WalkForwardResult:
    """Purged, anchored walk-forward validation of the threshold grid search.

    The first ``min_train_frac`` of the bars seeds the training window and the
    remainder is split into ``n_folds`` sequential test windows. For each fold,
    thresholds are chosen by ``grid_search`` on all bars *before* the test
    window minus a ``purge_days`` gap -- the purge drops the training bars whose
    next-open fills and still-open positions would otherwise bleed information
    across the boundary. The chosen thresholds are then evaluated once on the
    unseen test window (each fold starts flat, in cash), and the per-fold
    equity segments are compounded into ``oos_equity``.
    """
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must be between 0 and 1 (exclusive)")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if purge_days < 0:
        raise ValueError("purge_days cannot be negative")

    n = len(data)
    first_test = int(n * min_train_frac)
    if first_test - purge_days < MIN_TRAIN_BARS:
        raise ValueError(
            f"not enough training data: {first_test - purge_days} bars before the first "
            f"test window (need at least {MIN_TRAIN_BARS}); add history or lower purge_days"
        )
    bounds = np.linspace(first_test, n, n_folds + 1).astype(int)
    if np.any(np.diff(bounds) < 2):
        raise ValueError("test folds are too small; reduce n_folds or add history")

    folds: list[Fold] = []
    segments: list[pd.Series] = []
    level = float(initial_capital)

    for k in range(n_folds):
        test_lo, test_hi = int(bounds[k]), int(bounds[k + 1])
        train = data.iloc[: test_lo - purge_days]
        test = data.iloc[test_lo:test_hi]

        ranked = grid_search(train, entry_grid, exit_grid, objective, initial_capital)
        top = ranked.iloc[0]
        entry_threshold = float(top["entry_threshold"])
        exit_threshold = float(top["exit_threshold"])

        result = run_backtest(test, entry_threshold, exit_threshold, initial_capital)
        folds.append(
            Fold(
                number=k + 1,
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                train_objective=float(top[objective]),
                result=result,
            )
        )

        segment = result.equity / result.equity.iloc[0] * level
        level = float(segment.iloc[-1])
        segments.append(segment)

    oos_equity = pd.concat(segments).rename("OOS Capital")
    return WalkForwardResult(
        folds=folds,
        oos_equity=oos_equity,
        objective=objective,
        initial_capital=float(initial_capital),
    )
