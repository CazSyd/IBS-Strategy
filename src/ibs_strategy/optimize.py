"""Threshold selection: in-sample grid search and purged walk-forward validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import REQUIRED_COLUMNS, BacktestResult, Trade, run_backtest
from .metrics import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)

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

# Fine 0.001-step sweeps over the sensible bands: entry in (0, 0.20], exit in
# [0.80, 1.0). 40,000 pairs total -- tractable thanks to the vectorized replay
# in ``_fast_summary``.
DEFAULT_ENTRY_GRID: np.ndarray = np.round(0.001 * np.arange(1, 201), 3)
DEFAULT_EXIT_GRID: np.ndarray = np.round(0.8 + 0.001 * np.arange(0, 200), 3)

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


def _fast_inputs(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    open_prices = data["Open"].to_numpy(dtype=float)
    close_prices = data["Close"].to_numpy(dtype=float)
    ibs = data["IBS"].to_numpy(dtype=float)
    prev_ibs = np.empty_like(ibs)
    prev_ibs[0] = np.nan
    prev_ibs[1:] = ibs[:-1]
    if isinstance(data.index, pd.DatetimeIndex):
        years = (data.index[-1] - data.index[0]).days / 365.25
    else:
        years = (len(data) - 1) / TRADING_DAYS_PER_YEAR
    return open_prices, close_prices, prev_ibs, years


def _fast_summary(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    prev_ibs: np.ndarray,
    years: float,
    entry_threshold: float,
    exit_threshold: float,
    initial_capital: float,
) -> dict | None:
    """O(n) vectorized replay of ``run_backtest``, bit-identical in its metrics.

    Valid only when ``entry_threshold <= exit_threshold`` (buy and sell signals
    are then mutually exclusive, so the position is a set/reset latch on the
    most recent signal). Returns None when a fill cannot afford a single share;
    the caller falls back to the exact engine for that pair.
    """
    n = open_prices.size
    signal = np.zeros(n, dtype=np.int8)
    signal[prev_ibs < entry_threshold] = 1
    signal[prev_ibs > exit_threshold] = -1
    signal[0] = 0
    event_index = np.where(signal != 0, np.arange(n), 0)
    np.maximum.accumulate(event_index, out=event_index)
    in_position = signal[event_index] == 1
    was_in_position = np.empty_like(in_position)
    was_in_position[0] = False
    was_in_position[1:] = in_position[:-1]
    entry_bars = np.flatnonzero(in_position & ~was_in_position)
    exit_bars = np.flatnonzero(~in_position & was_in_position)

    cash_curve = np.empty(n, dtype=float)
    shares_curve = np.zeros(n, dtype=float)
    cash = float(initial_capital)
    wins = 0
    segment_start = 0
    for k, entry_bar in enumerate(entry_bars):
        exit_bar = int(exit_bars[k]) if k < exit_bars.size else n
        shares = int(cash // open_prices[entry_bar])
        if shares == 0:
            return None
        cash_curve[segment_start:entry_bar] = cash
        cash -= shares * open_prices[entry_bar]
        cash_curve[entry_bar:exit_bar] = cash
        shares_curve[entry_bar:exit_bar] = shares
        if exit_bar == n:
            segment_start = n
            break
        if open_prices[exit_bar] > open_prices[entry_bar]:
            wins += 1
        cash += shares * open_prices[exit_bar]
        segment_start = exit_bar
    cash_curve[segment_start:] = cash

    equity = cash_curve + shares_curve * close_prices
    returns = np.empty(n, dtype=float)
    returns[0] = 0.0
    returns[1:] = equity[1:] / equity[:-1] - 1
    std = returns.std(ddof=1) if n > 1 else float("nan")
    if std == 0 or math.isnan(std):
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))
    total = float(equity[-1] / initial_capital - 1)
    closed = int(exit_bars.size)
    return {
        "entry_threshold": float(entry_threshold),
        "exit_threshold": float(exit_threshold),
        "sharpe": sharpe,
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "win_rate": wins / closed if closed else 0.0,
        "num_trades": closed,
    }


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

    Non-overlapping pairs (entry <= exit, i.e. every pair in the default
    grids) run through a vectorized replay of the engine; anything else falls
    back to ``run_backtest``, so results are identical either way.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose one of {OBJECTIVES}")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError("need at least two bars to backtest")
    entry_grid = DEFAULT_ENTRY_GRID if entry_grid is None else np.asarray(entry_grid, dtype=float)
    exit_grid = DEFAULT_EXIT_GRID if exit_grid is None else np.asarray(exit_grid, dtype=float)

    open_prices, close_prices, prev_ibs, years = _fast_inputs(data)
    rows = []
    for entry_threshold in entry_grid:
        for exit_threshold in exit_grid:
            summary = None
            if entry_threshold <= exit_threshold:
                summary = _fast_summary(
                    open_prices, close_prices, prev_ibs, years,
                    entry_threshold, exit_threshold, initial_capital,
                )
            if summary is None:
                full = run_backtest(data, entry_threshold, exit_threshold, initial_capital).summary()
                summary = {column: full[column] for column in _RESULT_COLUMNS}
            rows.append(summary)

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
