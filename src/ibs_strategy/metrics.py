"""Performance metrics, matching the original notebook's definitions.

Sharpe uses a zero risk-free rate and the sample standard deviation (ddof=1)
of *all* daily strategy returns -- including flat days in cash -- annualized
with sqrt(252).
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "sharpe_ratio",
    "total_return",
    "drawdown_series",
    "max_drawdown",
    "cagr",
    "win_rate",
]

TRADING_DAYS_PER_YEAR = 252


class SupportsWin(Protocol):
    @property
    def is_open(self) -> bool: ...

    @property
    def is_win(self) -> bool: ...


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio; 0.0 when the return series has no variance."""
    std = returns.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def total_return(equity: pd.Series) -> float:
    """Fractional return of the equity curve from first to last value."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each point (<= 0)."""
    cumulative = equity / equity.iloc[0]
    return cumulative / cumulative.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Deepest peak-to-trough drawdown of the equity curve (a negative number)."""
    return float(drawdown_series(equity).min())


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate; falls back to 252-bar years without dates."""
    if isinstance(equity.index, pd.DatetimeIndex):
        years = (equity.index[-1] - equity.index[0]).days / 365.25
    else:
        years = (len(equity) - 1) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def win_rate(trades: Iterable[SupportsWin]) -> float:
    """Share of closed trades that exited above their entry price.

    Matches the notebook: a win compares exit fill to entry fill, and a
    position still open at the end of the data is not counted either way.
    """
    closed = [trade for trade in trades if not trade.is_open]
    if not closed:
        return 0.0
    return sum(trade.is_win for trade in closed) / len(closed)
