"""Event-driven backtest of the IBS mean-reversion strategy.

Mechanics (a faithful port of the notebook's ``IBS_strategy``):

- The signal is the *previous* bar's IBS: below ``entry_threshold`` buys,
  above ``exit_threshold`` sells (both strict comparisons).
- Fills happen at the *current* bar's open, so there is no look-ahead.
- Sizing is all-in with whole shares; leftover cash stays uninvested.
- Equity is marked to market at each bar's close. No commissions or slippage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import cagr, max_drawdown, sharpe_ratio, total_return, win_rate

__all__ = [
    "DEFAULT_ENTRY_THRESHOLD",
    "DEFAULT_EXIT_THRESHOLD",
    "Trade",
    "BacktestResult",
    "run_backtest",
]

REQUIRED_COLUMNS = ("Open", "Close", "IBS")

# Whole-listing-period CAGR optimum on TQQQ (2010-2026) from the 0.001-step
# grid; the original notebook's in-sample pick was 0.19/0.95.
DEFAULT_ENTRY_THRESHOLD = 0.132
DEFAULT_EXIT_THRESHOLD = 0.965


@dataclass(frozen=True)
class Trade:
    """A round trip, or the final open position when ``exit_date`` is None."""

    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def is_win(self) -> bool:
        return self.exit_price is not None and self.exit_price > self.entry_price

    @property
    def return_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    """Backtest output: the enriched bar data, the trade list, and the inputs."""

    data: pd.DataFrame
    trades: list[Trade]
    entry_threshold: float
    exit_threshold: float
    initial_capital: float

    @property
    def equity(self) -> pd.Series:
        return self.data["Capital"]

    @property
    def returns(self) -> pd.Series:
        return self.data["Strategy Return"]

    @property
    def closed_trades(self) -> list[Trade]:
        return [trade for trade in self.trades if not trade.is_open]

    def summary(self) -> dict[str, float]:
        return {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "sharpe": sharpe_ratio(self.returns),
            "total_return": total_return(self.equity),
            "max_drawdown": max_drawdown(self.equity),
            "win_rate": win_rate(self.trades),
            "num_trades": len(self.closed_trades),
            "cagr": cagr(self.equity),
            "exposure": float(self.data["Position"].mean()),
            "final_capital": float(self.equity.iloc[-1]),
        }


def run_backtest(
    data: pd.DataFrame,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Run the IBS strategy over ``data`` (requires Open, Close and IBS columns)."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError("need at least two bars to backtest")

    df = data.copy()
    open_prices = df["Open"].to_numpy(dtype=float)
    close_prices = df["Close"].to_numpy(dtype=float)
    ibs = df["IBS"].to_numpy(dtype=float)
    n = len(df)

    position = np.zeros(n, dtype=int)
    shares_held = np.zeros(n, dtype=float)
    cash_held = np.zeros(n, dtype=float)
    capital = np.zeros(n, dtype=float)

    cash = float(initial_capital)
    shares = 0
    in_position = False
    entry_date: pd.Timestamp | None = None
    entry_price = 0.0
    trades: list[Trade] = []

    cash_held[0] = cash
    capital[0] = cash

    for i in range(1, n):
        prev_ibs = ibs[i - 1]  # NaN (High == Low bar) compares False both ways -> hold
        open_price = open_prices[i]

        if not in_position and prev_ibs < entry_threshold:
            shares = int(cash // open_price)
            if shares > 0:
                cash -= shares * open_price
                in_position = True
                entry_date = df.index[i]
                entry_price = open_price
        elif in_position and prev_ibs > exit_threshold:
            cash += shares * open_price
            trades.append(Trade(entry_date, entry_price, shares, df.index[i], open_price))
            shares = 0
            in_position = False

        position[i] = int(in_position)
        shares_held[i] = shares
        cash_held[i] = cash
        capital[i] = cash + shares * close_prices[i]

    if in_position:
        trades.append(Trade(entry_date, entry_price, shares))

    df["Position"] = position
    df["Shares"] = shares_held
    df["Cash"] = cash_held
    df["Capital"] = capital
    df["Strategy Return"] = df["Capital"].pct_change().fillna(0)

    return BacktestResult(
        data=df,
        trades=trades,
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        initial_capital=float(initial_capital),
    )
