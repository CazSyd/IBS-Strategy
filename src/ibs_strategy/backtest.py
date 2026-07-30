"""Event-driven backtest of the IBS mean-reversion strategy.

Mechanics (a faithful port of the notebook's ``IBS_strategy``):

- The signal is the *previous* bar's IBS: below ``entry_threshold`` buys,
  above ``exit_threshold`` sells (both strict comparisons).
- Fills happen at the *current* bar's open, so there is no look-ahead.
- Sizing is all-in with whole shares; leftover cash stays uninvested.
- Equity is marked to market at each bar's close. No commissions or slippage.
- Idle cash earns ``cash_rate`` (annualized, accrued per trading bar). It
  defaults to zero -- the notebook's assumption -- but a strategy that sits
  flat most of the time is materially understated at 0%, so the CLI passes a
  real T-bill series by default.
- An optional ``regime`` flag (e.g. index close above its 200-day SMA) gates
  entries, and with ``regime_exit`` also liquidates when the flag drops. Like
  IBS, the flag is read from the *previous* bar and acted on at today's open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)

__all__ = [
    "DEFAULT_ENTRY_THRESHOLD",
    "DEFAULT_EXIT_THRESHOLD",
    "Trade",
    "BacktestResult",
    "cash_growth_factors",
    "run_backtest",
]

REQUIRED_COLUMNS = ("Open", "Close", "IBS")

# Crash-aware thresholds: the flat region that holds up on BOTH TQQQ (1999+)
# and SPXL (1993+) once the dot-com and GFC crashes are in the sample, scored
# by the worse of the two Sharpe ratios rather than by CAGR. Deliberately round
# -- the surrounding plateau is flat enough that a third digit would be noise.
# The crash-free 2010+ window instead favours a patient 0.965 exit (see the
# README): that pick draws down 99% through 2000-2002, which is why it is no
# longer the default.
DEFAULT_ENTRY_THRESHOLD = 0.13
DEFAULT_EXIT_THRESHOLD = 0.80


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


def cash_growth_factors(
    data: pd.DataFrame,
    cash_rate: pd.Series | float | None,
) -> np.ndarray:
    """Per-bar growth factors for idle cash from an annualized ``cash_rate``.

    Accepts a scalar rate, a Series (aligned on ``data``'s index and forward
    filled over market holidays), or None/0 for the no-interest case. Bar 0 is
    always 1.0: interest accrues *between* bars.
    """
    factors = np.ones(len(data), dtype=float)
    if cash_rate is None:
        return factors
    if isinstance(cash_rate, pd.Series):
        rates = cash_rate.reindex(data.index).ffill().bfill().to_numpy(dtype=float)
        rates = np.nan_to_num(rates, nan=0.0)
    else:
        rates = np.full(len(data), float(cash_rate))
    factors[1:] = 1.0 + rates[1:] / TRADING_DAYS_PER_YEAR
    return factors


def run_backtest(
    data: pd.DataFrame,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
    initial_capital: float = 10_000.0,
    cash_rate: pd.Series | float | None = None,
    regime: pd.Series | None = None,
    regime_exit: bool = False,
) -> BacktestResult:
    """Run the IBS strategy over ``data`` (requires Open, Close and IBS columns).

    ``cash_rate`` is an annualized yield on idle cash -- a scalar, a Series
    aligned on ``data``'s index, or None for the notebook's 0% assumption.
    Interest accrues on the cash balance *before* the bar's fill, so a day
    spent fully invested earns none.

    ``regime`` is an optional boolean Series (aligned on ``data``'s index;
    missing dates count as off). Entries require the *previous* bar's flag to
    be on -- the same no-look-ahead timing as the IBS signal -- and with
    ``regime_exit=True`` an open position is also sold at the next open once
    the flag turns off.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError("need at least two bars to backtest")

    df = data.copy()
    cash_factors = cash_growth_factors(df, cash_rate)
    open_prices = df["Open"].to_numpy(dtype=float)
    close_prices = df["Close"].to_numpy(dtype=float)
    ibs = df["IBS"].to_numpy(dtype=float)
    n = len(df)

    if regime is None:
        regime_ok = np.ones(n, dtype=bool)
    elif isinstance(regime, pd.Series):
        regime_ok = regime.reindex(df.index).fillna(False).astype(bool).to_numpy()
    else:
        regime_ok = np.asarray(regime, dtype=bool)
        if regime_ok.shape != (n,):
            raise ValueError("regime must provide one flag per bar of data")

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
        prev_regime = regime_ok[i - 1]
        open_price = open_prices[i]
        cash *= cash_factors[i]  # idle cash earns overnight, before today's fill

        if not in_position and prev_regime and prev_ibs < entry_threshold:
            shares = int(cash // open_price)
            if shares > 0:
                cash -= shares * open_price
                in_position = True
                entry_date = df.index[i]
                entry_price = open_price
        elif in_position and (prev_ibs > exit_threshold or (regime_exit and not prev_regime)):
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

    if regime is not None:
        df["Regime"] = regime_ok
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
