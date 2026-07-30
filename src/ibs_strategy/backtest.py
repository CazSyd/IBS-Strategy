"""Event-driven backtest of the IBS mean-reversion strategy.

Mechanics (a faithful port of the notebook's ``IBS_strategy``):

- The signal is the *previous* bar's IBS: below ``entry_threshold`` buys,
  above ``exit_threshold`` sells (both strict comparisons).
- Fills default to the *current* bar's open, so there is no look-ahead;
  ``entry_fill``/``exit_fill="close"`` instead transacts at the signal bar's
  close (capturing the overnight move, at the cost of close-auction execution).
- Sizing is all-in with whole shares by default; ``position_sizing="vol_target"``
  instead scales each entry to a target annualized volatility (capped at all-in,
  never levered), parking the remainder in cash. Leftover cash stays uninvested.
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
    "DEFAULT_TARGET_VOL",
    "DEFAULT_VOL_WINDOW",
    "Trade",
    "BacktestResult",
    "cash_growth_factors",
    "run_backtest",
]

REQUIRED_COLUMNS = ("Open", "Close", "IBS")

# Defaults are structural, not fitted -- an optimized threshold pair does not
# replicate out-of-sample (the surface is noise; see the README).
# Entry 0.13 buys the bottom ~12% of IBS days, where a replicating forward-return
# edge lives; anywhere in 0.10-0.20 is equivalent.
# Exit 0.5 is a prompt, zero-degrees-of-freedom exit at the neutral midpoint: the
# IBS edge is front-loaded (largely spent within a day, gone within three), so a
# prompt exit harvests it and steps aside, keeping the strategy in cash ~80% of
# the time and truncating crashes. A stop loss only makes a mean-reversion system
# worse. The crash-free 2010+ window alone would favour a patient 0.965 exit that
# then draws down 99% through 2000-2002.
DEFAULT_ENTRY_THRESHOLD = 0.13
DEFAULT_EXIT_THRESHOLD = 0.5

# Vol-target sizing defaults: scale each entry to ~40% annualized instrument
# volatility over a 20-bar trailing window. This is the risk-reducing sweet spot
# from the analysis (halves the crash drawdown at a higher Sharpe); it is the
# signal page's default, though run_backtest itself stays all-in ("full").
DEFAULT_TARGET_VOL = 0.40
DEFAULT_VOL_WINDOW = 20


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
    position_sizing: str = "full"
    target_vol: float = DEFAULT_TARGET_VOL
    vol_window: int = DEFAULT_VOL_WINDOW
    weights: pd.Series | None = None  # per-bar entry weight under vol_target, else None

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
    entry_fill: str = "open",
    exit_fill: str = "open",
    position_sizing: str = "full",
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_window: int = DEFAULT_VOL_WINDOW,
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

    ``entry_fill`` / ``exit_fill`` choose where each leg fills: ``"open"`` (the
    default) acts on the *previous* bar's signal at the next open -- the clean,
    liquid, no-look-ahead convention; ``"close"`` acts on *this* bar's signal at
    its own close. A close entry captures the overnight move into the next
    session (the IBS reversion begins overnight), but assumes you can transact at
    the close -- in practice a market-on-close order, whose fill you cannot know
    when you submit it. See the README on the resulting execution slippage.

    ``position_sizing`` defaults to ``"full"`` (all-in whole shares). Set it to
    ``"vol_target"`` to scale each entry by the instrument's trailing
    ``vol_window``-bar realized volatility: the deployed fraction is
    ``min(1, target_vol / realized_vol)`` (annualized), capped at all-in so the
    position is never levered, with the remainder held as cash. This cuts the
    crash-tail drawdown at a higher risk-adjusted return than a smaller constant
    size would; entry/exit *timing* is unchanged. See the README.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError("need at least two bars to backtest")
    for name, value in (("entry_fill", entry_fill), ("exit_fill", exit_fill)):
        if value not in ("open", "close"):
            raise ValueError(f"{name} must be 'open' or 'close', got {value!r}")
    if position_sizing not in ("full", "vol_target"):
        raise ValueError(
            f"position_sizing must be 'full' or 'vol_target', got {position_sizing!r}"
        )
    if position_sizing == "vol_target":
        if not target_vol > 0:
            raise ValueError(f"target_vol must be positive, got {target_vol!r}")
        if vol_window < 2:
            raise ValueError(f"vol_window must be at least 2, got {vol_window!r}")

    df = data.copy()
    cash_factors = cash_growth_factors(df, cash_rate)
    open_prices = df["Open"].to_numpy(dtype=float)
    close_prices = df["Close"].to_numpy(dtype=float)
    ibs = df["IBS"].to_numpy(dtype=float)
    n = len(df)

    if position_sizing == "vol_target":
        returns = df["Close"].pct_change()
        realized_vol = (
            returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.clip(target_vol / realized_vol, 0.0, 1.0)
        # undefined early vol (< vol_window bars) and flat stretches fall back to all-in
        weights = np.where(np.isfinite(realized_vol) & (realized_vol > 0.0), weights, 1.0)
    else:
        weights = np.ones(n, dtype=float)

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
        close_price = close_prices[i]
        cash *= cash_factors[i]  # idle cash earns overnight, before today's fill

        # fills at the OPEN act on the *previous* bar's signal (no look-ahead)
        if entry_fill == "open" and not in_position and prev_regime and prev_ibs < entry_threshold:
            shares = int(weights[i - 1] * cash // open_price)
            if shares > 0:
                cash -= shares * open_price
                in_position = True
                entry_date = df.index[i]
                entry_price = open_price
        elif exit_fill == "open" and in_position and (
            prev_ibs > exit_threshold or (regime_exit and not prev_regime)
        ):
            cash += shares * open_price
            trades.append(Trade(entry_date, entry_price, shares, df.index[i], open_price))
            shares = 0
            in_position = False

        # fills at the CLOSE act on *this* bar's signal, transacting at its close
        if entry_fill == "close" and not in_position and regime_ok[i] and ibs[i] < entry_threshold:
            shares = int(weights[i] * cash // close_price)
            if shares > 0:
                cash -= shares * close_price
                in_position = True
                entry_date = df.index[i]
                entry_price = close_price
        elif exit_fill == "close" and in_position and (
            ibs[i] > exit_threshold or (regime_exit and not regime_ok[i])
        ):
            cash += shares * close_price
            trades.append(Trade(entry_date, entry_price, shares, df.index[i], close_price))
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
        position_sizing=position_sizing,
        target_vol=float(target_vol),
        vol_window=int(vol_window),
        weights=(
            pd.Series(weights, index=df.index) if position_sizing == "vol_target" else None
        ),
    )
