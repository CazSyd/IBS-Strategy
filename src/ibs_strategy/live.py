"""Live signal check for the most recent completed bar.

Port of the notebook's ``enter_position``: download recent history, compute
IBS, and classify the latest *completed* session -- skipping today's bar when
the market has not yet closed (4pm America/New_York).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pandas as pd
import pytz

from .backtest import DEFAULT_ENTRY_THRESHOLD, DEFAULT_EXIT_THRESHOLD
from .data import load_data

__all__ = [
    "MARKET_TZ",
    "MARKET_CLOSE",
    "SignalReport",
    "classify",
    "signal_from_frame",
    "latest_signal",
]

MARKET_TZ = pytz.timezone("America/New_York")
MARKET_CLOSE = time(16, 0)


@dataclass(frozen=True)
class SignalReport:
    ticker: str
    bar_date: pd.Timestamp
    ibs: float
    signal: str  # "BUY" | "SELL" | "HOLD"
    entry_threshold: float
    exit_threshold: float
    data: pd.DataFrame

    @property
    def message(self) -> str:
        base = f"{self.ticker}: IBS {self.ibs:.3f} on {self.bar_date:%Y-%m-%d}"
        if self.signal == "BUY":
            return f"{base} is below {self.entry_threshold:g} -> BUY at the next open"
        if self.signal == "SELL":
            return f"{base} is above {self.exit_threshold:g} -> SELL at the next open"
        return f"{base} -> no action"


def classify(ibs: float, entry_threshold: float, exit_threshold: float) -> str:
    """BUY below the entry threshold, SELL above the exit threshold, else HOLD.

    Comparisons are strict, and NaN (a High == Low bar) never signals.
    """
    if ibs < entry_threshold:
        return "BUY"
    if ibs > exit_threshold:
        return "SELL"
    return "HOLD"


def signal_from_frame(
    data: pd.DataFrame,
    ticker: str,
    entry_threshold: float,
    exit_threshold: float,
    now: datetime | None = None,
) -> SignalReport:
    """Classify the most recent completed bar of ``data``.

    Before the 4pm America/New_York close, today's still-forming bar is
    ignored in favor of the previous session.
    """
    now = _as_market_time(now)
    row = -1
    if len(data) > 1 and data.index[-1].date() == now.date() and now.time() < MARKET_CLOSE:
        row = -2
    ibs = float(data["IBS"].iloc[row])
    return SignalReport(
        ticker=ticker,
        bar_date=data.index[row],
        ibs=ibs,
        signal=classify(ibs, entry_threshold, exit_threshold),
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        data=data,
    )


def latest_signal(
    ticker: str,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
    lookback_days: int = 100,
    now: datetime | None = None,
) -> SignalReport:
    """Download the last ``lookback_days`` calendar days and classify the latest bar."""
    now = _as_market_time(now)
    start = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    data = load_data(ticker, start=start, end=end)
    if len(data) < 2:
        raise ValueError(f"not enough recent history for {ticker!r}")
    return signal_from_frame(data, ticker, entry_threshold, exit_threshold, now)


def _as_market_time(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(MARKET_TZ)
    if now.tzinfo is None:
        return MARKET_TZ.localize(now)
    return now.astimezone(MARKET_TZ)
