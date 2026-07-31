"""Forward signal log + paper-trade reconstruction for the live pages.

The daily workflow appends each session's *published* signal to a CSV; because
the file is append-only and every row lands in git history on its date, it is a
tamper-evident, real-time record - genuine out-of-sample evidence, not a
backtest. ``paper_trade`` replays the logged BUY/SELL sequence against realized
next-open fills to reconstruct the live equity curve and current position.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metrics import max_drawdown, sharpe_ratio

__all__ = [
    "LOG_COLUMNS",
    "append_signal",
    "load_signal_log",
    "PaperTrack",
    "paper_trade",
]

LOG_COLUMNS = ("date", "ticker", "ibs", "signal", "size", "ref_price")


def load_signal_log(path, ticker: str | None = None) -> list[dict]:
    """Read the signal log as a date-sorted list of row dicts (optionally one ticker)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if ticker is None or r["ticker"] == ticker]
    rows.sort(key=lambda r: r["date"])
    return rows


def append_signal(path, *, date, ticker, ibs, signal, size, ref_price) -> bool:
    """Append one signal row unless ``(ticker, date)`` is already logged.

    Returns True if a row was written. The dedup key makes weekend/holiday
    reruns and push+schedule double-runs idempotent, so the log only ever grows
    by genuinely new sessions.
    """
    path = Path(path)
    seen = {(r["ticker"], r["date"]) for r in load_signal_log(path)}
    if (ticker, str(date)) in seen:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "date": str(date),
                "ticker": ticker,
                "ibs": f"{float(ibs):.4f}",
                "signal": signal,
                "size": "" if size is None else f"{float(size):.4f}",
                "ref_price": "" if ref_price is None else f"{float(ref_price):.4f}",
            }
        )
    return True


@dataclass
class PaperTrack:
    """Reconstructed live paper-trade from the logged signals."""

    ticker: str
    inception: pd.Timestamp | None
    n_sessions: int
    equity: pd.Series
    trades: list  # (entry_date, entry_price, exit_date, exit_price, return_pct)
    in_position: bool
    entry_date: pd.Timestamp | None
    entry_price: float | None
    entry_size: float | None
    latest_close: float | None

    @property
    def total_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float | None:
        if not self.trades:
            return None
        return sum(1 for t in self.trades if t[4] > 0) / len(self.trades)

    @property
    def unrealized(self) -> float | None:
        if not self.in_position or not self.entry_price or not self.latest_close:
            return None
        return self.latest_close / self.entry_price - 1.0

    def max_drawdown(self) -> float:
        return max_drawdown(self.equity) if len(self.equity) > 1 else 0.0

    def sharpe(self) -> float | None:
        if len(self.equity) < 20:  # too few sessions to be meaningful
            return None
        return sharpe_ratio(self.equity.pct_change().fillna(0.0))


def paper_trade(rows: list[dict], prices: pd.DataFrame, capital: float = 1.0,
                ticker: str = "") -> PaperTrack:
    """Replay logged signals against next-open fills.

    ``rows`` are log dicts for one ticker; ``prices`` needs Open/Close on a
    DatetimeIndex covering the log's span. A session's signal is acted on at the
    *next* bar's open (the same no-look-ahead timing as the backtest), sized by
    the logged vol-target weight, and marked to each close.
    """
    if not rows:
        return PaperTrack(ticker, None, 0, pd.Series(dtype=float), [], False, None, None, None, None)

    sig = pd.Series({pd.Timestamp(r["date"]): r["signal"] for r in rows}).sort_index()
    size = pd.Series(
        {pd.Timestamp(r["date"]): (float(r["size"]) if r["size"] else 1.0) for r in rows}
    ).sort_index()
    inception = sig.index.min()
    px = prices[prices.index >= inception]
    if len(px) == 0:
        return PaperTrack(ticker, inception, len(rows), pd.Series(dtype=float), [], False,
                          None, None, None, None)

    # the prior session's signal acts at THIS bar's open (shift = no look-ahead)
    applied = sig.reindex(px.index).ffill().shift(1).to_numpy(dtype=object)
    applied_size = size.reindex(px.index).ffill().shift(1).to_numpy(dtype=float)
    o = px["Open"].to_numpy(dtype=float)
    c = px["Close"].to_numpy(dtype=float)

    cash, shares, in_pos = float(capital), 0.0, False
    e_date = e_price = e_size = None
    equity, trades = [], []
    for i in range(len(px)):
        if not in_pos and applied[i] == "BUY":
            w = applied_size[i] if applied_size[i] == applied_size[i] else 1.0
            deploy = w * cash
            shares = deploy / o[i]
            cash -= deploy
            in_pos = True
            e_date, e_price, e_size = px.index[i], float(o[i]), float(w)
        elif in_pos and applied[i] == "SELL":
            cash += shares * o[i]
            trades.append((e_date, e_price, px.index[i], float(o[i]), float(o[i]) / e_price - 1.0))
            shares, in_pos = 0.0, False
            e_date = e_price = e_size = None
        equity.append(cash + shares * c[i])

    return PaperTrack(ticker, inception, len(rows), pd.Series(equity, index=px.index), trades,
                      in_pos, e_date, e_price, e_size, float(c[-1]))
