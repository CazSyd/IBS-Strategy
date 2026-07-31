"""Forward signal log + fills ledger + paper-trade reconstruction for the live pages.

Two append-only files, both committed by the workflow, so git history is a
tamper-evident real-time record - genuine out-of-sample evidence, not a backtest:

* **signals log** - every published BUY/SELL/HOLD (the audit trail of what the
  page told you each day).
* **fills ledger** - each realized entry/exit at its as-published open price,
  snapshotted the day after the signal. IBS is a *within-bar ratio*
  ``(Close-Low)/(High-Low)``, so dividend/split adjustments (which scale every
  price in a bar by the same factor) leave it unchanged - the signal and the
  fill *sequence* are revision-invariant, only the prices move. Freezing the
  fill prices is therefore all it takes to make the live track immune to Yahoo's
  later adjusted-price revisions, and the ledger is always a stable prefix of
  the recomputed sequence.

``paper_trade`` builds the live equity from the frozen fills, net of a per-side
cost and earning the T-bill on idle cash (matching the backtest's methodology).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import cash_growth_factors
from .metrics import max_drawdown, sharpe_ratio

__all__ = [
    "LOG_COLUMNS",
    "FILL_COLUMNS",
    "append_signal",
    "load_signal_log",
    "append_fill",
    "load_fills",
    "reconcile_fills",
    "PaperTrack",
    "paper_trade",
    "TrackStatus",
    "track_status",
]

LOG_COLUMNS = ("date", "ticker", "ibs", "signal", "size", "ref_price")
FILL_COLUMNS = ("date", "ticker", "side", "price", "size")


# --------------------------------------------------------------------------- #
# signals log (the published-signal audit trail)
# --------------------------------------------------------------------------- #
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
    _append_row(path, LOG_COLUMNS, {
        "date": str(date),
        "ticker": ticker,
        "ibs": f"{float(ibs):.4f}",
        "signal": signal,
        "size": "" if size is None else f"{float(size):.4f}",
        "ref_price": "" if ref_price is None else f"{float(ref_price):.4f}",
    })
    return True


# --------------------------------------------------------------------------- #
# fills ledger (frozen realized entries/exits)
# --------------------------------------------------------------------------- #
def load_fills(path, ticker: str | None = None) -> list[dict]:
    """Read the fills ledger, sorted by (date, side) (optionally one ticker)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if ticker is None or r["ticker"] == ticker]
    rows.sort(key=lambda r: (r["date"], r["side"]))
    return rows


def append_fill(path, *, date, ticker, side, price, size) -> bool:
    """Append one realized fill unless ``(ticker, date, side)`` is already recorded."""
    path = Path(path)
    seen = {(r["ticker"], r["date"], r["side"]) for r in load_fills(path)}
    if (ticker, str(date), side) in seen:
        return False
    _append_row(path, FILL_COLUMNS, {
        "date": str(date),
        "ticker": ticker,
        "side": side,
        "price": f"{float(price):.4f}",
        "size": "" if size is None else f"{float(size):.4f}",
    })
    return True


def _append_row(path: Path, columns, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _fill_sequence(signal_rows: list[dict], prices: pd.DataFrame) -> list[tuple]:
    """Deterministic (date, side, open_price, size) fills implied by the signals.

    A session's signal acts at the *next* bar's open (no look-ahead). The
    sequence (which dates, which side) is revision-invariant; only the prices
    depend on the current ``prices`` frame.
    """
    if not signal_rows:
        return []
    sig = pd.Series({pd.Timestamp(r["date"]): r["signal"] for r in signal_rows}).sort_index()
    size = pd.Series(
        {pd.Timestamp(r["date"]): (float(r["size"]) if r["size"] else 1.0) for r in signal_rows}
    ).sort_index()
    px = prices[prices.index >= sig.index.min()]
    applied = sig.reindex(px.index).ffill().shift(1).to_numpy(dtype=object)
    applied_size = size.reindex(px.index).ffill().shift(1).to_numpy(dtype=float)
    opens = px["Open"].to_numpy(dtype=float)

    fills, in_pos = [], False
    for i in range(len(px)):
        if not in_pos and applied[i] == "BUY":
            w = applied_size[i] if applied_size[i] == applied_size[i] else 1.0
            fills.append((px.index[i], "BUY", float(opens[i]), float(w)))
            in_pos = True
        elif in_pos and applied[i] == "SELL":
            fills.append((px.index[i], "SELL", float(opens[i]), None))
            in_pos = False
    return fills


def reconcile_fills(signal_rows: list[dict], prices: pd.DataFrame,
                    existing_fills: list[dict]) -> list[tuple]:
    """New fills to append: the implied sequence minus what's already frozen.

    Because the sequence is revision-invariant, anything not already in the
    ledger is a genuinely new fill, recorded at the open as reported *now* (the
    day after its signal). ``prices`` must span from the log's inception.
    """
    have = {(r["date"], r["side"]) for r in existing_fills}
    return [f for f in _fill_sequence(signal_rows, prices)
            if (str(f[0].date()), f[1]) not in have]


# --------------------------------------------------------------------------- #
# paper-trade reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class PaperTrack:
    """Live paper-trade reconstructed from the frozen fills."""

    ticker: str
    inception: pd.Timestamp | None
    n_sessions: int
    equity: pd.Series
    trades: list  # (entry_date, entry_price, exit_date, exit_price, net_return) - raw prices, net return
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


def paper_trade(signal_rows: list[dict], fills: list[dict], prices: pd.DataFrame, *,
                capital: float = 1.0, cost_bps: float = 2.0,
                cash_rate=None, ticker: str = "") -> PaperTrack:
    """Reconstruct the live equity from the frozen fills.

    ``signal_rows`` set the inception/session count; ``fills`` (frozen prices)
    drive the P&L. Each side pays ``cost_bps`` (half-spread + impact + the tiny
    IBKR commission), idle cash earns ``cash_rate`` (annualized T-bill), and the
    open position is marked to the latest close.
    """
    inception = pd.Timestamp(signal_rows[0]["date"]) if signal_rows else None
    n_sessions = len(signal_rows)
    empty = PaperTrack(ticker, inception, n_sessions, pd.Series(dtype=float), [], False,
                       None, None, None, None)
    if inception is None:
        return empty
    px = prices[prices.index >= inception]
    if len(px) == 0:
        return empty

    factors = cash_growth_factors(px, cash_rate)
    fill_map = {
        pd.Timestamp(r["date"]): (r["side"], float(r["price"]),
                                  float(r["size"]) if r["size"] else 1.0)
        for r in fills
    }
    close = px["Close"].to_numpy(dtype=float)
    cost = cost_bps / 1e4

    cash, shares, in_pos = float(capital), 0.0, False
    e_date = e_price = e_size = None
    equity, trades = [], []
    for i, date in enumerate(px.index):
        cash *= factors[i]  # idle cash earns interest before any fill
        leg = fill_map.get(date)
        if leg is not None:
            side, price, size = leg
            if side == "BUY" and not in_pos:
                deploy = size * cash  # cash == equity here (flat)
                shares = deploy / (price * (1 + cost))
                cash -= deploy
                in_pos = True
                e_date, e_price, e_size = date, price, size
            elif side == "SELL" and in_pos:
                cash += shares * price * (1 - cost)
                net = (price * (1 - cost)) / (e_price * (1 + cost)) - 1.0
                trades.append((e_date, e_price, date, price, net))
                shares, in_pos = 0.0, False
                e_date = e_price = e_size = None
        equity.append(cash + shares * close[i])

    return PaperTrack(ticker, inception, n_sessions, pd.Series(equity, index=px.index), trades,
                      in_pos, e_date, e_price, e_size, float(close[-1]))


# --------------------------------------------------------------------------- #
# Phase 3: forward-vs-expected + integrity checks
# --------------------------------------------------------------------------- #
@dataclass
class TrackStatus:
    """Diagnostics: is the live run tracking the backtest, and is it clean?"""

    n: int                          # live sessions used for the comparison
    expected: float | None          # backtest's expected n-session return
    band: float | None              # +/- one standard error over n sessions
    z: float | None                 # (live - expected) / band, in SE units
    gaps: int                       # trading days in the logged span with no signal row
    last_gap: pd.Timestamp | None
    ibs_mismatches: int             # logged IBS != current IBS (a real data alarm - IBS is revision-invariant)
    revised_bars: int               # logged ref_price != current close (expected under dividends/splits)
    max_revision: float             # largest relative ref_price revision

    @property
    def verdict(self) -> str | None:
        if self.z is None:
            return None
        if abs(self.z) <= 1.0:
            return "on track"
        return "ahead of it" if self.z > 0 else "behind it"


def track_status(track: PaperTrack, ref_returns, prices: pd.DataFrame,
                 signals: list[dict], *, min_sessions: int = 20) -> TrackStatus:
    """Compare the live paper-track to the backtest and audit the log's integrity.

    ``ref_returns`` is the backtest strategy's daily-return series (its mean/std
    set the expectation band); ``prices`` (current data) reconciles the logged
    signals against today's values.
    """
    n = len(track.equity)
    expected = band = z = None
    if n >= min_sessions and ref_returns is not None and len(ref_returns.dropna()) > 20:
        r = ref_returns.dropna()
        m, s = float(r.mean()), float(r.std(ddof=1))
        expected, band = m * n, s * (n ** 0.5)
        if band > 0:
            z = (track.total_return - expected) / band

    # pipeline health: trading days in the logged span missing a signal row
    log_dates = {pd.Timestamp(row["date"]) for row in signals}
    gaps, last_gap = 0, None
    if log_dates:
        lo, hi = min(log_dates), max(log_dates)
        for day in prices.index:
            if lo < day <= hi and day not in log_dates:
                gaps, last_gap = gaps + 1, day

    # revision integrity: logged ref_price/IBS vs current data
    ibs = prices["IBS"] if "IBS" in prices.columns else (
        (prices["Close"] - prices["Low"]) / (prices["High"] - prices["Low"]))
    close = prices["Close"]
    ibs_mismatches = revised_bars = 0
    max_revision = 0.0
    for row in signals:
        day = pd.Timestamp(row["date"])
        if day not in prices.index:
            continue
        if row.get("ref_price"):
            logged = float(row["ref_price"])
            delta = abs(float(close.loc[day]) - logged) / logged if logged else 0.0
            if delta > 0.001:
                revised_bars += 1
                max_revision = max(max_revision, delta)
        if row.get("ibs"):
            current = float(ibs.loc[day])
            if current == current and abs(current - float(row["ibs"])) > 0.01:
                ibs_mismatches += 1

    return TrackStatus(n, expected, band, z, gaps, last_gap,
                       ibs_mismatches, revised_bars, max_revision)
