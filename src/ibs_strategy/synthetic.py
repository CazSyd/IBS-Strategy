"""Synthetic pre-listing history for leveraged ETFs, built from a proxy.

TQQQ lists only from 2010-02, but its underlying (QQQ) trades since
1999-03-10 -- Yahoo's single ``QQQ`` symbol also covers its Amex and
QQQQ-era history. A daily-rebalanced leveraged fund's price at any point in
a session is, relative to the previous close, ``leverage`` times the
proxy's move: leverage resets at each close, so intraday extremes coincide
with the proxy's and open/high/low/close all map through the same affine
transform. Daily fund costs (expense ratio plus financing of the borrowed
``leverage - 1`` exposure at a short-term rate) are deducted uniformly
across the bar, which keeps each bar internally consistent and leaves IBS
exactly equal to the proxy's -- pre-listing signals are the proxy's own IBS
signals, traded at leverage.
"""

from __future__ import annotations

import pandas as pd

from .data import compute_ibs, load_data

__all__ = [
    "DEFAULT_EXPENSE_RATIO",
    "DEFAULT_FINANCING_SPREAD",
    "DEFAULT_RATE_TICKER",
    "synthetic_leveraged_ohlc",
    "extend_with_synthetic",
    "load_extended_data",
]

DEFAULT_EXPENSE_RATIO = 0.0095  # TQQQ charges ~0.95%/yr
DEFAULT_RATE_TICKER = "^IRX"  # 13-week T-bill yield: financing-cost proxy
# Swap financing runs over T-bills; 0.5%/yr on the borrowed exposure closes
# the CAGR gap to real TQQQ over the 2010-2026 overlap (drift +1.5%/yr -> ~0).
DEFAULT_FINANCING_SPREAD = 0.005
TRADING_DAYS_PER_YEAR = 252

_PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def _align_rate(financing_rate: pd.Series | float, index: pd.Index) -> pd.Series:
    if isinstance(financing_rate, pd.Series):
        return financing_rate.reindex(index).ffill().bfill().fillna(0.0)
    return pd.Series(float(financing_rate), index=index)


def synthetic_leveraged_ohlc(
    base: pd.DataFrame,
    leverage: float = 3.0,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    financing_rate: pd.Series | float = 0.0,
    final_close: float | None = None,
) -> pd.DataFrame:
    """Daily-rebalanced ``leverage``x OHLC series derived from ``base``.

    The first base bar only seeds the previous close and is not part of the
    output. ``financing_rate`` is an annualized short rate (scalar, or a
    series aligned by date) accrued daily on the borrowed ``leverage - 1``
    exposure, on top of ``expense_ratio``. ``final_close`` rescales the whole
    path so the last synthetic close lands exactly there.
    """
    if len(base) < 2:
        raise ValueError("need at least two proxy bars to build synthetic history")

    frame = base.iloc[1:]
    prev_close = base["Close"].shift(1).iloc[1:]
    rate = _align_rate(financing_rate, frame.index)
    daily_cost = (expense_ratio + (leverage - 1) * rate) / TRADING_DAYS_PER_YEAR

    factors = {
        column: 1 + leverage * (frame[column] / prev_close - 1) - daily_cost
        for column in _PRICE_COLUMNS
    }
    if (factors["Low"] <= 0).any():
        raise ValueError(
            "synthetic path wiped out: a single proxy bar moved beyond "
            f"-1/{leverage:g} of the previous close"
        )

    growth = factors["Close"].cumprod()
    previous_growth = growth.shift(1).fillna(1.0)

    out = pd.DataFrame(index=frame.index)
    out["Open"] = previous_growth * factors["Open"]
    out["High"] = previous_growth * factors["High"]
    out["Low"] = previous_growth * factors["Low"]
    out["Close"] = growth
    if final_close is not None:
        scale = final_close / float(out["Close"].iloc[-1])
        out[list(_PRICE_COLUMNS)] = out[list(_PRICE_COLUMNS)] * scale
    if "Volume" in frame.columns:
        out["Volume"] = frame["Volume"]
    out["IBS"] = compute_ibs(out)
    return out


def extend_with_synthetic(
    real: pd.DataFrame,
    base: pd.DataFrame,
    leverage: float = 3.0,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    financing_rate: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Prepend synthetic pre-listing bars (derived from ``base``) to ``real``.

    The synthetic path is scaled so the seam overnight move (last synthetic
    close to first real open) equals the modeled ``leverage`` times the
    proxy's overnight move. A boolean ``Synthetic`` column marks the
    reconstructed bars.
    """
    listing = real.index[0]
    prior = base[base.index < listing]
    if len(prior) < 2:
        raise ValueError(f"proxy history has no bars before {listing.date()} to extend with")

    first_open = float(real["Open"].iloc[0])
    final_close = first_open
    seam = base[base.index >= listing]
    if not seam.empty:
        overnight = float(seam["Open"].iloc[0]) / float(prior["Close"].iloc[-1]) - 1
        if 1 + leverage * overnight > 0:
            final_close = first_open / (1 + leverage * overnight)

    synth = synthetic_leveraged_ohlc(
        prior, leverage, expense_ratio, financing_rate, final_close=final_close
    )
    synth["Synthetic"] = True
    real = real.copy()
    real["Synthetic"] = False

    combined = pd.concat([synth, real])
    columns = [
        column
        for column in ("Open", "High", "Low", "Close", "Volume", "IBS", "Synthetic")
        if column in combined.columns
    ]
    return combined[columns]


def load_extended_data(
    ticker: str = "TQQQ",
    proxy: str = "QQQ",
    start: str | None = None,
    end: str | None = None,
    leverage: float = 3.0,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    rate_ticker: str | None = DEFAULT_RATE_TICKER,
    financing_spread: float = DEFAULT_FINANCING_SPREAD,
) -> pd.DataFrame:
    """Full listing history of ``ticker``, extended back with synthetic bars.

    Downloads ``ticker`` and ``proxy`` (and ``rate_ticker`` for the financing
    leg unless None; ``financing_spread`` rides on top of that rate),
    reconstructs the pre-listing years, and returns one continuous frame with
    a ``Synthetic`` marker column.
    """
    real = load_data(ticker, end=end)
    base = load_data(proxy, end=end)
    financing_rate: pd.Series | float = 0.0
    if rate_ticker:
        financing_rate = load_data(rate_ticker, end=end)["Close"] / 100.0 + financing_spread
    combined = extend_with_synthetic(real, base, leverage, expense_ratio, financing_rate)
    if start is not None:
        combined = combined[combined.index >= pd.Timestamp(start)]
    return combined
