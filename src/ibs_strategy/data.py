"""Download OHLCV data and compute Internal Bar Strength (IBS)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

__all__ = ["compute_ibs", "flatten_columns", "load_data"]


def compute_ibs(data: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength: (Close - Low) / (High - Low).

    Ranges from 0 (close at the low) to 1 (close at the high). Bars where
    High == Low have no defined IBS and yield NaN, which downstream signal
    checks treat as "no signal".
    """
    bar_range = data["High"] - data["Low"]
    ibs = (data["Close"] - data["Low"]) / bar_range.where(bar_range != 0)
    return ibs.rename("IBS")


def flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (Price, Ticker) column MultiIndex yfinance returns for single tickers."""
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


def load_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV history for ``ticker`` and append an ``IBS`` column.

    With no ``start``, the full listing history is downloaded (``period="max"``).
    """
    if start is None:
        data = yf.download(
            ticker,
            period="max",
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
        if data is not None and not data.empty and end is not None:
            data = data.loc[data.index < pd.Timestamp(end)]
    else:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
    if data is None or data.empty:
        raise ValueError(f"no data returned for {ticker!r} (start={start}, end={end})")
    data = flatten_columns(data)
    data["IBS"] = compute_ibs(data)
    return data
