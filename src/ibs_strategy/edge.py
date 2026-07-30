"""Does the IBS signal predict anything, independent of any threshold?

Threshold grid searches answer "which parameters won on this sample", which
turns out to be noise: fit the Sharpe surface on the first and second halves of
TQQQ's history separately and the two correlate at -0.07. With a standard error
of roughly +/-10%/yr on any single cell's return, decades of data still cannot
resolve one threshold pair from another, so the surface has no stable shape to
find.

The prior question -- whether low IBS predicts higher forward returns at all --
is answerable, because it pools every bar instead of slicing the sample by
parameter. ``decile_response`` buckets days by IBS and measures the return of
the session you would have held (buy at the next open, mark at that close). A
real effect shows a monotone gradient that repeats out of sample; noise does
not.

Two data hazards, both learned from trying to extend the history with the S&P
500 index (Yahoo ``^GSPC``):

- **Reconstructed ranges.** IBS needs genuine intraday High/Low. Close-only or
  smoothed feeds compress IBS toward 0.5 (std ~0.16 vs a genuine ~0.30) and
  never pin a close at the day's extreme. ``assess_ibs_quality`` flags such
  periods per year; ``decile_response(..., require_genuine=True)`` refuses them
  so the trap can't recur silently. Because IBS is range-normalized, a genuinely
  low-volatility instrument still scores std ~0.30 -- only synthetic ranges fail.
- **Synthetic opens.** When a feed reports ``Open == previous Close`` (Yahoo
  ``^GSPC`` before ~2010), the open-to-close forward return collapses to a
  close-to-close return whose base price ``Close_t`` also drives ``IBS_t`` --
  the shared price manufactures reversal (it inflated the modern edge ~40%).
  ``decile_response(..., gap_immune=True)`` measures a one-day-displaced return
  that never touches ``Close_t``; it is a robustness proxy, not the tradable
  fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["assess_ibs_quality", "decile_response", "response_gradient"]

# Genuine intraday IBS is ~uniform: std near 0.30, with a few percent of days
# closing exactly at the high or low. Reconstructed ranges cluster at 0.5
# (std ~0.16) and pin nothing. A period passing EITHER bar is treated as
# genuine -- synthetic data fails both, so the OR only widens the benefit of the
# doubt for real data.
GENUINE_IBS_STD = 0.25
GENUINE_PINNED_FRACTION = 0.01


def assess_ibs_quality(data: pd.DataFrame, min_bars: int = 120) -> pd.DataFrame:
    """Per-year check for reconstructed OHLC where IBS cannot be trusted.

    Returns a frame indexed by period (calendar year when ``data`` has a
    ``DatetimeIndex``, else a single period) with the usable-bar count, the IBS
    standard deviation, the fraction of bars pinned at 0 or 1, and a ``genuine``
    verdict. The verdict is ``None`` where fewer than ``min_bars`` usable bars
    make the call unreliable (e.g. a partial boundary year).
    """
    if "IBS" not in data.columns:
        raise ValueError("data must have an IBS column")

    if isinstance(data.index, pd.DatetimeIndex):
        keys = data.index.year
    else:
        keys = np.zeros(len(data), dtype=int)
    grouped = pd.DataFrame({"ibs": data["IBS"].to_numpy(), "period": keys})

    rows = []
    for period, block in grouped.groupby("period"):
        ibs = block["ibs"].dropna()
        n = len(ibs)
        if n == 0:
            continue
        std = float(ibs.std(ddof=1)) if n > 1 else float("nan")
        pinned = float(ibs.isin([0.0, 1.0]).mean())
        genuine = None
        if n >= min_bars:
            genuine = bool(std >= GENUINE_IBS_STD or pinned >= GENUINE_PINNED_FRACTION)
        rows.append({"period": int(period), "bars": n, "ibs_std": std,
                     "pinned_frac": pinned, "genuine": genuine})
    return pd.DataFrame(rows).set_index("period")


def decile_response(
    data: pd.DataFrame,
    buckets: int = 10,
    gap_immune: bool = False,
    require_genuine: bool = False,
) -> pd.DataFrame:
    """Mean forward return per IBS bucket, with a t-statistic for each.

    The forward return is ``Close / Open - 1`` of the *next* session -- the day
    a signal on this bar would have had you long -- so the measurement matches
    what the strategy can actually trade and carries no look-ahead.

    ``gap_immune=True`` instead measures the session *after* the one you would
    hold (``Close_{t+2} / Close_{t+1} - 1``), whose base price is never
    ``Close_t``; use it on feeds with synthetic opens, where the default measure
    shares a price with the IBS signal and manufactures reversal. It is a
    robustness proxy for whether the effect exists, not the tradable P&L.

    ``require_genuine=True`` raises if any full year of ``data`` has
    reconstructed-looking IBS (see ``assess_ibs_quality``), so untrustworthy
    intraday data can't slip through unnoticed.
    """
    if not {"Open", "Close", "IBS"} <= set(data.columns):
        raise ValueError("data must have Open, Close and IBS columns")

    if require_genuine:
        quality = assess_ibs_quality(data)
        bad = quality[quality["genuine"] == False]  # noqa: E712 -- exclude None verdicts
        if len(bad):
            spans = ", ".join(str(p) for p in bad.index)
            raise ValueError(
                f"IBS looks reconstructed (not genuine intraday data) for: {spans}. "
                "Slice to the trustworthy span, or pass require_genuine=False to "
                "override. See assess_ibs_quality()."
            )

    frame = data.dropna(subset=["IBS"]).copy()
    if gap_immune:
        frame["forward_return"] = frame["Close"].shift(-2) / frame["Close"].shift(-1) - 1
    else:
        frame["forward_return"] = (frame["Close"] / frame["Open"] - 1).shift(-1)
    frame = frame.dropna(subset=["forward_return"])
    if len(frame) < buckets:
        raise ValueError(f"need at least {buckets} usable bars, got {len(frame)}")

    frame["bucket"] = pd.qcut(frame["IBS"], buckets, labels=False, duplicates="drop")

    rows = []
    for bucket, block in frame.groupby("bucket"):
        returns = block["forward_return"]
        count = len(returns)
        deviation = returns.std(ddof=1) / np.sqrt(count) if count > 1 else np.nan
        rows.append({
            "bucket": int(bucket) + 1,
            "ibs_low": float(block["IBS"].min()),
            "ibs_high": float(block["IBS"].max()),
            "mean_forward_return": float(returns.mean()),
            "t_stat": float(returns.mean() / deviation) if deviation else np.nan,
            "count": count,
        })
    return pd.DataFrame(rows)


def response_gradient(response: pd.DataFrame) -> dict[str, float]:
    """Summarize a ``decile_response`` table: is it monotone, and how steep?

    ``rank_correlation`` near -1 means forward returns fall steadily as IBS
    rises, which is the signature of a real effect. ``spread`` is the bottom
    bucket's edge over the top one.
    """
    means = response["mean_forward_return"].to_numpy(dtype=float)
    order = np.arange(len(means))
    return {
        "rank_correlation": float(np.corrcoef(order, means)[0, 1]),
        "slope_per_bucket": float(np.polyfit(order, means, 1)[0]),
        "spread": float(means[0] - means[-1]),
        "bottom_bucket": float(means[0]),
        "top_bucket": float(means[-1]),
    }
