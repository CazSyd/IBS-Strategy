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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["decile_response", "response_gradient"]


def decile_response(data: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    """Mean forward return per IBS bucket, with a t-statistic for each.

    The forward return is ``Close / Open - 1`` of the *next* session -- the day
    a signal on this bar would have had you long -- so the measurement matches
    what the strategy can actually trade and carries no look-ahead.
    """
    if not {"Open", "Close", "IBS"} <= set(data.columns):
        raise ValueError("data must have Open, Close and IBS columns")

    frame = data.dropna(subset=["IBS"]).copy()
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
