import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlc

from ibs_strategy.edge import decile_response, response_gradient


def _frame_with_ibs_effect(n=600, effect=0.02, seed=3):
    """Bars where the next session's return depends on today's IBS by construction."""
    rng = np.random.default_rng(seed)
    ibs = rng.uniform(0, 1, n)
    rows = []
    for i in range(n):
        # today's IBS is set by where the close sits in the range
        low, high = 100.0, 110.0
        close = low + ibs[i] * (high - low)
        # the NEXT bar's open->close return falls as today's IBS rises
        drift = effect * (0.5 - ibs[i - 1]) if i else 0.0
        open_ = 100.0
        rows.append((open_, max(high, close), min(low, close), close, open_ * (1 + drift)))
    frame = make_ohlc([(o, h, l, c) for o, h, l, c, _ in rows])
    # overwrite Close of each bar with the engineered next-day close so that
    # Close/Open - 1 carries the effect
    frame["Close"] = [r[4] for r in rows]
    frame["IBS"] = ibs
    return frame


def test_decile_response_detects_a_planted_gradient():
    response = decile_response(_frame_with_ibs_effect(), buckets=5)
    assert list(response["bucket"]) == [1, 2, 3, 4, 5]
    assert response["count"].sum() == pytest.approx(599, abs=2)

    gradient = response_gradient(response)
    # planted effect: forward returns fall as IBS rises
    assert gradient["rank_correlation"] < -0.9
    assert gradient["spread"] > 0
    assert gradient["bottom_bucket"] > gradient["top_bucket"]


def test_decile_response_finds_nothing_in_noise():
    rng = np.random.default_rng(11)
    n = 600
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    frame = make_ohlc(np.column_stack([open_, high, low, close]))

    gradient = response_gradient(decile_response(frame, buckets=5))
    assert abs(gradient["rank_correlation"]) < 0.9  # no reliable gradient in noise


def test_decile_response_uses_the_next_session_only(scenario_frame):
    """The forward return must be the NEXT bar's open->close, never this bar's."""
    response = decile_response(scenario_frame, buckets=2)
    assert response["count"].sum() < len(scenario_frame)  # last bar has no forward return

    frame = scenario_frame.dropna(subset=["IBS"]).copy()
    expected = (frame["Close"] / frame["Open"] - 1).shift(-1).dropna()
    assert response["mean_forward_return"].isna().sum() == 0
    assert expected.mean() == pytest.approx(
        (response["mean_forward_return"] * response["count"]).sum() / response["count"].sum()
    )


def test_decile_response_requires_columns():
    with pytest.raises(ValueError, match="Open, Close and IBS"):
        decile_response(pd.DataFrame({"Close": [1.0, 2.0]}))
