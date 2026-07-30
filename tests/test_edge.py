import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlc

from ibs_strategy.edge import (
    GENUINE_IBS_STD,
    assess_ibs_quality,
    decile_response,
    response_gradient,
)


def _two_era_frame(seed=3):
    """Two calendar years of IBS: 2000 synthetic (compressed to 0.5, zero
    pinning) and 2001 genuine (std ~0.3, some pinning) -- the ^GSPC pre-1980
    pattern. IBS is assigned by calendar year so it aligns with the grouping."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", "2001-12-31")
    is_2000 = idx.year == 2000
    ibs = np.empty(len(idx))
    ibs[is_2000] = np.clip(rng.normal(0.5, 0.16, is_2000.sum()), 0.001, 0.999)  # never pins
    genuine = rng.uniform(0, 1, (~is_2000).sum())
    genuine[::15] = 1.0   # ~7% of days pinned at the high
    genuine[1::23] = 0.0  # a few pinned at the low
    ibs[~is_2000] = genuine
    close = 100 + np.arange(len(idx)) * 0.1
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "IBS": ibs},
        index=idx,
    )


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


def test_assess_ibs_quality_flags_synthetic_ranges():
    quality = assess_ibs_quality(_two_era_frame())
    assert not quality.loc[2000, "genuine"]  # compressed, zero pinning
    assert quality.loc[2001, "genuine"]      # std ~0.3, pinned days present
    assert quality.loc[2000, "ibs_std"] < GENUINE_IBS_STD  # the synthetic tell
    assert quality.loc[2000, "pinned_frac"] == 0.0
    assert quality.loc[2001, "pinned_frac"] > 0.0


def test_assess_ibs_quality_skips_thin_periods():
    idx = pd.bdate_range("2020-01-01", periods=30)
    thin = pd.DataFrame({"IBS": np.linspace(0.4, 0.6, 30)}, index=idx)
    assert assess_ibs_quality(thin, min_bars=120).loc[2020, "genuine"] is None


def test_require_genuine_raises_on_reconstructed_data():
    frame = _two_era_frame()
    # default: no gate, runs fine
    decile_response(frame, buckets=5)
    # gated: the synthetic 2000 block must trip it
    with pytest.raises(ValueError, match="reconstructed"):
        decile_response(frame, buckets=5, require_genuine=True)


def test_require_genuine_passes_clean_data():
    response = decile_response(_frame_with_ibs_effect(), buckets=5, require_genuine=True)
    assert response["count"].sum() > 0  # uniform IBS clears the std bar


def test_gap_immune_uses_a_shared_price_free_return(scenario_frame):
    """gap_immune measures Close_{t+2}/Close_{t+1}-1, never touching Close_t."""
    default = decile_response(scenario_frame, buckets=2)
    gap = decile_response(scenario_frame, buckets=2, gap_immune=True)
    # one extra row is lost to the second shift, so the gap-immune sample is smaller
    assert gap["count"].sum() == default["count"].sum() - 1

    frame = scenario_frame.dropna(subset=["IBS"]).copy()
    expected = (frame["Close"].shift(-2) / frame["Close"].shift(-1) - 1).dropna()
    pooled = (gap["mean_forward_return"] * gap["count"]).sum() / gap["count"].sum()
    assert pooled == pytest.approx(expected.mean())
