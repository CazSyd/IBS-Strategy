import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlc

from ibs_strategy.synthetic import extend_with_synthetic, synthetic_leveraged_ohlc

BASE_BARS = [
    (100.0, 104.0, 96.0, 100.0),  # seed bar: only its close (the previous close) is used
    (101.0, 106.0, 99.0, 104.0),
    (103.0, 105.0, 95.0, 96.0),
    (97.0, 102.0, 94.0, 100.0),
]


def test_synthetic_matches_hand_computation():
    base = make_ohlc(BASE_BARS)
    synth = synthetic_leveraged_ohlc(base, leverage=2.0, expense_ratio=0.0, financing_rate=0.0)
    assert len(synth) == 3
    # bar 1 vs previous close 100: open +1% -> +2%, high +6% -> +12%, low -1% -> -2%, close +4% -> +8%
    assert synth["Open"].iloc[0] == pytest.approx(1.02)
    assert synth["High"].iloc[0] == pytest.approx(1.12)
    assert synth["Low"].iloc[0] == pytest.approx(0.98)
    assert synth["Close"].iloc[0] == pytest.approx(1.08)
    # bar 2 chains off the synthetic close 1.08 and proxy previous close 104
    assert synth["Open"].iloc[1] == pytest.approx(1.08 * (1 + 2 * (103 / 104 - 1)))
    assert synth["Close"].iloc[1] == pytest.approx(1.08 * (1 + 2 * (96 / 104 - 1)))


def test_ibs_is_invariant_under_leverage_and_costs():
    base = make_ohlc(BASE_BARS)
    synth = synthetic_leveraged_ohlc(base, leverage=3.0, expense_ratio=0.02, financing_rate=0.05)
    assert np.allclose(synth["IBS"].to_numpy(), base["IBS"].iloc[1:].to_numpy())


def test_daily_cost_reduces_close_return():
    base = make_ohlc(BASE_BARS)
    gross = synthetic_leveraged_ohlc(base, 3.0, expense_ratio=0.0)
    net = synthetic_leveraged_ohlc(base, 3.0, expense_ratio=0.0252)  # exactly 1bp per day
    assert (gross["Close"].iloc[0] - net["Close"].iloc[0]) == pytest.approx(0.0001)


def test_financing_series_and_scalar_agree():
    base = make_ohlc(BASE_BARS)
    scalar = synthetic_leveraged_ohlc(base, 3.0, financing_rate=0.05)
    series = synthetic_leveraged_ohlc(base, 3.0, financing_rate=pd.Series(0.05, index=base.index))
    assert np.allclose(scalar["Close"].to_numpy(), series["Close"].to_numpy())


def test_wipeout_raises():
    bars = [(100.0, 104.0, 96.0, 100.0), (60.0, 62.0, 55.0, 58.0)]  # low is -45% on the day
    with pytest.raises(ValueError, match="wiped out"):
        synthetic_leveraged_ohlc(make_ohlc(bars), leverage=3.0)


def test_final_close_rescales_whole_path():
    base = make_ohlc(BASE_BARS)
    synth = synthetic_leveraged_ohlc(base, 2.0, expense_ratio=0.0, final_close=50.0)
    assert synth["Close"].iloc[-1] == pytest.approx(50.0)
    unscaled = synthetic_leveraged_ohlc(base, 2.0, expense_ratio=0.0)
    ratio = synth["High"] / unscaled["High"]
    assert np.allclose(ratio.to_numpy(), ratio.iloc[0])


def test_extend_with_synthetic_splices_consistently():
    base = make_ohlc(BASE_BARS + [(101.0, 103.0, 97.0, 99.0)], start="2024-01-02")  # Jan 2,3,4,5,8
    real = make_ohlc([(50.0, 52.0, 48.0, 51.0), (51.5, 53.0, 50.0, 52.0)], start="2024-01-08")
    combined = extend_with_synthetic(real, base, leverage=3.0, expense_ratio=0.0)

    assert combined.index.is_monotonic_increasing
    assert combined["Synthetic"].tolist() == [True, True, True, False, False]
    assert combined.loc[real.index, "Close"].tolist() == real["Close"].tolist()

    # seam: the last synthetic close is chosen so the modeled 3x overnight move
    # lands exactly on the first real open
    overnight = base["Open"].iloc[4] / base["Close"].iloc[3] - 1
    expected_final = real["Open"].iloc[0] / (1 + 3 * overnight)
    assert combined["Close"].iloc[2] == pytest.approx(expected_final)


def test_extend_requires_prior_history():
    base = make_ohlc(BASE_BARS, start="2024-02-05")
    real = make_ohlc([(50.0, 52.0, 48.0, 51.0)], start="2024-01-08")  # listed before proxy
    with pytest.raises(ValueError, match="no bars before"):
        extend_with_synthetic(real, base)
