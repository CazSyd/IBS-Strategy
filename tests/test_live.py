from datetime import datetime

from conftest import make_ohlc

from ibs_strategy.live import MARKET_TZ, classify, signal_from_frame


def _frame_with_last_ibs(ibs_value):
    close = 92.0 + ibs_value * 8.0  # last bar: High 100, Low 92 -> chosen IBS
    bars = [
        (100.0, 104.0, 96.0, 100.0),
        (100.0, 104.0, 96.0, 100.0),  # IBS 0.5
        (98.0, 100.0, 92.0, close),
    ]
    return make_ohlc(bars, start="2024-01-02")  # sessions: Jan 2, 3, 4


def test_buy_signal_after_close():
    frame = _frame_with_last_ibs(0.125)
    now = MARKET_TZ.localize(datetime(2024, 1, 4, 17, 0))
    report = signal_from_frame(frame, "TEST", 0.19, 0.95, now)
    assert report.signal == "BUY"
    assert report.bar_date == frame.index[-1]
    assert "BUY" in report.message


def test_incomplete_bar_uses_previous_session():
    frame = _frame_with_last_ibs(0.125)  # last bar would say BUY...
    now = MARKET_TZ.localize(datetime(2024, 1, 4, 10, 0))  # ...but market still open
    report = signal_from_frame(frame, "TEST", 0.19, 0.95, now)
    assert report.bar_date == frame.index[-2]
    assert report.signal == "HOLD"


def test_later_date_uses_last_bar_even_in_the_morning():
    frame = _frame_with_last_ibs(0.99)
    now = MARKET_TZ.localize(datetime(2024, 1, 5, 10, 0))  # day after the last bar
    report = signal_from_frame(frame, "TEST", 0.19, 0.95, now)
    assert report.bar_date == frame.index[-1]
    assert report.signal == "SELL"


def test_classify_boundaries():
    assert classify(0.05, 0.19, 0.95) == "BUY"
    assert classify(0.99, 0.19, 0.95) == "SELL"
    assert classify(0.5, 0.19, 0.95) == "HOLD"
    assert classify(float("nan"), 0.19, 0.95) == "HOLD"
    assert classify(0.19, 0.19, 0.95) == "HOLD"  # strict inequality
