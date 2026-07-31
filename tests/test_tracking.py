import pandas as pd

from ibs_strategy.tracking import append_signal, load_signal_log, paper_trade


def test_append_is_deduped_and_ticker_scoped(tmp_path):
    log = tmp_path / "signals.csv"
    assert append_signal(log, date="2026-08-03", ticker="TQQQ", ibs=0.05, signal="BUY",
                         size=0.6, ref_price=50.0) is True
    # same (ticker, date) is idempotent - reruns don't double-log
    assert append_signal(log, date="2026-08-03", ticker="TQQQ", ibs=0.05, signal="BUY",
                         size=0.6, ref_price=50.0) is False
    # a different ticker on the same date is a distinct row
    assert append_signal(log, date="2026-08-03", ticker="SPXL", ibs=0.02, signal="BUY",
                         size=1.0, ref_price=30.0) is True

    assert len(load_signal_log(log)) == 2
    tqqq = load_signal_log(log, ticker="TQQQ")
    assert len(tqqq) == 1 and tqqq[0]["signal"] == "BUY" and tqqq[0]["size"] == "0.6000"


def test_append_allows_empty_size(tmp_path):
    log = tmp_path / "signals.csv"
    append_signal(log, date="2026-08-03", ticker="X", ibs=0.5, signal="HOLD",
                  size=None, ref_price=None)
    row = load_signal_log(log)[0]
    assert row["size"] == "" and row["ref_price"] == ""


def _prices(dates, opens, closes):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Open": opens, "Close": closes}, index=idx)


def test_paper_trade_replays_next_open_fills():
    prices = _prices(
        ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        [100.0, 110.0, 120.0, 130.0, 140.0],
        [100.0, 110.0, 120.0, 130.0, 140.0],
    )
    rows = [  # BUY signalled on the 3rd (fills at the 4th's open), SELL on the 5th (fills 6th open)
        {"date": "2026-08-03", "ticker": "T", "ibs": "0.05", "signal": "BUY", "size": "1.0", "ref_price": "100"},
        {"date": "2026-08-04", "ticker": "T", "ibs": "0.50", "signal": "HOLD", "size": "1.0", "ref_price": "110"},
        {"date": "2026-08-05", "ticker": "T", "ibs": "0.95", "signal": "SELL", "size": "1.0", "ref_price": "120"},
    ]
    track = paper_trade(rows, prices, capital=1.0, ticker="T")

    assert track.n_trades == 1
    entry_date, entry_px, exit_date, exit_px, ret = track.trades[0]
    assert entry_px == 110.0 and exit_px == 130.0  # next-open fills, no look-ahead
    assert ret == 130.0 / 110.0 - 1.0
    assert not track.in_position
    assert track.total_return == pytest_approx(130.0 / 110.0 - 1.0)


def test_paper_trade_marks_open_position_and_size():
    prices = _prices(
        ["2026-08-03", "2026-08-04", "2026-08-05"],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 120.0],
    )
    rows = [{"date": "2026-08-03", "ticker": "T", "ibs": "0.05", "signal": "BUY",
             "size": "0.50", "ref_price": "100"}]
    track = paper_trade(rows, prices, capital=1.0, ticker="T")
    assert track.in_position and track.entry_size == 0.5
    assert track.n_trades == 0
    # deployed half at 100, close rises to 120 -> unrealized +20% on the position
    assert track.unrealized == pytest_approx(0.20)
    # portfolio equity: 0.5 cash + 0.5*(120/100) = 1.10
    assert track.total_return == pytest_approx(0.10)


def test_paper_trade_empty_log():
    track = paper_trade([], _prices(["2026-08-03"], [100.0], [100.0]), ticker="T")
    assert track.inception is None and track.n_sessions == 0 and not track.in_position
    assert track.total_return == 0.0


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
