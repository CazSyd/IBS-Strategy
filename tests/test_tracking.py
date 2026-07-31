import pandas as pd
import pytest

from ibs_strategy.tracking import (
    append_fill,
    append_signal,
    load_fills,
    load_signal_log,
    paper_trade,
    reconcile_fills,
)

DATES = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]


def _prices(opens, closes):
    return pd.DataFrame({"Open": opens, "Close": closes}, index=pd.to_datetime(DATES))


def _signals():
    # BUY on the 3rd (fills at the 4th open), HOLD, SELL on the 5th (fills at the 6th open)
    return [
        {"date": "2026-08-03", "ticker": "T", "ibs": "0.05", "signal": "BUY", "size": "1.0", "ref_price": "100"},
        {"date": "2026-08-04", "ticker": "T", "ibs": "0.50", "signal": "HOLD", "size": "1.0", "ref_price": "110"},
        {"date": "2026-08-05", "ticker": "T", "ibs": "0.95", "signal": "SELL", "size": "1.0", "ref_price": "120"},
    ]


def test_append_signal_is_deduped_and_ticker_scoped(tmp_path):
    log = tmp_path / "signals.csv"
    assert append_signal(log, date="2026-08-03", ticker="TQQQ", ibs=0.05, signal="BUY",
                         size=0.6, ref_price=50.0) is True
    assert append_signal(log, date="2026-08-03", ticker="TQQQ", ibs=0.05, signal="BUY",
                         size=0.6, ref_price=50.0) is False  # rerun is idempotent
    assert append_signal(log, date="2026-08-03", ticker="SPXL", ibs=0.02, signal="BUY",
                         size=1.0, ref_price=30.0) is True   # other ticker is distinct
    assert len(load_signal_log(log)) == 2
    assert load_signal_log(log, ticker="TQQQ")[0]["size"] == "0.6000"


def test_append_fill_dedups_on_ticker_date_side(tmp_path):
    ledger = tmp_path / "fills.csv"
    assert append_fill(ledger, date="2026-08-04", ticker="T", side="BUY", price=110.0, size=0.6) is True
    assert append_fill(ledger, date="2026-08-04", ticker="T", side="BUY", price=110.0, size=0.6) is False
    # a SELL on the same date is a distinct leg
    assert append_fill(ledger, date="2026-08-04", ticker="T", side="SELL", price=110.0, size=None) is True
    assert len(load_fills(ledger)) == 2


def test_reconcile_emits_next_open_fills_and_only_the_new_tail():
    prices = _prices([100.0, 110.0, 120.0, 130.0, 140.0], [100.0, 110.0, 120.0, 130.0, 140.0])
    seq = reconcile_fills(_signals(), prices, existing_fills=[])
    assert [(f[1], f[2]) for f in seq] == [("BUY", 110.0), ("SELL", 130.0)]  # next-open fills
    assert [str(f[0].date()) for f in seq] == ["2026-08-04", "2026-08-06"]

    # once the BUY is frozen, only the SELL is new (revision-invariant sequence => stable prefix)
    frozen = [{"date": "2026-08-04", "ticker": "T", "side": "BUY", "price": "110", "size": "1.0"}]
    tail = reconcile_fills(_signals(), prices, existing_fills=frozen)
    assert len(tail) == 1 and tail[0][1] == "SELL" and str(tail[0][0].date()) == "2026-08-06"


def test_paper_trade_returns_are_net_of_cost():
    prices = _prices([100.0, 110.0, 120.0, 130.0, 140.0], [100.0, 110.0, 120.0, 130.0, 140.0])
    fills = [
        {"date": "2026-08-04", "ticker": "T", "side": "BUY", "price": "110", "size": "1.0"},
        {"date": "2026-08-06", "ticker": "T", "side": "SELL", "price": "130", "size": ""},
    ]
    track = paper_trade(_signals(), fills, prices, capital=1.0, cost_bps=10.0, cash_rate=None)
    assert track.n_trades == 1
    entry_date, entry_px, exit_date, exit_px, ret = track.trades[0]
    assert entry_px == 110.0 and exit_px == 130.0  # raw fill prices are shown
    assert ret == pytest.approx((130.0 * 0.999) / (110.0 * 1.001) - 1.0)  # 10bp/side charged
    assert not track.in_position


def test_paper_trade_marks_open_position_and_earns_cash_interest():
    prices = _prices([100.0, 100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0, 120.0])
    fills = [{"date": "2026-08-04", "ticker": "T", "side": "BUY", "price": "100", "size": "0.50"}]
    track = paper_trade(_signals()[:1], fills, prices, capital=1.0, cost_bps=0.0, cash_rate=None)
    assert track.in_position and track.entry_size == 0.5 and track.n_trades == 0
    assert track.unrealized == pytest.approx(0.20)  # deployed at 100, marks to 120
    # half deployed (rises to 1.2x) + half cash => 0.5*1.2 + 0.5 = 1.10
    assert track.total_return == pytest.approx(0.10)

    # a positive cash rate lifts the flat-cash half's contribution
    earning = paper_trade(_signals()[:1], fills, prices, capital=1.0, cost_bps=0.0, cash_rate=0.252)
    assert earning.total_return > track.total_return


def test_paper_trade_empty_log():
    track = paper_trade([], [], _prices([100.0] * 5, [100.0] * 5), ticker="T")
    assert track.inception is None and track.n_sessions == 0 and track.total_return == 0.0
