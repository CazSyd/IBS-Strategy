import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlc

from ibs_strategy.backtest import run_backtest


def test_scenario_equity_and_trades(scenario_frame):
    result = run_backtest(scenario_frame, entry_threshold=0.2, exit_threshold=0.9, initial_capital=1_000.0)

    assert result.equity.tolist() == [1000.0, 1050.0, 1080.0, 1160.0, 1170.0, 1170.0, 1093.0, 1115.0, 1126.0]
    assert result.data["Position"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 1]
    assert result.data["Shares"].tolist() == [0, 10, 10, 10, 0, 11, 11, 11, 11]
    assert result.data["Cash"].tolist() == [1000.0, 50.0, 50.0, 50.0, 1170.0, 70.0, 70.0, 70.0, 70.0]

    assert len(result.trades) == 2
    closed, open_trade = result.trades
    assert closed.entry_date == scenario_frame.index[1]
    assert closed.entry_price == 95.0
    assert closed.shares == 10
    assert closed.exit_date == scenario_frame.index[4]
    assert closed.exit_price == 112.0
    assert closed.is_win
    assert closed.return_pct == pytest.approx(112 / 95 - 1)
    assert open_trade.is_open
    assert open_trade.entry_price == 100.0
    assert open_trade.shares == 11
    assert open_trade.return_pct is None


def test_scenario_summary_matches_notebook_formulas(scenario_frame):
    result = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0)
    summary = result.summary()
    assert summary["total_return"] == pytest.approx(0.126)
    assert summary["max_drawdown"] == pytest.approx(1093 / 1170 - 1)
    assert summary["win_rate"] == 1.0
    assert summary["num_trades"] == 1
    returns = result.data["Strategy Return"]
    assert summary["sharpe"] == pytest.approx(float(returns.mean() / returns.std() * np.sqrt(252)))
    assert summary["exposure"] == pytest.approx(7 / 9)
    assert summary["final_capital"] == 1126.0


def test_thresholds_are_strict():
    # IBS exactly at the entry threshold must not trigger a buy
    frame = make_ohlc(
        [
            (100.0, 108.0, 100.0, 102.0),  # IBS exactly 0.25
            (100.0, 104.0, 96.0, 100.0),
            (100.0, 104.0, 96.0, 100.0),
        ]
    )
    result = run_backtest(frame, entry_threshold=0.25, exit_threshold=0.9, initial_capital=1_000.0)
    assert result.trades == []
    assert result.equity.tolist() == [1000.0, 1000.0, 1000.0]

    # IBS exactly at the exit threshold must not trigger a sell
    frame = make_ohlc(
        [
            (100.0, 110.0, 90.0, 92.0),   # IBS 0.10 -> buy next open
            (95.0, 105.0, 95.0, 104.0),   # IBS exactly 0.90
            (100.0, 104.0, 96.0, 100.0),
        ]
    )
    result = run_backtest(frame, entry_threshold=0.2, exit_threshold=0.9, initial_capital=1_000.0)
    assert len(result.trades) == 1
    assert result.trades[0].is_open


def test_no_signals_leaves_capital_flat():
    frame = make_ohlc([(100.0, 104.0, 96.0, 100.0)] * 5)  # IBS 0.5 throughout
    result = run_backtest(frame, 0.2, 0.9, 1_000.0)
    assert result.trades == []
    assert (result.equity == 1_000.0).all()
    assert result.summary()["sharpe"] == 0.0


def test_requires_columns():
    frame = pd.DataFrame({"Open": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing"):
        run_backtest(frame, 0.2, 0.9)
    with pytest.raises(ValueError, match="two bars"):
        run_backtest(make_ohlc([(100.0, 104.0, 96.0, 100.0)]), 0.2, 0.9)
