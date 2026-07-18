import math

import pandas as pd
import pytest

from ibs_strategy.backtest import Trade
from ibs_strategy.metrics import (
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)


def test_total_return():
    assert total_return(pd.Series([100.0, 110.0, 121.0])) == pytest.approx(0.21)


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 150.0, 100.0])
    assert max_drawdown(equity) == pytest.approx(100 / 150 - 1)


def test_drawdown_series_zero_at_highs():
    dd = drawdown_series(pd.Series([100.0, 120.0, 90.0]))
    assert dd.iloc[0] == 0
    assert dd.iloc[1] == 0
    assert dd.iloc[2] == pytest.approx(90 / 120 - 1)


def test_sharpe_matches_manual_formula():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015])
    expected = float(returns.mean() / returns.std() * math.sqrt(252))
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_degenerate_cases():
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0  # zero variance
    assert sharpe_ratio(pd.Series([0.01])) == 0.0  # std undefined
    assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_cagr_with_datetime_index():
    equity = pd.Series([100.0, 144.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    assert cagr(equity) == pytest.approx(0.2, abs=1e-3)


def test_cagr_without_datetime_index():
    equity = pd.Series([100.0] * 252 + [110.0])  # 252 bars ~ one year
    assert cagr(equity) == pytest.approx(0.1, abs=1e-3)


def test_win_rate_ignores_open_trades():
    ts = pd.Timestamp("2024-01-02")
    trades = [
        Trade(ts, 100.0, 10, ts, 110.0),  # win
        Trade(ts, 100.0, 10, ts, 90.0),  # loss
        Trade(ts, 100.0, 10),  # still open -> ignored
    ]
    assert win_rate(trades) == 0.5
    assert win_rate([]) == 0.0
