# IBS Strategy on Leveraged ETF Tickers

This project applies the **Internal Bar Strength (IBS)** mean-reversion strategy to daily ticker data from Yahoo Finance. It is based on [u/heygentlewhale's post on Reddit](https://www.reddit.com/r/TQQQ/comments/1l63i0i/tqqq_internal_bar_strength_strategy_that_made_me/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).

The repo is a small, tested Python package with a backtesting engine, threshold optimization, **purged walk-forward validation**, signal visualization, and a live-signal command that opens an **interactive candlestick chart** of the past year's trades. The original notebook is kept at the repo root and still runs in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CazSyd/IBS-Strategy/blob/main/IBS_strategy.ipynb)

## The strategy

IBS measures where a bar closes within its daily range:

```
IBS = (Close - Low) / (High - Low)
```

Values near 0 mean the close sat at the low of the day (oversold), values near 1 at the high (overbought). The strategy is long-only on daily bars:

- **Entry** - if flat and _yesterday's_ IBS < entry threshold (default **0.13**), buy at _today's open_, all-in with whole shares.
- **Exit** - if long and _yesterday's_ IBS > exit threshold (default **0.97**), sell at _today's open_.
- Equity is marked to market at each close. No commissions or slippage are modeled.

The default thresholds are the CAGR-optimal pair over TQQQ's full listing history (2010-2026, see the results snapshot below); pass `--entry`/`--exit` (or the function arguments) to try others, e.g. the notebook's original 0.19/0.95.

Signals always come from the previous completed bar, so there is no look-ahead. Mean reversion of this kind works best on high-volatility leveraged ETFs such as **TQQQ** and **SPXL**.

## Getting started

The project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock` replace the old `requirements.txt`):

```bash
uv sync              # creates .venv and installs everything
uv run ibs --help    # the CLI
uv run pytest        # run the test suite
```

To open the notebook locally instead of Colab: `uv sync --group notebook && uv run jupyter lab`.

## CLI

```bash
# Backtest the default thresholds: metrics + trades/equity/drawdown chart
uv run ibs backtest TQQQ

# In-sample grid search over entry x exit thresholds, with a heatmap
uv run ibs optimize TQQQ --objective cagr

# Purged walk-forward validation: re-optimize per fold, evaluate out-of-sample
uv run ibs walkforward TQQQ --folds 5 --purge 5 --objective cagr

# Live signal (BUY / SELL / HOLD) + interactive candlestick page of recent trades
uv run ibs signal TQQQ
```

History defaults to the ticker's **full listing period** (TQQQ: 2010); narrow it with `--start`/`--end`. `backtest`/`optimize`/`walkforward` render matplotlib charts, while `signal` builds a self-contained interactive HTML page (Plotly) and opens it in your browser: daily candlesticks and volume for the past year (`--lookback` to change), every entry/exit fill marked with labeled B/S triangles, dashed guide lines and shaded holding spans, range-selector buttons, hover showing the day's OHLC, IBS, volume, and change, plus a light/dark theme toggle that follows your OS preference and remembers your choice. All commands accept `--save DIR` (write files instead of opening anything) and `--no-plot`. Objectives for `optimize`/`walkforward`: `total_return` (default, Sharpe tiebreak - same ranking as the notebook), `cagr`, `sharpe`, `max_drawdown`, `win_rate`. Note that over a fixed window CAGR and total return rank thresholds identically; CAGR is the right number for comparing _across_ windows of different lengths.

## Python API

```python
from ibs_strategy import load_data, run_backtest, grid_search, walk_forward, plot_backtest

data = load_data("TQQQ")  # full listing history; pass start/end to narrow

result = run_backtest(data)      # defaults: entry 0.13 / exit 0.97
print(result.summary())          # sharpe, total_return, cagr, max_drawdown, win_rate, ...
plot_backtest(result, ticker="TQQQ")

ranked = grid_search(data)       # every (entry, exit) pair, best row first
wf = walk_forward(data, n_folds=5, purge_days=5)
print(wf.summary())              # stitched out-of-sample metrics
```

## Methodology

### Backtest engine

`run_backtest` is a faithful port of the notebook's event-driven loop: previous-bar IBS signal, next-open fill, all-in whole-share sizing (leftover cash stays uninvested), one position at a time, strict threshold comparisons. Bars where `High == Low` have undefined IBS and never signal.

### Metrics (as defined in the notebook)

| Metric       | Definition                                                                               |
| ------------ | ---------------------------------------------------------------------------------------- |
| Sharpe ratio | `mean(daily returns) / std(ddof=1) * sqrt(252)`, zero risk-free rate, flat days included |
| Total return | `final capital / initial capital - 1`                                                    |
| Max drawdown | deepest peak-to-trough decline of the equity curve                                       |
| Win rate     | share of _closed_ trades whose exit fill beat the entry fill                             |

The package additionally reports CAGR, time-in-market, trade count, and final capital.

### Purged walk-forward validation

The notebook picked its "ideal" thresholds by optimizing over the whole backtest period - an in-sample estimate that flatters results. `walk_forward` addresses that:

1. The first `min_train_frac` (default 50%) of the history seeds the training window; the rest is split into `n_folds` sequential test windows.
2. For each fold, the threshold grid is re-optimized on all data _before_ the test window, minus a `purge_days` gap (default 5 trading days) so boundary fills and still-open positions can't leak information across the split.
3. The chosen thresholds are evaluated once on the unseen test window (starting flat), and the per-fold equity segments are compounded into a single out-of-sample curve.

### Results snapshot (TQQQ, full listing history 2010-02 to 2026-07, checked July 2026)

Default thresholds **entry 0.13 / exit 0.97** - the whole-period CAGR optimum from the grid search:

| Metric                     | IBS strategy (0.13 / 0.97) | Buy & hold |
| -------------------------- | -------------------------- | ---------- |
| CAGR                       | **53.8%**                  | 42.3%      |
| Total return               | +117,387%                  | +32,748%   |
| Sharpe ratio               | 1.10                       | 0.89       |
| Max drawdown               | -60.6%                     | -81.7%     |
| Win rate                   | 73.6% (155 closed trades)  | -          |
| Time in market             | 65.0%                      | 100%       |
| Final capital ($10k start) | $11.75M                    | $3.28M     |

- **Purged walk-forward, out-of-sample 2018-04 to 2026-07**: 47.0% CAGR vs 34.8% for buy & hold (Sharpe 0.93, max drawdown -60.5%). The folds independently keep choosing entry 0.13 with exit 0.97-0.99, agreeing with the default, and the strategy sat roughly flat through the 2022 bear while buy & hold collapsed.
- Engine parity: over the original notebook's window (2020-01 to 2025-07) with its 0.19/0.95 thresholds, the engine reproduces the notebook's reported numbers (Sharpe 1.286 vs 1.283, max drawdown -46.48% vs -46.47%) up to Yahoo's adjusted-price revisions.

Caveat: the headline row is still fitted in-sample and no commissions or slippage are modeled - the walk-forward row is the fairer performance estimate.

![2022-2023 Backtest Results on TQQQ](20222023backtest.png)

## Project layout

```
├── IBS_strategy.ipynb        # original Colab notebook (kept as-is)
├── pyproject.toml            # uv-managed project + dependencies
├── uv.lock                   # locked environment
├── src/ibs_strategy/
│   ├── data.py               # yfinance download + IBS computation
│   ├── backtest.py           # event-driven backtest engine
│   ├── metrics.py            # notebook metric definitions
│   ├── optimize.py           # grid search + purged walk-forward
│   ├── visualize.py          # trades, equity, drawdown, heatmap, walk-forward charts
│   ├── live.py               # realtime BUY/SELL/HOLD signal check
│   ├── web.py                # interactive candlestick signal page (plotly)
│   └── cli.py                # `ibs` command
└── tests/                    # pytest suite (synthetic data, no network)
```

## Data source

All data is fetched via the [`yfinance`](https://github.com/ranaroussi/yfinance) library (auto-adjusted daily OHLCV).

## ⚠️ Disclaimer

This project is for educational and research purposes only. Trading involves significant risk and past performance does not guarantee future returns.

The publication is not intended to be and does not constitute financial advice, investment advice, trading advice or any other advice or recommendation of any sort. The publisher also does not warrant that the publication is accurate, up to date or applicable to the circumstances of any particular case.

---

Feel free to fork, modify, and test this strategy on your own selected tickers
