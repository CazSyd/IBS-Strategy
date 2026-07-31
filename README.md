# IBS Strategy on Leveraged ETF Tickers

This project applies the **Internal Bar Strength (IBS)** mean-reversion strategy to daily ticker data from Yahoo Finance. It is based on [u/heygentlewhale's post on Reddit](https://www.reddit.com/r/TQQQ/comments/1l63i0i/tqqq_internal_bar_strength_strategy_that_made_me/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).

The repo is a small, tested Python package with a backtesting engine, threshold optimization, **purged walk-forward validation**, signal visualization, and a live-signal command that opens an **interactive candlestick chart** of the past year's trades. The original notebook is kept at the repo root and still runs in Colab:

[![CI](https://github.com/CazSyd/IBS-Strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/CazSyd/IBS-Strategy/actions/workflows/ci.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CazSyd/IBS-Strategy/blob/main/IBS_strategy.ipynb)

**Live signal page: <https://cazsyd.github.io/IBS-Strategy/>** - interactive TQQQ & SPXL charts, rebuilt every weekday after the US close.

## The strategy

IBS measures where a bar closes within its daily range:

```
IBS = (Close - Low) / (High - Low)
```

Values near 0 mean the close sat at the low of the day (oversold), values near 1 at the high (overbought). The strategy is long-only on daily bars:

- **Entry** - if flat and _yesterday's_ IBS < entry threshold (default **0.13**), buy at _today's open_, all-in with whole shares.
- **Exit** - if long and _yesterday's_ IBS > exit threshold (default **0.5**), sell at _today's open_.
- Equity is marked to market at each close. Idle cash earns the 13-week T-bill; no commissions or slippage are modeled.

The defaults are round on purpose and are **not** optimizer output. Entry 0.13 fires on the bottom ~12% of days, where [essentially all of the measured edge lives](#does-the-signal-actually-predict-anything). Exit 0.5 is a prompt, zero-DOF exit at the midpoint: the IBS edge is _front-loaded_ (largely spent within a day), so a prompt exit harvests it and steps aside - in cash ~80% of the time, truncating crashes. An optimized exit can't be justified (the surface [doesn't replicate out-of-sample](#why-the-thresholds-are-not-optimized)); a stop loss makes mean-reversion worse; and the 200-day-SMA overlay [removes the edge, not the risk](#where-the-edge-lives-below-the-200-day-sma). Pass `--entry`/`--exit` for any other pair (e.g. the notebook's 0.19/0.95).

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

History defaults to the ticker's **full listing period**; narrow with `--start`/`--end`. `backtest`/`optimize`/`walkforward` render matplotlib charts; `signal` builds a self-contained interactive Plotly page - a year of candles and volume with every fill marked (B/S triangles, guide lines, shaded holds), 1M/3M/6M/All range buttons, per-day OHLC/IBS hover, a light/dark toggle, and a phone-friendly layout. All commands accept `--save DIR` and `--no-plot`. `optimize`/`walkforward` objectives: `total_return` (default, Sharpe tiebreak), `cagr`, `sharpe`, `max_drawdown`, `win_rate` - use CAGR to compare _across_ windows of different lengths.

The default grids sweep entry over (0, 0.20] and exit over [0.80, 1.0) in **0.001 steps** - 40,000 pairs, scanned in under half a minute by a vectorized replay of the backtest engine (identical results to `run_backtest`, verified by tests). Coarser or narrower searches via `--entry-grid`/`--exit-grid A:B:STEP`. Finer grids fit noise more easily, so judge candidates by their walk-forward showing, not the in-sample leaderboard.

`--extend QQQ` prepends **synthetic pre-listing history** so TQQQ runs can start in 1999-03 instead of 2010-02 (see the methodology below); `--leverage` sets the synthetic daily leverage (default 3).

## Python API

```python
from ibs_strategy import load_data, run_backtest, grid_search, walk_forward, plot_backtest

data = load_data("TQQQ")  # full listing history; pass start/end to narrow

result = run_backtest(data)      # defaults: entry 0.13 / exit 0.5
print(result.summary())          # sharpe, total_return, cagr, max_drawdown, win_rate, ...
plot_backtest(result, ticker="TQQQ")

ranked = grid_search(data)       # every (entry, exit) pair, best row first
wf = walk_forward(data, n_folds=5, purge_days=5)
print(wf.summary())              # stitched out-of-sample metrics
```

## Methodology

### Backtest engine

`run_backtest` is a faithful port of the notebook's event-driven loop: previous-bar IBS signal, next-open fill, all-in whole-share sizing (leftover cash stays uninvested), one position at a time, strict threshold comparisons. Bars where `High == Low` have undefined IBS and never signal.

**Idle cash earns interest.** The strategy sits in cash >80% of the time, so paying 0% on it (the notebook's assumption) understates it. `--cash-rate` (default `^IRX`, the 13-week T-bill) accrues a yield on the cash balance before each bar's fill; `--cash-rate 0` restores the old behavior. Matters most in the high-rate 1999-2007 stretch (3-6.5%).

**Optional regime gate.** `run_backtest(..., regime=flags)` blocks entries while a boolean Series is off (and `regime_exit=True` also liquidates when it flips), same no-look-ahead timing as the signal. It exists only to *test* overlays like the 200-day SMA - which is how we learned [the filter removes the edge, not the risk](#where-the-edge-lives-below-the-200-day-sma). No CLI flag by design.

**Fill timing (the overnight edge).** `entry_fill`/`exit_fill` (default `"open"`; CLI `--entry-fill`/`--exit-fill`) fill a leg at the next open or at the signal bar's own **close**. The IBS reversion *begins overnight*: the bottom-quintile close→next-open return is positive and significant (t 2.7-3.6) and scales ~3x with leverage, so a close entry lifts extended TQQQ from **29.5% / 0.79 Sharpe to 41.1% / 0.95** (SPXL 22.4/0.82 → 28.2/0.92). The catch is execution - a close fills in the auction or a post-market print, at a spread. Modelled with a wide spread only on genuine crash days, **TQQQ's edge survives** (blended breakeven ~22 bp) while less-liquid **SPXL is roughly a wash** (~13 bp). The default stays `"open"` for universal executability; close entry is a real option on a liquid name if you can transact near the close.

**Position sizing.** `position_sizing="vol_target"` (CLI `--position-sizing vol_target`) scales each entry to a target annualized volatility - `min(1, target_vol / trailing 20-day vol)` of capital, the rest in cash, **capped at all-in (never levered)**. The strategy buys weak closes, so it enters disproportionately on high-vol days; sizing down there is the win. On extended TQQQ a 0.4 target holds **16.7% CAGR / 0.87 Sharpe / -34.5% drawdown** against **29.6% / 0.79 / -70.7%** all-in - about half the drawdown at a higher Sharpe, same time-in-market - and it beats a constant smaller size at matched exposure (-35% vs -52% drawdown), so it is timing, not deleveraging. Two limits: it is a risk *reducer*, not amplifier (levering it back to the baseline's drawdown needs ~9x underlying, Reg-T-inaccessible), and the plain estimator wins - forward vol (VXN/VIX) and IBS-*depth* tilts both fail to beat trailing realized vol. The engine default stays `"full"`; the [hosted page](https://cazsyd.github.io/IBS-Strategy/) defaults to `vol_target` and prints the fraction to deploy on the current signal.

### Does the signal actually predict anything?

Which parameters "won" is unanswerable (below), so settle the prior question separately: does a low IBS predict a higher forward return _at all_? `decile_response` pools every bar - bucket by IBS, then measure the return of the session a signal would have had you long (buy next open, mark at that close):

Reported by quintile (IBS < ~0.20 vs > ~0.80); which _decile_ peaks is not stable enough to report finer:

| Instrument | Bottom quintile (IBS < 0.20) | Top quintile | Rank corr | Split-half agreement |
| ---------- | ---------------------------- | ------------ | --------- | -------------------- |
| TQQQ (3x)  | **+0.395%** (t=2.75)         | -0.106%      | -0.83     | +0.43                |
| SPXL (3x)  | **+0.222%** (t=2.64)         | -0.100%      | -0.74     | **+0.79**            |
| QQQ (1x)   | +0.144% (t=3.02)             | -0.037%      | -0.83     | -                    |
| SPY (1x)   | +0.088% (t=3.05)             | -0.039%      | -0.70     | -                    |

Four things make this credible where the threshold surface was not: the gradient runs the same way on all four instruments (rank corr -0.70 to -0.83); the leveraged versions earn **2.5-2.7x** their underlyings' edge (close to the 3x a genuine price effect predicts, less fund costs and decay); the effect is present in the plain 1x underlyings, so it is not a leverage artifact; and the curve **replicates across sample halves** (+0.43, +0.79) where the Sharpe surface managed -0.07.

That is the evidence the strategy rests on, and it is what sets entry 0.13 - not an optimizer. The threshold sits inside the region carrying essentially all of the positive forward returns, and anywhere in roughly 0.10-0.20 would do as well.

```python
from ibs_strategy import decile_response, response_gradient, load_data

response = decile_response(load_data("TQQQ"))
print(response, response_gradient(response), sep="\n")
```

Caveats: the effect lives almost entirely in the bottom quintile (an extreme-value effect, not a smooth dose-response), fat tails flatter t-stats, and the bottom TQQQ *decile* alone isn't significant (t=1.00). And the **exit** gets no support here at all - forward returns stay positive well above the entry band, so any IBS exit is a risk/exposure choice; the default 0.5 is justified by drawdown control, not predictive power.

#### Reaching further back (and the data-quality guard)

More history is the only lever on resolution, but Yahoo's `^GSPC` hides two traps - both now guarded in code (`assess_ibs_quality`; `decile_response(require_genuine=/gap_immune=)`). Pre-1982 bars carry *reconstructed* intraday ranges (IBS std ~0.16 vs a real ~0.32; naively they flip the edge's sign to momentum), and `Open == prev Close` until ~2010 creates a shared-price artifact that inflates the modern edge ~40%. Cleaned, **1982-1992 confirms the edge out of sample** - gap-immune bottom-quintile **+0.118% (t=2.06)**, through the 1987 crash - though 11 extra years barely tighten the threshold surface. A genuine extension needs real futures OHLC, not Yahoo's index.

### Where the edge lives: below the 200-day SMA

Splitting the same test by regime - is the underlying index above or below its 200-day SMA? - localizes the effect completely. Mean next-session open->close return of bottom-quintile-IBS days (SMA on `^NDX` for the Nasdaq pair, `^GSPC` for the S&P pair, flag read at the prior close):

| Instrument       | Above the 200-day SMA | Below the 200-day SMA |
| ---------------- | --------------------- | --------------------- |
| TQQQ (3x, 1999+) | +0.035% (t=0.28)      | **+1.047%** (t=3.14)  |
| SPXL (3x, 1993+) | +0.055% (t=0.69)      | **+0.551%** (t=2.83)  |
| QQQ (1x)         | +0.019% (t=0.45)      | **+0.369%** (t=3.36)  |
| SPY (1x)         | +0.016% (t=0.61)      | **+0.226%** (t=3.38)  |

Above the SMA the edge is statistically zero on every instrument; below it, large and significant on every one. Buying panic is a *downtrend* phenomenon - only ~a third of signal days sit below the SMA, and they carry essentially all the edge; the above-SMA trades just captured drift a plain hold would have earned anyway.

This kills the most popular "fix" for crash exposure. Gating entries with the 200-day SMA keeps the no-edge trades and discards the ones with all the edge: on extended TQQQ it cuts CAGR 31.3% → 14.7% and *lowers* Sharpe 0.77 → 0.58, buying a smaller drawdown with most of the return. The direction replicates in all four half-samples, a length sweep asks for *less* filter all the way to none, and pure 200-SMA timing of the 3x fund still rode the dot-com crash to **-94.5%** (the index falls 25-30% before the cross triggers). The one setup it rescues is the patient 0.965 exit - proving the SMA is just a months-slow version of what the prompt IBS exit does in days.

The uncomfortable conclusion: **the edge and the crash risk are the same trades.** The premium comes from buying panic in downtrends, which is also the only place a crash can catch the strategy. No trend filter can remove the tail without removing the return - position size and leverage choice are the only levers that actually control it.

```python
from ibs_strategy import load_data

data, index = load_data("QQQ"), load_data("^NDX")["Close"]
above = (index > index.rolling(200).mean()).reindex(data.index).ffill()
forward = (data["Close"] / data["Open"] - 1).shift(-1)
signal = data["IBS"] <= data["IBS"].quantile(0.2)
print(forward[signal & above].mean(), forward[signal & ~above].mean())
```

### Does the edge generalize? Universe breadth

Everything above was measured on four instruments picked *because they worked* - textbook selection bias. The honest test freezes the signal and runs it unchanged across a deliberately diverse universe, including assets outside US equities where a genuine effect should stand or fall on its own. Same metric as the dose-response table (the bottom-IBS-quintile session's next-open→close return, in basis points, with its t-stat):

| Instrument | Class | Bottom-quintile fwd | t | vs top quintile |
| ---------- | ----- | ------------------- | - | --------------- |
| XLF  | US financials      | +16.0 bp | 3.56  | +26.4 bp |
| XLK  | US tech sector     | +13.8 bp | 3.30  | +19.6 bp |
| FXI  | China              | +13.5 bp | 3.26  | +16.4 bp |
| SPY  | US large-cap       | +8.8 bp  | 3.09  | +9.5 bp  |
| EEM  | Emerging markets   | +10.7 bp | 2.96  | +12.6 bp |
| QQQ  | US Nasdaq-100      | +14.0 bp | 2.89  | +18.2 bp |
| SOXX | US semiconductors  | +12.7 bp | 2.56  | +24.6 bp |
| DIA  | US Dow-30          | +7.5 bp  | 2.52  | +9.6 bp  |
| IWM  | US small-cap       | +9.8 bp  | 2.51  | +13.5 bp |
| EFA  | Developed intl     | +6.8 bp  | 2.28  | +3.8 bp  |
| XLE  | US energy          | +8.4 bp  | 1.96  | +15.2 bp |
| TLT  | US 20y Treasuries  | +1.7 bp  | 0.89  | +1.0 bp  |
| GLD  | Gold               | -2.0 bp  | -0.78 | -4.4 bp  |

Positive and significant (t>2) in **10 of 13** - every equity index, sector, and region tested, across size, style, and geography. Not a Nasdaq artifact. The decisive rows are the last three: **bonds (TLT) and gold (GLD) show nothing**. That equity/non-equity boundary is the strongest single piece of evidence here that the effect is *real* - a data-mined coincidence would not confine itself to equities; a behavioral overreaction to weak equity closes would. It is an equity-microstructure edge, full stop.

The default strategy runs positive on all seven tradable 3x ETFs (SOXL 0.81 Sharpe, TECL 0.83, TQQQ 0.78, TNA 0.70, UDOW 0.62, SPXL 0.55, TMF 0.51 - though TMF earns its 9%/yr *parking in cash* while bonds fell, not from an edge it lacks). Kept explicit: these are **survivors** - leveraged ETFs close, so the list is pre-filtered for success.

**A basket diversifies the one risk a stop can't.** Each sleeve reverts on its *own* weak closes - mostly different days - so strategy returns across six low-correlation sleeves (QQQ, IWM, TLT, GLD, EEM, XLE) correlate just **0.21**. Equal-weighting them lifts Sharpe from a 0.72 single-name average to **1.14** and shrinks max drawdown from −22% to **−16%** - a higher Sharpe than any single component. The non-equity sleeves don't revert, but they don't crash *with* equities either, so they cushion the systemic tail that correlates every equity sleeve at once - the one drawdown no per-instrument exit or stop can reach.

**And as a sleeve in a conventional portfolio.** The vol-target strategy correlates just **0.40** to a 60/40, so a small allocation improves it on all three axes at once: at 20% a 60/40's Sharpe rises 0.82 → 0.99 and its worst drawdown *shrinks* from -31% to -23% (100% equity: 0.66 → 0.79). The Sharpe-maximizing weight is ~40-60%, but that over-bets one leveraged strategy - **10-20% captures the diversification** without the concentration. This, not the standalone CAGR, is where the strategy earns its keep.

### Why the thresholds are not optimized

Fitting the Sharpe surface on the first and second halves of the history separately and correlating them gives **-0.07 on TQQQ and -0.01 on SPXL**. The shape of the surface in one half predicts nothing about the other, and a peak scoring Sharpe 1.25 in-sample scores **0.30** on the unseen half.

This is not evidence the strategy is broken - only that the grid cannot be read. The SE on any single cell's annualized return is ~**±10.5% (TQQQ)** / **±6.3% (SPXL)**, larger than the 2-3 point gaps between cells, so even 27 years can't resolve one threshold pair from another and anything picked from the surface's shape is a coin flip - which is why in-sample optima deliver ~half their advertised CAGR out-of-sample, and why re-fitting per ticker [made things worse](#extended-history-where-the-defaults-were-actually-chosen).

`plateau_thresholds` (`--selector plateau`) is therefore a **tie-breaker, not a discovery**: it averages each cell with its neighbours before the argmax, so it at least refuses to chase spikes. Neither number deserves three significant digits.

An unreadable *grid* is not the same as a fitted *edge*, though - and a deflated Sharpe ratio (Bailey-Lopez de Prado) settles the second claim. Across 475 threshold configs on extended TQQQ the best in-sample Sharpe (0.93) sits well above the expected-max-under-null of **0.46** - the luck bar for that many trials - for a **deflated Sharpe of 0.995**; the structural default (0.80) clears the same bar at 96.7% (both after charging for the returns' fat tails, kurtosis ~39). The search finds a real edge it simply cannot pin to a specific cell: real signal, un-tunable knob.

### Metrics (as defined in the notebook)

| Metric       | Definition                                                                               |
| ------------ | ---------------------------------------------------------------------------------------- |
| Sharpe ratio | `mean(daily returns) / std(ddof=1) * sqrt(252)`, zero risk-free rate in the ratio, flat days included |
| Total return | `final capital / initial capital - 1`                                                    |
| Max drawdown | deepest peak-to-trough decline of the equity curve                                       |
| Win rate     | share of _closed_ trades whose exit fill beat the entry fill                             |

The package additionally reports CAGR, time-in-market, trade count, and final capital.

### Purged walk-forward validation

The notebook picked its "ideal" thresholds by optimizing over the whole backtest period - an in-sample estimate that flatters results. `walk_forward` addresses that:

1. The first `min_train_frac` (default 50%) of the history seeds the training window; the rest is split into `n_folds` sequential test windows.
2. For each fold, the threshold grid is re-optimized on all data _before_ the test window, minus a `purge_days` gap (default 5 trading days) so boundary fills and still-open positions can't leak information across the split.
3. The chosen thresholds are evaluated once on the unseen test window (starting flat), and the per-fold equity segments are compounded into a single out-of-sample curve.

### Synthetic pre-listing history (1999+)

TQQQ only lists from 2010-02, so its real history misses the dot-com crash. `--extend QQQ` (API: `load_extended_data`) prepends synthetic 3x bars derived from QQQ, which trades since 1999-03-10 - Yahoo's single `QQQ` symbol also covers its Amex and QQQQ-era listings:

- A daily-rebalanced fund resets leverage at each close, so every intraday price relative to the previous close moves at `leverage` times the proxy's. Open/high/low/close therefore map through one affine transform per bar, and **IBS - hence every signal - is exactly the proxy's**: pre-2010 the strategy is trading QQQ's IBS at 3x.
- Daily costs: 0.95%/yr expense ratio plus **financing of the borrowed 2x exposure** at the 13-week T-bill yield (`^IRX`) + a 0.5%/yr swap spread, deducted uniformly across each bar. The spread is calibrated on the 2010-2026 overlap, where the model tracks real TQQQ to **+0.07%/yr CAGR drift at 0.9989 daily-return correlation**; skipping the financing leg (as naive 3x reconstructions do) would overshoot by ~5.7%/yr.
- The path is scaled so the seam overnight move into the first real bar equals the modeled 3x proxy move, and a boolean `Synthetic` column marks reconstructed bars.

Caveats: the synthetic era models no tracking error or intraday-rebalancing slippage, and it assumes the *wrapper survives* - no 3x equity ETF existed before 2008, and sponsors have cut leverage mid-crisis, which would quietly break the 3x assumption through exactly the crash the fund's own listing history happens not to contain. Defaults are chosen on this extended history anyway, because that crash is a 3x fund's defining risk.

### Results snapshot (TQQQ, real listing history 2010-02 to 2026-07, checked July 2026)

This window flatters the patient exit - no crash in it, so holding longer just captures more drift. Each value carries **± one standard error** from a 1,000-run moving-block bootstrap (21-day blocks); at 3x leverage even decades of daily data pin these loosely. The first two columns are the same 0.13/0.5 signal at **full** (all-in) and **vol-target 0.4** sizing, so the gap between them isolates what risk-adjusted sizing buys.

| Metric                     | IBS 0.13 / 0.5 (full)      | IBS 0.13 / 0.5 (vol-target 0.4) | IBS 0.132 / 0.965 (patient exit) | Buy & hold        | QQQ 1.13x (risk-matched) |
| -------------------------- | ----------------------------- | ------------------------------- | -------------------------------- | ----------------- | ------------------------ |
| CAGR                       | 20.3% ± 7.5%                  | 15.2% ± 4.3%                    | **59.3% ± 16.3%**                | 40.9% ± 20.1%     | 21.0% ± 6.0%             |
| Total return               | +1,989%                       | +926%                           | +213,619%                        | +27,991%          | +2,191%                  |
| Sharpe ratio               | 0.78 ± 0.22                   | 0.86 ± 0.21                     | 1.17 ± 0.21                      | 0.87 ± 0.24       | 0.94 ± 0.23              |
| Max drawdown               | -39.2% ± 10.0%            | **-34.5% ± 6.8%**                   | -56.8% ± 9.3%                    | -81.7% ± 9.9%     | -39.2% ± 8.1%            |
| Win rate                   | 65.3% ± 2.4% (392 trades)     | 65.3% ± 2.4% (392)              | 74.6% ± 3.3% (177)               | -                 | -                        |
| Time in market             | **16.2% ± 0.9%**              | 16.2% ± 0.8%                    | 62.3% ± 2.1%                     | 100%              | 100%                     |
| Final capital ($10k start) | $0.21M                        | $0.10M                          | $21.4M                           | $2.81M            | $0.23M                   |

On this crash-free decade the prompt default is deliberately timid - in cash ~84% of the time, it trails buy & hold and the patient exit on return, winning only on drawdown. The patient exit beats buy & hold ~8x in final capital here - precisely the trap: add a crash and it inverts.

The last column reframes the benchmark: since the default sits ~16% invested, comparing it to a 100% hold measures exposure, not skill. The fairer yardstick is the plain 1x index **levered to the strategy's own drawdown** (here QQQ at 1.13x, financed at T-bills). Matched that way, passive still wins this crash-free decade (21.0% vs 20.3% at a higher Sharpe) - with no crash to truncate, the cash-heavy default pays for insurance the window never claims.

![TQQQ backtest at the default thresholds: equity vs buy & hold, drawdown](docs/backtest.png)

- **Purged walk-forward, OOS 2018-04 to 2026-07** (grid re-optimized per fold): 46.6% CAGR vs 34.8% buy & hold (Sharpe 0.93, -61.8% drawdown). Trained on crash-free data, the folds converge on the patient exit - exactly how a strategy talks itself into the config that dies in 2000.
- Engine parity: over the original notebook's window (2020-01 to 2025-07) with its 0.19/0.95 thresholds, the engine reproduces the notebook's reported numbers (Sharpe 1.286 vs 1.283, max drawdown -46.48% vs -46.47%) up to Yahoo's adjusted-price revisions.

![TQQQ walk-forward out-of-sample equity with per-fold thresholds](docs/walkforward.png)

Caveat: these rows are fitted in-sample and model no commissions, slippage, or taxes - and the default trades more than twice as often as the patient exit, so it carries roughly double the slippage and realizes short-term gains twice as often.

### Extended history, where the defaults were actually chosen

Add the dot-com crash and the GFC (`--extend`) and the ranking reverses on **every** axis - the default earns more *and* risks less than the patient exit, on both tickers:

**TQQQ, 1999-03 to 2026-07** (synthetic + real bars):

| Metric                     | IBS 0.13 / 0.5 (full)      | IBS 0.13 / 0.5 (vol-target 0.4) | IBS 0.132 / 0.965 (patient exit) | Buy & hold        | QQQ 0.74x (risk-matched) |
| -------------------------- | ----------------------------- | ------------------------------- | -------------------------------- | ----------------- | ------------------------ |
| CAGR                       | **29.6% ± 8.5%**              | 16.7% ± 3.6%                    | 23.0% ± 15.0%                    | 2.2% ± 15.6%      | 9.0% ± 3.7%              |
| Total return               | +120,552%                     | +6,743%                         | +28,753%                         | +81%              | +957%                    |
| Sharpe ratio               | 0.79 ± 0.14               | **0.87 ± 0.16**                     | 0.64 ± 0.17                      | 0.43 ± 0.18       | 0.53 ± 0.18              |
| Max drawdown               | -70.7% ± 9.3%             | **-34.5% ± 6.7%**                   | -99.2% ± 6.4%                    | -99.98% ± 3.6%    | -70.7% ± 10.1%           |
| Win rate                   | 62.9% ± 1.8% (753 trades)     | 62.9% ± 1.8% (753)              | 68.7% ± 2.7% (294)               | -                 | -                        |
| Time in market             | **19.1% ± 0.7%**              | 19.1% ± 0.7%                    | 66.7% ± 1.6%                     | 100%              | 100%                     |
| Final capital ($10k start) | $12.07M                       | $0.68M                          | $2.89M                           | $18.1k            | $0.11M                   |

**SPXL, 1993-02 to 2026-07** (extended via SPY, so it spans two crashes):

| Metric                     | IBS 0.13 / 0.5 (full)      | IBS 0.13 / 0.5 (vol-target 0.4) | IBS 0.132 / 0.965 (patient exit) | Buy & hold        | SPY 0.91x (risk-matched) |
| -------------------------- | ----------------------------- | ------------------------------- | -------------------------------- | ----------------- | ------------------------ |
| CAGR                       | **22.6% ± 5.7%**              | 16.0% ± 3.4%                    | 18.7% ± 8.2%                     | 13.9% ± 10.3%     | 10.1% ± 2.8%             |
| Sharpe ratio               | 0.83 ± 0.16               | **0.88 ± 0.16**                     | 0.60 ± 0.15                      | 0.51 ± 0.16       | 0.65 ± 0.16              |
| Max drawdown               | -51.5% ± 9.5%             | **-37.1% ± 7.8%**                   | -91.4% ± 9.0%                    | -98.2% ± 7.9%     | -51.5% ± 9.1%            |
| Time in market             | **18.7% ± 0.6%**              | 18.7% ± 0.6%                    | 63.4% ± 1.5%                     | 100%              | 100%                     |
| Final capital ($10k start) | $9.23M                        | $1.45M                          | $3.08M                           | $0.77M            | $0.25M                   |

Read with the error bars: the default's CAGR *advantage* over the patient exit sits inside one SE (not distinguishable), while its drawdown advantage (-70.7% vs -99.2%, -51.5% vs -91.4%) is two-to-three SE clear. The measurable edge over holding longer is **lower risk, not higher return** - the risk-transformer thesis, quantified. The `vol-target 0.4` column sharpens it from the sizing side (drawdown to **-34.5% / -37.1%** at the highest Sharpe in each table); and unlike the patient exit's *illusory* low risk - a 68-73% win rate hiding a -99% tail in still-open positions - this reduction is real, two-plus SE clear.

The risk-matched column is the fair benchmark. Raw buy & hold swings 41% / 2% / 14% CAGR on nothing but the start date; levered to the strategy's *own* drawdown, that passive alternative earns just 9-10% where the strategy earns 22-30% - the edge, invisible against a 100% hold, reappears. (Match on *volatility* instead and passive needs 1.69x for a -97% drawdown: you cannot lever a plain index to this risk without courting ruin, because the strategy spends its risk budget in episodic bites, not continuously.)

- Walk-forward with training anchored at 1999 (out-of-sample 2012-11 to 2026-07, all real bars): **32.4% CAGR, Sharpe 0.93, -39.9% max drawdown at just 27% time in market** - far safer than buy & hold, though trailing its 43.8% CAGR through a crash-free bull era. The crash-taught exit keeps paying for insurance that period never needed.
- Per-ticker tuning does **not** help: re-fitting on SPXL's own history lost 5.4 CAGR points out-of-sample vs the shared default. IBS is range-normalized, so a fixed threshold already fires on 11.7-13.5% of days across instruments whose volatility differs 3.4x - there is no per-instrument quantity to adapt to.

![1999-2026 equity on log scale: the default 0.13/0.5 vs buy & hold](docs/extended.png)

Charts regenerate with `uv run python scripts/build_readme_charts.py`.

## CI & the hosted signal page

Two GitHub Actions workflows live in `.github/workflows/`:

- **`ci.yml`** - runs `uv sync --locked` + the pytest suite on every push and pull request.
- **`pages.yml`** - rebuilds the interactive signal pages (`scripts/build_site.py`, TQQQ + SPXL) and deploys them to GitHub Pages on every push to `main` and on a weekday schedule (21:30 UTC, after the 4pm ET close), so the hosted page always shows the latest completed session:

**Live signals: <https://cazsyd.github.io/IBS-Strategy/>**

One-time setup after pushing: in the repo's **Settings → Pages**, set **Source** to **GitHub Actions**.

## Project layout

```
├── IBS_strategy.ipynb        # original Colab notebook (kept as-is)
├── pyproject.toml            # uv-managed project + dependencies
├── uv.lock                   # locked environment
├── .github/workflows/        # CI (tests) + GitHub Pages deploy
├── scripts/build_site.py     # builds the hosted signal pages
├── src/ibs_strategy/
│   ├── data.py               # yfinance download + IBS computation
│   ├── backtest.py           # event-driven backtest engine
│   ├── metrics.py            # notebook metric definitions
│   ├── optimize.py           # grid search + purged walk-forward
│   ├── edge.py               # IBS decile forward-return test + OHLC data-quality guard
│   ├── synthetic.py          # synthetic pre-listing history (3x QQQ back to 1999)
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
