"""Matplotlib charts for backtests, threshold grids, and walk-forward results.

Colors follow a validated light-mode palette: one blue series hue, muted gray
for benchmark context, and reserved status green/red for buy/sell markers
(shape carries the same information, so meaning is never color-alone).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .metrics import drawdown_series

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .optimize import WalkForwardResult

__all__ = [
    "plot_signals",
    "plot_equity",
    "plot_drawdown",
    "plot_backtest",
    "plot_heatmap",
    "plot_walk_forward",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BLUE = "#2a78d6"
BUY_GREEN = "#0ca30c"
SELL_RED = "#d03b3b"

SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list(
    "ibs_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

PERCENT_METRICS = {"total_return", "cagr", "max_drawdown", "win_rate"}
METRIC_LABELS = {
    "total_return": "Total return (%)",
    "cagr": "CAGR (%)",
    "sharpe": "Sharpe ratio",
    "max_drawdown": "Max drawdown (%)",
    "win_rate": "Win rate (%)",
    "num_trades": "Closed trades",
}


def _new_axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=11)


def _legend(ax: Axes) -> None:
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)


def plot_signals(
    result: "BacktestResult",
    ax: Axes | None = None,
    price_col: str = "Open",
    title: str | None = None,
) -> Axes:
    """Price line with the backtest's actual trade entries (^) and exits (v).

    Markers sit at fill prices (the bar's open) and held spans are shaded, so
    repeated raw threshold crossings inside a position don't show up as
    duplicate signals.
    """
    ax = _new_axes(ax, (12, 5))
    data = result.data
    ax.plot(
        data.index,
        data[price_col],
        color=INK_SECONDARY,
        linewidth=1.4,
        label=f"{price_col} price",
    )
    for trade in result.trades:
        end = trade.exit_date if trade.exit_date is not None else data.index[-1]
        ax.axvspan(trade.entry_date, end, color=SERIES_BLUE, alpha=0.07, linewidth=0)
    entries = [(trade.entry_date, trade.entry_price) for trade in result.trades]
    exits = [(trade.exit_date, trade.exit_price) for trade in result.trades if not trade.is_open]
    if entries:
        x, y = zip(*entries)
        ax.scatter(x, y, marker="^", s=80, color=BUY_GREEN, zorder=3, label="Buy (entry fill)")
    if exits:
        x, y = zip(*exits)
        ax.scatter(x, y, marker="v", s=80, color=SELL_RED, zorder=3, label="Sell (exit fill)")
    ax.set_ylabel("Price", color=INK_SECONDARY, fontsize=9.5)
    _title(
        ax,
        title
        or f"Trades - entry < {result.entry_threshold:g}, exit > {result.exit_threshold:g}",
    )
    _legend(ax)
    return ax


def plot_equity(result: "BacktestResult", ax: Axes | None = None, title: str | None = None) -> Axes:
    """Strategy equity vs buy & hold, both indexed to 100 at the start."""
    ax = _new_axes(ax, (12, 5))
    data = result.data
    strategy = data["Capital"] / data["Capital"].iloc[0] * 100
    ax.plot(data.index, strategy, color=SERIES_BLUE, linewidth=1.8, label="IBS strategy")
    if "Close" in data.columns:
        market = data["Close"] / data["Close"].iloc[0] * 100
        ax.plot(data.index, market, color=INK_MUTED, linewidth=1.4, label="Buy & hold")
        _legend(ax)
    ax.set_ylabel("Value (start = 100)", color=INK_SECONDARY, fontsize=9.5)
    _title(ax, title or "Equity - IBS strategy vs buy & hold")
    return ax


def plot_drawdown(result: "BacktestResult", ax: Axes | None = None, title: str | None = None) -> Axes:
    """Strategy drawdown from the running equity peak, with the trough marked."""
    ax = _new_axes(ax, (12, 3.2))
    drawdown = drawdown_series(result.equity) * 100
    ax.plot(drawdown.index, drawdown, color=SELL_RED, linewidth=1.4)
    ax.fill_between(drawdown.index, drawdown, 0, color=SELL_RED, alpha=0.12, linewidth=0)
    trough = drawdown.idxmin()
    ax.scatter([trough], [drawdown.min()], color=SELL_RED, s=45, zorder=3)
    ax.text(
        0.995,
        0.06,
        f"max drawdown {drawdown.min():.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set_ylim(min(drawdown.min() * 1.35, -1.0), 1.0)
    ax.set_ylabel("Drawdown (%)", color=INK_SECONDARY, fontsize=9.5)
    _title(ax, title or "Strategy drawdown")
    return ax


def plot_backtest(result: "BacktestResult", ticker: str | None = None) -> Figure:
    """Three stacked panels: trades on price, equity vs buy & hold, drawdown."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.6, 1.0]},
    )
    fig.set_facecolor(SURFACE)
    prefix = f"{ticker} - " if ticker else ""
    plot_signals(
        result,
        ax=axes[0],
        title=f"{prefix}trades at entry < {result.entry_threshold:g}, exit > {result.exit_threshold:g}",
    )
    plot_equity(result, ax=axes[1], title="Equity - IBS strategy vs buy & hold (start = 100)")
    plot_drawdown(result, ax=axes[2])
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def plot_heatmap(
    results: pd.DataFrame,
    metric: str = "total_return",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Entry x exit heatmap of ``metric`` from a ``grid_search`` results table."""
    table = results.pivot_table(index="exit_threshold", columns="entry_threshold", values=metric)
    table = table.sort_index(ascending=False)
    values = table.to_numpy(dtype=float)
    if metric in PERCENT_METRICS:
        values = values * 100
    n_rows, n_cols = values.shape

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 1.6 + 0.6 * n_cols), max(3.2, 1.2 + 0.42 * n_rows)))
    fig = ax.figure
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    image = ax.imshow(values, cmap=SEQUENTIAL_BLUES, aspect="auto")
    ax.set_xticks(range(n_cols), [f"{column:g}" for column in table.columns])
    ax.set_yticks(range(n_rows), [f"{row:g}" for row in table.index])
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    if values.size <= 150:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        span = vmax - vmin
        for i in range(n_rows):
            for j in range(n_cols):
                value = values[i, j]
                if np.isnan(value):
                    continue
                fraction = 0.5 if span == 0 else (value - vmin) / span
                color = "#ffffff" if fraction > 0.55 else INK
                label = f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    colorbar.set_label(METRIC_LABELS.get(metric, metric), color=INK_SECONDARY, fontsize=9)

    ax.set_xlabel("Entry threshold (buy below)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Exit threshold (sell above)", color=INK_SECONDARY, fontsize=9.5)
    _title(ax, title or "Threshold grid")
    return ax


def plot_walk_forward(wf: "WalkForwardResult", ax: Axes | None = None, title: str | None = None) -> Axes:
    """Stitched out-of-sample equity vs buy & hold, with fold boundaries.

    Each fold boundary is annotated with the entry/exit thresholds that fold
    chose on its training window.
    """
    ax = _new_axes(ax, (12, 5.5))
    equity = wf.oos_equity / wf.initial_capital * 100
    ax.plot(
        equity.index,
        equity.to_numpy(),
        color=SERIES_BLUE,
        linewidth=1.8,
        label="IBS strategy (out-of-sample)",
    )
    closes = pd.concat([fold.result.data["Close"] for fold in wf.folds])
    market = closes / closes.iloc[0] * 100
    ax.plot(market.index, market.to_numpy(), color=INK_MUTED, linewidth=1.4, label="Buy & hold")
    for fold in wf.folds:
        ax.axvline(fold.test_start, color=BASELINE, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.annotate(
            f"{fold.entry_threshold:g}/{fold.exit_threshold:g}",
            xy=(fold.test_start, 0.99),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_ylabel("Value (start = 100)", color=INK_SECONDARY, fontsize=9.5)
    _title(ax, title or "Walk-forward out-of-sample equity (labels: entry/exit per fold)")
    _legend(ax)
    return ax
