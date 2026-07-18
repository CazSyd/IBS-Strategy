"""Interactive HTML signal page: candlesticks, trade markers, hover details.

Renders a self-contained Plotly page (works offline) with daily candles, a
volume row, prominent entry/exit markers (labeled triangles plus dashed guide
lines), shaded holding periods, per-day hover showing OHLC, IBS, volume and
the daily change, and a light/dark theme toggle that follows the OS
preference and remembers the choice.
"""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import BacktestResult
from .live import SignalReport

__all__ = ["build_signal_figure", "render_signal_page", "open_in_browser"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CANDLE_UP = "#1baf7a"
CANDLE_DOWN = "#e34948"
BUY_GREEN = "#0ca30c"
SELL_RED = "#d03b3b"
SPAN_BLUE = "#2a78d6"

DARK_SURFACE = "#1a1a19"
DARK_INK = "#ffffff"
DARK_INK_SECONDARY = "#c3c2b7"
DARK_GRIDLINE = "#2c2c2a"
DARK_BASELINE = "#383835"

SIGNAL_COLORS = {"BUY": BUY_GREEN, "SELL": SELL_RED, "HOLD": INK_MUTED}

# Layout patches applied client-side by the theme toggle (Plotly.relayout).
_LIGHT_PATCH = {
    "paper_bgcolor": SURFACE,
    "plot_bgcolor": SURFACE,
    "font.color": INK_SECONDARY,
    "title.font.color": INK,
    "hoverlabel.bgcolor": "#ffffff",
    "hoverlabel.bordercolor": GRIDLINE,
    "hoverlabel.font.color": INK,
    "xaxis.gridcolor": GRIDLINE,
    "xaxis.linecolor": BASELINE,
    "xaxis.spikecolor": INK_MUTED,
    "xaxis.rangeselector.bgcolor": SURFACE,
    "xaxis.rangeselector.activecolor": GRIDLINE,
    "xaxis.rangeselector.font.color": INK_SECONDARY,
    "xaxis2.gridcolor": GRIDLINE,
    "xaxis2.linecolor": BASELINE,
    "xaxis2.spikecolor": INK_MUTED,
    "yaxis.gridcolor": GRIDLINE,
    "yaxis.linecolor": BASELINE,
    "yaxis2.gridcolor": GRIDLINE,
    "yaxis2.linecolor": BASELINE,
}
_DARK_PATCH = {
    "paper_bgcolor": DARK_SURFACE,
    "plot_bgcolor": DARK_SURFACE,
    "font.color": DARK_INK_SECONDARY,
    "title.font.color": DARK_INK,
    "hoverlabel.bgcolor": "#262624",
    "hoverlabel.bordercolor": DARK_BASELINE,
    "hoverlabel.font.color": DARK_INK,
    "xaxis.gridcolor": DARK_GRIDLINE,
    "xaxis.linecolor": DARK_BASELINE,
    "xaxis.spikecolor": INK_MUTED,
    "xaxis.rangeselector.bgcolor": DARK_SURFACE,
    "xaxis.rangeselector.activecolor": DARK_BASELINE,
    "xaxis.rangeselector.font.color": DARK_INK_SECONDARY,
    "xaxis2.gridcolor": DARK_GRIDLINE,
    "xaxis2.linecolor": DARK_BASELINE,
    "xaxis2.spikecolor": INK_MUTED,
    "yaxis.gridcolor": DARK_GRIDLINE,
    "yaxis.linecolor": DARK_BASELINE,
    "yaxis2.gridcolor": DARK_GRIDLINE,
    "yaxis2.linecolor": DARK_BASELINE,
}

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --page: #f9f9f7; --chip-bg: #fcfcfb; --chip-ink: #52514e; --chip-border: rgba(11, 11, 11, 0.12); }
  :root[data-theme="dark"] { --page: #0d0d0d; --chip-bg: #1a1a19; --chip-ink: #c3c2b7; --chip-border: rgba(255, 255, 255, 0.14); }
  html, body { margin: 0; background: var(--page); transition: background 0.2s ease; }
  body { padding: 0 12px 6px; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; justify-content: flex-end; padding: 10px 4px 6px; }
  #theme-toggle { padding: 6px 14px; border-radius: 999px; border: 1px solid var(--chip-border);
                  background: var(--chip-bg); color: var(--chip-ink); font: inherit; font-size: 13px;
                  cursor: pointer; }
  #theme-toggle:hover { filter: brightness(0.95); }
</style>
</head>
<body>
<header><button id="theme-toggle" type="button">Dark</button></header>
__PLOT__
<script>
(function () {
  var LIGHT = __LIGHT__;
  var DARK = __DARK__;
  var chart = document.getElementById("ibs-chart");
  var button = document.getElementById("theme-toggle");
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    button.textContent = theme === "dark" ? "\\u2600 Light" : "\\u263E Dark";
    try { localStorage.setItem("ibs-theme", theme); } catch (err) {}
    Plotly.relayout(chart, theme === "dark" ? DARK : LIGHT);
  }
  var saved = null;
  try { saved = localStorage.getItem("ibs-theme"); } catch (err) {}
  apply(saved || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  button.addEventListener("click", function () {
    apply(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
})();
</script>
</body>
</html>
"""


def _date_str(value) -> str:
    """ISO date string for plotly x-values (keeps the figure JSON-serializable)."""
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def build_signal_figure(
    result: BacktestResult,
    ticker: str,
    report: SignalReport | None = None,
) -> go.Figure:
    """Candlestick chart of ``result.data`` with entry/exit markers."""
    data = result.data
    has_volume = "Volume" in data.columns
    dates = [_date_str(value) for value in data.index]

    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22] if has_volume else None,
        vertical_spacing=0.03,
    )

    change = (data["Close"].pct_change() * 100).fillna(0)
    hover_text = []
    for i in range(len(data)):
        parts = [f"IBS {data['IBS'].iloc[i]:.3f}", f"Change {change.iloc[i]:+.2f}%"]
        if has_volume:
            parts.append(f"Volume {data['Volume'].iloc[i]:,.0f}")
        hover_text.append("<br>".join(parts))

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing=dict(line=dict(color=CANDLE_UP, width=1.2), fillcolor=CANDLE_UP),
            decreasing=dict(line=dict(color=CANDLE_DOWN, width=1.2), fillcolor=CANDLE_DOWN),
            text=hover_text,
            name=ticker,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for trade in result.trades:
        span_end = trade.exit_date if trade.exit_date is not None else data.index[-1]
        fig.add_vrect(
            x0=_date_str(trade.entry_date), x1=_date_str(span_end),
            fillcolor=SPAN_BLUE, opacity=0.08, line_width=0,
            row=1, col=1,
        )
        fig.add_vline(
            x=_date_str(trade.entry_date),
            line_width=1, line_dash="dot", line_color=BUY_GREEN, opacity=0.5,
            row=1, col=1,
        )
        if trade.exit_date is not None:
            fig.add_vline(
                x=_date_str(trade.exit_date),
                line_width=1, line_dash="dot", line_color=SELL_RED, opacity=0.5,
                row=1, col=1,
            )

    # marker offset so triangles sit just outside the candle they belong to
    ranges = (data["High"] - data["Low"]).to_numpy(dtype=float)
    pad = float(np.nanmedian(ranges))
    if not np.isfinite(pad) or pad == 0:
        pad = float(data["Close"].iloc[-1]) * 0.01

    entries = result.trades
    exits = [trade for trade in result.trades if not trade.is_open]
    if entries:
        fig.add_trace(
            go.Scatter(
                x=[_date_str(trade.entry_date) for trade in entries],
                y=[data["Low"].loc[trade.entry_date] - 0.9 * pad for trade in entries],
                mode="markers+text",
                name="Buy",
                text=["B"] * len(entries),
                textposition="bottom center",
                textfont=dict(size=11, color=BUY_GREEN, weight=700),
                marker=dict(symbol="triangle-up", size=16, color=BUY_GREEN,
                            line=dict(color="#ffffff", width=1.5)),
                customdata=[[trade.shares, trade.entry_price] for trade in entries],
                hovertemplate="BUY %{customdata[0]:.0f} sh @ %{customdata[1]:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if exits:
        fig.add_trace(
            go.Scatter(
                x=[_date_str(trade.exit_date) for trade in exits],
                y=[data["High"].loc[trade.exit_date] + 0.9 * pad for trade in exits],
                mode="markers+text",
                name="Sell",
                text=["S"] * len(exits),
                textposition="top center",
                textfont=dict(size=11, color=SELL_RED, weight=700),
                marker=dict(symbol="triangle-down", size=16, color=SELL_RED,
                            line=dict(color="#ffffff", width=1.5)),
                customdata=[
                    [trade.shares, trade.exit_price, (trade.return_pct or 0) * 100]
                    for trade in exits
                ],
                hovertemplate=(
                    "SELL %{customdata[0]:.0f} sh @ %{customdata[1]:.2f} "
                    "(%{customdata[2]:+.1f}%)<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if has_volume:
        colors = np.where(data["Close"] >= data["Open"], CANDLE_UP, CANDLE_DOWN).tolist()
        fig.add_trace(
            go.Bar(
                x=dates,
                y=data["Volume"],
                marker_color=colors,
                marker_line_width=0,
                opacity=0.6,
                name="Volume",
                showlegend=False,
                hovertemplate="Volume %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    title = (
        f"{ticker} - IBS strategy trades "
        f"<span style='color:{INK_MUTED};font-size:13px'>entry &lt; "
        f"{result.entry_threshold:g} · exit &gt; {result.exit_threshold:g}</span>"
    )
    if report is not None:
        color = SIGNAL_COLORS.get(report.signal, INK_MUTED)
        title = (
            f"{ticker}  <span style='color:{color}'><b>{report.signal}</b></span>  "
            f"<span style='color:{INK_MUTED};font-size:13px'>IBS {report.ibs:.3f} on "
            f"{report.bar_date:%Y-%m-%d} · entry &lt; {result.entry_threshold:g} · "
            f"exit &gt; {result.exit_threshold:g}</span>"
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=18), x=0.01, xanchor="left"),
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  size=12, color=INK_SECONDARY),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRIDLINE,
                        font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=90, b=30),
        height=760,
        xaxis_rangeslider_visible=False,
        bargap=0.15,
    )
    fig.update_xaxes(
        gridcolor=GRIDLINE,
        linecolor=BASELINE,
        showspikes=True,
        spikecolor=INK_MUTED,
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )
    fig.update_yaxes(gridcolor=GRIDLINE, linecolor=BASELINE, zeroline=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=SURFACE,
            activecolor=GRIDLINE,
            font=dict(color=INK_SECONDARY),
        ),
        row=1,
        col=1,
    )
    return fig


def render_signal_page(
    result: BacktestResult,
    ticker: str,
    report: SignalReport | None = None,
    path: Path | None = None,
) -> Path:
    """Write the signal page as a self-contained HTML file and return its path.

    The page embeds plotly.js (works offline) and a light/dark toggle that
    follows the OS color scheme on first load and persists the user's choice.
    """
    fig = build_signal_figure(result, ticker, report)
    if path is None:
        path = Path(tempfile.gettempdir()) / f"{ticker.lower()}_ibs_signals.html"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="ibs-chart",
        config={"displaylogo": False},
    )
    page = (
        _PAGE_TEMPLATE
        .replace("__TITLE__", f"{ticker} IBS signals")
        .replace("__LIGHT__", json.dumps(_LIGHT_PATCH))
        .replace("__DARK__", json.dumps(_DARK_PATCH))
        .replace("__PLOT__", plot_div)
    )
    path.write_text(page, encoding="utf-8")
    return path


def open_in_browser(path: Path | str) -> None:
    webbrowser.open(Path(path).resolve().as_uri())
