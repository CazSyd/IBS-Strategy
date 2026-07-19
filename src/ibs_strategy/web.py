"""Interactive HTML signal page: candlesticks, trade markers, hover details.

Renders a self-contained Plotly page (works offline) with daily candles, a
volume row, prominent entry/exit markers (labeled triangles plus dashed guide
lines), shaded holding periods, per-day hover showing OHLC, IBS, volume and
the daily change, HTML range buttons (1M/3M/6M/All) that also refit the
y-axis, a light/dark theme toggle, and a mobile-friendly full-viewport layout.

The x-axis is a category axis over trading days (with month tick labels
computed here) rather than a date axis with ``rangebreaks`` -- rangebreaks
plus range buttons can hang plotly's tick calculator on narrow windows.
"""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
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
    "hoverlabel.bgcolor": "#ffffff",
    "hoverlabel.bordercolor": GRIDLINE,
    "hoverlabel.font.color": INK,
    "xaxis.gridcolor": GRIDLINE,
    "xaxis.linecolor": BASELINE,
    "xaxis.spikecolor": INK_MUTED,
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
    "hoverlabel.bgcolor": "#262624",
    "hoverlabel.bordercolor": DARK_BASELINE,
    "hoverlabel.font.color": DARK_INK,
    "xaxis.gridcolor": DARK_GRIDLINE,
    "xaxis.linecolor": DARK_BASELINE,
    "xaxis.spikecolor": INK_MUTED,
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
  /* page matches the chart's paper color so the plot blends seamlessly */
  :root { --page: #fcfcfb; --ink: #0b0b0b; --muted: #898781;
          --chip-bg: #fcfcfb; --chip-ink: #52514e; --chip-border: rgba(11, 11, 11, 0.12); }
  :root[data-theme="dark"] { --page: #1a1a19; --ink: #ffffff; --muted: #898781;
          --chip-bg: #1a1a19; --chip-ink: #c3c2b7; --chip-border: rgba(255, 255, 255, 0.14); }
  * { box-sizing: border-box; }
  html { height: 100%; }
  body { margin: 0; height: 100vh; height: 100dvh; display: flex; flex-direction: column;
         background: var(--page); transition: background 0.2s ease;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px;
           padding: 10px 14px 6px; }
  .head-left { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; min-width: 0; }
  .ticker { font-weight: 700; font-size: 18px; color: var(--ink); }
  .badge { font-weight: 700; font-size: 13px; padding: 2px 10px; border-radius: 999px;
           border: 1.5px solid currentColor; }
  .meta { color: var(--muted); font-size: 12.5px; }
  .head-right { display: flex; align-items: center; gap: 6px; margin-left: auto; flex-wrap: wrap; }
  .chip { padding: 6px 11px; border-radius: 999px; border: 1px solid var(--chip-border);
          background: var(--chip-bg); color: var(--chip-ink); font: inherit; font-size: 12.5px;
          cursor: pointer; }
  .chip:hover { filter: brightness(0.95); }
  .range-chip.active { border-color: var(--chip-ink); font-weight: 700; }
  #chart-wrap { flex: 1 1 auto; min-height: 0; padding: 0 6px 6px; }
  #chart-wrap > div { height: 100%; }
</style>
</head>
<body>
<header>
  <div class="head-left">__HEADER_LEFT__</div>
  <div class="head-right">
    <button class="chip range-chip" type="button" data-months="1">1M</button>
    <button class="chip range-chip" type="button" data-months="3">3M</button>
    <button class="chip range-chip" type="button" data-months="6">6M</button>
    <button class="chip range-chip active" type="button" data-months="0">All</button>
    <button id="theme-toggle" class="chip" type="button">Dark</button>
  </div>
</header>
<div id="chart-wrap">__PLOT__</div>
<script>
(function () {
  var LIGHT = __LIGHT__;
  var DARK = __DARK__;
  var chart = document.getElementById("ibs-chart");
  var toggle = document.getElementById("theme-toggle");

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    toggle.textContent = theme === "dark" ? "\\u2600 Light" : "\\u263E Dark";
    try { localStorage.setItem("ibs-theme", theme); } catch (err) {}
    Plotly.relayout(chart, theme === "dark" ? DARK : LIGHT);
  }
  var saved = null;
  try { saved = localStorage.getItem("ibs-theme"); } catch (err) {}
  applyTheme(saved || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  toggle.addEventListener("click", function () {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  function setMonths(months, chip) {
    var candles = chart.data[0];
    var xs = candles.x;
    var n = xs.length;
    var update;
    if (!months) {
      update = { "xaxis.autorange": true, "yaxis.autorange": true, "yaxis2.autorange": true };
    } else {
      var cutoff = new Date(xs[n - 1]);
      cutoff.setMonth(cutoff.getMonth() - months);
      var start = n - 1;
      while (start > 0 && new Date(xs[start - 1]) >= cutoff) { start--; }
      var lo = Infinity, hi = -Infinity;
      for (var i = start; i < n; i++) {
        if (candles.low[i] < lo) { lo = candles.low[i]; }
        if (candles.high[i] > hi) { hi = candles.high[i]; }
      }
      if (!isFinite(lo) || !isFinite(hi)) { return; }
      var pad = (hi - lo) * 0.06 || hi * 0.04;
      update = {
        "xaxis.range": [start - 0.5, n - 0.5],
        "yaxis.range": [lo - 2.6 * pad, hi + 1.6 * pad]
      };
      var bars = null;
      for (var t = 0; t < chart.data.length; t++) {
        if (chart.data[t].type === "bar") { bars = chart.data[t]; break; }
      }
      if (bars) {
        var volHi = 0;
        for (var j = start; j < n; j++) { if (bars.y[j] > volHi) { volHi = bars.y[j]; } }
        if (volHi > 0) { update["yaxis2.range"] = [0, volHi * 1.08]; }
      }
    }
    Plotly.relayout(chart, update);
    var chips = document.querySelectorAll(".range-chip");
    for (var c = 0; c < chips.length; c++) { chips[c].classList.remove("active"); }
    if (chip) { chip.classList.add("active"); }
  }
  var rangeChips = document.querySelectorAll(".range-chip");
  for (var r = 0; r < rangeChips.length; r++) {
    rangeChips[r].addEventListener("click", function () {
      setMonths(parseInt(this.getAttribute("data-months"), 10), this);
    });
  }

  if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) {
    Plotly.relayout(chart, { dragmode: "pan" });
  }
  function syncLegend() {
    var show = window.innerWidth > 560;
    if (chart._fullLayout && chart._fullLayout.showlegend !== show) {
      Plotly.relayout(chart, { showlegend: show });
    }
  }
  window.addEventListener("resize", syncLegend);
  syncLegend();
})();
</script>
</body>
</html>
"""


def _date_str(value) -> str:
    """ISO date string for plotly x-values (keeps the figure JSON-serializable)."""
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def _month_ticks(index) -> tuple[list[int], list[str]] | None:
    """First trading day of each month -> (positions, labels), or None without dates."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    positions: list[int] = []
    labels: list[str] = []
    previous = None
    for i, timestamp in enumerate(index):
        month = (timestamp.year, timestamp.month)
        if month != previous:
            positions.append(i)
            labels.append(
                f"{timestamp:%b %Y}" if timestamp.month == 1 or not positions[:-1] else f"{timestamp:%b}"
            )
            previous = month
    return positions, labels


def build_signal_figure(
    result: BacktestResult,
    ticker: str,
    report: SignalReport | None = None,
    include_title: bool = True,
) -> go.Figure:
    """Candlestick chart of ``result.data`` with entry/exit markers.

    The signal page passes ``include_title=False`` and shows the same
    information in its own HTML header instead.
    """
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

    # trade fills are folded into the candle hover so the markers themselves can
    # skip hover -- big markers otherwise steal the unified hover from
    # neighboring candles
    trade_notes: dict = {}
    for trade in result.trades:
        trade_notes[trade.entry_date] = (
            f"<span style='color:{BUY_GREEN}'><b>BUY {trade.shares} sh @ "
            f"{trade.entry_price:.2f}</b></span>"
        )
        if trade.exit_date is not None:
            pct = (trade.return_pct or 0) * 100
            trade_notes[trade.exit_date] = (
                f"<span style='color:{SELL_RED}'><b>SELL {trade.shares} sh @ "
                f"{trade.exit_price:.2f} ({pct:+.1f}%)</b></span>"
            )

    change = (data["Close"].pct_change() * 100).fillna(0)
    hover_text = []
    for i in range(len(data)):
        parts = [f"IBS {data['IBS'].iloc[i]:.3f}", f"Change {change.iloc[i]:+.2f}%"]
        if has_volume:
            parts.append(f"Volume {data['Volume'].iloc[i]:,.0f}")
        note = trade_notes.get(data.index[i])
        if note:
            parts.append(note)
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
                textfont=dict(size=10, color=BUY_GREEN, weight=700),
                marker=dict(symbol="triangle-up", size=12, color=BUY_GREEN,
                            line=dict(color="#ffffff", width=1.2)),
                hoverinfo="skip",
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
                textfont=dict(size=10, color=SELL_RED, weight=700),
                marker=dict(symbol="triangle-down", size=12, color=SELL_RED,
                            line=dict(color="#ffffff", width=1.2)),
                hoverinfo="skip",
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

    layout_kwargs: dict = {}
    if include_title:
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
        layout_kwargs["title"] = dict(text=title, font=dict(color=INK, size=18), x=0.01, xanchor="left")

    fig.update_layout(
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
        margin=dict(l=10, r=10, t=90 if include_title else 44, b=10),
        xaxis_rangeslider_visible=False,
        bargap=0.15,
        **layout_kwargs,
    )
    fig.update_xaxes(
        type="category",
        gridcolor=GRIDLINE,
        linecolor=BASELINE,
        automargin=True,
        showspikes=True,
        spikecolor=INK_MUTED,
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    )
    ticks = _month_ticks(data.index)
    if ticks is not None:
        positions, labels = ticks
        fig.update_xaxes(tickmode="array", tickvals=positions, ticktext=labels)
    fig.update_yaxes(gridcolor=GRIDLINE, linecolor=BASELINE, zeroline=False, automargin=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)
    return fig


def _header_left(result: BacktestResult, ticker: str, report: SignalReport | None) -> str:
    thresholds = (
        f"entry &lt; {result.entry_threshold:g} · exit &gt; {result.exit_threshold:g}"
    )
    if report is None:
        return f'<span class="ticker">{ticker}</span><span class="meta">{thresholds}</span>'
    color = SIGNAL_COLORS.get(report.signal, INK_MUTED)
    return (
        f'<span class="ticker">{ticker}</span>'
        f'<span class="badge" style="color:{color}">{report.signal}</span>'
        f'<span class="meta">IBS {report.ibs:.3f} · {report.bar_date:%Y-%m-%d} · {thresholds}</span>'
    )


def render_signal_page(
    result: BacktestResult,
    ticker: str,
    report: SignalReport | None = None,
    path: Path | None = None,
) -> Path:
    """Write the signal page as a self-contained HTML file and return its path.

    The page embeds plotly.js (works offline), fills the viewport on any
    screen size, and carries HTML range buttons plus a light/dark toggle that
    follows the OS color scheme on first load and persists the user's choice.
    """
    fig = build_signal_figure(result, ticker, report, include_title=False)
    if path is None:
        path = Path(tempfile.gettempdir()) / f"{ticker.lower()}_ibs_signals.html"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="ibs-chart",
        default_width="100%",
        default_height="100%",
        config={"displaylogo": False, "responsive": True},
    )
    page = (
        _PAGE_TEMPLATE
        .replace("__TITLE__", f"{ticker} IBS signals")
        .replace("__HEADER_LEFT__", _header_left(result, ticker, report))
        .replace("__LIGHT__", json.dumps(_LIGHT_PATCH))
        .replace("__DARK__", json.dumps(_DARK_PATCH))
        .replace("__PLOT__", plot_div)
    )
    path.write_text(page, encoding="utf-8")
    return path


def open_in_browser(path: Path | str) -> None:
    webbrowser.open(Path(path).resolve().as_uri())
