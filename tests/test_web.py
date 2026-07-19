from ibs_strategy.backtest import run_backtest
from ibs_strategy.web import build_signal_figure, render_signal_page


def test_figure_has_candles_volume_and_trade_markers(scenario_frame):
    result = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0)
    fig = build_signal_figure(result, "TEST")

    trace_types = [trace.type for trace in fig.data]
    assert "candlestick" in trace_types
    assert "bar" in trace_types  # volume row

    buys = next(trace for trace in fig.data if trace.name == "Buy")
    sells = next(trace for trace in fig.data if trace.name == "Sell")
    assert len(buys.x) == 2  # closed trade + still-open trade
    assert len(sells.x) == 1  # only the closed trade has an exit
    assert buys.mode == "markers+text"
    assert buys.text == ("B", "B")
    assert sells.text == ("S",)
    # markers are visual only; trade details live in the candle hover so big
    # markers can't steal the unified hover from neighboring candles
    assert buys.hoverinfo == "skip"
    assert sells.hoverinfo == "skip"

    # per trade: one holding-span rect + entry guide line (+ exit line if closed)
    assert len(fig.layout.shapes) == 5

    candles = next(trace for trace in fig.data if trace.type == "candlestick")
    assert "IBS" in candles.text[0]  # hover carries per-day details
    assert "BUY 10 sh @ 95.00" in candles.text[1]  # entry day hover
    assert "SELL 10 sh @ 112.00" in candles.text[4]  # exit day hover

    # category axis with no rangebreaks/rangeselector -- their combination can
    # hang plotly's tick calculator on narrow windows (the frozen 1m/3m bug)
    assert fig.layout.xaxis.type == "category"
    assert not fig.layout.xaxis.rangebreaks
    assert not fig.layout.xaxis.rangeselector.buttons


def test_figure_without_volume_column(scenario_frame):
    frame = scenario_frame.drop(columns=["Volume"])
    result = run_backtest(frame, 0.2, 0.9, 1_000.0)
    fig = build_signal_figure(result, "TEST")
    assert all(trace.type != "bar" for trace in fig.data)


def test_render_writes_self_contained_html(tmp_path, scenario_frame):
    result = run_backtest(scenario_frame, 0.2, 0.9, 1_000.0)
    path = render_signal_page(result, "TEST", path=tmp_path / "test_signals.html")
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "candlestick" in html
    assert "theme-toggle" in html  # dark-mode toggle wired in
    assert "#1a1a19" in html  # dark surface present in the relayout patch
    assert "prefers-color-scheme" in html
    assert html.count("range-chip") >= 4  # HTML range buttons replace plotly's selector
    # the range buttons must read plain JSON arrays -- plotly serializes trace
    # numerics as base64 "bdata" blobs that page JS cannot index
    assert "var DATA = {" in html
    assert '"low": [90.0' in html
    assert '"high": [110.0' in html
