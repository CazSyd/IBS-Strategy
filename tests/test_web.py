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

    # per trade: one holding-span rect + entry guide line (+ exit line if closed)
    assert len(fig.layout.shapes) == 5

    candles = next(trace for trace in fig.data if trace.type == "candlestick")
    assert "IBS" in candles.text[0]  # hover carries per-day details


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
