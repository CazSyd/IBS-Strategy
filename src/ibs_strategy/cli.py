"""Command-line interface: ``ibs backtest | optimize | walkforward | signal``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__
from .backtest import DEFAULT_ENTRY_THRESHOLD, DEFAULT_EXIT_THRESHOLD, run_backtest
from .data import load_data
from .live import DEFAULT_LOOKBACK_DAYS, latest_signal
from .metrics import cagr, total_return
from .optimize import (
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    OBJECTIVES,
    grid_search,
    walk_forward,
)
from .visualize import (
    METRIC_LABELS,
    plot_backtest,
    plot_heatmap,
    plot_walk_forward,
)
from .web import open_in_browser, render_signal_page


def parse_grid(spec: str) -> np.ndarray:
    """Parse a ``start:stop:step`` range (stop exclusive) into a threshold grid."""
    try:
        start, stop, step = (float(part) for part in spec.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid grid {spec!r}; expected start:stop:step, e.g. 0.01:0.20:0.02"
        ) from exc
    grid = np.round(np.arange(start, stop, step), 4)
    if grid.size == 0:
        raise argparse.ArgumentTypeError(f"grid {spec!r} is empty")
    return grid


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_summary(title: str, summary: dict) -> None:
    rows = [
        ("Sharpe ratio", f"{summary['sharpe']:.3f}"),
        ("Total return", _pct(summary["total_return"])),
        ("CAGR", _pct(summary["cagr"])),
        ("Max drawdown", _pct(summary["max_drawdown"])),
        ("Win rate", f"{_pct(summary['win_rate'])} ({summary['num_trades']} closed trades)"),
        ("Time in market", _pct(summary["exposure"])),
        ("Final capital", f"${summary['final_capital']:,.2f}"),
    ]
    print(f"\n{title}")
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def _format_metric_columns(view: pd.DataFrame) -> pd.DataFrame:
    view = view.copy()
    view["sharpe"] = view["sharpe"].map("{:.3f}".format)
    for column in ("total_return", "cagr", "max_drawdown", "win_rate"):
        if column in view.columns:
            view[column] = view[column].map(_pct)
    return view


def _buy_hold_line(closes: pd.Series, label: str) -> None:
    print(f"\n  {label}: {_pct(total_return(closes))} total ({_pct(cagr(closes))} CAGR)")


def _emit(fig, args, filename: str) -> None:
    if getattr(args, "no_plot", False):
        plt.close(fig)
        return
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        path = args.save / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()


def _load(args) -> pd.DataFrame:
    data = load_data(args.ticker, start=args.start, end=args.end)
    first, last = data.index[0].date(), data.index[-1].date()
    print(f"{args.ticker}: {len(data)} daily bars, {first} to {last}")
    return data


def cmd_backtest(args) -> None:
    data = _load(args)
    result = run_backtest(data, args.entry, args.exit, args.capital)
    print(f"Thresholds: entry < {args.entry:g}, exit > {args.exit:g}")
    _print_summary("Backtest metrics", result.summary())
    _buy_hold_line(data["Close"], "Buy & hold over the same period")
    fig = plot_backtest(result, ticker=args.ticker)
    _emit(fig, args, f"{args.ticker.lower()}_backtest.png")


def cmd_optimize(args) -> None:
    data = _load(args)
    entry_grid = args.entry_grid if args.entry_grid is not None else DEFAULT_ENTRY_GRID
    exit_grid = args.exit_grid if args.exit_grid is not None else DEFAULT_EXIT_GRID
    results = grid_search(
        data, entry_grid, exit_grid, objective=args.objective, initial_capital=args.capital
    )
    print(
        f"\nTop {min(args.top, len(results))} of {len(results)} threshold pairs "
        f"by {args.objective} (in-sample):"
    )
    print(_format_metric_columns(results.head(args.top)).to_string(index=False))
    best = results.iloc[0]
    entry, exit_ = best["entry_threshold"], best["exit_threshold"]
    follow_up = f"ibs backtest {args.ticker}"
    if args.start:
        follow_up += f" --start {args.start}"
    if args.end:
        follow_up += f" --end {args.end}"
    follow_up += f" --entry {entry:g} --exit {exit_:g}"
    print(f"\nBest thresholds: entry {entry:g} / exit {exit_:g}")
    print(f"Backtest them with: {follow_up}")
    if len(entry_grid) > 1 or len(exit_grid) > 1:
        fig_ax = plot_heatmap(
            results,
            metric=args.objective,
            title=f"{args.ticker} threshold grid - {METRIC_LABELS.get(args.objective, args.objective)}",
        )
        _emit(fig_ax.figure, args, f"{args.ticker.lower()}_heatmap.png")


def cmd_walkforward(args) -> None:
    data = _load(args)
    wf = walk_forward(
        data,
        entry_grid=args.entry_grid,
        exit_grid=args.exit_grid,
        n_folds=args.folds,
        min_train_frac=args.min_train_frac,
        purge_days=args.purge,
        objective=args.objective,
        initial_capital=args.capital,
    )
    print(
        f"\nPer-fold out-of-sample results ({args.folds} folds, {args.purge}-day purge, "
        f"thresholds re-optimized on each training window by {args.objective}):"
    )
    print(_format_metric_columns(wf.fold_table()).to_string(index=False))
    _print_summary("Stitched out-of-sample metrics", wf.summary())
    closes = pd.concat([fold.result.data["Close"] for fold in wf.folds])
    _buy_hold_line(closes, "Buy & hold over the out-of-sample span")
    ax = plot_walk_forward(
        wf, title=f"{args.ticker} walk-forward out-of-sample equity (labels: entry/exit per fold)"
    )
    _emit(ax.figure, args, f"{args.ticker.lower()}_walkforward.png")


def cmd_signal(args) -> None:
    report = latest_signal(args.ticker, args.entry, args.exit, lookback_days=args.lookback)
    print(report.message)
    print(
        f"  Bar used: {report.bar_date:%Y-%m-%d} | IBS {report.ibs:.3f} | "
        f"entry < {args.entry:g}, exit > {args.exit:g}"
    )
    if args.no_plot:
        return
    result = run_backtest(report.data, args.entry, args.exit)
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        path = render_signal_page(
            result, args.ticker, report, args.save / f"{args.ticker.lower()}_signals.html"
        )
        print(f"\nSaved interactive chart to {path}")
    else:
        path = render_signal_page(result, args.ticker, report)
        print(f"\nOpening interactive chart in your browser: {path}")
        open_in_browser(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibs",
        description="IBS (Internal Bar Strength) mean-reversion strategy toolkit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("ticker", help="ticker symbol, e.g. TQQQ")
        p.add_argument("--start", default=None,
                       help="history start date (default: full listing history)")
        p.add_argument("--end", default=None, help="history end date (default: today)")
        p.add_argument("--capital", type=float, default=10_000.0, help="starting capital (default 10000)")
        add_output(p)

    def add_output(p: argparse.ArgumentParser) -> None:
        p.add_argument("--save", type=Path, default=None, metavar="DIR",
                       help="save plots into DIR instead of showing them")
        p.add_argument("--no-plot", action="store_true", help="skip plots entirely")

    def add_thresholds(p: argparse.ArgumentParser) -> None:
        p.add_argument("--entry", type=float, default=DEFAULT_ENTRY_THRESHOLD,
                       help="buy when the previous bar's IBS is below this "
                            f"(default {DEFAULT_ENTRY_THRESHOLD:g})")
        p.add_argument("--exit", type=float, default=DEFAULT_EXIT_THRESHOLD,
                       help="sell when the previous bar's IBS is above this "
                            f"(default {DEFAULT_EXIT_THRESHOLD:g})")

    def add_grids(p: argparse.ArgumentParser) -> None:
        p.add_argument("--entry-grid", type=parse_grid, default=None, metavar="A:B:STEP",
                       help="entry threshold grid (default 0.002:0.202:0.002)")
        p.add_argument("--exit-grid", type=parse_grid, default=None, metavar="A:B:STEP",
                       help="exit threshold grid (default 0.8:1.0:0.002)")
        p.add_argument("--objective", choices=OBJECTIVES, default="total_return",
                       help="ranking metric (default total_return, Sharpe tiebreak)")

    p = sub.add_parser("backtest", help="backtest fixed thresholds; plot trades, equity, drawdown")
    add_common(p)
    add_thresholds(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("optimize", help="grid-search entry/exit thresholds (in-sample) with a heatmap")
    add_common(p)
    add_grids(p)
    p.add_argument("--top", type=int, default=10, help="rows to display (default 10)")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("walkforward",
                       help="purged walk-forward validation: re-optimize per fold, evaluate out-of-sample")
    add_common(p)
    add_grids(p)
    p.add_argument("--folds", type=int, default=5, help="number of test folds (default 5)")
    p.add_argument("--purge", type=int, default=5,
                   help="trading days dropped between train and test windows (default 5)")
    p.add_argument("--min-train-frac", type=float, default=0.5,
                   help="fraction of history reserved for the first training window (default 0.5)")
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser(
        "signal",
        help="check the live signal and open an interactive candlestick page of recent trades",
    )
    p.add_argument("ticker", help="ticker symbol, e.g. TQQQ")
    add_thresholds(p)
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help="calendar days of history shown in the chart "
                        f"(default {DEFAULT_LOOKBACK_DAYS})")
    add_output(p)
    p.set_defaults(func=cmd_signal)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
