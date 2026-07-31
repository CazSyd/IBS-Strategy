"""Append today's published signal for each ticker, and reconcile realized fills.

Run by the Pages workflow after each close (before the site build). Idempotent:
reruns on the same completed bar are ignored, so both files only grow by genuinely
new information - which is what makes them real-time out-of-sample evidence.

Each run:
  1. appends the latest published signal to the signals log; and
  2. records any fill that has since become known (a signal's next-open fill,
     frozen at its as-published price) into the fills ledger beside it.

Usage: python scripts/log_signal.py [SIGNALS_PATH] [TICKER ...]
Defaults: SIGNALS_PATH=data/signals.csv (fills.csv sits beside it), tickers TQQQ SPXL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ibs_strategy import latest_signal, load_data, run_backtest
from ibs_strategy.tracking import (
    append_fill,
    append_signal,
    load_fills,
    load_signal_log,
    reconcile_fills,
)


def main(argv: list[str]) -> None:
    signals_path = Path(argv[0]) if argv else Path("data/signals.csv")
    fills_path = signals_path.with_name("fills.csv")
    tickers = [t.upper() for t in argv[1:]] or ["TQQQ", "SPXL"]

    for ticker in tickers:
        report = latest_signal(ticker)
        # size = the vol-target fraction the page would tell you to deploy on this signal
        result = run_backtest(
            report.data, report.entry_threshold, report.exit_threshold,
            position_sizing="vol_target",
        )
        size = None
        if result.weights is not None:
            try:
                size = float(result.weights.loc[report.bar_date])
            except (KeyError, TypeError, ValueError):
                size = None
        ref_price = float(report.data["Close"].loc[report.bar_date])
        wrote = append_signal(
            signals_path, date=report.bar_date.date(), ticker=ticker, ibs=report.ibs,
            signal=report.signal, size=size, ref_price=ref_price,
        )
        size_str = f"{size:.2f}" if size is not None else "-"
        print(f"{ticker}: {report.bar_date.date()} {report.signal} (size {size_str}) "
              f"-> {'logged' if wrote else 'already logged'}")

        # reconcile fills: the price frame must span from the log's inception
        signals = load_signal_log(signals_path, ticker=ticker)
        if signals:
            inception = report.data.index[0]  # 365-day lookback covers a recent inception
            prices = report.data
            first_signal = signals[0]["date"]
            if str(inception.date()) > first_signal:  # log older than the lookback
                prices = load_data(ticker, start=first_signal)
            for date, side, price, fsize in reconcile_fills(signals, prices, load_fills(fills_path, ticker)):
                append_fill(fills_path, date=date.date(), ticker=ticker, side=side,
                            price=price, size=fsize)
                print(f"  fill: {date.date()} {side} @ {price:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
