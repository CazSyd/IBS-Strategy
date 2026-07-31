"""Append today's published signal for each ticker to the append-only log.

Run by the Pages workflow after each close (before the site build). Idempotent:
reruns on the same completed bar are ignored, so the log only grows by genuinely
new sessions - which is what makes it real-time out-of-sample evidence.

Usage: python scripts/log_signal.py [LOG_PATH] [TICKER ...]
Defaults: LOG_PATH=data/signals.csv, tickers TQQQ SPXL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ibs_strategy import latest_signal, run_backtest
from ibs_strategy.tracking import append_signal


def main(argv: list[str]) -> None:
    log_path = Path(argv[0]) if argv else Path("data/signals.csv")
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
            log_path, date=report.bar_date.date(), ticker=ticker, ibs=report.ibs,
            signal=report.signal, size=size, ref_price=ref_price,
        )
        size_str = f"{size:.2f}" if size is not None else "-"
        print(f"{ticker}: {report.bar_date.date()} {report.signal} (size {size_str}) "
              f"-> {'logged' if wrote else 'already logged'}")


if __name__ == "__main__":
    main(sys.argv[1:])
