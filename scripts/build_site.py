"""Build the GitHub Pages site: one interactive signal page per ticker plus an index.

Usage: python scripts/build_site.py [OUTPUT_DIR] [TICKER ...]
Defaults: OUTPUT_DIR=site, tickers TQQQ SPXL.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from ibs_strategy import (
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_EXIT_THRESHOLD,
    latest_signal,
    run_backtest,
)
from ibs_strategy.web import SIGNAL_COLORS, render_signal_page

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IBS live signals</title>
<style>
  :root { --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
          --muted: #898781; --border: rgba(11, 11, 11, 0.10); }
  @media (prefers-color-scheme: dark) {
    :root { --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
            --muted: #898781; --border: rgba(255, 255, 255, 0.12); }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink-2);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 640px; margin: 0 auto; padding: 48px 20px 32px; }
  h1 { color: var(--ink); font-size: 24px; margin: 0 0 4px; }
  .updated { color: var(--muted); font-size: 13px; margin: 0 0 28px; }
  a.card { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 14px;
           padding: 16px 18px; margin: 0 0 12px; background: var(--surface);
           border: 1px solid var(--border); border-radius: 12px;
           text-decoration: none; color: var(--ink-2); }
  a.card:hover { border-color: var(--muted); }
  .ticker { color: var(--ink); font-weight: 700; font-size: 17px; min-width: 64px; }
  .signal { font-weight: 700; }
  .detail { color: var(--muted); font-size: 13px; margin-left: auto; }
  footer { margin-top: 36px; color: var(--muted); font-size: 12px; line-height: 1.5; }
  footer a { color: inherit; }
</style>
</head>
<body>
<main>
  <h1>IBS live signals</h1>
  <p class="updated">Updated __UPDATED__</p>
__CARDS__
  <footer>Signals classify the previous completed session at the default thresholds
  (entry &lt; __ENTRY__, exit &gt; __EXIT__). Educational use only - not financial advice.
  Source: <a href="https://github.com/CazSyd/IBS-Strategy">CazSyd/IBS-Strategy</a>.</footer>
</main>
</body>
</html>
"""

CARD_TEMPLATE = (
    '  <a class="card" href="__HREF__"><span class="ticker">__TICKER__</span>'
    '<span class="signal" style="color:__COLOR__">__SIGNAL__</span>'
    '<span class="detail">IBS __IBS__ on __DATE__</span></a>'
)


def main(argv: list[str]) -> None:
    output = Path(argv[0]) if argv else Path("site")
    tickers = [ticker.upper() for ticker in argv[1:]] or ["TQQQ", "SPXL"]
    output.mkdir(parents=True, exist_ok=True)

    cards = []
    for ticker in tickers:
        report = latest_signal(ticker)
        result = run_backtest(report.data)
        page = render_signal_page(result, ticker, report, output / f"{ticker.lower()}.html")
        print(f"{report.message} -> {page.name}")
        cards.append(
            CARD_TEMPLATE
            .replace("__HREF__", f"{ticker.lower()}.html")
            .replace("__TICKER__", escape(ticker))
            .replace("__COLOR__", SIGNAL_COLORS.get(report.signal, "#898781"))
            .replace("__SIGNAL__", report.signal)
            .replace("__IBS__", f"{report.ibs:.3f}")
            .replace("__DATE__", f"{report.bar_date:%Y-%m-%d}")
        )

    index = (
        INDEX_TEMPLATE
        .replace("__UPDATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        .replace("__CARDS__", "\n".join(cards))
        .replace("__ENTRY__", f"{DEFAULT_ENTRY_THRESHOLD:g}")
        .replace("__EXIT__", f"{DEFAULT_EXIT_THRESHOLD:g}")
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    print(f"Site written to {output.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
