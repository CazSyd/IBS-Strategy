"""Regenerate the README charts into docs/ (network required).

Usage: uv run python scripts/build_readme_charts.py
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt

from ibs_strategy import (
    load_data,
    load_extended_data,
    plot_drawdown,
    plot_equity,
    plot_walk_forward,
    run_backtest,
    walk_forward,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(DOCS / name, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote docs/{name}")


# 1. Full real-history equity + drawdown at the default thresholds
# (no trades panel: 350+ markers over 16 years is unreadable clutter)
data = load_data("TQQQ")
result = run_backtest(data)
fig, axes = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]}
)
fig.set_facecolor("#fcfcfb")
plot_equity(result, ax=axes[0], title="TQQQ - IBS strategy vs buy & hold (start = 100)")
plot_drawdown(result, ax=axes[1])
fig.align_ylabels(axes)
fig.tight_layout()
save(fig, "backtest.png")

# 2. Purged walk-forward, thresholds re-optimized per fold by CAGR
wf = walk_forward(data, objective="cagr")
ax = plot_walk_forward(wf)
save(ax.figure, "walkforward.png")

# 3. Extended 1999+ history at the crash-aware optimum, log scale
extended = load_extended_data("TQQQ", "QQQ")
crash_result = run_backtest(extended, 0.133, 0.802)
ax = plot_equity(
    crash_result,
    log=True,
    title="1999-2026 - IBS 0.133/0.802 vs buy & hold (log scale)",
)
synthetic_end = extended.index[extended["Synthetic"]][-1]
ax.axvspan(extended.index[0], synthetic_end, color="#898781", alpha=0.10, linewidth=0)
ax.text(
    0.02, 0.95, "synthetic (3x QQQ)",
    transform=ax.transAxes, fontsize=9, color="#52514e", va="top",
)
save(ax.figure, "extended.png")
