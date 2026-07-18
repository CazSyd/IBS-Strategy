import argparse

import pytest

from ibs_strategy.cli import build_parser, parse_grid


def test_parse_grid():
    grid = parse_grid("0.01:0.20:0.02")
    assert len(grid) == 10
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.19)


def test_parse_grid_rejects_bad_specs():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_grid("0.01-0.2")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_grid("0.5:0.1:0.02")


def test_parser_wires_subcommands():
    parser = build_parser()
    args = parser.parse_args(["backtest", "TQQQ", "--entry", "0.15", "--no-plot"])
    assert args.ticker == "TQQQ"
    assert args.entry == 0.15
    assert args.no_plot is True
    assert args.start is None  # default: full listing history
    assert callable(args.func)

    args = parser.parse_args(["walkforward", "SPXL", "--folds", "4", "--purge", "3"])
    assert args.folds == 4
    assert args.purge == 3
    assert args.min_train_frac == 0.5

    args = parser.parse_args(["optimize", "TQQQ", "--objective", "cagr"])
    assert args.objective == "cagr"


def test_default_thresholds_are_whole_period_optimum():
    from ibs_strategy.backtest import DEFAULT_ENTRY_THRESHOLD, DEFAULT_EXIT_THRESHOLD

    assert DEFAULT_ENTRY_THRESHOLD == 0.13
    assert DEFAULT_EXIT_THRESHOLD == 0.97

    args = build_parser().parse_args(["backtest", "TQQQ"])
    assert args.entry == DEFAULT_ENTRY_THRESHOLD
    assert args.exit == DEFAULT_EXIT_THRESHOLD
