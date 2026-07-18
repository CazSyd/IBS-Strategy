import numpy as np
import pandas as pd
import pytest

from ibs_strategy.data import compute_ibs, flatten_columns


def test_compute_ibs_formula():
    frame = pd.DataFrame({"High": [110.0], "Low": [90.0], "Close": [95.0]})
    assert compute_ibs(frame).iloc[0] == pytest.approx(0.25)


def test_compute_ibs_flat_bar_is_nan():
    frame = pd.DataFrame({"High": [100.0], "Low": [100.0], "Close": [100.0]})
    assert np.isnan(compute_ibs(frame).iloc[0])


def test_flatten_columns_collapses_multiindex():
    columns = pd.MultiIndex.from_product([["Open", "Close"], ["TQQQ"]])
    frame = pd.DataFrame([[1.0, 2.0]], columns=columns)
    flat = flatten_columns(frame)
    assert list(flat.columns) == ["Open", "Close"]
    assert isinstance(frame.columns, pd.MultiIndex)  # original untouched


def test_flatten_columns_noop_on_flat_frame():
    frame = pd.DataFrame({"Open": [1.0]})
    assert flatten_columns(frame) is frame
