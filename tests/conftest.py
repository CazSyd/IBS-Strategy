import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ibs_strategy.data import compute_ibs


def make_ohlc(rows, start="2024-01-02"):
    frame = pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close"],
        index=pd.bdate_range(start, periods=len(rows)),
    )
    frame["Volume"] = 1_000
    frame["IBS"] = compute_ibs(frame)
    return frame


# Hand-crafted bars with exactly-representable IBS values (entry 0.2 / exit 0.9):
SCENARIO_BARS = [
    (100.0, 110.0, 90.0, 92.0),    # IBS 0.10 -> buy at the next open
    (95.0, 105.0, 94.0, 100.0),    # entry day at open 95; IBS ~0.545
    (101.0, 104.0, 96.0, 103.0),   # IBS 0.875 (below exit, strict) -> hold
    (104.0, 112.0, 96.0, 111.0),   # IBS 0.9375 -> sell at the next open
    (112.0, 114.0, 110.0, 110.0),  # exit day at open 112; IBS 0 -> buy at the next open
    (100.0, 100.0, 100.0, 100.0),  # entry day at open 100; High == Low -> IBS NaN
    (98.0, 100.0, 92.0, 93.0),     # NaN previous IBS -> hold
    (94.0, 96.0, 88.0, 95.0),      # previous IBS 0.125 but already long -> hold
    (96.0, 98.0, 94.0, 96.0),      # previous IBS 0.875 -> hold; position stays open
]


@pytest.fixture
def scenario_frame():
    return make_ohlc(SCENARIO_BARS)


@pytest.fixture
def random_frame():
    rng = np.random.default_rng(7)
    n = 250
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, 0.005, n - 1))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, n))
    return make_ohlc(np.column_stack([open_, high, low, close]))
