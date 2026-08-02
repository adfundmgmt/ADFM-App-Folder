from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.data_registry import market_symbols


def market_frames() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2023-01-02", periods=700)
    t = np.arange(len(index), dtype=float)
    frames = {}
    for position, symbol in enumerate(market_symbols()):
        values = 100.0 * np.exp((0.00020 + position * 0.00001) * t)
        if symbol == "^VIX":
            values = 30.0 * np.exp(-0.00020 * t)
        frames[symbol] = pd.DataFrame(
            {
                "Open": values,
                "High": values * 1.01,
                "Low": values * 0.99,
                "Close": values,
                "Adj Close": values,
                "Volume": 1_000_000,
            },
            index=index,
        )
    return frames


class HomeCommandCenterTests(unittest.TestCase):
    @patch(
        "adfm_core.signal_ledger.record_signal_snapshot",
        return_value=pd.DataFrame(),
    )
    @patch("adfm_core.market_data.fetch_daily_ohlcv")
    def test_home_renders_command_center_and_clickable_tool_links(
        self, market_loader, _ledger
    ):
        market_loader.return_value = (
            market_frames(),
            pd.DataFrame(columns=["Ticker", "Reason"]),
        )

        app = AppTest.from_file("Home.py", default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(len(app.get("page_link")), 19)


if __name__ == "__main__":
    unittest.main()
