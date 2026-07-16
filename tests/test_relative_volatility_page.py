"""Streamlit render smoke test for the Relative Volatility Lab."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "20_Relative_Volatility_Lab.py"


def market_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000.0,
        },
        index=close.index,
    )


def fake_market_data(tickers: tuple[str, ...], period: str = "5y"):
    del period
    index = pd.bdate_range("2021-01-04", periods=1_350)
    step = np.arange(len(index), dtype=float)
    primary_returns = 0.0003 + 0.011 * np.sin(step / 13.0)
    comparison_returns = 0.0002 + 0.007 * np.sin(step / 17.0)
    primary = pd.Series(100 * np.exp(np.cumsum(primary_returns)), index=index)
    comparison = pd.Series(100 * np.exp(np.cumsum(comparison_returns)), index=index)
    implied = pd.Series(19 + 4 * np.sin(step / 31.0), index=index)
    close_map = {"^NDX": primary, "^GSPC": comparison, "^VIX": implied}
    frames = {
        ticker: market_frame(close_map[ticker])
        for ticker in tickers
        if ticker in close_map
    }
    missing = pd.DataFrame(
        {
            "Ticker": [ticker for ticker in tickers if ticker not in frames],
            "Reason": "No valid OHLCV data returned",
        }
    )
    return frames, missing


class RelativeVolatilityPageTests(unittest.TestCase):
    def test_page_renders_controls_metrics_and_tabs(self):
        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            side_effect=fake_market_data,
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual([item.value for item in app.text_input], ["^NDX", "^GSPC", "^VIX"])
        self.assertEqual(len(app.selectbox), 3)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Volatility spread", "Normalized stress", "Data", "Methodology"],
        )
        self.assertTrue(
            any("^NDX synthetic VIX" in block.value for block in app.markdown)
        )


if __name__ == "__main__":
    unittest.main()
