"""Streamlit render smoke test for the Cross-Asset Correlation Lab."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "21_Cross_Asset_Correlation_Lab.py"


def market_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.003,
            "Low": close * 0.997,
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
    market = 0.0002 + 0.006 * np.sin(step / 17.0)
    defensive = 0.0001 + 0.004 * np.cos(step / 23.0)
    inflation = 0.0001 + 0.005 * np.sin(step / 31.0 + 0.7)
    frames: dict[str, pd.DataFrame] = {}
    for position, ticker in enumerate(tickers):
        if ticker == "^VIX":
            close = pd.Series(22 + 7 * np.sin(step / 29.0), index=index)
        else:
            equity_weight = 0.80 - (position % 5) * 0.18
            defensive_weight = -0.35 + (position % 4) * 0.22
            inflation_weight = -0.20 + (position % 3) * 0.25
            idiosyncratic = 0.002 * np.sin(step / (7.0 + position) + position)
            returns = (
                equity_weight * market
                + defensive_weight * defensive
                + inflation_weight * inflation
                + idiosyncratic
            )
            close = pd.Series(100 * np.exp(np.cumsum(returns)), index=index)
        frames[ticker] = market_frame(close)
    return frames, pd.DataFrame(columns=["Ticker", "Reason"])


class CorrelationLabPageTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()

    def test_page_renders_data_only_controls_and_all_diagnostic_tabs(self) -> None:
        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            side_effect=fake_market_data,
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 6)
        self.assertEqual(len(app.selectbox), 4)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "Correlation Matrix",
                "Regime Structure",
                "Pair Lab",
                "Data + Methodology",
            ],
        )
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 6)
        self.assertEqual(len(app.get("download_button")), 4)
        self.assertTrue(
            any("Cross-Asset Correlation Lab" in block.value for block in app.markdown)
        )

        figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
        matrix_layout = figures[0]["layout"]
        self.assertFalse(matrix_layout["showlegend"])
        self.assertEqual(matrix_layout["margin"]["t"], 70)
        self.assertEqual(len(matrix_layout["annotations"]), 120)
        self.assertEqual(len(matrix_layout["shapes"]), 10)

        for figure_index in (2, 4, 5):
            layout = figures[figure_index]["layout"]
            self.assertEqual(layout["margin"]["t"], 92)
            self.assertEqual(layout["title"]["y"], 0.985)
            self.assertEqual(layout["legend"]["y"], 1.025)

    def test_page_keeps_core_analysis_when_optional_series_are_missing(self) -> None:
        def optional_series_missing(tickers: tuple[str, ...], period: str = "5y"):
            frames, _ = fake_market_data(tickers, period)
            frames = {
                ticker: frame
                for ticker, frame in frames.items()
                if ticker not in {"FXY", "^VIX"}
            }
            missing = pd.DataFrame(
                {
                    "Ticker": ["FXY", "^VIX"],
                    "Reason": "No valid OHLCV data returned",
                }
            )
            return frames, missing

        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            side_effect=optional_series_missing,
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 6)
        self.assertTrue(any("14/15" in metric.value for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
