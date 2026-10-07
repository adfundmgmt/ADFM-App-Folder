"""Streamlit render smoke test for the Relative Volatility Lab."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "13_Relative_Volatility_Lab.py"


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
    primary_implied = pd.Series(23 + 5 * np.sin(step / 29.0), index=index)
    comparison_implied = pd.Series(19 + 4 * np.sin(step / 31.0), index=index)
    soxx = pd.Series(
        90 * np.exp(np.cumsum(0.0004 + 0.014 * np.sin(step / 11.0))),
        index=index,
    )
    equal_weight = pd.Series(
        95 * np.exp(np.cumsum(0.00025 + 0.008 * np.sin(step / 19.0))),
        index=index,
    )
    cap_weight = pd.Series(
        100 * np.exp(np.cumsum(0.0003 + 0.010 * np.sin(step / 17.0))),
        index=index,
    )
    close_map = {
        "^NDX": primary,
        "^GSPC": comparison,
        "^VXN": primary_implied,
        "^VIX": comparison_implied,
        "SOXX": soxx,
        "QEW": equal_weight,
        "QQQ": cap_weight,
    }
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
    def test_page_renders_controls_and_collapsed_details_without_summary_cards(self):
        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            side_effect=fake_market_data,
        ) as loader:
            app = AppTest.from_file(str(PAGE)).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [item.value for item in app.text_input],
            ["^NDX", "^GSPC", "^VXN", "^VIX"],
        )
        self.assertEqual(len(app.selectbox), 3)
        self.assertEqual(len(app.tabs), 0)
        self.assertFalse(app.checkbox[0].value)
        self.assertEqual(tuple(loader.call_args.args[0]), ("^NDX", "^GSPC"))
        summary = app.dataframe[0].value
        self.assertEqual(summary.Asset.tolist(), ["NDX", "SPX"])
        self.assertEqual(list(summary.columns), ["Asset", "Volatility (%)", "5-session change (pp)", "History percentile"])
        frames, _ = fake_market_data(("^NDX", "^GSPC"))
        for row, ticker in enumerate(("^NDX", "^GSPC")):
            returns = np.log(frames[ticker]["Close"]).diff()
            expected = returns.iloc[-21:].std(ddof=1) * np.sqrt(252) * 100
            prior = returns.iloc[-26:-5].std(ddof=1) * np.sqrt(252) * 100
            self.assertAlmostEqual(summary.iloc[row]["Volatility (%)"], expected)
            self.assertAlmostEqual(summary.iloc[row]["5-session change (pp)"], expected - prior)
            self.assertTrue(0 <= summary.iloc[row]["History percentile"] <= 100)
        plot = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(len(plot["data"]), 4)  # two assets, ratio, latest marker
        self.assertTrue(any(item["text"] == "1.00x = equal volatility" for item in plot["layout"]["annotations"]))
        self.assertTrue({"Historical stress detail", "Data", "Methodology"}.issubset({item.label for item in app.expander}))
        self.assertTrue(all(not item.proto.expanded for item in app.expander if item.label in {"Historical stress detail", "Data", "Methodology"}))
        markdown = [block.value for block in app.markdown]
        self.assertFalse(any("Current relative-volatility read" in value for value in markdown))
        self.assertFalse(any("Ratio decomposition" in value for value in markdown))
        self.assertFalse(any("Cross-market diagnostics" in value for value in markdown))

    def test_page_keeps_core_pair_when_optional_diagnostics_are_missing(self):
        def required_pair_only(tickers: tuple[str, ...], period: str = "5y"):
            frames, _ = fake_market_data(tickers, period)
            frames = {
                ticker: frame
                for ticker, frame in frames.items()
                if ticker in {"^NDX", "^GSPC"}
            }
            missing = pd.DataFrame(
                {
                    "Ticker": [ticker for ticker in tickers if ticker not in frames],
                    "Reason": "No valid OHLCV data returned",
                }
            )
            return frames, missing

        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            side_effect=required_pair_only,
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            self.assertEqual(len(app.warning), 0)
            app.checkbox[0].check()
            app.button[0].click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Implied-volatility overlays are unavailable" in item.value for item in app.warning)
        )
        self.assertEqual(len(app.dataframe[0].value), 2)

    def test_implied_overlay_remains_available_when_enabled(self):
        with patch("adfm_core.market_data.fetch_daily_ohlcv", side_effect=fake_market_data) as loader:
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            app.checkbox[0].check()
            app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(set(loader.call_args.args[0]), {"^NDX", "^GSPC", "^VXN", "^VIX"})
        plot = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(len(plot["data"]), 7)

    def test_custom_pair_uses_selected_assets_without_default_index_overlays(self):
        with patch("adfm_core.market_data.fetch_daily_ohlcv", side_effect=fake_market_data) as loader:
            app = AppTest.from_file(str(PAGE)).run(timeout=30)
            app.text_input[0].set_value("SOXX")
            app.text_input[1].set_value("QQQ")
            app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(set(loader.call_args.args[0]), {"SOXX", "QQQ"})
        self.assertEqual(app.dataframe[0].value.Asset.tolist(), ["SOXX", "QQQ"])


if __name__ == "__main__":
    unittest.main()
