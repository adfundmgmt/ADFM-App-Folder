"""Deterministic coverage for the options relative-value compass."""

import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "pages/16_Options_Positioning_Compass.py"


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 30, 12, tzinfo=tz)


def option_chain(option_type):
    return pd.DataFrame(
        {
            "contractSymbol": [f"TEST_{option_type}_100", f"TEST_{option_type}_105"],
            "strike": [100.0, 105.0],
            "lastPrice": [2.0, 3.0],
            "bid": [1.8, 2.8],
            "ask": [2.2, 3.2],
            "volume": [10, 20],
            "openInterest": [50, 60],
            "impliedVolatility": [0.25, 0.28],
            "lastTradeDate": pd.to_datetime(["2026-09-30", "2026-09-30"], utc=True),
        }
    )


class FixtureTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def options(self):
        return ("2026-11-14",)

    def option_chain(self, expiry):
        assert expiry == "2026-11-14"
        return SimpleNamespace(
            calls=option_chain("call"),
            puts=option_chain("put"),
            underlying={"regularMarketPrice": 100.0},
        )


def market_data(tickers, period):
    assert period == "1y"
    index = pd.bdate_range("2026-01-01", periods=160)
    close = pd.Series(100 + np.sin(np.arange(160) / 7), index=index)
    frame = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Adj Close": close, "Volume": 1_000_000},
        index=index,
    )
    return {symbol: frame.copy() for symbol in tickers}, pd.DataFrame()


def set_expander(app, label, opened):
    # AppTest does not yet provide an expander setter; set its registered state.
    expander = next(item for item in app.expander if item.label == label)
    app.session_state[expander.proto.id] = opened


class OptionsPositioningPageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        for patcher in (
            patch("datetime.datetime", FixedDatetime),
            patch("yfinance.Ticker", FixtureTicker),
            patch("adfm_core.market_data.fetch_daily_ohlcv", side_effect=market_data),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_first_view_is_one_four_quadrant_chart_and_one_compact_table(self):
        app = AppTest.from_file(str(PAGE)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(
            list(app.dataframe[0].value.columns),
            [
                "Ticker",
                "Regime",
                "1M Return",
                "ATM IV",
                "21D Realized",
                "IV-RV Spread",
                "IV Richness Pctl",
                "Expiry",
                "DTE",
            ],
        )
        self.assertEqual({item.label for item in app.expander}, {"Methodology & coverage"})
        self.assertEqual(len(app.get("download_button")), 0)

    def test_three_month_horizon_switches_price_direction_column(self):
        app = AppTest.from_file(str(PAGE)).run(timeout=20)
        app.selectbox[0].set_value("3 months")
        app.run(timeout=20)

        self.assertFalse(app.exception)
        self.assertIn("3M Return", app.dataframe[0].value.columns)
        self.assertNotIn("1M Return", app.dataframe[0].value.columns)

    def test_underlying_spot_is_preserved_when_price_history_is_missing(self):
        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            return_value=({}, pd.DataFrame([{"Ticker": "QQQ", "Reason": "Unavailable"}])),
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        table = app.dataframe[0].value
        row = table.loc[table["Ticker"].eq("QQQ")].iloc[0]
        self.assertAlmostEqual(row["ATM IV"], 0.25)
        self.assertEqual(row["Regime"], "Unavailable")

    def test_missing_strikes_report_unusable_chain_without_crashing(self):
        class MissingStrikesTicker(FixtureTicker):
            def option_chain(self, expiry):
                chain = super().option_chain(expiry)
                chain.calls["strike"] = np.nan
                chain.puts["strike"] = np.nan
                return chain

        with patch("yfinance.Ticker", MissingStrikesTicker):
            app = AppTest.from_file(str(PAGE)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        issues = app.dataframe[0].value["Issue"].tolist()
        self.assertTrue(all("valid strikes" in issue for issue in issues))

    def test_missing_underlying_price_reports_provider_diagnostic(self):
        class MissingSpotTicker(FixtureTicker):
            def option_chain(self, expiry):
                chain = super().option_chain(expiry)
                chain.underlying["regularMarketPrice"] = None
                return chain

        with (
            patch("yfinance.Ticker", MissingSpotTicker),
            patch("adfm_core.market_data.fetch_daily_ohlcv", return_value=({}, pd.DataFrame())),
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertTrue(
            all("underlying price" in issue for issue in app.dataframe[0].value["Issue"])
        )

    def test_failed_highlight_ticker_does_not_hide_loaded_peer_map(self):
        class OneTickerFails(FixtureTicker):
            def option_chain(self, expiry):
                if self.symbol == "QQQ":
                    raise RuntimeError("QQQ unavailable")
                return super().option_chain(expiry)

        with (
            patch("yfinance.Ticker", OneTickerFails),
            patch(
                "adfm_core.options_sources.fetch_cboe_delayed_options",
                side_effect=RuntimeError("Cboe unavailable"),
            ),
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(len(app.dataframe), 1)
        self.assertNotIn("QQQ", app.dataframe[0].value["Ticker"].tolist())

    def test_source_no_longer_contains_old_detail_surfaces(self):
        source = PAGE.read_text(encoding="utf-8")
        self.assertNotIn("Term structure + surface", source)
        self.assertNotIn("Premium activity", source)
        self.assertNotIn("Download compass snapshot", source)
        self.assertNotIn("build_positioning_commentary", source)
        self.assertIn("POSITIVE TREND · IV RICH", source)
        self.assertIn("NEGATIVE TREND · IV CHEAP", source)


if __name__ == "__main__":
    unittest.main()
