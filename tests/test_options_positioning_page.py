"""Deterministic interaction coverage for lazy options detail expanders."""

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

    def test_underlying_spot_is_preserved_when_price_history_is_missing(self):
        with patch(
            "adfm_core.market_data.fetch_daily_ohlcv",
            return_value=({}, pd.DataFrame([{"Ticker": "QQQ", "Reason": "Unavailable"}])),
        ):
            app = AppTest.from_file(str(PAGE)).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            set_expander(app, "Term structure + surface", True)
            app.run(timeout=20)

        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[-1].value["spot"].tolist(), [100.0])
        self.assertAlmostEqual(app.dataframe[-1].value.iloc[0]["atm_iv"], 0.25)

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

    def test_data_opens_without_opening_premium_activity(self):
        app = AppTest.from_file(str(PAGE)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get("download_button")), 0)

        set_expander(app, "Data", True)
        app.run(timeout=20)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(
            {item.proto.label for item in app.get("download_button")},
            {"Download compass snapshot", "Download selected term structure", "Download premium activity"},
        )
        activity = next(item for item in app.expander if item.label == "Premium activity")
        self.assertFalse(activity.proto.expanded)

    def test_premium_activity_renders_then_data_survives_its_collapse(self):
        app = AppTest.from_file(str(PAGE)).run(timeout=20)
        set_expander(app, "Premium activity", True)
        app.run(timeout=20)
        self.assertFalse(app.exception)
        table = app.dataframe[-1].value
        self.assertEqual(table["premium_activity"].tolist(), [6000.0, 6000.0, 2000.0, 2000.0])
        self.assertEqual(set(table["type"]), {"call", "put"})

        set_expander(app, "Data", True)
        set_expander(app, "Premium activity", True)
        set_expander(app, "Term structure + surface", True)
        set_expander(app, "Methodology", True)
        app.run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 3)
        self.assertEqual(len(app.get("download_button")), 3)

        set_expander(app, "Premium activity", False)
        set_expander(app, "Term structure + surface", False)
        set_expander(app, "Data", True)
        app.run(timeout=20)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get("download_button")), 3)


if __name__ == "__main__":
    unittest.main()
