"""ETF trading-pressure page regression checks with mocked market data."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def prices(tickers, **kwargs):
    del kwargs
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=800)
    columns = {}
    for i, ticker in enumerate(tickers):
        close = pd.Series(100 + np.arange(len(dates)) * .02, index=dates)
        for field, values in {
            "Open": close,
            "High": close + 1,
            "Low": close - 3,
            "Close": close,
            "Adj Close": close,
            "Volume": (i + 1) * 100000.0,
        }.items():
            columns[field, ticker] = values
    frame = pd.DataFrame(columns, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


class ETFPressurePageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    def test_full_dollar_weighted_universe_is_visible(self):
        with patch("adfm_core.market_data.download_market_data", side_effect=prices):
            app = AppTest.from_file(str(ROOT / "pages/14_ETF_Flow_Pressure_Proxy.py")).run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 0)
        readings = app.dataframe[0].value

        self.assertEqual(len(readings), 99)
        self.assertIn("1 Month Dollar Pressure", readings.columns)
        self.assertIn("WTD $ Pressure", readings.columns)
        self.assertIn("Pressure / ADV (x)", readings.columns)
        self.assertIn("1Y Pressure %ile", readings.columns)
        self.assertIn("Price / Pressure", readings.columns)
        self.assertTrue(readings["1Y Pressure %ile"].dropna().between(0, 100).all())
        self.assertTrue(readings["Pressure / ADV (x)"].notna().all())

        subheads = [item.value for item in app.subheader]
        self.assertNotIn("Reported ETF capital flows", subheads)
        self.assertIn("Dollar-Weighted Trading Pressure", subheads)
        self.assertIn("ETF Pressure Detail", subheads)

    def test_asset_filter_keeps_underlying_dollar_pressure_columns(self):
        with patch("adfm_core.market_data.download_market_data", side_effect=prices):
            app = AppTest.from_file(str(ROOT / "pages/14_ETF_Flow_Pressure_Proxy.py")).run(timeout=30)
            app.selectbox[0].select("FX").run(timeout=30)

        self.assertEqual(list(app.exception), [])
        readings = app.dataframe[0].value
        self.assertTrue(readings["Asset Class"].eq("FX").all())
        self.assertTrue(readings["1 Month Dollar Pressure"].notna().all())
        self.assertTrue(readings["Pressure / ADV (x)"].notna().all())
        self.assertTrue(readings["1Y Pressure %ile"].notna().all())
        self.assertTrue(readings["Price / Pressure"].ne("N/A").all())

    def test_missing_provider_data_stays_visible_instead_of_shrinking_universe(self):
        with patch("adfm_core.market_data.download_market_data", return_value=pd.DataFrame()):
            app = AppTest.from_file(str(ROOT / "pages/14_ETF_Flow_Pressure_Proxy.py")).run(timeout=30)

        self.assertEqual(list(app.exception), [])
        readings = app.dataframe[0].value
        self.assertEqual(len(readings), 99)
        self.assertTrue(readings["Data Status"].eq("Missing").all())


if __name__ == "__main__":
    unittest.main()
