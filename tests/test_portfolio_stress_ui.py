"""Actual Streamlit renderer with in-memory upload and deterministic inputs."""

from __future__ import annotations

import io
import math
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

CSV = b"symbol,kind,quantity,multiplier,mark,mark_date,currency,fx_rate,factor,margin_rate\nAAPL,shares,10,1,100,2026-09-29,USD,1,equity,0.5\n"
SOURCE = "from adfm_core.portfolio_stress_page import render_portfolio_stress\nrender_portfolio_stress()"


class StressUITests(unittest.TestCase):
    def test_futures_symbol_price_target_accepts_yahoo_equals_sign(self):
        csv = CSV.replace(b"AAPL,shares,10,1,100", b"CL=F,futures,-1,1000,100").replace(b",equity,", b",commodity,")
        with patch("streamlit.file_uploader", return_value=io.BytesIO(csv)):
            app = AppTest.from_string(SOURCE).run()
            app.number_input(key="stress_nav").set_value(100000)
            app.date_input(key="stress_date").set_value(date(2026, 9, 29))
            app.text_area(key="stress_targets").set_value("CL=F=70")
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertAlmostEqual(app.dataframe[0].value.iloc[-1]["P&L USD"], 30000.)

    def test_upload_requires_explicit_nav_and_date_then_one_complete_table(self):
        with patch("streamlit.file_uploader", return_value=io.BytesIO(CSV)):
            app = AppTest.from_string(SOURCE).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            app.number_input(key="stress_nav").set_value(100000)
            app.date_input(key="stress_date").set_value(date(2026, 9, 29))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 1)
            table = app.dataframe[0].value
            self.assertEqual(table.iloc[0]["P&L USD"], 0)
            self.assertAlmostEqual(table.iloc[-1]["P&L USD"], -100)
            self.assertAlmostEqual(table.iloc[-1]["Scenario NAV USD"], 99900)

    def test_rates_target_overrides_equity_shock_without_duration_guess(self):
        csv = CSV.replace(b"AAPL,shares", b"TLT,shares").replace(
            b",equity,", b",rates,"
        )
        with patch("streamlit.file_uploader", return_value=io.BytesIO(csv)):
            app = AppTest.from_string(SOURCE).run()
            app.number_input(key="stress_nav").set_value(100000)
            app.date_input(key="stress_date").set_value(date(2026, 9, 29))
            app.text_area(key="stress_targets").set_value("TLT=95")
            app.number_input(key="stress_equity").set_value(20)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 1)
            self.assertAlmostEqual(app.dataframe[0].value.iloc[-1]["P&L USD"], -50)
            self.assertTrue(
                math.isnan(app.dataframe[0].value.iloc[-1]["Gross DV01 USD/bp"])
            )

    def test_individual_simulator_operates_when_expanded(self):
        index = pd.bdate_range("2025-05-01", periods=350)
        close = pd.Series(
            100 + np.arange(350) * 0.04 + 3 * np.sin(np.arange(350) / 8), index=index
        )
        frame = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Adj Close": close,
                "Volume": 1000000,
            },
            index=index,
        )
        frames = {
            symbol: frame.copy()
            for symbol in ("AAPL", "SPY", "QQQ", "TLT", "UUP", "USO", "^VIX")
        }

        class TickerFixture:
            def __init__(self, symbol):
                self.symbol = symbol

            def get_earnings_dates(self, limit):
                return pd.DataFrame()

        page = Path(__file__).resolve().parents[1] / "pages/22_Position_Sizing_Lab.py"
        with (
            patch(
                "adfm_core.market_data.fetch_daily_ohlcv",
                return_value=(frames, pd.DataFrame()),
            ),
            patch("yfinance.Ticker", TickerFixture),
        ):
            app = AppTest.from_file(str(page)).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            app.checkbox(key="psl_load_history").set_value(True).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(
                any("Constraint" in table.value.columns for table in app.dataframe)
            )
            button = next(
                button for button in app.button if button.label == "Run one session"
            )
            button.click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["psl_returns"]), 1)
            self.assertEqual(len(app.session_state["psl_balances"]), 2)
            self.assertAlmostEqual(
                app.session_state["psl_balances"][-1],
                app.session_state["psl_balances"][0]
                * (1 + app.session_state["psl_nav_returns"][0]),
            )

    def test_invalid_uploaded_row_shows_error_without_partial_table(self):
        invalid = CSV + b"BAD,shares,10,1,,2026-09-29,USD,1,equity,0.5\n"
        with patch("streamlit.file_uploader", return_value=io.BytesIO(invalid)):
            app = AppTest.from_string(SOURCE).run()
            app.number_input(key="stress_nav").set_value(100000)
            app.date_input(key="stress_date").set_value(date(2026, 9, 29))
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(len(app.dataframe), 0)
            self.assertIn("Row 2", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
