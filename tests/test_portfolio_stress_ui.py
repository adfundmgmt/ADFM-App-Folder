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

    def test_volatility_sizing_loads_and_updates_without_simulator(self):
        index = pd.bdate_range("2025-01-01", periods=400)
        returns = np.random.default_rng(1).normal(0, .01, 400)
        returns[-20:] *= 3
        close = pd.Series(100 * np.cumprod(1 + returns), index=index)
        frame = pd.DataFrame({"Close": close, "Adj Close": close}, index=index)
        page = Path(__file__).resolve().parents[1] / "pages/22_Position_Sizing_Lab.py"
        with patch("adfm_core.market_data.fetch_daily_ohlcv", return_value=({"TLT": frame}, pd.DataFrame())):
            app = AppTest.from_file(str(page)).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 1)
            table = app.dataframe[0].value
            target = table.iloc[0]["Permitted"]
            self.assertLess(target, 10)
            app.number_input(key="psl_ceiling").set_value(1.0).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.iloc[0]["Permitted"], 1.0)
            app.selectbox(key="psl_side").set_value("Short")
            app.number_input(key="psl_nav").set_value(100000).run(timeout=20)
            self.assertFalse(app.exception)
            table = app.dataframe[0].value
            self.assertEqual(table.loc[table.Scenario.eq("Position notional (USD)"), "Permitted"].iloc[0], 1000)
            app.checkbox(key="psl_stress").set_value(True).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any(input.key == "stress_nav" for input in app.number_input))

    def test_optional_loss_budget_binds_and_wrong_side_is_rejected(self):
        index = pd.bdate_range("2025-01-01", periods=500)
        close = pd.Series(100*np.cumprod(1+np.random.default_rng(2).normal(0,.01,500)),index=index)
        frame = pd.DataFrame({"Close":close, "Adj Close":close},index=index)
        page = Path(__file__).resolve().parents[1] / "pages/22_Position_Sizing_Lab.py"
        latest = float(close.iloc[-1])
        with patch("adfm_core.market_data.fetch_daily_ohlcv", return_value=({"TLT":frame},pd.DataFrame())):
            app = AppTest.from_file(str(page)).run(timeout=20)
            app.number_input(key="psl_invalidation_TLT_Long").set_value(latest*.80)
            app.number_input(key="psl_loss_budget").set_value(1.0).run(timeout=20)
            self.assertFalse(app.exception)
            table = app.dataframe[0].value
            self.assertAlmostEqual(table.iloc[0]["Permitted"],5)
            self.assertAlmostEqual(table.loc[table.Scenario.eq("At invalidation"),"Permitted"].iloc[0],-1)
            self.assertTrue(any("Invalidation loss budget" in item.value for item in app.markdown))
            app.number_input(key="psl_invalidation_TLT_Long").set_value(latest*1.10).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(len(app.dataframe),0)
            app.number_input(key="psl_invalidation_TLT_Long").set_value(0).run(timeout=20)
            self.assertFalse(app.error)
            self.assertEqual(len(app.dataframe),1)

    def test_missing_history_shows_error_without_fabricated_sizing(self):
        page = Path(__file__).resolve().parents[1] / "pages/22_Position_Sizing_Lab.py"
        with patch("adfm_core.market_data.fetch_daily_ohlcv", return_value=({}, pd.DataFrame())):
            app = AppTest.from_file(str(page)).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(len(app.dataframe), 0)

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
