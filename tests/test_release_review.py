"""Regression checks for independently reviewed release findings."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from adfm_core import commodity_top_exhaustion_page as commodity
from adfm_core.catalyst_calendar_page import _close_from_yfinance
from tests.test_page_accuracy import page_functions


class ReleaseReviewTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()

    def test_catalyst_alignment_does_not_extend_stale_prices(self):
        dates = pd.bdate_range("2026-09-01", periods=4)
        frame = pd.DataFrame({"AAA": [100., 101., np.nan, np.nan], "BBB": [100., 101., 102., 103.]}, index=dates)
        result = _close_from_yfinance(frame)
        self.assertTrue(result["AAA"].iloc[2:].isna().all())

    def test_comparison_chart_stops_at_last_shared_observed_endpoint(self):
        import plotly.graph_objects as go
        dates = pd.bdate_range("2026-09-01", periods=4)
        secondary = pd.Series([100., 101.], index=dates[:2], name="BBB")
        functions = page_functions("10_ADFM_Chart_Terminal.py", {"build_compare_chart"}, {"go": go, "COLORS": {"text": "black", "grid": "white"}, "build_rangebreaks": lambda *args: [], "warmup_start_date": lambda *args: None, "start_date_from_period": lambda *args: None, "fetch_compare_close": lambda **kwargs: secondary})
        settings = SimpleNamespace(period="max", interval="1d", ticker="AAA", auto_adjust=False)
        figure = functions["build_compare_chart"](pd.DataFrame({"Close": [100., 101., 102., 103.]}, index=dates), settings, ["BBB"])
        self.assertEqual(pd.Timestamp(figure.data[0].x[-1]), dates[1])

    def test_stale_commodity_history_cannot_show_current_active(self):
        dates = pd.bdate_range("2020-01-01", periods=600)
        data = pd.DataFrame({"Close": 100 + np.arange(600) * .1, "Volume": 1000}, index=dates)
        with patch.object(commodity, "load_cftc_crowding", return_value=(pd.Series(dtype=float), "CFTC unavailable")):
            diagnostics, condition, label, source = commodity.build_exhaustion_frame(data, "CL=F", "Confirmed Exhaustion", commodity.PROFILE_PRESETS["Confirmed Exhaustion"], 21)
        condition.iloc[-1] = True
        with patch.object(commodity, "load_contract_history", return_value=data), patch.object(commodity, "build_exhaustion_frame", return_value=(diagnostics, condition, label, source)):
            app = AppTest.from_string("from adfm_core.commodity_top_exhaustion_page import render_commodity_event_study\nrender_commodity_event_study()").run(timeout=30)
        self.assertFalse(app.exception)
        current = next(item.value for item in app.markdown if "<strong>Current</strong>" in item.value)
        self.assertIn("Stale", current)
        self.assertNotIn("<strong>Current</strong> ACTIVE", current)


if __name__ == "__main__":
    unittest.main()
