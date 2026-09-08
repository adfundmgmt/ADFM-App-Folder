"""Regression checks for issues found during the all-page release audit."""

from __future__ import annotations

import ast
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from adfm_core.market_data import fill_short_calendar_gaps

ROOT = Path(__file__).resolve().parents[1]


def page_functions(filename, names, extra=None):
    """Load actual calculation bodies while leaving page UI and providers idle."""
    path = ROOT / "pages" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    namespace = {"pd": pd, "np": np, "datetime": datetime, "date": date, **(extra or {})}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class HistoricalPageAccuracyTests(unittest.TestCase):
    def test_ratio_ytd_includes_first_trading_day(self):
        functions = page_functions("11_Cross-Asset_Ratio_Chartbook.py", {"ytd_change"})
        prices = pd.Series([100., 110., 121.], index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]))
        self.assertAlmostEqual(functions["ytd_change"](prices), .21)
        self.assertTrue(np.isnan(functions["ytd_change"](prices.iloc[1:])))

    def test_short_ratio_gaps_never_extend_a_stale_endpoint(self):
        dates = pd.bdate_range("2026-01-01", periods=9)
        prices = pd.DataFrame({"AAA": [np.nan, 10., np.nan, 12., np.nan, np.nan, np.nan, np.nan, np.nan],
                               "BBB": [10., np.nan, np.nan, np.nan, 14., 15., 16., 17., 18.]}, index=dates)
        actual = fill_short_calendar_gaps(prices)
        self.assertEqual(actual["AAA"].iloc[2], 10.)
        self.assertTrue(actual["AAA"].iloc[4:].isna().all())
        self.assertTrue(pd.isna(actual["AAA"].iloc[0]))
        self.assertTrue(pd.isna(actual["BBB"].iloc[3]))
        pd.testing.assert_frame_equal(prices.where(prices.notna()), actual.where(prices.notna()))

    def test_missing_macro_observations_are_unknown(self):
        functions = page_functions("24_Monthly_Seasonality_Explorer.py", {"fetch_regime_data"},
                                   {"_today": lambda: pd.Timestamp("2026-09-08"), "_fred_series": Mock(return_value=None)})
        empty = functions["fetch_regime_data"]("2025-01-01", "2026-09-08")
        self.assertTrue(empty["is_recession"].isna().all())
        self.assertTrue(empty["regime_cycle"].eq("Unknown").all())
        recession = pd.Series([0., 1.], index=pd.to_datetime(["2025-01-01", "2025-02-01"]))
        rates = pd.Series([4., 4.], index=recession.index)
        functions["_fred_series"] = Mock(side_effect=[recession, rates])
        observed = functions["fetch_regime_data"]("2025-01-01", "2026-09-08")
        self.assertEqual(observed.loc["2025-01", "regime_cycle"], "Expansion")
        self.assertEqual(observed.loc["2025-02", "regime_cycle"], "Recession")
        self.assertEqual(observed.loc["2025-03", "regime_cycle"], "Unknown")
        self.assertEqual(observed.loc["2025-02", "fed_regime"], "Unknown")

    def test_market_memory_year_anchor_matches_observed_prior_year(self):
        functions = page_functions("23_Market_Memory_Explorer.py", {"build_feature_frame", "rolling_max_drawdown_array"}, {"DEFAULT_HORIZONS": [5, 21, 63, 252]})
        dates = pd.bdate_range("2023-11-01", periods=400)
        prices = pd.Series(np.linspace(100., 160., len(dates)), index=dates)
        actual = functions["build_feature_frame"](prices, {})
        expected = []
        for day in dates:
            prior = prices[prices.index.year < day.year]
            expected.append(prior.iloc[-1] if len(prior) else np.nan)
        np.testing.assert_allclose(actual["prior_year_close"], expected, equal_nan=True)

    def test_market_memory_yield_changes_use_basis_points(self):
        functions = page_functions("23_Market_Memory_Explorer.py", {"build_feature_frame", "rolling_max_drawdown_array", "bucket_bps_change"}, {"DEFAULT_HORIZONS": [5, 21, 63, 252]})
        dates = pd.bdate_range("2025-01-01", periods=80)
        prices = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        yields = pd.Series(np.linspace(4., 5., len(dates)), index=dates)
        for quotes in (yields, yields * 10):
            actual = functions["build_feature_frame"](prices, {"tnx": quotes})
            self.assertAlmostEqual(actual["tnx_63_bps"].iloc[-1], (yields.iloc[-1] - yields.iloc[-64]) * 100)

    def test_market_stress_breadth_excludes_missing_inputs(self):
        functions = page_functions("19_Market_Stress_Composite.py", {"foreign_breadth_components"})
        dates = pd.bdate_range("2025-01-01", periods=80)
        prices = pd.DataFrame({"missing": np.nan, "falling": np.linspace(100, 80, len(dates))}, index=dates)
        returns = prices.pct_change(21, fill_method=None)
        negative, below = functions["foreign_breadth_components"](prices, returns)
        self.assertEqual(negative.iloc[-1], 1.)
        self.assertEqual(below.iloc[-1], 1.)
        self.assertTrue(negative.iloc[:21].isna().all())
        self.assertTrue(below.iloc[:59].isna().all())


class YieldPageRecoveryTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()

    @patch("adfm_core.primary_data.fetch_fred_series")
    @patch("yfinance.download", return_value=pd.DataFrame())
    def test_yield_page_renders_official_fallback_and_optional_source_tables(self, yahoo, fred):
        dates = pd.bdate_range("2025-01-02", periods=420)
        official = pd.DataFrame({"Y3M": np.linspace(4.3, 4., len(dates)), "Y5": np.linspace(4.4, 4.1, len(dates)),
                                 "Y10": np.linspace(4.5, 4.3, len(dates)), "Y30": np.linspace(4.7, 4.6, len(dates))}, index=dates)
        official["Y2"] = 4.1
        official["R5"] = 1.7
        official["R10"] = 1.8
        official["BE5"] = 2.2
        official["BE10"] = 2.3
        fred.return_value = (official, pd.DataFrame([{"symbol": "DGS10", "status": "OK", "data_through": str(dates[-1].date())}]))
        app = AppTest.from_file(str(ROOT / "pages" / "4_Yield_Curve_Rates_Regime_Monitor.py"), default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.error), [])
        self.assertTrue(any("Federal Reserve" in notice.value for notice in app.info))
        for checkbox in app.checkbox:
            if checkbox.label in {"Show source yield table", "Show yield download status"}:
                checkbox.set_value(True)
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(len(app.dataframe) > 0)
        self.assertEqual(fred.call_count, 1)
        yahoo.assert_not_called()
        self.assertTrue(any("2Y Treasury" in item.value for item in app.markdown))
