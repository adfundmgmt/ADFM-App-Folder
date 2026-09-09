"""Deterministic accuracy tests for the Public Equities Baskets page."""

from __future__ import annotations

import ast
import math
import unittest
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PAGE = (
    Path(__file__).resolve().parents[1]
    / "pages"
    / "1_ADFM_Public_Equities_Baskets.py"
)


def load_page_functions(names: set[str]) -> dict:
    tree = ast.parse(PAGE.read_text(encoding="utf-8"), filename=str(PAGE))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {
        "pd": pd,
        "np": np,
        "math": math,
        "date": date,
        "datetime": datetime,
        "dt_time": dt_time,
        "timedelta": timedelta,
        "ZoneInfo": ZoneInfo,
        "NY_TZ": ZoneInfo("America/New_York"),
        "COMPLETED_SESSION_TIME": dt_time(16, 15),
        "MIN_LIVE_MEMBER_COVERAGE": 0.75,
        "MIN_DAILY_MEMBER_COVERAGE": 0.75,
        "MIN_BREADTH_MEMBER_COVERAGE": 0.75,
        "MAX_PLAUSIBLE_ABSOLUTE_DAILY_RETURN": 10.0,
        "MAX_FX_FORWARD_FILL_SESSIONS": 3,
        "CACHE_MAX_AGE_DAYS": 7,
        "BENCH": "SPY",
        "BASKET_KEY_SEPARATOR": " :: ",
        "FUND_QUOTE_TYPES": {"ETF", "MUTUALFUND", "MONEYMARKET"},
        "MACD_CONFIGS": {
            "1W": {"frequency": "daily", "fast": 2, "slow": 5, "signal": 2,
                   "acceleration_lookback": 2, "strength_window": 20, "min_observations": 20},
            "1M": {"frequency": "daily", "fast": 4, "slow": 9, "signal": 3,
                   "acceleration_lookback": 3, "strength_window": 30, "min_observations": 30},
            "3M": {"frequency": "daily", "fast": 8, "slow": 17, "signal": 5,
                   "acceleration_lookback": 5, "strength_window": 63, "min_observations": 63},
            "6M": {"frequency": "daily", "fast": 12, "slow": 26, "signal": 9,
                   "acceleration_lookback": 5, "strength_window": 63, "min_observations": 63},
            "YTD": {"frequency": "daily", "fast": 12, "slow": 26, "signal": 9,
                    "acceleration_lookback": 5, "strength_window": 63, "min_observations": 63},
            "1Y": {"frequency": "weekly", "fast": 4, "slow": 9, "signal": 3,
                   "acceleration_lookback": 3, "strength_window": 26, "min_observations": 26},
            "3Y": {"frequency": "weekly", "fast": 8, "slow": 17, "signal": 5,
                   "acceleration_lookback": 4, "strength_window": 52, "min_observations": 52},
            "5Y": {"frequency": "weekly", "fast": 12, "slow": 26, "signal": 9,
                   "acceleration_lookback": 5, "strength_window": 104, "min_observations": 104},
        },
    }
    exec(compile(ast.fix_missing_locations(module), str(PAGE), "exec"), namespace)
    return namespace


class BasketCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_page_functions(
            {
                "completed_download_end",
                "_cache_is_usable",
                "align_levels_to_calendar",
                "ew_rets_from_levels",
                "basket_key",
                "normalize_basket_members",
                "build_live_baskets",
                "required_fx_tickers",
                "convert_foreign_levels_to_usd",
                "convert_market_metadata_to_usd",
                "exclude_strongly_suspect_price_series",
                "rsi",
                "macd_hist",
                "get_macd_config",
                "prepare_macd_series",
                "dynamic_macd_momentum",
                "ema_regime",
                "momentum_label",
                "anchored_total_return",
                "pct_since",
                "compute_display_start",
                "basket_vs_dma_pct",
                "calculate_basket_breadth",
                "build_panel_df",
                "select_performance_extremes",
            }
        )

    def test_all_preset_returns_use_the_observed_prior_anchor(self):
        dates = pd.bdate_range("2019-12-31", "2026-06-30")
        levels = pd.Series(100.0 * np.cumprod(np.repeat(1.001, len(dates))), index=dates)
        reference = pd.Timestamp("2026-06-30")
        for preset in ["1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]:
            with self.subTest(preset=preset):
                start = pd.Timestamp(
                    self.ns["compute_display_start"](preset, reference.date())
                )
                anchor = levels[levels.index <= start].iloc[-1]
                expected = levels.iloc[-1] / anchor - 1.0
                actual = self.ns["anchored_total_return"](
                    levels,
                    start_ts=start,
                    end_ts=reference,
                )
                self.assertAlmostEqual(actual, expected)

    def test_ytd_keeps_the_first_trading_sessions_return(self):
        levels = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]),
        )
        result = self.ns["anchored_total_return"](
            levels,
            start_ts=pd.Timestamp("2026-01-01"),
            end_ts=pd.Timestamp("2026-01-05"),
        )
        self.assertAlmostEqual(result, 0.21)

    def test_equal_weight_is_daily_rebalanced_arithmetic_mean(self):
        levels = pd.DataFrame(
            {"AAA": [100.0, 110.0, 121.0], "BBB": [100.0, 90.0, 99.0]},
            index=pd.bdate_range("2026-01-02", periods=3),
        )
        result = self.ns["ew_rets_from_levels"](
            levels,
            {"Pair": ["AAA", "BBB"]},
            min_daily_coverage=0.75,
        )["Pair"]
        self.assertAlmostEqual(result.iloc[1], 0.0)
        self.assertAlmostEqual(result.iloc[2], 0.10)

    def test_missing_member_is_not_zero_and_daily_coverage_fails_closed(self):
        dates = pd.bdate_range("2026-01-02", periods=2)
        levels = pd.DataFrame(
            {f"S{i}": [100.0, 101.0 if i < 7 else np.nan] for i in range(10)},
            index=dates,
        )
        result = self.ns["ew_rets_from_levels"](
            levels,
            {"Ten": list(levels.columns)},
            min_daily_coverage=0.75,
        )["Ten"].reindex(dates)
        self.assertTrue(pd.isna(result.loc[dates[-1]]))

        levels.loc[dates[-1], "S7"] = 101.0
        result = self.ns["ew_rets_from_levels"](
            levels,
            {"Ten": list(levels.columns)},
            min_daily_coverage=0.75,
        )["Ten"].reindex(dates)
        self.assertAlmostEqual(result.loc[dates[-1]], 0.01)

    def test_live_basket_requires_three_quarters_of_defined_members(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        levels = pd.DataFrame(
            {"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3]},
            index=dates,
        )
        categories = {"Test": {"Four": ["A", "B", "C", "D"]}}
        live, _, dropped = self.ns["build_live_baskets"](
            levels, categories, {}, None, 30, dates[-1]
        )
        self.assertIn("Four", live["Test"])
        self.assertFalse(dropped)

        live, _, dropped = self.ns["build_live_baskets"](
            levels.drop(columns="C"), categories, {}, None, 30, dates[-1]
        )
        self.assertFalse(live)
        self.assertEqual(dropped, ["Test :: Four"])

    def test_macd_methodology_changes_with_preset(self):
        configs = [self.ns["get_macd_config"](preset) for preset in ["1W", "1M", "3M", "1Y", "3Y", "5Y"]]
        signatures = {
            (cfg["frequency"], cfg["fast"], cfg["slow"], cfg["signal"], cfg["strength_window"])
            for cfg in configs
        }
        self.assertEqual(len(signatures), len(configs))
        self.assertEqual(self.ns["get_macd_config"]("1Y")["frequency"], "weekly")

    def test_macd_insufficient_history_is_neutral(self):
        short = pd.Series(
            np.arange(10.0),
            index=pd.bdate_range("2026-01-02", periods=10),
        )
        self.assertEqual(self.ns["dynamic_macd_momentum"](short, "1M"), "Neutral")
        self.assertEqual(self.ns["dynamic_macd_momentum"](short, "5Y"), "Neutral")

    def test_weekly_macd_excludes_incomplete_week(self):
        levels = pd.Series(
            np.arange(8.0),
            index=pd.to_datetime(
                [
                    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
                    "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-15",
                ]
            ),
        )
        weekly = self.ns["prepare_macd_series"](levels, "weekly")
        self.assertEqual(weekly.index.max(), pd.Timestamp("2026-01-09"))

    def test_breadth_uses_only_valid_member_returns_and_requires_coverage(self):
        dates = pd.to_datetime(["2025-12-31", "2026-01-30"])
        data = {}
        for index in range(10):
            if index < 6:
                data[f"S{index}"] = [100.0, 110.0]
            elif index < 8:
                data[f"S{index}"] = [100.0, 90.0]
            else:
                data[f"S{index}"] = [np.nan, 90.0]
        levels = pd.DataFrame(data, index=dates)
        basket = {"Ten": list(data)}
        detail = self.ns["calculate_basket_breadth"](
            levels,
            basket,
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-30"),
            0.75,
        )["Ten"]
        self.assertEqual(detail["Valid Members"], 8)
        self.assertEqual(detail["Breadth %"], 75.0)

        levels.loc[pd.Timestamp("2025-12-31"), "S7"] = np.nan
        detail = self.ns["calculate_basket_breadth"](
            levels,
            basket,
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-30"),
            0.75,
        )["Ten"]
        self.assertTrue(pd.isna(detail["Breadth %"]))

    def test_breadth_changes_with_the_selected_horizon(self):
        dates = pd.to_datetime(["2026-05-29", "2026-06-23", "2026-06-30"])
        levels = pd.DataFrame(
            {
                "A": [100.0, 120.0, 110.0],
                "B": [100.0, 105.0, 115.0],
            },
            index=dates,
        )
        baskets = {"Pair": ["A", "B"]}
        one_month = self.ns["calculate_basket_breadth"](
            levels,
            baskets,
            pd.Timestamp("2026-05-30"),
            pd.Timestamp("2026-06-30"),
            0.75,
        )["Pair"]["Breadth %"]
        one_week = self.ns["calculate_basket_breadth"](
            levels,
            baskets,
            pd.Timestamp("2026-06-23"),
            pd.Timestamp("2026-06-30"),
            0.75,
        )["Pair"]["Breadth %"]
        self.assertEqual(one_month, 100.0)
        self.assertEqual(one_week, 50.0)

    def test_table_metrics_remain_numeric_and_duplicate_names_are_disambiguated(self):
        dates = pd.bdate_range("2025-01-02", periods=300)
        returns = pd.DataFrame(
            {
                "A :: Same": np.repeat(0.001, len(dates)),
                "B :: Same": np.repeat(0.002, len(dates)),
            },
            index=dates,
        )
        metadata = {
            "A :: Same": {"Category": "A", "Basket": "Same"},
            "B :: Same": {"Category": "B", "Basket": "Same"},
        }
        breadth = {
            "A :: Same": {"Breadth %": 50.0},
            "B :: Same": {"Breadth %": 75.0},
        }
        panel = self.ns["build_panel_df"](
            returns,
            dates[-100],
            "3M",
            metadata,
            breadth,
        )
        for column in ["%5D", "%1M", "%3M", "RSI(14W)", "Breadth %", "vs 21DMA %", "vs 50DMA %"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(panel[column]))
        self.assertEqual(set(panel["Basket"]), {"A | Same", "B | Same"})

    def test_fx_multiply_divide_and_no_future_fill(self):
        self.ns["FX_CONVERSIONS"] = {
            "EUR.LOCAL": ("EURUSD=X", "multiply"),
            "JPY.LOCAL": ("JPY=X", "divide"),
        }
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        levels = pd.DataFrame(
            {
                "EUR.LOCAL": [100.0, 110.0],
                "EURUSD=X": [1.10, 1.20],
                "JPY.LOCAL": [10_000.0, 11_000.0],
                "JPY=X": [100.0, 110.0],
            },
            index=dates,
        )
        converted, issues = self.ns["convert_foreign_levels_to_usd"](levels)
        self.assertFalse(issues)
        self.assertAlmostEqual(converted.loc[dates[-1], "EUR.LOCAL"], 132.0)
        self.assertAlmostEqual(converted.loc[dates[-1], "JPY.LOCAL"], 100.0)

        missing_fx = levels.copy()
        missing_fx.loc[dates[0], "EURUSD=X"] = np.nan
        converted, _ = self.ns["convert_foreign_levels_to_usd"](missing_fx)
        self.assertTrue(pd.isna(converted.loc[dates[0], "EUR.LOCAL"]))

    def test_gbp_pence_market_cap_is_scaled_before_fx(self):
        self.ns["FX_CONVERSIONS"] = {"ABC.L": ("GBPUSD=X", "multiply")}
        metadata = {"ABC.L": {"market_cap": 100_000_000.0, "currency": "GBp"}}
        raw = pd.DataFrame(
            {"GBPUSD=X": [1.25]},
            index=[pd.Timestamp("2026-01-02")],
        )
        converted = self.ns["convert_market_metadata_to_usd"](metadata, raw)
        self.assertEqual(converted["ABC.L"]["market_cap"], 1_250_000.0)

    def test_stale_prices_and_missing_market_cap_fail_closed(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-20"])
        levels = pd.DataFrame({"STALE": [10.0, np.nan], "LIVE": [10.0, 11.0]}, index=dates)
        members = self.ns["normalize_basket_members"](
            levels,
            ["STALE", "LIVE"],
            market_metadata={"LIVE": {"quote_type": "EQUITY", "market_cap": None}},
            min_market_cap=1_000_000_000,
            stale_days=10,
            reference_date=dates[-1],
        )
        self.assertEqual(members, [])

        fund_members = self.ns["normalize_basket_members"](
            levels,
            ["LIVE"],
            market_metadata={"LIVE": {"quote_type": "ETF", "market_cap": None}},
            min_market_cap=1_000_000_000,
            stale_days=10,
            reference_date=dates[-1],
        )
        self.assertEqual(fund_members, ["LIVE"])

    def test_foreign_calendar_alignment_preserves_missing_sessions(self):
        calendar = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        levels = pd.DataFrame(
            {"FOREIGN": [100.0, 101.0]},
            index=[calendar[0], calendar[2]],
        )
        aligned = self.ns["align_levels_to_calendar"](levels, calendar)
        self.assertTrue(pd.isna(aligned.loc[calendar[1], "FOREIGN"]))

    def test_implausible_data_is_excluded_conservatively(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        levels = pd.DataFrame(
            {"GOOD": [100.0, 400.0, 200.0], "BAD": [100.0, 1_200.0, 100.0]},
            index=dates,
        )
        cleaned, issues = self.ns["exclude_strongly_suspect_price_series"](levels)
        self.assertIn("GOOD", cleaned)
        self.assertNotIn("BAD", cleaned)
        self.assertIn("BAD", issues)

    def test_cache_staleness_and_session_cutoff(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-09"])
        levels = pd.DataFrame({"SPY": [100.0, 101.0], "AAA": [50.0, 51.0]}, index=dates)
        usable = self.ns["_cache_is_usable"](
            levels,
            ["SPY", "AAA"],
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-01-12"),
        )
        stale = self.ns["_cache_is_usable"](
            levels.iloc[:1],
            ["SPY", "AAA"],
            pd.Timestamp("2026-01-02"),
            pd.Timestamp("2026-01-12"),
        )
        self.assertTrue(usable)
        self.assertFalse(stale)

        before = datetime(2026, 1, 5, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        after = datetime(2026, 1, 5, 16, 16, tzinfo=ZoneInfo("America/New_York"))
        weekend = datetime(2026, 1, 10, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(self.ns["completed_download_end"](before), date(2026, 1, 5))
        self.assertEqual(self.ns["completed_download_end"](after), date(2026, 1, 6))
        self.assertEqual(self.ns["completed_download_end"](weekend), date(2026, 1, 11))

    def test_performance_chart_selection_is_top_bottom_and_uses_table_values(self):
        panel = pd.DataFrame(
            {"Basket": [f"B{i}" for i in range(40)], "%YTD": np.arange(40.0)},
            index=[f"K{i}" for i in range(40)],
        )
        selected = self.ns["select_performance_extremes"](panel, "%YTD", 15)
        self.assertEqual(len(selected), 30)
        self.assertEqual(set(selected.index), {f"K{i}" for i in range(15)} | {f"K{i}" for i in range(25, 40)})
        pd.testing.assert_series_equal(selected["%YTD"], panel.loc[selected.index, "%YTD"])

    def test_primary_table_is_native_and_has_no_consolidated_heading(self):
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn("st.dataframe(", source)
        self.assertNotIn("go.Table(", source)
        self.assertNotIn('heading="All Baskets | Consolidated Panel"', source)
        self.assertIn("heading=None", source)
        self.assertIn('st.subheader("Basket Performance")', source)


if __name__ == "__main__":
    unittest.main()
