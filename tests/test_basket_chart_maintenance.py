"""Deterministic preservation and freshness checks for basket/chart maintenance."""

from __future__ import annotations

import ast
import importlib
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def page_functions(filename, names):
    tree = ast.parse((ROOT / "pages" / filename).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    scope = {
        "pd": pd,
        "np": np,
        "REQUIRED_PRICE_COLUMNS": ["Open", "High", "Low", "Close"],
        "CAP_MAX_ROWS": 250_000,
        "MIN_DAILY_MEMBER_COVERAGE": 0.6,
        "BENCH": "SPY",
        "math": math,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *nodes,
                    ],
                    type_ignores=[],
                )
            ),
            filename,
            "exec",
        ),
        scope,
    )
    return scope


def prices():
    dates = pd.bdate_range("2025-01-01", periods=320)
    close = pd.Series(
        100 + np.arange(320) * 0.1 + np.sin(np.arange(320) / 9), index=dates
    )
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000 + np.arange(320),
        },
        index=dates,
    )


class PagePreservationTests(unittest.TestCase):
    def test_chart_indicators_preserve_raw_volume_and_standard_formulas(self):
        scope = page_functions(
            "10_ADFM_Chart_Terminal.py",
            {
                "compute_rsi",
                "compute_macd",
                "compute_atr",
                "compute_bollinger_bands",
                "add_indicators",
            },
        )
        frame = prices()
        actual = scope["add_indicators"](frame)
        pd.testing.assert_series_equal(actual["Volume"], frame["Volume"])
        pd.testing.assert_series_equal(
            actual["SMA20"],
            frame.Close.rolling(20, min_periods=20).mean(),
            check_names=False,
        )
        macd = (
            frame.Close.ewm(span=12, adjust=False, min_periods=12).mean()
            - frame.Close.ewm(span=26, adjust=False, min_periods=26).mean()
        )
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        pd.testing.assert_series_equal(
            actual.MACD_HIST, macd - signal, check_names=False
        )
        self.assertAlmostEqual(actual.ATR14.iloc[-1], 2.0)
        self.assertEqual(len(actual), len(frame))
        pd.testing.assert_frame_equal(frame, prices())

    def test_basket_returns_preserve_coverage_and_adjacent_observed_sessions(self):
        scope = page_functions(
            "1_ADFM_Public_Equities_Baskets.py", {"ew_rets_from_levels"}
        )
        dates = pd.bdate_range("2026-01-05", periods=4)
        frame = pd.DataFrame(
            {
                "A": [100, 110, 121, 121],
                "B": [100, 100, 100, 100],
                "C": [100, np.nan, 100, 100],
            },
            index=dates,
        )
        actual = scope["ew_rets_from_levels"](
            frame, {"basket": ["A", "B", "C"], "missing": ["NONE"]}
        )
        np.testing.assert_allclose(actual.basket, [0, 0.05, 0.05, 0])
        self.assertNotIn("missing", actual)
        frame.loc[dates[1], "B"] = np.nan
        self.assertTrue(
            pd.isna(
                scope["ew_rets_from_levels"](
                    frame, {"basket": ["A", "B", "C"]}
                ).basket.iloc[1]
            )
        )


class ExtractedHelpersTests(unittest.TestCase):
    def helper(self, module):
        self.assertIsNotNone(
            importlib.util.find_spec(module),
            f"{module} must expose extracted page calculations",
        )
        return importlib.import_module(module)

    def test_extracted_chart_preserves_both_yahoo_layouts_and_missing_volume(self):
        core = self.helper("adfm_core.chart_terminal_data")
        frame = prices().iloc[:4].drop(columns="Volume")
        expected = frame.assign(Volume=np.nan)
        for tuples in (
            [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"), ("Close", "SPY")],
            [("SPY", "Open"), ("SPY", "High"), ("SPY", "Low"), ("SPY", "Close")],
        ):
            raw = frame.copy()
            raw.columns = pd.MultiIndex.from_tuples(tuples)
            pd.testing.assert_frame_equal(core.clean_price_data(raw), expected)
        self.assertTrue(core.clean_price_data(frame.drop(columns="Close")).empty)

    def test_extracted_chart_preserves_dates_duplicates_and_raw_volume(self):
        core = self.helper("adfm_core.chart_terminal_data")
        frame = prices().iloc[:4].copy()
        frame.index = frame.index.tz_localize("America/New_York")
        frame = pd.concat([frame.iloc[::-1], frame.iloc[1:2].assign(Volume=987_654)])
        cleaned = core.clean_price_data(frame)
        self.assertIsNone(cleaned.index.tz)
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned.Volume.iloc[1], 987_654)

    def test_observation_summary_never_promotes_missing_endpoint(self):
        core = self.helper("adfm_core.basket_calculations")
        dates = pd.bdate_range("2026-01-05", periods=4)
        returns = pd.Series([0, 0.05, 0.1, np.nan], index=dates)
        summary = core.basket_observation(returns)
        self.assertEqual(summary["observed_as_of"], dates[-2])
        self.assertEqual(summary["observation_age_days"], 1)
        self.assertTrue(pd.isna(summary["prior_close_change_pct"]))
        returns.iloc[-1] = -0.02
        summary = core.basket_observation(returns)
        self.assertEqual(summary["observation_age_days"], 0)
        self.assertAlmostEqual(summary["prior_close_change_pct"], -2.0)
        summary = core.basket_observation(pd.Series([np.nan], index=dates[:1]))
        self.assertIsNone(summary["observed_as_of"])
        summary = core.basket_observation(pd.Series([0.0], index=dates[:1]))
        self.assertTrue(
            pd.isna(summary["prior_close_change_pct"]),
            "inception anchor has no observed prior close",
        )

    def test_extracted_calendar_filters_universe_outage_but_not_one_symbol(self):
        core = self.helper("adfm_core.basket_calculations")
        dates = pd.bdate_range("2026-09-14", periods=9)
        frame = pd.DataFrame(
            {
                name: np.arange(100, 109, dtype=float)
                for name in ["SPY", "A", "B", "C", "D"]
            },
            index=dates,
        )
        frame.loc[dates[-2], "A"] = np.nan
        calendar, omitted = core.reliable_price_sessions(
            frame, dates, ["A", "B", "C", "D"]
        )
        self.assertTrue(omitted.empty)
        frame.loc[dates[-2], ["B", "C", "D"]] = np.nan
        calendar, omitted = core.reliable_price_sessions(
            frame, dates, ["A", "B", "C", "D"]
        )
        self.assertEqual(list(omitted), [dates[-2]])
        rets = core.ew_rets_from_levels(
            frame.reindex(calendar), {"basket": ["A", "B", "C", "D"]}
        )
        self.assertAlmostEqual(rets.basket.iloc[-1], 108 / 106 - 1)


if __name__ == "__main__":
    unittest.main()


class MaintenanceIntegrationTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()

    def test_basket_download_uses_shared_transport_without_a_guessed_us_cutoff(self):
        scope = page_functions(
            "1_ADFM_Public_Equities_Baskets.py",
            {"_download_close_once", "_clean_index", "_to_float_frame"},
        )
        raw = prices().iloc[:3]
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, "ABC.PA") for col in raw.columns]
        )
        provider = Mock(return_value=raw)
        scope["download_market_data"] = provider
        scope["yf"] = Mock(download=Mock(return_value=raw))
        actual = scope["_download_close_once"](
            ["ABC.PA"], raw.index[0], raw.index[-1] + pd.Timedelta(days=1)
        )
        self.assertEqual(actual.columns.tolist(), ["ABC.PA"])
        self.assertTrue(provider.called, "basket loader must call the shared transport")
        self.assertIs(provider.call_args.kwargs["completed_only"], False)
        self.assertNotIn("session_timezone", provider.call_args.kwargs)

    @patch("adfm_core.market_data.download_market_data")
    def test_chart_page_has_compact_header_and_lazy_signal_detail(self, provider):
        data = prices()
        data.index = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
            periods=len(data),
        )
        provider.return_value = data
        with patch("yfinance.download", return_value=data):
            app = AppTest.from_file(
                str(ROOT / "pages" / "10_ADFM_Chart_Terminal.py"), default_timeout=30
            ).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(app.error), [])
        self.assertTrue(any("Observed bar" in item.value for item in app.caption))
        self.assertFalse(
            any('class="metric-strip"' in item.value for item in app.markdown)
        )
        self.assertFalse(
            any('<table class="signal-table">' in item.value for item in app.markdown)
        )
        detail = [
            item
            for item in app.checkbox
            if item.label == "Show signal matrix and technical memo"
        ]
        self.assertEqual(len(detail), 1)
        self.assertFalse(detail[0].value)
        count = provider.call_count
        detail[0].set_value(True)
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any('<table class="signal-table">' in item.value for item in app.markdown)
        )
        self.assertEqual(
            provider.call_count,
            count,
            "warm detail rerun should reuse the bounded cache",
        )


class BasketFreshnessIntegrationTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()

    def test_fx_alignment_does_not_extend_a_missing_exchange_rate_endpoint(self):
        scope = page_functions(
            "1_ADFM_Public_Equities_Baskets.py",
            {"convert_foreign_levels_to_usd", "required_fx_tickers"},
        )
        scope.update(
            FX_CONVERSIONS={"ABC.PA": ("EURUSD=X", "multiply")},
            MAX_FORWARD_FILL_SESSIONS=5,
        )
        dates = pd.bdate_range("2026-01-05", periods=6)
        frame = pd.DataFrame(
            {
                "ABC.PA": [100.0] * 6,
                "EURUSD=X": [1.1, np.nan, 1.2, np.nan, np.nan, np.nan],
            },
            index=dates,
        )
        actual, issues = scope["convert_foreign_levels_to_usd"](frame)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(actual["ABC.PA"].iloc[1], 110.0)
        self.assertTrue(
            actual["ABC.PA"].iloc[3:].isna().all(),
            "missing trailing FX must not fabricate USD endpoints",
        )
        self.assertEqual(actual.columns.tolist(), ["ABC.PA"])

    def test_panel_omits_close_column_and_keeps_observed_age_for_missing_endpoint(self):
        from tests.test_public_equities_integrity_regressions import scope

        dates = pd.bdate_range("2024-01-02", periods=600)
        returns = pd.DataFrame({"active": 0.001, "stale": 0.001}, index=dates)
        returns.loc[dates[-2] :, "stale"] = np.nan
        panel = scope["build_panel_df"](returns, dates[-63], "3M", {}, returns.active)
        self.assertNotIn("%Close", panel.columns)
        note = dict(zip(panel.index, panel.attrs["row_notes"], strict=True))["stale"]
        self.assertIn(str(dates[-3].date()), note)
        self.assertIn("calendar days old", note)

    def test_basket_scanner_height_expands_to_show_every_row(self):
        source = (ROOT / "pages" / "1_ADFM_Public_Equities_Baskets.py").read_text()
        self.assertIn("height=64 + int(23.4 * max(3, len(panel_df)))", source)
        self.assertNotIn("height=min(920", source)

    @patch("adfm_core.market_data.download_market_data")
    def test_basket_page_keeps_one_table_and_selected_chart_on_demand(self, provider):
        dates = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=650
        )

        def public_prices(tickers, **kwargs):
            close = 100 + np.arange(len(dates)) * 0.1
            frame = pd.DataFrame(
                {("Close", ticker): close for ticker in tickers}, index=dates
            )
            frame.columns = pd.MultiIndex.from_tuples(frame.columns)
            return frame

        provider.side_effect = public_prices
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                app = AppTest.from_file(
                    str(ROOT / "pages" / "1_ADFM_Public_Equities_Baskets.py"),
                    default_timeout=45,
                ).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(list(app.error), [])
                detail = [
                    item
                    for item in app.checkbox
                    if item.label == "Show selected basket chart"
                ]
                self.assertEqual(len(detail), 1)
                self.assertEqual(len(app.get("plotly_chart")), 0)
                count = provider.call_count
                detail[0].set_value(True)
                app.run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(len(app.get("plotly_chart")), 1)
                self.assertEqual(provider.call_count, count)
        finally:
            os.chdir(previous)


class BasketCacheTests(unittest.TestCase):
    def test_disk_snapshot_retention_removes_old_pairs_and_preserves_other_files(self):
        self.assertIsNotNone(importlib.util.find_spec("adfm_core.basket_cache"))
        from adfm_core.basket_cache import prune_basket_snapshots

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            untouched = directory / "other.pkl"
            untouched.write_bytes(b"keep")
            for number in range(5):
                for suffix in ("pkl", "json"):
                    file = directory / f"basket_levels_{number}.{suffix}"
                    file.write_bytes(b"fixture")
                    os.utime(file, (number + 1, number + 1))
            prune_basket_snapshots(directory, max_snapshots=2)
            self.assertEqual(
                sorted(path.name for path in directory.glob("basket_levels_*.pkl")),
                ["basket_levels_3.pkl", "basket_levels_4.pkl"],
            )
            self.assertEqual(len(list(directory.glob("basket_levels_*.json"))), 2)
            self.assertEqual(untouched.read_bytes(), b"keep")


class ChartRecoveryTests(unittest.TestCase):
    def tearDown(self):
        st.cache_data.clear()

    @patch("adfm_core.market_data.download_market_data")
    def test_transient_empty_chart_fetch_does_not_block_next_rerun(self, provider):
        frame = prices()
        frame.index = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
            periods=len(frame),
        )
        provider.side_effect = [pd.DataFrame(), pd.DataFrame(), frame]
        app = AppTest.from_file(
            str(ROOT / "pages" / "10_ADFM_Chart_Terminal.py"), default_timeout=30
        ).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.error), 1)
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            list(app.error),
            [],
            "transient empty response must not become a one-hour cached outage",
        )
        self.assertEqual(len(app.get("plotly_chart")), 1)


class ChartReturnEndpointTests(unittest.TestCase):
    def test_return_horizon_is_unavailable_without_an_observed_prior_anchor(self):
        scope = page_functions("10_ADFM_Chart_Terminal.py", {"close_asof"})
        frame = prices().iloc[:5]
        self.assertIsNone(
            scope["close_asof"](frame, frame.index[0] - pd.Timedelta(days=1))
        )
        self.assertEqual(
            scope["close_asof"](frame, frame.index[2]), frame.Close.iloc[2]
        )

    def test_ytd_is_unavailable_without_the_prior_year_close(self):
        scope = page_functions(
            "10_ADFM_Chart_Terminal.py",
            {"close_asof", "return_from_close", "compute_return_metrics"},
        )
        frame = prices().iloc[:30]
        metrics = scope["compute_return_metrics"](frame)
        self.assertIsNone(metrics["YTD"])
        prior = frame.iloc[:1].copy()
        prior.index = pd.to_datetime(["2024-12-31"])
        metrics = scope["compute_return_metrics"](pd.concat([prior, frame]))
        self.assertAlmostEqual(
            metrics["YTD"], frame.Close.iloc[-1] / prior.Close.iloc[0] - 1
        )


class ChartComparisonEndpointTests(unittest.TestCase):
    def test_comparison_uses_common_observed_dates_without_stale_endpoint_or_gap_carry(self):
        from types import SimpleNamespace

        import plotly.graph_objects as go
        scope = page_functions('10_ADFM_Chart_Terminal.py', {'build_compare_chart', 'build_rangebreaks'})
        dates = pd.bdate_range('2026-01-05', periods=8)
        primary = pd.DataFrame({'Close': np.arange(100., 108.)}, index=dates)
        comparison = pd.Series([100., 101., np.nan, np.nan, np.nan, 105., np.nan, np.nan], index=dates, name='PEER')
        scope.update(go=go, COLORS={'text': '#111111', 'grid': 'white'},
                     fetch_compare_close=Mock(return_value=comparison),
                     warmup_start_date=lambda *args: None, start_date_from_period=lambda *args: None)
        settings = SimpleNamespace(period='max', interval='1d', ticker='PRIMARY', auto_adjust=False)
        figure = scope['build_compare_chart'](primary, settings, ['PEER'])
        self.assertIsNotNone(figure)
        for trace in figure.data:
            self.assertEqual(list(pd.to_datetime(trace.x)), [dates[0], dates[1], dates[5]])
        np.testing.assert_allclose(figure.data[1].y, [100., 101., 105.])
