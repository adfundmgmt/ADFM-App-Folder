"""Causality and forward-outcome checks for the deployed commodity study."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from adfm_core import commodity_top_exhaustion_page as study
from adfm_core.commodity_top_exhaustion_page import (
    FORWARD_HORIZONS,
    PROFILE_PRESETS,
    build_event_observations,
    build_exhaustion_frame,
    detect_events,
    summarize_forward_performance,
)


class CommodityExhaustionTests(unittest.TestCase):
    @patch("adfm_core.commodity_top_exhaustion_page.load_cftc_crowding")
    def test_all_signal_profiles_are_unchanged_by_future_prices(self, crowding):
        crowding.return_value = (pd.Series(dtype=float), "CFTC unavailable")
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=700)
        data = pd.DataFrame(
            {
                "Close": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(dates)))),
                "Volume": rng.integers(1000, 10000, len(dates)).astype(float),
            },
            index=dates,
        )
        changed = data.copy()
        changed.iloc[500:, 0] *= 10
        for profile, settings in PROFILE_PRESETS.items():
            with self.subTest(profile=profile):
                original, signals, _, source = build_exhaustion_frame(
                    data, "TEST", profile, settings, 21
                )
                future, _, _, _ = build_exhaustion_frame(
                    changed, "TEST", profile, settings, 21
                )
                pd.testing.assert_frame_equal(original.iloc[:500], future.iloc[:500])
                warmup = 63 if profile == "Failed Breakout" else 252
                self.assertFalse(signals.iloc[:warmup].any())
                self.assertEqual(source, "CFTC unavailable")

    @patch("adfm_core.commodity_top_exhaustion_page.load_cftc_crowding")
    def test_failed_breakout_fires_without_positioning_once_per_setup(self, crowding):
        crowding.return_value = (pd.Series(dtype=float), "CFTC unavailable")
        close = [100.0] * 80 + [105.0, 104.0, 99.0, 98.0, 97.0]
        data = pd.DataFrame({"Close": close, "Volume": 1000.0},
                            index=pd.bdate_range("2020-01-01", periods=len(close)))
        diagnostics, signals, _, source = build_exhaustion_frame(
            data, "UNMAPPED", "Failed Breakout",
            {"breakout_days": 63, "memory": 10}, 63,
        )
        self.assertEqual(source, "CFTC unavailable")
        self.assertTrue(diagnostics["CrowdingPctile"].isna().all())
        self.assertEqual(np.flatnonzero(signals).tolist(), [82])
        self.assertEqual(diagnostics.iloc[82]["BreakoutLevel"], 100.0)
        self.assertFalse(diagnostics.iloc[-1]["BreakoutPending"])
        crowding.return_value = (pd.Series(99.0, index=data.index), "CFTC test fixture")
        _, with_crowding, _, _ = build_exhaustion_frame(
            data, "MAPPED", "Failed Breakout", {"breakout_days": 63, "memory": 10}, 63,
        )
        pd.testing.assert_series_equal(signals, with_crowding)

    def test_failed_breakout_freezes_levels_and_requires_ma_confirmation(self):
        self.assertTrue(hasattr(study, "failed_breakout_frame"))
        close = pd.Series([100.0] * 63 + [105.0, 110.0, 107.0, 104.0, 99.0])
        result = study.failed_breakout_frame(close, 63, 10)
        # 107 is below the latest high but above both original breakout levels.
        # 104 fails 105, but is still above the 10-day average.
        self.assertEqual(np.flatnonzero(result["Signal"]).tolist(), [67])
        self.assertEqual(result.iloc[67]["BreakoutLevel"], 105.0)

    def test_failed_breakout_window_includes_tenth_session_but_not_eleventh(self):
        self.assertTrue(hasattr(study, "failed_breakout_frame"))
        for count, expected in [(9, [73]), (10, [])]:
            with self.subTest(sessions_before_failure=count):
                close = pd.Series([100.0] * 63 + [105.0] + [105.0] * count + [99.0])
                result = study.failed_breakout_frame(close, 63, 10)
                self.assertEqual(np.flatnonzero(result["Signal"]).tolist(), expected)
                self.assertFalse(result.iloc[-1]["BreakoutPending"])

    def test_cftc_positioning_is_available_only_after_publication(self):
        self.assertTrue(hasattr(study, "cftc_availability_date"))
        availability = study.cftc_availability_date

        # Tuesday Sep 8 positions are normally published after market hours Friday Sep 11.
        self.assertEqual(availability(pd.Timestamp("2026-09-08")), pd.Timestamp("2026-09-14"))
        # The 2025 shutdown delayed the Oct 7 report until Friday Nov 21.
        self.assertEqual(availability(pd.Timestamp("2025-10-07")), pd.Timestamp("2025-11-24"))
        # A delayed report published Monday becomes usable the following business session.
        self.assertEqual(availability(pd.Timestamp("2025-11-25")), pd.Timestamp("2025-12-16"))

    def test_cftc_contract_map_covers_core_commodity_futures_and_micro_aliases(self):
        self.assertTrue(hasattr(study, "CFTC_CONTRACT_CODES"))
        mapping = study.CFTC_CONTRACT_CODES
        self.assertEqual(mapping["MGC=F"], mapping["GC=F"])
        self.assertEqual(mapping["SIL=F"], mapping["SI=F"])
        self.assertEqual(mapping["RB=F"], "111659")
        self.assertEqual(mapping["HO=F"], "022651")
        self.assertEqual(mapping["ZC=F"], "002602")
        self.assertEqual(mapping["CC=F"], "073732")
        self.assertEqual(mapping["LE=F"], "057642")

    def test_event_spacing_counts_observed_sessions_and_one_signal_per_run(self):
        dates = pd.bdate_range("2026-01-01", periods=9)
        condition = pd.Series(
            [True, True, False, True, False, False, True, True, False], index=dates
        )
        self.assertEqual(detect_events(condition, spacing_days=4).tolist(), [dates[0], dates[6]])

    def test_forward_returns_drawdowns_and_incomplete_samples(self):
        dates = pd.bdate_range("2024-01-01", periods=260)
        close = pd.Series(np.linspace(100, 50, 260), index=dates)
        events = [dates[0], dates[-1]]
        diagnostics = pd.DataFrame({"RealizedVol": 0.20}, index=dates)
        history, arrays = build_event_observations(close, events, diagnostics)
        summary = summarize_forward_performance(arrays)
        for label, horizon in FORWARD_HORIZONS.items():
            expected = close.iloc[horizon] / close.iloc[0] - 1
            self.assertAlmostEqual(history.loc[0, label], expected)
            self.assertTrue(pd.isna(history.loc[1, label]))
            np.testing.assert_allclose(arrays[label]["signal_dd"], [expected])
            np.testing.assert_allclose(arrays[label]["path_dd"], [expected])
            self.assertEqual(summary.loc["Sample", label], 1)
            self.assertEqual(summary.loc["% Negative", label], 1.0)
        self.assertTrue(pd.isna(history.loc[1, "DaysFromLocalPeak"]))

    def test_edge_summary_uses_independent_signal_samples_and_a_baseline(self):
        self.assertTrue(hasattr(study, "summarize_forward_edge"))
        dates = pd.bdate_range("2024-01-01", periods=90)
        close = pd.Series(100.0 - np.arange(len(dates)) * 0.25, index=dates)
        events = [dates[i] for i in (0, 5, 10, 15, 20, 25)]
        diagnostics = pd.DataFrame({"RealizedVol": 0.20}, index=dates)
        history, arrays = build_event_observations(close, events, diagnostics)

        first = study.summarize_forward_edge(close, history, arrays, dates[0])
        second = study.summarize_forward_edge(close, history, arrays, dates[0])

        self.assertEqual(first.loc["Independent N", "2W"], 3)
        self.assertTrue(np.isfinite(first.loc["Baseline Median", "2W"]))
        self.assertTrue(np.isfinite(first.loc["Median Edge", "2W"]))
        self.assertTrue(np.isfinite(first.loc["95% CI Low", "2W"]))
        self.assertTrue(np.isfinite(first.loc["95% CI High", "2W"]))
        pd.testing.assert_frame_equal(first, second)

    def test_event_observations_include_volatility_normalized_excursions(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        close = pd.Series(np.linspace(100.0, 70.0, len(dates)), index=dates)
        diagnostics = pd.DataFrame({"RealizedVol": 0.25}, index=dates)
        _, arrays = build_event_observations(close, [dates[0]], diagnostics)

        self.assertIn("signal_dd_vol", arrays["3M"])
        self.assertIn("upside_vol", arrays["3M"])
        self.assertEqual(arrays["3M"]["signal_dd_vol"].size, 1)
        self.assertTrue(np.isfinite(arrays["3M"]["signal_dd_vol"][0]))

    def test_empty_events_do_not_invent_forward_results(self):
        dates = pd.bdate_range("2026-01-01", periods=10)
        history, arrays = build_event_observations(
            pd.Series(100.0, index=dates), [], pd.DataFrame(index=dates)
        )
        self.assertTrue(history.empty)
        self.assertTrue(summarize_forward_performance(arrays).isna().all().all())


if __name__ == "__main__":
    unittest.main()
