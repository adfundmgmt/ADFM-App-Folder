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
                self.assertFalse(signals.iloc[:252].any())
                self.assertEqual(source, "CFTC unavailable")

    @patch("adfm_core.commodity_top_exhaustion_page.load_cftc_crowding")
    def test_crowded_profile_requires_real_positioning_and_never_uses_volume_as_crowding(
        self, crowding
    ):
        crowding.return_value = (pd.Series(dtype=float), "CFTC unavailable")
        dates = pd.bdate_range("2020-01-01", periods=800)
        close = 100 * np.exp(np.linspace(0.0, 1.8, len(dates)))
        volume = np.linspace(1000.0, 100000.0, len(dates))
        data = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)

        diagnostics, signals, _, source = build_exhaustion_frame(
            data,
            "UNMAPPED",
            "Crowded Blow-Off",
            PROFILE_PRESETS["Crowded Blow-Off"],
            63,
        )

        self.assertEqual(source, "CFTC unavailable")
        self.assertTrue(diagnostics["CrowdingPctile"].isna().all())
        self.assertIn("ProfileSetup", diagnostics.columns)
        self.assertFalse(diagnostics["ProfileSetup"].any())
        self.assertFalse(signals.any())

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
