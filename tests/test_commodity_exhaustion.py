"""Causality and forward-outcome checks for the deployed commodity study."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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
        crowding.return_value = (pd.Series(dtype=float), "")
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=700)
        data = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(.001, .02, len(dates)))),
                             "Volume": rng.integers(1000, 10000, len(dates)).astype(float)}, index=dates)
        changed = data.copy()
        changed.iloc[500:, 0] *= 10
        for profile, settings in PROFILE_PRESETS.items():
            with self.subTest(profile=profile):
                original, signals, _, source = build_exhaustion_frame(data, "TEST", profile, settings, 21)
                future, _, _, _ = build_exhaustion_frame(changed, "TEST", profile, settings, 21)
                pd.testing.assert_frame_equal(original.iloc[:500], future.iloc[:500])
                self.assertFalse(signals.iloc[:252].any())
                self.assertEqual(source, "Volume intensity fallback")

    def test_event_spacing_counts_observed_sessions_and_one_signal_per_run(self):
        dates = pd.bdate_range("2026-01-01", periods=9)
        condition = pd.Series([True, True, False, True, False, False, True, True, False], index=dates)
        self.assertEqual(detect_events(condition, spacing_days=4).tolist(), [dates[0], dates[6]])

    def test_forward_returns_drawdowns_and_incomplete_samples(self):
        dates = pd.bdate_range("2024-01-01", periods=260)
        close = pd.Series(np.linspace(100, 50, 260), index=dates)
        events = [dates[0], dates[-1]]
        history, arrays = build_event_observations(close, events, pd.DataFrame(index=dates))
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

    def test_empty_events_do_not_invent_forward_results(self):
        dates = pd.bdate_range("2026-01-01", periods=10)
        history, arrays = build_event_observations(pd.Series(100., index=dates), [], pd.DataFrame())
        self.assertTrue(history.empty)
        self.assertTrue(summarize_forward_performance(arrays).isna().all().all())
