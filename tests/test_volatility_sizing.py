"""Risk invariants for inverse-volatility exposure sizing."""
import unittest

import numpy as np
import pandas as pd

from adfm_core import volatility_sizing as sizing


class VolatilitySizingTests(unittest.TestCase):
    def test_double_volatility_halves_exposure_and_preserves_risk(self):
        result = sizing.scale_exposure(0.10, 0.10, 0.40, 0.20, 0.30)
        self.assertAlmostEqual(result.target, 0.05)
        self.assertAlmostEqual(result.change, -0.05)
        self.assertAlmostEqual(result.target * 0.40, 0.10 * 0.20)

    def test_calm_markets_respect_ceiling_and_zero_base_stays_zero(self):
        self.assertAlmostEqual(sizing.scale_exposure(.10, .10, .01, .20, .25).target, .25)
        self.assertEqual(sizing.scale_exposure(0, .10, .20, .20, .25).target, 0)

    def test_invalid_volatility_cannot_produce_a_size(self):
        for vol in (0, -1, np.nan, np.inf):
            with self.subTest(vol=vol), self.assertRaises(ValueError):
                sizing.scale_exposure(.10, .10, vol, .20, .25)

    def test_baseline_excludes_recent_shock_and_history_has_no_lookahead(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, .01, 500)
        close = pd.Series(100 * np.cumprod(1 + returns), index=pd.bdate_range('2024-01-01', periods=500))
        original = sizing.volatility_history(close, 20)
        shocked = close.copy()
        shocked.iloc[-20:] *= np.cumprod(1 + np.tile([.08, -.08], 10))
        actual = sizing.volatility_history(shocked, 20)
        self.assertAlmostEqual(actual.iloc[-1].baseline, original.iloc[-1].baseline)
        self.assertGreater(actual.iloc[-1].recent, original.iloc[-1].recent * 2)
        pd.testing.assert_frame_equal(actual.iloc[:-20], original.iloc[:-20])

    def test_insufficient_or_flat_history_has_no_usable_sizing(self):
        for n in (50, 500):
            self.assertTrue(sizing.volatility_history(pd.Series(np.ones(n) * 100), 20).empty)

    def test_invalidation_budget_limits_size_and_respects_direction(self):
        long_distance = sizing.invalidation_distance(100, 92, "Long")
        short_distance = sizing.invalidation_distance(100, 108, "Short")
        for distance in (long_distance, short_distance):
            result = sizing.scale_exposure(.20, .20, .20, .20, .25,
                                          loss_budget=.01, stop_distance=distance)
            self.assertAlmostEqual(result.target, .125)
            self.assertAlmostEqual(result.loss_cap, .125)
            self.assertEqual(result.binding, "Invalidation loss budget")
            self.assertAlmostEqual(result.target * distance, .01)
        self.assertIsNone(sizing.invalidation_distance(100, 0, "Long"))
        for side, price in [("Long", 108), ("Short", 92), ("Long", 100)]:
            with self.subTest(side=side), self.assertRaises(ValueError):
                sizing.invalidation_distance(100, price, side)

    def test_zero_loss_budget_allows_no_exposure_and_ceiling_can_bind(self):
        result = sizing.scale_exposure(.20, .20, .20, .20, .25,
                                      loss_budget=0, stop_distance=.08)
        self.assertEqual(result.target, 0)
        ceiling = sizing.scale_exposure(.20, .20, .20, .20, .10,
                                       loss_budget=.01, stop_distance=.08)
        self.assertEqual(ceiling.binding, "Exposure ceiling")

    def test_directional_tails_use_upside_for_shorts_and_compounded_week(self):
        daily = np.tile([.10, -.02, .01, -.03, .02], 30)
        close = pd.Series(100*np.cumprod(1+daily), index=pd.bdate_range("2025-01-01", periods=len(daily)))
        frame = pd.DataFrame({"Close": close, "Open": close.shift(1)*1.04})
        long = sizing.downside_statistics(frame, "Long")
        short = sizing.downside_statistics(frame, "Short")
        self.assertAlmostEqual(long["Average worst 5% day"], .03)
        self.assertAlmostEqual(short["Average worst 5% day"], .10)
        self.assertEqual(long["Worst adverse gap"], 0)
        self.assertAlmostEqual(short["Worst adverse gap"], .04)
        expected_week = np.prod(1 + daily[:5]) - 1
        self.assertAlmostEqual(short["Average worst 5% week"], expected_week)

    def test_missing_open_or_short_history_is_unavailable_not_zero(self):
        frame = pd.DataFrame({"Close":np.arange(100, 150, dtype=float)})
        stats = sizing.downside_statistics(frame, "Long")
        self.assertTrue(np.isnan(stats["Worst adverse gap"]))
        self.assertTrue(np.isnan(stats["Average worst 5% day"]))

    def test_context_percentile_and_acceleration_use_known_prior_observations(self):
        rng = np.random.default_rng(12)
        returns = rng.normal(0, .005, 600)
        returns[-20:] *= 7
        close = pd.Series(100*np.cumprod(1+returns),index=pd.bdate_range("2024-01-01",periods=600))
        contexts = sizing.volatility_context(close)
        self.assertEqual(set(contexts.index), {10, 20, 60})
        self.assertGreater(contexts.loc[20, "percentile"], 95)
        self.assertGreater(contexts.loc[20, "change"], 1)
        self.assertAlmostEqual(contexts.loc[20,"recent"], sizing.volatility_history(close,20).iloc[-1].recent)

    def test_comparison_uses_same_shock_for_all_sizes(self):
        result = sizing.scale_exposure(.20, .20, .20, .20, .25,
                                      loss_budget=.01, stop_distance=.08)
        table = sizing.comparison_table(.20, result, {}, "Short", stop_distance=.08)
        shock = table.loc[table.Scenario.eq("10% adverse move")].iloc[0]
        self.assertAlmostEqual(shock["Market move"], 10)
        self.assertAlmostEqual(shock["Current"], -2)
        self.assertAlmostEqual(shock["Half size"], -1)
        self.assertAlmostEqual(shock["Vol reference"], -2)
        self.assertAlmostEqual(shock["Permitted"], -1.25)
        stop = table.loc[table.Scenario.eq("At invalidation")].iloc[0]
        self.assertAlmostEqual(stop["Permitted"], -1)


if __name__ == '__main__':
    unittest.main()
