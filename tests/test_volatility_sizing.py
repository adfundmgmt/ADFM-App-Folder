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


if __name__ == '__main__':
    unittest.main()
