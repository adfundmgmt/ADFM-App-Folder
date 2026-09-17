import unittest

import numpy as np
import pandas as pd

from adfm_core.sector_rotation import (
    CONFIRMATION_SESSIONS,
    NEUTRAL_BAND,
    build_catalog,
    classify_coordinates,
    compute_breadth,
    compute_relative_metrics,
    confirm_state_series,
    movement_from_coordinates,
)


class SectorRotationUpgradeTests(unittest.TestCase):
    def test_catalog_contains_approved_178_entries_and_scopes(self):
        catalog = build_catalog()
        self.assertEqual(len(catalog), 178)
        self.assertEqual(set(catalog["Universe"]), {"Sectors", "Industries", "Themes", "Countries"})
        self.assertEqual((catalog["Kind"] == "Stock Basket").sum(), 36)
        self.assertEqual((catalog["Universe"] == "Countries").sum(), 40)

    def test_neutral_band_is_small_and_does_not_swallow_normal_rotation(self):
        self.assertLessEqual(NEUTRAL_BAND, 0.01)
        self.assertEqual(classify_coordinates(0.03, 0.02), "Leading")
        self.assertEqual(classify_coordinates(-0.03, 0.02), "Improving")
        self.assertEqual(classify_coordinates(0.001, -0.001), "Neutral")

    def test_movement_measures_five_session_coordinate_change(self):
        idx = pd.date_range("2026-01-01", periods=7, freq="B")
        x = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09], index=idx)
        y = pd.Series([0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08], index=idx)
        out = movement_from_coordinates(x, y, periods=5)
        self.assertAlmostEqual(out["dx"], 0.07)
        self.assertAlmostEqual(out["dy"], 0.07)
        self.assertAlmostEqual(out["speed"], np.sqrt(0.07**2 + 0.07**2))
        self.assertAlmostEqual(out["angle"], 45.0)

    def test_state_change_requires_persistence_and_counts_days(self):
        raw = pd.Series([
            "Lagging", "Lagging", "Lagging",
            "Improving", "Lagging", "Improving",
            "Improving", "Improving", "Improving", "Improving",
        ])
        confirmed, days = confirm_state_series(raw, confirmation_sessions=CONFIRMATION_SESSIONS)
        self.assertEqual(confirmed.iloc[5], "Lagging")
        self.assertEqual(confirmed.iloc[8], "Improving")
        self.assertEqual(days.iloc[8], 1)
        self.assertEqual(days.iloc[9], 2)

    def test_relative_metrics_expose_1w_1m_3m_and_weekly_change(self):
        idx = pd.date_range("2025-01-01", periods=90, freq="B")
        bench = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
        asset = pd.Series(np.linspace(100.0, 125.0, len(idx)), index=idx)
        metrics = compute_relative_metrics(asset, bench)
        for key in ("rel_1w", "rel_1m", "rel_3m", "rel_1w_change"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertGreater(metrics["rel_3m"], 0)

    def test_breadth_uses_constituent_prices_without_filling_missing_sessions(self):
        idx = pd.date_range("2025-01-01", periods=230, freq="B")
        a = pd.Series(np.linspace(80.0, 120.0, len(idx)), index=idx)
        b = pd.Series(np.linspace(120.0, 80.0, len(idx)), index=idx)
        b.iloc[-1] = np.nan
        prices = pd.DataFrame({"A": a, "B": b})
        out = compute_breadth(prices)
        self.assertEqual(out["coverage"], 1)
        self.assertEqual(out["above_50d"], 100.0)
        self.assertEqual(out["above_200d"], 100.0)
        self.assertTrue(np.isfinite(out["breadth_1m_change"]))


if __name__ == "__main__":
    unittest.main()
