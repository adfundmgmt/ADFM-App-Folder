import unittest

import numpy as np
import pandas as pd

from adfm_core.bond_monitor import GLOBAL_SOVEREIGNS, daily_snapshot, monthly_snapshot, spread_series


def series(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))


class BondMonitorTests(unittest.TestCase):
    def test_global_universe_covers_developed_and_emerging_bond_markets(self):
        countries = dict(GLOBAL_SOVEREIGNS)
        self.assertGreaterEqual(len(countries), 24)
        self.assertTrue({"United States", "Germany", "Japan", "China", "India", "Brazil", "Türkiye"} <= countries.keys())
        self.assertEqual(countries["India"], "INDIRLTLT01STM")

    def test_daily_move_is_basis_points_with_exact_observation_date(self):
        data = series(["2026-08-21", "2026-09-18", "2026-09-21", "2026-09-22"], [4.0, 4.2, 4.3, 4.4])
        result = daily_snapshot(data, pd.Timestamp("2026-09-23"))
        self.assertAlmostEqual(result["1D"], 10)
        self.assertAlmostEqual(result["1M"], 40)
        self.assertEqual(result["Observation"], "2026-09-22")

    def test_stale_daily_yield_has_no_actionable_changes(self):
        data = series(["2026-07-01", "2026-08-01"], [4.0, 4.5])
        result = daily_snapshot(data, pd.Timestamp("2026-09-23"))
        self.assertEqual(result["Status"], "Stale")
        self.assertTrue(np.isnan(result["1M"]))

    def test_missing_anchor_does_not_shorten_one_month(self):
        data = series(["2026-09-10", "2026-09-22"], [4.0, 4.5])
        self.assertTrue(np.isnan(daily_snapshot(data, pd.Timestamp("2026-09-23"))["1M"]))

    def test_spread_requires_same_day_on_both_legs(self):
        a = series(["2026-09-20", "2026-09-22"], [4.0, 4.2])
        b = series(["2026-09-20", "2026-09-21"], [3.5, 3.7])
        spread = spread_series(a, b)
        self.assertEqual(list(spread.index.strftime("%Y-%m-%d")), ["2026-09-20"])
        self.assertAlmostEqual(spread.iloc[0], 0.5)

    def test_monthly_change_requires_exact_prior_month_and_marks_stale(self):
        data = series(["2026-06-01", "2026-08-01"], [3.5, 4.0])
        current = monthly_snapshot(data, pd.Timestamp("2026-09-23"))
        self.assertEqual(current["Observation"], "2026-08")
        self.assertTrue(np.isnan(current["1M"]))
        self.assertEqual(current["Status"], "Current")
        stale = monthly_snapshot(data, pd.Timestamp("2027-02-01"))
        self.assertEqual(stale["Status"], "Stale")
        self.assertTrue(np.isnan(stale["3M"]))


if __name__ == "__main__":
    unittest.main()
