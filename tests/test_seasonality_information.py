import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from tests.test_page_accuracy import page_functions


class SeasonalityInformationTests(unittest.TestCase):
    def test_partial_current_month_path_is_kept_without_future_extension(self):
        functions = page_functions("24_Monthly_Seasonality_Explorer.py", {"_month_paths_prev_eom_equal_weight_from_filtered"})
        prices = pd.Series([100., 102., 104., 106., 200., 210.], index=pd.to_datetime(["2025-08-29", "2025-09-01", "2025-09-02", "2025-09-03", "2026-08-31", "2026-09-01"]))
        periods = pd.PeriodIndex(["2025-09", "2026-09"], freq="M")
        paths, _ = functions["_month_paths_prev_eom_equal_weight_from_filtered"](prices, pd.DataFrame({"month": [9, 9]}, index=periods), 9)
        self.assertAlmostEqual(paths.loc[1, "2026-09"], 5.)
        self.assertTrue(paths.loc[2:, "2026-09"].isna().all())

    def test_intraday_transport_keeps_current_overlay_on_adjusted_basis(self):
        stamps = pd.to_datetime(["2026-09-29 15:30:00+00:00"])
        frame = pd.DataFrame({("Close", "SPY"): [110.0]}, index=stamps)
        download = Mock(return_value=frame)
        functions = page_functions("24_Monthly_Seasonality_Explorer.py", {"fetch_intraday", "current_price_overlay"}, {"download_market_data": download})
        intraday = functions["fetch_intraday"]("SPY")
        prices = pd.Series([90.0, 95.0], index=pd.to_datetime(["2026-09-28", "2026-09-29"]))
        prices.attrs.update(source="Yahoo Finance", adjustment_factor=.9, daily_date="2026-09-29")
        result, status = functions["current_price_overlay"](prices, intraday, now=pd.Timestamp("2026-09-29 16:00:00+00:00"))
        self.assertEqual(result.iloc[-1], 99.0)
        self.assertIn("1-minute", status)
        stale, _ = functions["current_price_overlay"](prices, intraday, now=pd.Timestamp("2026-09-30 16:00:00+00:00"))
        pd.testing.assert_series_equal(stale, prices)

    def test_decision_regimes_use_prior_month_and_missing_trend_stays_unknown(self):
        functions = page_functions("24_Monthly_Seasonality_Explorer.py", {"build_monthly_regime_features"})
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        values = pd.DataFrame({"vix": 10.0, "tnx": np.where(dates.month < 6, 2.0, 6.0), "dxy": 100.0}, index=dates)
        values.loc[values.index.month == 6, "vix"] = 30.0
        strict = functions["build_monthly_regime_features"](values, information_mode="Known at month start")
        self.assertEqual(strict.loc["2020-06", "vix_bucket"], "VIX <15")
        self.assertEqual(strict.loc["2020-06", "teny_trend"], "10Y flat")
        self.assertEqual(strict.loc["2020-02", "teny_trend"], "Unknown")
        revised = functions["build_monthly_regime_features"](values)
        self.assertEqual(revised.loc["2020-06", "vix_bucket"], "VIX >25")
        self.assertEqual(revised.loc["2020-06", "teny_trend"], "10Y rising")


if __name__ == "__main__":
    unittest.main()
