import math
import unittest

import numpy as np
import pandas as pd

from adfm_core.relative_volatility import (
    annualized_realized_volatility,
    prior_percentile_rank,
    relative_volatility_frame,
    rolling_zscore_previous,
)


class RelativeVolatilityTests(unittest.TestCase):
    def test_annualized_realized_volatility_matches_log_return_definition(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        close = pd.Series(100.0 * np.exp(np.r_[0.0, returns.cumsum()]))

        result = annualized_realized_volatility(close, window=4)
        expected = returns.std(ddof=1) * math.sqrt(252) * 100.0

        self.assertTrue(np.isclose(result.iloc[-1], expected))

    def test_rolling_zscore_excludes_current_observation(self):
        values = pd.Series([1.0, 2.0, 3.0, 10.0])
        result = rolling_zscore_previous(values, window=3, min_periods=3)

        self.assertTrue(result.iloc[:3].isna().all())
        self.assertTrue(np.isclose(result.iloc[-1], 8.0))

    def test_prior_percentile_rank_gives_ties_half_credit(self):
        self.assertEqual(
            prior_percentile_rank(pd.Series([1.0, 2.0, 2.0])),
            75.0,
        )

    def test_relative_volatility_frame_builds_ratio_and_both_zscores(self):
        index = pd.date_range("2025-01-01", periods=12, freq="D")
        primary = pd.Series(
            100 * np.exp(np.cumsum([0.00, 0.01, -0.02] * 4)), index=index
        )
        comparison = pd.Series(
            100 * np.exp(np.cumsum([0.00, 0.005, -0.01] * 4)), index=index
        )

        result = relative_volatility_frame(
            primary,
            comparison,
            rvol_window=3,
            normalization_window=3,
            normalization_min_periods=2,
        )

        self.assertTrue(
            {
                "primary_rvol",
                "comparison_rvol",
                "rvol_ratio",
                "primary_zscore",
                "comparison_zscore",
            }.issubset(result.columns)
        )
        self.assertTrue(np.isclose(result["rvol_ratio"].dropna().iloc[-1], 2.0))


if __name__ == "__main__":
    unittest.main()
