import unittest

import numpy as np
import pandas as pd

from adfm_core.regime_math import (
    grouped_weighted_composite,
    rolling_percentile_previous,
)


class RegimeMathTests(unittest.TestCase):
    def test_percentiles_match_reference_with_ties_gaps_and_future_changes(self) -> None:
        rng = np.random.default_rng(41)
        values = pd.Series(rng.integers(-4, 5, 500), dtype=float)
        values.iloc[::7] = np.nan
        window, minimum = 31, 10
        expected = []
        for position, current in enumerate(values):
            previous = values.iloc[max(0, position - window):position].dropna()
            if pd.isna(current) or len(previous) < minimum:
                expected.append(np.nan)
            else:
                expected.append(((previous < current).sum() + .5 * (previous == current).sum()) / len(previous) * 100)
        result = rolling_percentile_previous(values, window, minimum, 100)
        pd.testing.assert_series_equal(result, pd.Series(expected))
        changed = values.copy()
        changed.iloc[300:] = 999
        pd.testing.assert_series_equal(result.iloc[:300], rolling_percentile_previous(changed, window, minimum, 100).iloc[:300])

    def test_composite_renormalizes_missing_members_and_enforces_coverage(self) -> None:
        scores = pd.DataFrame({"a": [2., np.nan, np.nan], "b": [4., 4., np.nan], "c": [-2., np.nan, np.nan]})
        specs = [{"name": "a", "category": "A", "weight": 1},
                 {"name": "b", "category": "A", "weight": 3},
                 {"name": "c", "category": "B", "weight": 1}]
        groups, composite, breadth, coverage = grouped_weighted_composite(scores, specs, {"A": 2, "B": 1}, min_groups=2)
        np.testing.assert_allclose(groups["A"], [3.5, 4, np.nan], equal_nan=True)
        np.testing.assert_allclose(composite, [5 / 3, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(breadth, [50, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(coverage, [100, 50, np.nan], equal_nan=True)

    def test_rolling_percentile_uses_prior_observations_only(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = rolling_percentile_previous(
            values,
            window=3,
            min_periods=3,
            scale=100.0,
        )

        self.assertTrue(result.iloc[:3].isna().all())
        self.assertEqual(result.iloc[3], 100.0)

    def test_grouped_composite_prevents_proxy_count_from_overweighting_group(
        self,
    ) -> None:
        scores = pd.DataFrame(
            {
                "credit_a": [3.0],
                "credit_b": [3.0],
                "credit_c": [3.0],
                "volatility": [-3.0],
            }
        )
        specs = [
            {"name": "credit_a", "category": "Credit", "weight": 1.0},
            {"name": "credit_b", "category": "Credit", "weight": 1.0},
            {"name": "credit_c", "category": "Credit", "weight": 1.0},
            {"name": "volatility", "category": "Volatility", "weight": 1.0},
        ]

        sleeves, composite, breadth, coverage = grouped_weighted_composite(
            scores,
            specs,
            min_groups=2,
        )

        self.assertEqual(sleeves.loc[0, "Credit"], 3.0)
        self.assertEqual(sleeves.loc[0, "Volatility"], -3.0)
        self.assertTrue(np.isclose(composite.iloc[0], 0.0))
        self.assertEqual(breadth.iloc[0], 50.0)
        self.assertEqual(coverage.iloc[0], 100.0)


if __name__ == "__main__":
    unittest.main()
