from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.position_sizing import (
    HORIZON_TRADING_DAYS,
    bootstrap_portfolio_paths,
    calculate_sizing,
    conviction_ceiling,
    first_touch_statistics,
    historical_windows,
)


class PositionSizingTests(unittest.TestCase):
    def test_conviction_mapping(self) -> None:
        actual_values = [conviction_ceiling(i) for i in range(1, 6)]
        expected_values = [0.05, 0.10, 0.15, 0.20, 0.25]
        for actual, expected in zip(actual_values, expected_values, strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(HORIZON_TRADING_DAYS["1 month"], 21)
        self.assertEqual(HORIZON_TRADING_DAYS["5 years"], 1260)

    def test_sizing_uses_lowest_cap(self) -> None:
        result = calculate_sizing(
            conviction=5,
            max_nav_loss=0.01,
            stop_distance=0.10,
            current_volatility=0.40,
            historical_median_volatility=0.20,
            event_move=0.05,
            tail_move=0.08,
            portfolio_nav=5_000_000,
            median_dollar_volume=100_000_000,
            hold_through_earnings=True,
        )
        self.assertAlmostEqual(result.conviction_ceiling, 0.25)
        self.assertAlmostEqual(result.volatility_cap, 0.125)
        self.assertAlmostEqual(result.invalidation_cap, 0.10)
        self.assertAlmostEqual(result.suggested_size, 0.10)
        self.assertEqual(result.binding_constraint, "Invalidation loss budget")

    def test_historical_windows_long_and_short(self) -> None:
        close = pd.Series(
            [100, 105, 110, 100],
            index=pd.date_range("2026-01-01", periods=4),
        )
        long_frame = historical_windows(close, 2, "long", step=1)
        short_frame = historical_windows(close, 2, "short", step=1)
        self.assertAlmostEqual(float(long_frame.iloc[0]["return"]), 0.10)
        self.assertAlmostEqual(float(short_frame.iloc[0]["return"]), -0.10)

    def test_first_touch(self) -> None:
        frame = pd.DataFrame(
            {
                "Close": [100, 102, 104, 106, 108],
                "High": [101, 103, 105, 107, 109],
                "Low": [99, 101, 103, 105, 107],
            },
            index=pd.date_range("2026-01-01", periods=5),
        )
        stats = first_touch_statistics(frame, 3, "long", 0.05, 0.05, step=1)
        self.assertGreater(stats["target_first"], 0)
        self.assertEqual(stats["stop_first"], 0)

    def test_bootstrap_shapes(self) -> None:
        returns = pd.Series(np.linspace(-0.02, 0.02, 200))
        wealth, endings, drawdowns = bootstrap_portfolio_paths(
            returns,
            direction="long",
            position_size=0.10,
            horizon_days=63,
            n_paths=100,
            block_size=5,
            seed=1,
        )
        self.assertEqual(wealth.shape, (100, 63))
        self.assertEqual(endings.shape, (100,))
        self.assertEqual(drawdowns.shape, (100,))


if __name__ == "__main__":
    unittest.main()
