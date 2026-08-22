"""Tests for the equity leadership and rotation calculations."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.leadership import (
    build_leadership_frame,
    centered_cross_sectional_rank,
    classify_state,
    period_return,
    summarize_families,
)


class LeadershipTests(unittest.TestCase):
    def test_period_return_uses_completed_observations(self) -> None:
        series = pd.Series([100.0, 102.0, 104.0, 106.0])
        self.assertAlmostEqual(period_return(series, 2), 106.0 / 102.0 - 1.0)

    def test_centered_rank_is_symmetric(self) -> None:
        ranked = centered_cross_sectional_rank(pd.Series([1.0, 2.0, 3.0]))
        self.assertEqual(ranked.tolist(), [-100.0, 0.0, 100.0])

    def test_state_classification_covers_four_rotation_quadrants(self) -> None:
        self.assertEqual(classify_state(10.0, 5.0), "Leading")
        self.assertEqual(classify_state(10.0, -5.0), "Weakening")
        self.assertEqual(classify_state(-10.0, 5.0), "Improving")
        self.assertEqual(classify_state(-10.0, -5.0), "Lagging")
        self.assertEqual(classify_state(np.nan, 5.0), "Unavailable")

    def test_frame_ranks_stronger_ratio_above_weaker_ratio(self) -> None:
        index = pd.date_range("2025-01-01", periods=260, freq="B")
        ratios = {
            "strong": pd.Series(np.linspace(80.0, 140.0, len(index)), index=index),
            "flat": pd.Series(np.linspace(100.0, 101.0, len(index)), index=index),
            "weak": pd.Series(np.linspace(130.0, 75.0, len(index)), index=index),
        }
        metadata = pd.DataFrame(
            {
                "Family": ["Technology", "Technology", "Cyclicals"],
                "Relationship": ["Strong", "Flat", "Weak"],
                "Pair": ["A/B", "C/D", "E/F"],
                "Note": ["", "", ""],
            },
            index=["strong", "flat", "weak"],
        )

        frame = build_leadership_frame(ratios, metadata)

        self.assertEqual(frame.index[0], "strong")
        self.assertEqual(frame.index[-1], "weak")
        self.assertGreater(frame.loc["strong", "Leadership Score"], 0)
        self.assertLess(frame.loc["weak", "Leadership Score"], 0)
        self.assertEqual(frame.loc["strong", "Trend"], "Above 50D + 200D")

        families = summarize_families(frame)
        self.assertEqual(families.iloc[0]["Family"], "Technology")
        self.assertEqual(int(families["Relationships"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
