"""Tests for pure calculations used by the Rate of Change Regime Explorer."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.rate_of_change import (
    add_trading_session_axis,
    compute_features,
    make_date_ticks,
    padded_range,
)


class RateOfChangeFeatureTests(unittest.TestCase):
    def test_features_match_documented_calculations(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=30)
        frame = pd.DataFrame({"Close": np.arange(100.0, 130.0)}, index=index)

        features = compute_features(frame, roc_period=5)

        self.assertAlmostEqual(features.loc[index[-1], "SMA_21"], frame["Close"].iloc[-21:].mean())
        self.assertAlmostEqual(features.loc[index[-1], "ROC"], (129.0 / 124.0) - 1.0)
        self.assertAlmostEqual(
            features.loc[index[-1], "Second_Derivative"],
            features.loc[index[-1], "ROC"] - features.loc[index[-2], "ROC"],
        )
        self.assertEqual(features.loc[index[0], "SMA_200"], 100.0)

    def test_inflection_flags_follow_acceleration_zero_crossings(self) -> None:
        frame = pd.DataFrame({"Close": [100.0, 90.0, 95.0, 105.0, 110.0, 108.0, 100.0, 105.0]})
        features = compute_features(frame, roc_period=1)

        self.assertTrue(features["Pos_Inflect"].any())
        self.assertTrue(features["Neg_Inflect"].any())

    def test_features_require_close_and_positive_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "Close"):
            compute_features(pd.DataFrame({"Open": [100.0]}), roc_period=5)
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_features(pd.DataFrame({"Close": [100.0]}), roc_period=0)


class RateOfChangeChartHelperTests(unittest.TestCase):
    def test_padded_range_handles_empty_constant_and_zero_inclusion(self) -> None:
        self.assertIsNone(padded_range([pd.Series([np.nan, np.inf])]))
        self.assertEqual(padded_range([pd.Series([10.0, 10.0])]), [9.5, 10.5])
        self.assertEqual(padded_range([pd.Series([10.0, 20.0])], include_zero=True), [-1.0, 21.0])

    def test_trading_session_axis_and_date_ticks(self) -> None:
        index = pd.bdate_range("2025-01-02", periods=10)
        frame = pd.DataFrame({"Close": np.arange(10.0)}, index=index)
        sessions = add_trading_session_axis(frame)

        self.assertEqual(sessions["Session"].tolist(), list(range(10)))
        self.assertEqual(sessions["Date_Label"].iloc[0], "2025-01-02")
        tickvals, ticktext = make_date_ticks(sessions.index, sessions["Session"], max_ticks=4)
        self.assertEqual(tickvals[0], 0)
        self.assertEqual(tickvals[-1], 9)
        self.assertEqual(len(tickvals), len(ticktext))

    def test_tick_labels_adjust_to_long_history(self) -> None:
        medium_index = pd.bdate_range("2024-01-02", periods=300)
        long_index = pd.bdate_range("2022-01-03", periods=901)
        medium_sessions = pd.Series(range(len(medium_index)))
        long_sessions = pd.Series(range(len(long_index)))

        _, medium_labels = make_date_ticks(medium_index, medium_sessions)
        _, long_labels = make_date_ticks(long_index, long_sessions)

        self.assertTrue(all(" " in label for label in medium_labels))
        self.assertTrue(all(len(label) == 4 for label in long_labels))


if __name__ == "__main__":
    unittest.main()
