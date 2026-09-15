"""Behavior tests for the Hedge Timer warning and calibration model."""

from __future__ import annotations

import unittest

import pandas as pd

from adfm_core.hedge_timer_model import (
    find_drawdown_episodes,
    fresh_short_onset,
    select_full_recall_candidate,
    warning_summary,
    watch_signal,
)


class HedgeTimerModelTests(unittest.TestCase):
    def test_watch_signal_is_not_suppressed_by_oversold_or_late_gates(self) -> None:
        idx = pd.date_range("2026-01-02", periods=4, freq="B")
        score = pd.Series([45.0, 72.0, 78.0, 82.0], index=idx)
        meta = {
            "early_stage": pd.Series([True, True, False, False], index=idx),
            "oversold_block": pd.Series([False, False, True, True], index=idx),
        }

        watch = watch_signal(score, threshold=70)
        short_on = fresh_short_onset(score, meta, threshold=70)

        self.assertEqual(watch.tolist(), [False, True, True, True])
        self.assertEqual(short_on.tolist(), [False, True, False, False])

    def test_full_recall_is_a_hard_constraint_before_precision(self) -> None:
        candidates = [
            {
                "threshold": 75,
                "spx_coverage": 1.0,
                "ndx_coverage": 0.8,
                "median_lead": 15.0,
                "false_warning_rate": 0.01,
            },
            {
                "threshold": 68,
                "spx_coverage": 1.0,
                "ndx_coverage": 1.0,
                "median_lead": 9.0,
                "false_warning_rate": 0.08,
            },
            {
                "threshold": 62,
                "spx_coverage": 1.0,
                "ndx_coverage": 1.0,
                "median_lead": 14.0,
                "false_warning_rate": 0.12,
            },
        ]

        chosen = select_full_recall_candidate(candidates)

        self.assertEqual(chosen["threshold"], 62)
        self.assertEqual(chosen["spx_coverage"], 1.0)
        self.assertEqual(chosen["ndx_coverage"], 1.0)

    def test_full_recall_selector_falls_back_to_highest_coverage_when_impossible(self) -> None:
        candidates = [
            {
                "threshold": 80,
                "spx_coverage": 0.75,
                "ndx_coverage": 0.75,
                "median_lead": 18.0,
                "false_warning_rate": 0.01,
            },
            {
                "threshold": 70,
                "spx_coverage": 1.0,
                "ndx_coverage": 0.9,
                "median_lead": 10.0,
                "false_warning_rate": 0.05,
            },
        ]

        chosen = select_full_recall_candidate(candidates)

        self.assertEqual(chosen["threshold"], 70)

    def test_drawdown_ledger_resets_to_a_new_local_peak(self) -> None:
        idx = pd.date_range("2020-01-02", periods=150, freq="B")
        prices = pd.Series(100.0, index=idx)
        prices.iloc[20:31] = [100, 98, 96, 94, 92, 90, 88, 89, 91, 94, 97]
        prices.iloc[31:105] = 97.0
        prices.iloc[105:116] = [97, 95, 93, 91, 89, 87, 85, 87, 90, 94, 97]
        prices.iloc[116:] = 97.0

        episodes = find_drawdown_episodes(prices)

        self.assertEqual(len(episodes), 2)
        self.assertLessEqual(episodes[0][3], -0.10)
        self.assertLessEqual(episodes[1][3], -0.10)

    def test_warning_summary_counts_captured_and_false_warnings(self) -> None:
        idx = pd.date_range("2020-01-02", periods=80, freq="B")
        prices = pd.Series(100.0, index=idx)
        prices.iloc[30:46] = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 89, 88, 87, 88, 89, 90]
        prices.iloc[46:] = 100.0
        warnings = pd.Series(False, index=idx)
        warnings.iloc[30] = True
        warnings.iloc[60] = True

        summary = warning_summary(prices, warnings, lookback=15)

        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["captured"], 1)
        self.assertEqual(summary["warnings"], 2)
        self.assertEqual(summary["false_warnings"], 1)
        self.assertEqual(summary["median_lead"], 10.0)


if __name__ == "__main__":
    unittest.main()
