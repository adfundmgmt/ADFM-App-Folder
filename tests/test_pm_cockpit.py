from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.data_registry import market_symbols
from adfm_core.pm_cockpit import (
    build_signal_snapshot,
    causal_percentile_score,
    group_scores,
    largest_changes,
    summarize_snapshot,
)


def synthetic_prices() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=700)
    t = np.arange(len(index), dtype=float)
    data = {}
    for position, symbol in enumerate(market_symbols()):
        drift = 0.00025 + position * 0.000015
        cycle = 0.025 * np.sin(t / (21.0 + position))
        values = 100.0 * np.exp(drift * t + cycle)
        data[symbol] = values
    data["^VIX"] = 28.0 * np.exp(-0.00035 * t + 0.08 * np.sin(t / 18.0))
    return pd.DataFrame(data, index=index)


class PmCockpitTests(unittest.TestCase):
    def test_percentile_score_is_bounded_and_requires_prior_history(self):
        short = pd.Series(np.arange(30.0))
        self.assertTrue(np.isnan(causal_percentile_score(short, 21)))

        score = causal_percentile_score(
            synthetic_prices()["SPY"], 21, history_window=504, min_history=126
        )
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_snapshot_and_summary_cover_every_registered_proxy(self):
        snapshot = build_signal_snapshot(synthetic_prices())
        summary = summarize_snapshot(snapshot)

        self.assertEqual(summary.total_signals, 12)
        self.assertEqual(summary.available_signals, 12)
        self.assertTrue(-1.0 <= summary.composite <= 1.0)
        self.assertTrue(0.0 <= summary.confidence <= 1.0)
        self.assertIsNotNone(summary.as_of)

    def test_group_and_change_views_preserve_signal_counts(self):
        snapshot = build_signal_snapshot(synthetic_prices())
        groups = group_scores(snapshot)
        movers = largest_changes(snapshot, limit=2)

        self.assertEqual(int(groups["Signals"].sum()), len(snapshot))
        self.assertLessEqual(len(movers["improving"]), 2)
        self.assertLessEqual(len(movers["deteriorating"]), 2)


if __name__ == "__main__":
    unittest.main()
