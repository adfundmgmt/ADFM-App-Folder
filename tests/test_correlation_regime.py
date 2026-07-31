"""Tests for the cross-asset correlation calculations."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.correlation_regime import (
    average_off_diagonal,
    conditional_pair_statistics,
    correlation_snapshot,
    current_beta_table,
    eigen_diagnostics,
    log_returns,
    pair_table,
    rolling_pair_metrics,
    window_correlation,
)


class CorrelationRegimeTests(unittest.TestCase):
    def test_log_returns_preserve_missing_observations(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, np.nan, 104.0],
                "B": [50.0, 51.0, 52.0, 53.0],
            },
            index=pd.bdate_range("2026-01-02", periods=4),
        )

        result = log_returns(prices)

        self.assertTrue(pd.isna(result.loc[result.index[2], "A"]))
        self.assertTrue(pd.isna(result.loc[result.index[3], "A"]))
        self.assertAlmostEqual(result["B"].iloc[1], np.log(51.0 / 50.0))

    def test_current_and_prior_windows_do_not_overlap(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=10)
        returns = pd.DataFrame(
            {
                "A": np.arange(1.0, 11.0),
                "B": np.r_[np.arange(1.0, 6.0), np.arange(6.0, 11.0)],
                "C": np.r_[np.arange(5.0, 0.0, -1.0), np.arange(6.0, 11.0)],
            },
            index=index,
        )

        current = window_correlation(returns, 5)
        prior = window_correlation(returns, 5, offset=5)

        self.assertAlmostEqual(current.loc["A", "C"], 1.0)
        self.assertAlmostEqual(prior.loc["A", "C"], -1.0)
        snapshot = correlation_snapshot(returns, 5)
        self.assertAlmostEqual(snapshot.average_correlation, 1.0)
        self.assertAlmostEqual(snapshot.prior_average_correlation, -1 / 3)

    def test_short_nearly_complete_window_is_accepted(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=250)
        returns = pd.DataFrame(
            {
                "A": np.sin(np.arange(250) / 9.0),
                "B": np.sin(np.arange(250) / 9.0 + 0.2),
            },
            index=index,
        )

        result = window_correlation(returns, 252)

        self.assertEqual(result.shape, (2, 2))

    def test_eigen_diagnostics_report_effective_independent_bets(self) -> None:
        matrix = pd.DataFrame(np.eye(3), columns=list("ABC"), index=list("ABC"))

        market_mode, effective_bets = eigen_diagnostics(matrix)

        self.assertAlmostEqual(market_mode, 1 / 3)
        self.assertAlmostEqual(effective_bets, 3.0)
        self.assertAlmostEqual(average_off_diagonal(matrix), 0.0)

    def test_pair_table_reconciles_current_and_prior_values(self) -> None:
        current = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], columns=["A", "B"], index=["A", "B"]
        )
        prior = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]], columns=["A", "B"], index=["A", "B"]
        )

        result = pair_table(current, prior, {"A": "Asset A", "B": "Asset B"})

        self.assertEqual(result.iloc[0]["Asset 1"], "Asset A")
        self.assertAlmostEqual(result.iloc[0]["Change"], 0.5)

    def test_pair_beta_and_conditioning_use_selected_benchmark(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=80)
        benchmark = pd.Series(
            np.sin(np.arange(80) / 5.0) / 100,
            index=index,
        )
        returns = pd.DataFrame(
            {"QQQ": benchmark, "TLT": 2.0 * benchmark},
            index=index,
        )

        rolling = rolling_pair_metrics(returns, "TLT", "QQQ", 21)
        conditional = conditional_pair_statistics(returns, "QQQ")
        term = current_beta_table(returns, "QQQ", windows=(21, 63))

        self.assertAlmostEqual(rolling["Beta"].dropna().iloc[-1], 2.0)
        self.assertIn("QQQ Up Sessions", conditional["Regime"].tolist())
        self.assertNotIn("SPY Up Sessions", conditional["Regime"].tolist())
        self.assertAlmostEqual(term.loc[0, "Beta 63D"], 2.0)


if __name__ == "__main__":
    unittest.main()
