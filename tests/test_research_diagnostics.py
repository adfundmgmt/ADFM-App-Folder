from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.data_registry import market_symbols
from adfm_core.market_data import close_panel
from adfm_core.research_diagnostics import (
    build_snapshot_history,
    dashboard_route,
    evidence_weights,
    historical_regime_analogs,
    regime_feature_panel,
    signal_attribution,
    signal_performance,
)


def market_frames() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2023-01-02", periods=700)
    time = np.arange(len(index), dtype=float)
    frames = {}
    for position, symbol in enumerate(market_symbols()):
        values = 100.0 * np.exp((0.00020 + position * 0.00001) * time)
        if symbol == "^VIX":
            values = 30.0 * np.exp(-0.00020 * time)
        frames[symbol] = pd.DataFrame(
            {
                "Open": values,
                "High": values * 1.01,
                "Low": values * 0.99,
                "Close": values,
                "Adj Close": values,
                "Volume": 1_000_000,
            },
            index=index,
        )
    return frames


class ResearchDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = close_panel(market_frames(), market_symbols(), adjusted=True)
        cls.history = build_snapshot_history(
            cls.prices, frequency="ME", minimum_observations=190
        )

    def test_history_and_attribution_are_point_in_time_and_reconcile(self) -> None:
        self.assertFalse(self.history.empty)
        dates = sorted(self.history["Snapshot Date"].unique())
        current = self.history.loc[self.history["Snapshot Date"] == dates[-1]]
        previous = self.history.loc[self.history["Snapshot Date"] == dates[-2]]

        attribution = signal_attribution(current, previous)

        self.assertFalse(attribution.empty)
        reconciled = attribution.groupby("Key")["Current Contribution"].sum()
        composites = current.set_index("Key")["Composite"]
        for key in reconciled.index:
            self.assertAlmostEqual(reconciled[key], composites[key], places=10)
        self.assertTrue(attribution["Data Through"].notna().all())

    def test_signal_performance_and_evidence_weights_are_bounded(self) -> None:
        diagnostics = signal_performance(self.history, self.prices)
        weights = evidence_weights(diagnostics)

        self.assertFalse(diagnostics.empty)
        self.assertTrue(diagnostics["Hit Rate"].dropna().between(0, 1).all())
        self.assertTrue(weights["Proposed Weight"].between(0.50, 1.25).all())

    def test_regime_analogs_return_matches_and_distributions(self) -> None:
        index = self.prices.index
        macro = pd.DataFrame(
            {
                "dgs2": np.linspace(1.0, 4.0, len(index)),
                "dgs10": np.linspace(2.0, 5.0, len(index)),
                "t10yie": 2.0 + np.sin(np.arange(len(index)) / 40) * 0.2,
                "walcl": np.linspace(5_000, 8_000, len(index)),
                "tga": np.linspace(300, 600, len(index)),
                "rrp": np.linspace(100, 1_000, len(index)),
            },
            index=index,
        )
        features = regime_feature_panel(self.prices, macro)

        result = historical_regime_analogs(
            features, self.prices, matches=5, exclusion_sessions=63
        )

        self.assertEqual(len(result.matches), 5)
        self.assertFalse(result.distribution.empty)
        self.assertTrue(result.distribution["Positive Rate"].between(0, 1).all())

    def test_cross_dashboard_route_preserves_context(self) -> None:
        route = dashboard_route("credit_risk", lookback="5y")

        self.assertEqual(route["page"], "pages/13_Credit_Conditions_Monitor.py")
        self.assertEqual(route["instrument"], "HYG/LQD")
        self.assertEqual(route["lookback"], "5y")


if __name__ == "__main__":
    unittest.main()
