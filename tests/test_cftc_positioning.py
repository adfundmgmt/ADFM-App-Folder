"""Deterministic tests for CFTC positioning normalization and crowding metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from adfm_core.cftc_positioning import (
    add_metrics,
    build_scanner,
    estimate_notional,
    percentile_rank,
    positioning_signal,
    price_proxy,
    rolling_metrics,
)


def sample_tff(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-06", periods=rows, freq="W-TUE")
    return pd.DataFrame(
        {
            "report_date": dates,
            "contract_code": "209742",
            "market_name": "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE",
            "commodity_name": "NASDAQ-100",
            "open_interest": 1_000.0,
            "asset_mgr_positions_long": np.arange(rows, dtype=float) + 200.0,
            "asset_mgr_positions_short": 100.0,
            "lev_money_positions_long": 50.0,
            "lev_money_positions_short": 120.0,
            "dealer_positions_long_all": 100.0,
            "dealer_positions_short_all": 90.0,
            "other_rept_positions_long": 75.0,
            "other_rept_positions_short": 70.0,
        }
    )


class CFTCPositioningTests(unittest.TestCase):
    def test_combined_financial_cohort_aggregates_long_and_short_legs(self) -> None:
        frame = sample_tff(rows=2)
        out = add_metrics(frame, "TFF", "Asset Managers + Leveraged Funds")

        self.assertEqual(float(out.iloc[-1]["cohort_long"]), 251.0)
        self.assertEqual(float(out.iloc[-1]["cohort_short"]), 220.0)
        self.assertEqual(float(out.iloc[-1]["net_contracts"]), 31.0)
        self.assertAlmostEqual(float(out.iloc[-1]["net_pct_oi"]), 0.031)

    def test_scanner_flags_a_trailing_record_long(self) -> None:
        scanner = build_scanner(
            sample_tff(rows=30),
            "TFF",
            "Asset Managers + Leveraged Funds",
            lookback_weeks=52,
        )
        latest = scanner.iloc[0]

        self.assertEqual(latest["signal"], "Extreme Long")
        self.assertEqual(latest["record"], "30W High")
        self.assertEqual(int(latest["history_weeks"]), 30)
        self.assertEqual(float(latest["one_week_change"]), 1.0)
        self.assertEqual(float(latest["four_week_change"]), 4.0)

    def test_rolling_metrics_create_normalized_history(self) -> None:
        history = rolling_metrics(
            sample_tff(rows=30),
            "TFF",
            "Asset Managers + Leveraged Funds",
            window_weeks=52,
        )
        self.assertTrue(np.isfinite(float(history.iloc[-1]["rolling_zscore"])))
        self.assertGreater(float(history.iloc[-1]["rolling_percentile"]), 97.0)

    def test_percentile_and_signal_thresholds(self) -> None:
        self.assertAlmostEqual(percentile_rank(pd.Series([1, 2, 3, 4])), 87.5)
        self.assertEqual(positioning_signal(2.5), "Extreme Short")
        self.assertEqual(positioning_signal(10.0), "Crowded Short")
        self.assertEqual(positioning_signal(50.0), "Neutral")
        self.assertEqual(positioning_signal(90.0), "Crowded Long")
        self.assertEqual(positioning_signal(97.5), "Extreme Long")

    def test_nasdaq_price_proxy_supports_notional_estimate(self) -> None:
        proxy = price_proxy("209742")
        self.assertIsNotNone(proxy)
        assert proxy is not None
        ticker, _, multiplier = proxy
        self.assertEqual(ticker, "NQ=F")
        self.assertEqual(multiplier, 20.0)
        notional = estimate_notional(
            pd.Series([10.0]), pd.Series([20_000.0]), multiplier
        )
        self.assertEqual(float(notional.iloc[0]), 4_000_000.0)


if __name__ == "__main__":
    unittest.main()
