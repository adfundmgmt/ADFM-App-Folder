"""Input-date regressions for Hedge Timer; observations must remain genuine."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from adfm_core.hedge_timer_data import extract_close, load_hedge_inputs
from adfm_core.market_data import _cached_download


class HedgeTimerDataTests(unittest.TestCase):
    def test_failed_endpoint_recovery_retries_on_next_page_load(self):
        old = pd.DataFrame({("SPY", "Close"): [100.], ("HYG", "Close"): [80.]}, index=pd.to_datetime(["2026-10-01"]))
        fresh = pd.DataFrame({("SPY", "Close"): [100., 102.], ("HYG", "Close"): [80., 81.]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
        _cached_download.clear()
        try:
            with patch("yfinance.download", side_effect=[old, old, fresh]):
                _, _, first = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
                panel, _, second = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:01", tz="America/New_York"))
            self.assertEqual(first["Status"].tolist(), ["Lagging", "Lagging"])
            self.assertEqual(second["Status"].tolist(), ["Current", "Current"])
            self.assertEqual(panel.loc[pd.Timestamp("2026-10-02"), "HYG"], 81.)
        finally:
            _cached_download.clear()

    def test_all_lagging_inputs_retry_provider_instead_of_reusing_cached_panel(self):
        old = pd.DataFrame({("SPY", "Close"): [100.], ("HYG", "Close"): [80.]}, index=pd.to_datetime(["2026-10-01"]))
        fresh = pd.DataFrame({("SPY", "Close"): [100., 102.], ("HYG", "Close"): [80., 81.]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
        _cached_download.clear()
        try:
            with patch("yfinance.download", side_effect=[old, fresh]):
                panel, _, health = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
            self.assertEqual(health["Status"].tolist(), ["Current", "Current"])
            self.assertEqual(panel.loc[pd.Timestamp("2026-10-02"), "SPY"], 102.)
        finally:
            _cached_download.clear()

    def test_wholly_missing_symbol_recovers_its_history_for_rolling_signals(self):
        dates = pd.bdate_range(end="2026-10-02", periods=150)
        raw = pd.DataFrame({("SPY", "Close"): np.arange(150.) + 100.}, index=dates)
        full = pd.DataFrame({"Close": np.arange(150.) + 100.}, index=dates)

        def provider(tickers, **kwargs):
            if len(tickers) == 2:
                return raw
            return full.iloc[-63:] if kwargs.get("period") == "3mo" else full

        with patch("adfm_core.market_data.download_market_data", side_effect=provider):
            panel, _, _ = load_hedge_inputs(["SPY", "^VVIX"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
        self.assertEqual(panel["^VVIX"].notna().sum(), 150)

    def test_extractor_accepts_both_column_orders(self):
        dates = pd.to_datetime(["2026-10-01", "2026-10-02"])
        raw = pd.DataFrame({("SPY", "Close"): [100., 102.], ("HYG", "Close"): [80., np.nan]}, index=dates)
        for frame in (raw, raw.swaplevel(axis=1)):
            panel = extract_close(frame, ["SPY", "HYG"])
            self.assertEqual(panel["SPY"].tolist(), [100., 102.])
            self.assertTrue(pd.isna(panel.loc[dates[-1], "HYG"]))

    def test_failed_recovery_preserves_old_date_and_missing_endpoint(self):
        raw = pd.DataFrame({("SPY", "Close"): [100., 102.], ("HYG", "Close"): [80., np.nan]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
        with patch("adfm_core.market_data.download_market_data", side_effect=[raw, pd.DataFrame()]):
            panel, expected, health = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
        self.assertEqual(expected, pd.Timestamp("2026-10-02"))
        self.assertTrue(pd.isna(panel.loc[expected, "HYG"]))
        self.assertEqual(health.set_index("Input").loc["HYG"].to_dict(), {"Last observation": "2026-10-01", "Status": "Lagging"})

    def test_provisional_daily_bar_does_not_advance_signal_before_close(self):
        raw = pd.DataFrame({("SPY", "Close"): [100., 999.], ("HYG", "Close"): [80., 999.]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
        with patch("adfm_core.market_data.download_market_data", return_value=raw):
            panel, expected, health = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 12:00", tz="America/New_York"))
        self.assertEqual(expected, pd.Timestamp("2026-10-01"))
        self.assertEqual(panel["SPY"].tolist(), [100.])
        self.assertEqual(health["Status"].tolist(), ["Current", "Current"])

    def test_holiday_uses_previous_exchange_session(self):
        raw = pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2026-09-04"]))
        with patch("adfm_core.market_data.download_market_data", return_value=raw):
            _, expected, health = load_hedge_inputs(["SPY"], date(2020, 1, 1), now=pd.Timestamp("2026-09-07 18:00", tz="America/New_York"))
        self.assertEqual(expected, pd.Timestamp("2026-09-04"))
        self.assertEqual(health["Status"].tolist(), ["Current"])

    def test_same_date_last_good_cache_is_not_current_when_retry_fails(self):
        raw = pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2026-10-02"]))
        raw.attrs["market_data_health"] = {"SPY": {"status": "last_good"}}
        with patch("adfm_core.market_data.download_market_data", side_effect=[raw, pd.DataFrame()]):
            _, _, health = load_hedge_inputs(["SPY"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
        self.assertEqual(health["Status"].tolist(), ["Last-good cache"])

    def test_total_outage_returns_diagnostics_without_crashing(self):
        with patch("adfm_core.market_data.download_market_data", return_value=pd.DataFrame()):
            panel, _, health = load_hedge_inputs(["SPY", "HYG"], date(2020, 1, 1), now=pd.Timestamp("2026-10-02 18:00", tz="America/New_York"))
        self.assertTrue(panel.empty)
        self.assertEqual(health["Status"].tolist(), ["Missing", "Missing"])


if __name__ == "__main__":
    unittest.main()
