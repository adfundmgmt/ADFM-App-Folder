"""Deterministic tests for CFTC positioning normalization and crowding metrics."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from adfm_core.cftc_positioning import (
    _request,
    add_metrics,
    build_scanner,
    estimate_notional,
    fetch_contract_history,
    fetch_recent,
    infer_asset_class,
    normalize,
    percentile_rank,
    positioning_signal,
    price_proxy,
    rolling_metrics,
    zscore_latest,
)


def sample_tff(rows: int = 30, code: str = "209742") -> pd.DataFrame:
    dates = pd.date_range("2026-01-06", periods=rows, freq="W-TUE")
    return pd.DataFrame(
        {
            "report_date": dates,
            "contract_code": code,
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


def raw_tff() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_and_exchange_names": ["NASDAQ-100 Consolidated - CME"],
            "contract_market_name": ["NASDAQ-100"],
            "commodity_name": ["NASDAQ-100"],
            "report_date_as_yyyy_mm_dd": ["2026-08-11T00:00:00.000"],
            "cftc_contract_market_code": ["209742"],
            "open_interest_all": ["1000"],
            "asset_mgr_positions_long": ["300"],
            "asset_mgr_positions_short": ["200"],
            "lev_money_positions_long": ["100"],
            "lev_money_positions_short": ["250"],
            "dealer_positions_long_all": ["400"],
            "dealer_positions_short_all": ["300"],
            "other_rept_positions_long": ["80"],
            "other_rept_positions_short": ["75"],
        }
    )


class CFTCPositioningTests(unittest.TestCase):
    def test_normalize_converts_public_api_payload_to_numeric_history(self) -> None:
        out = normalize(raw_tff(), "TFF")

        self.assertEqual(list(out["contract_code"]), ["209742"])
        self.assertEqual(float(out.iloc[0]["open_interest"]), 1000.0)
        self.assertEqual(float(out.iloc[0]["asset_mgr_positions_long"]), 300.0)
        self.assertEqual(out.iloc[0]["report_date"], pd.Timestamp("2026-08-11"))

    def test_normalize_backfills_optional_missing_columns(self) -> None:
        raw = raw_tff().drop(columns=["commodity_name", "other_rept_positions_long"])
        out = normalize(raw, "TFF")

        self.assertEqual(str(out.iloc[0]["commodity_name"]), "")
        self.assertTrue(pd.isna(out.iloc[0]["other_rept_positions_long"]))
        self.assertTrue(normalize(pd.DataFrame(), "TFF").empty)

    def test_combined_financial_cohort_aggregates_long_and_short_legs(self) -> None:
        frame = sample_tff(rows=2)
        out = add_metrics(frame, "TFF", "Asset Managers + Leveraged Funds")

        self.assertEqual(float(out.iloc[-1]["cohort_long"]), 251.0)
        self.assertEqual(float(out.iloc[-1]["cohort_short"]), 220.0)
        self.assertEqual(float(out.iloc[-1]["net_contracts"]), 31.0)
        self.assertAlmostEqual(float(out.iloc[-1]["net_pct_oi"]), 0.031)

    def test_disaggregated_managed_money_math(self) -> None:
        frame = pd.DataFrame(
            {
                "open_interest": [2_000.0],
                "m_money_positions_long_all": [500.0],
                "m_money_positions_short_all": [800.0],
            }
        )
        out = add_metrics(frame, "Disaggregated", "Managed Money")

        self.assertEqual(float(out.iloc[0]["net_contracts"]), -300.0)
        self.assertAlmostEqual(float(out.iloc[0]["net_pct_oi"]), -0.15)

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
        self.assertEqual(latest["asset_class"], "Equity / Vol")

    def test_scanner_drops_discontinued_contracts(self) -> None:
        current = sample_tff(rows=30, code="209742")
        stale = sample_tff(rows=26, code="OLD001").copy()
        stale["report_date"] = stale["report_date"] - pd.Timedelta(days=90)
        scanner = build_scanner(
            pd.concat([current, stale], ignore_index=True),
            "TFF",
            "Asset Managers + Leveraged Funds",
            lookback_weeks=52,
            max_stale_days=21,
        )

        self.assertEqual(set(scanner["contract_code"]), {"209742"})
        self.assertTrue(build_scanner(pd.DataFrame(), "TFF", "Asset Managers").empty)

    def test_rolling_metrics_create_normalized_history(self) -> None:
        history = rolling_metrics(
            sample_tff(rows=30),
            "TFF",
            "Asset Managers + Leveraged Funds",
            window_weeks=52,
        )
        self.assertTrue(np.isfinite(float(history.iloc[-1]["rolling_zscore"])))
        self.assertGreater(float(history.iloc[-1]["rolling_percentile"]), 97.0)

    def test_percentile_zscore_and_signal_thresholds(self) -> None:
        self.assertAlmostEqual(percentile_rank(pd.Series([1, 2, 3, 4])), 87.5)
        self.assertTrue(np.isnan(percentile_rank(pd.Series(dtype=float))))
        self.assertTrue(np.isnan(zscore_latest(pd.Series([1.0, 2.0]))))
        self.assertTrue(np.isnan(zscore_latest(pd.Series([2.0] * 8))))
        self.assertEqual(positioning_signal(np.nan), "N/A")
        self.assertEqual(positioning_signal(2.5), "Extreme Short")
        self.assertEqual(positioning_signal(10.0), "Crowded Short")
        self.assertEqual(positioning_signal(50.0), "Neutral")
        self.assertEqual(positioning_signal(90.0), "Crowded Long")
        self.assertEqual(positioning_signal(97.5), "Extreme Long")

    def test_asset_class_inference_covers_major_cot_groups(self) -> None:
        cases = [
            ("TFF", "CME BITCOIN", "", "Crypto"),
            ("TFF", "NASDAQ-100", "", "Equity / Vol"),
            ("TFF", "10-YEAR U.S. TREASURY NOTES", "", "Rates"),
            ("TFF", "JAPANESE YEN", "", "FX"),
            ("TFF", "OTHER FINANCIAL CONTRACT", "", "Financial"),
            ("Disaggregated", "NYMEX CRUDE OIL", "", "Energy"),
            ("Disaggregated", "COMEX GOLD", "", "Metals"),
            ("Disaggregated", "CBOT CORN", "", "Grains / Oilseeds"),
            ("Disaggregated", "ICE COCOA", "", "Softs"),
            ("Disaggregated", "LIVE CATTLE", "", "Livestock / Dairy"),
            ("Disaggregated", "OTHER", "", "Physical Commodity"),
        ]
        for report_type, market, commodity, expected in cases:
            with self.subTest(market=market):
                self.assertEqual(
                    infer_asset_class(report_type, market, commodity), expected
                )

    @patch("adfm_core.cftc_positioning._request")
    def test_fetch_recent_builds_a_bounded_public_reporting_query(self, request: Mock) -> None:
        request.return_value = raw_tff()
        out = fetch_recent("TFF", years=3, timeout=12)

        self.assertEqual(len(out), 1)
        report_type, params, timeout = request.call_args.args
        self.assertEqual(report_type, "TFF")
        self.assertEqual(timeout, 12)
        self.assertEqual(params["$limit"], 50000)
        self.assertIn("report_date_as_yyyy_mm_dd >=", params["$where"])
        self.assertIn("asset_mgr_positions_long", params["$select"])

    @patch("adfm_core.cftc_positioning._request")
    def test_fetch_contract_history_sanitizes_code_and_sorts(self, request: Mock) -> None:
        request.return_value = raw_tff()
        out = fetch_contract_history("TFF", "209'742", timeout=7)

        self.assertEqual(len(out), 1)
        report_type, params, timeout = request.call_args.args
        self.assertEqual(report_type, "TFF")
        self.assertEqual(timeout, 7)
        self.assertIn("209''742", params["$where"])
        self.assertEqual(params["$order"], "report_date_as_yyyy_mm_dd ASC")

    @patch("adfm_core.cftc_positioning.requests.get")
    def test_request_uses_public_reporting_endpoint(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = [{"ok": "1"}]
        get.return_value = response

        out = _request("TFF", {"$limit": 1}, timeout=5)

        self.assertEqual(out.iloc[0]["ok"], "1")
        response.raise_for_status.assert_called_once_with()
        args, kwargs = get.call_args
        self.assertEqual(args[0], "https://publicreporting.cftc.gov/resource/gpe5-46if.json")
        self.assertEqual(kwargs["timeout"], 5)
        self.assertEqual(kwargs["params"]["$limit"], 1)

    @patch("adfm_core.cftc_positioning.requests.get")
    def test_request_rejects_non_list_payload(self, get: Mock) -> None:
        response = Mock()
        response.json.return_value = {"error": "bad payload"}
        get.return_value = response

        with self.assertRaises(ValueError):
            _request("TFF", {"$limit": 1})

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
        missing = estimate_notional(pd.Series([10.0]), pd.Series([20_000.0]), None)
        self.assertTrue(missing.isna().all())
        self.assertIsNone(price_proxy("UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
