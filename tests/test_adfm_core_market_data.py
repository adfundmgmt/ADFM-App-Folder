"""Regression tests for the non-network ADFM market-data primitives."""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from adfm_core.market_data import (
    adjusted_ohlcv,
    align_to_benchmark_calendar,
    benchmark_calendar,
    canonicalize_date_index,
    close_panel,
    drop_unfinished_daily_session,
    fetch_daily_ohlcv,
    percent_change,
    stale_session_count,
    unique_tickers,
)


def sample_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.arange(len(index), dtype=float) + 100.0
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Adj Close": close * .5, "Volume": 1_000.0 + np.arange(len(index)) * 100.0},
        index=index,
    )


class MarketDataPrimitiveTests(unittest.TestCase):
    def tearDown(self) -> None:
        fetch_daily_ohlcv.clear()

    def test_ticker_normalization_preserves_order(self) -> None:
        self.assertEqual(unique_tickers([" spy ", "SPY", "qqq", ""]), ("SPY", "QQQ"))

    def test_canonical_index_removes_timezone_and_duplicate_dates(self) -> None:
        index = pd.DatetimeIndex(["2026-01-02 00:00:00+00:00", "2026-01-02 00:00:00+00:00", "2026-01-05 00:00:00+00:00"])
        frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=index)
        result = canonicalize_date_index(frame)
        self.assertEqual(len(result), 2)
        self.assertIsNone(result.index.tz)
        self.assertEqual(result.loc["2026-01-02", "Close"], 2.0)

    def test_current_session_is_removed_before_cash_close(self) -> None:
        index = pd.DatetimeIndex(["2026-07-13", "2026-07-14"])
        frame = sample_ohlcv(index).iloc[:2]
        now = datetime(2026, 7, 14, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        result = drop_unfinished_daily_session(frame, now=now)
        self.assertEqual(result.index.max().date().isoformat(), "2026-07-13")

    def test_unsorted_duplicate_dates_keep_last_provider_observation(self) -> None:
        frame = pd.DataFrame({"Close": [20.0, 10.0, 21.0]},
                             index=pd.to_datetime(["2026-01-05", "2026-01-02", "2026-01-05"]))
        result = canonicalize_date_index(frame)
        self.assertEqual(result["Close"].tolist(), [10.0, 21.0])
        self.assertTrue(result.index.is_unique)

    def test_daily_dates_preserve_exchange_date_and_normalize_midnight(self) -> None:
        frame = pd.DataFrame({"Close": [10.0, 11.0]}, index=pd.DatetimeIndex(
            ["2026-01-05 00:00", "2026-01-06 16:00"], tz="Asia/Tokyo"))
        result = canonicalize_date_index(frame)
        self.assertEqual(result.index.tolist(), list(pd.to_datetime(["2026-01-05", "2026-01-06"])))

    def test_session_cutoff_converts_aware_clock_to_new_york(self) -> None:
        frame = sample_ohlcv(pd.to_datetime(["2026-07-13", "2026-07-14"]))
        now = datetime(2026, 7, 14, 19, 0, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(len(drop_unfinished_daily_session(frame, now=now)), 1)

    def test_alignment_does_not_fill_by_default(self) -> None:
        sessions = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"])
        frame = pd.DataFrame({"Close": [10.0, 11.0]}, index=[sessions[0], sessions[2]])
        aligned = align_to_benchmark_calendar(frame, sessions)
        self.assertTrue(pd.isna(aligned.loc[sessions[1], "Close"]))
        filled = align_to_benchmark_calendar(frame, sessions, forward_fill_limit=1)
        self.assertEqual(filled.loc[sessions[1], "Close"], 10.0)

    def test_staleness_uses_observed_benchmark_sessions(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        benchmark = sample_ohlcv(index)
        self.assertEqual(stale_session_count(benchmark.iloc[:-1], benchmark_calendar({"SPY": benchmark}, "SPY")), 1)

    def test_adjusted_ohlcv_scales_prices_and_preserves_provider_volume(self) -> None:
        frame = sample_ohlcv(pd.bdate_range("2026-01-02", periods=3))
        adjusted = adjusted_ohlcv(frame)
        self.assertAlmostEqual(adjusted["Open"].iloc[0], 50.0)
        pd.testing.assert_series_equal(adjusted["Volume"], frame["Volume"])

    def test_missing_adjusted_prices_are_not_replaced_with_raw_prices(self) -> None:
        frame = sample_ohlcv(pd.bdate_range("2026-01-02", periods=3))
        frame["Adj Close"] = np.nan
        self.assertTrue(adjusted_ohlcv(frame)["Close"].isna().all())

    def test_return_horizon_does_not_compress_missing_sessions(self) -> None:
        self.assertTrue(np.isnan(percent_change(pd.Series([100.0, np.nan, 110.0]), 1)))
        self.assertAlmostEqual(percent_change(pd.Series([100.0, np.nan, 110.0]), 2), .1)
        self.assertTrue(np.isnan(percent_change(pd.Series([100.0, 110.0, np.nan]), 1)))
        self.assertTrue(np.isnan(percent_change(pd.Series([0.0, 110.0]), 1)))

    def test_close_panel_and_returns_preserve_missing_data(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        frames = {"AAA": sample_ohlcv(index), "BBB": sample_ohlcv(index).assign(Close=[10.0, np.nan, 12.0])}
        panel = close_panel(frames, ["AAA", "BBB"])
        self.assertTrue(pd.isna(panel["BBB"].iloc[1]))
        self.assertAlmostEqual(percent_change(panel["AAA"], 2), .02)

    @patch("adfm_core.market_data.yf.download")
    def test_shared_loader_normalizes_a_daily_ohlcv_response(self, download_mock: object) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        raw = sample_ohlcv(index)
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAA"]])
        download_mock.return_value = raw  # type: ignore[attr-defined]

        frames, dropped = fetch_daily_ohlcv((" aaa ",), period="1y")

        self.assertEqual(list(frames), ["AAA"])
        self.assertEqual(frames["AAA"].index.tolist(), index.tolist())
        self.assertTrue(dropped.empty)
        self.assertEqual(download_mock.call_count, 1)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
