"""Regression tests for the non-network ADFM market-data primitives."""

from __future__ import annotations

from datetime import datetime
import unittest
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

    def test_adjusted_ohlcv_scales_prices_and_volume(self) -> None:
        frame = sample_ohlcv(pd.bdate_range("2026-01-02", periods=3))
        adjusted = adjusted_ohlcv(frame)
        self.assertAlmostEqual(adjusted["Open"].iloc[0], 50.0)
        self.assertAlmostEqual(adjusted["Volume"].iloc[0], 2_000.0)

    def test_close_panel_and_returns_preserve_missing_data(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        frames = {"AAA": sample_ohlcv(index), "BBB": sample_ohlcv(index).assign(Close=[10.0, np.nan, 12.0])}
        panel = close_panel(frames, ["AAA", "BBB"])
        self.assertTrue(pd.isna(panel["BBB"].iloc[1]))
        self.assertAlmostEqual(percent_change(panel["AAA"], 2), .02)


if __name__ == "__main__":
    unittest.main()
