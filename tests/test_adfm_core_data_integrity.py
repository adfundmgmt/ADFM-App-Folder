"""Tests for the shared ADFM raw-data eligibility contract."""

from __future__ import annotations

import unittest

import pandas as pd

from adfm_core.data_integrity import DataIntegrityPolicy, build_data_quality_report, report_caption


def valid_frame(periods: int = 5) -> pd.DataFrame:
    index = pd.bdate_range("2026-01-02", periods=periods)
    close = pd.Series(range(100, 100 + periods), index=index, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000.0}, index=index)


class DataIntegrityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DataIntegrityPolicy(min_valid_sessions=4, max_stale_sessions=1)

    def test_valid_frame_is_eligible_and_reports_benchmark_date(self) -> None:
        frames = {"SPY": valid_frame(), "AAA": valid_frame()}
        report = build_data_quality_report(frames, "SPY", policy=self.policy)
        self.assertTrue(report.benchmark_ready)
        self.assertEqual(set(report.eligible_tickers), {"SPY", "AAA"})
        self.assertIn("Data through: 2026-01-08", report_caption(report))

    def test_thin_history_is_excluded(self) -> None:
        report = build_data_quality_report({"SPY": valid_frame(), "AAA": valid_frame(3)}, "SPY", policy=self.policy)
        self.assertEqual(report.reason_for("AAA"), "Thin history (<4 complete sessions)")

    def test_stale_series_is_excluded_against_benchmark_sessions(self) -> None:
        benchmark = valid_frame(6)
        report = build_data_quality_report({"SPY": benchmark, "AAA": benchmark.iloc[:-2]}, "SPY", policy=self.policy)
        self.assertEqual(report.reason_for("AAA"), "Stale (>1 benchmark sessions)")

    def test_invalid_range_and_missing_required_column_are_excluded(self) -> None:
        bad_range = valid_frame()
        bad_range.loc[bad_range.index[-1], "High"] = bad_range.loc[bad_range.index[-1], "Low"] - 1
        missing_volume = valid_frame().drop(columns="Volume")
        report = build_data_quality_report({"SPY": valid_frame(), "RANGE": bad_range, "MISSING": missing_volume}, "SPY", policy=self.policy)
        self.assertEqual(report.reason_for("RANGE"), "Invalid high/low range")
        self.assertEqual(report.reason_for("MISSING"), "Missing required fields: Volume")

    def test_contract_never_fills_missing_observations(self) -> None:
        incomplete = valid_frame()
        incomplete.loc[incomplete.index[2], "Close"] = pd.NA
        report = build_data_quality_report({"SPY": valid_frame(), "AAA": incomplete}, "SPY", policy=self.policy)
        row = report.diagnostics.set_index("Ticker").loc["AAA"]
        self.assertEqual(row["Valid Sessions"], 4)
        self.assertEqual(row["Missing Benchmark Sessions"], 1)


if __name__ == "__main__":
    unittest.main()
