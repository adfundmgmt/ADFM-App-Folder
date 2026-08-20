"""Regression tests for 13F value-unit reconciliation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adfm_core.sec_13f import PreparedDataset
from adfm_core.sec_13f_corrected import (
    manager_portfolio,
    rank_fund_exposure,
    search_manager_candidates,
)


class Corrected13FValueTests(unittest.TestCase):
    def prepared_fixture(self, root: Path) -> PreparedDataset:
        filings = pd.DataFrame(
            [
                {
                    "ACCESSION_NUMBER": "legacy-summary",
                    "FILING_DATE": pd.Timestamp("2026-04-20"),
                    "SUBMISSIONTYPE": "13F-HR",
                    "CIK": "0001537191",
                    "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                    "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                    "AMENDMENTNO": None,
                    "AMENDMENTTYPE": None,
                    "FILINGMANAGER_NAME": "LEGACY SUMMARY MANAGER",
                    "TABLEVALUETOTAL": 5_704_793.0,
                },
                {
                    "ACCESSION_NUMBER": "dollar-summary",
                    "FILING_DATE": pd.Timestamp("2026-05-15"),
                    "SUBMISSIONTYPE": "13F-HR",
                    "CIK": "0000000002",
                    "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                    "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                    "AMENDMENTNO": None,
                    "AMENDMENTTYPE": None,
                    "FILINGMANAGER_NAME": "DOLLAR SUMMARY MANAGER",
                    "TABLEVALUETOTAL": 11_410_312_995.0,
                },
            ]
        )
        holdings = pd.DataFrame(
            [
                {
                    "ACCESSION_NUMBER": "legacy-summary",
                    "NAMEOFISSUER": "TEST SECURITY",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": "123456789",
                    "VALUE": 25_039_362.0,
                    "SSHPRNAMT": 567_400.0,
                    "SSHPRNAMTTYPE": "SH",
                    "PUTCALL": None,
                },
                {
                    "ACCESSION_NUMBER": "legacy-summary",
                    "NAMEOFISSUER": "OTHER HOLDINGS",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": "000000001",
                    "VALUE": 5_679_753_638.0,
                    "SSHPRNAMT": 1.0,
                    "SSHPRNAMTTYPE": "SH",
                    "PUTCALL": None,
                },
                {
                    "ACCESSION_NUMBER": "dollar-summary",
                    "NAMEOFISSUER": "TEST SECURITY",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": "123456789",
                    "VALUE": 3_837_391_316.0,
                    "SSHPRNAMT": 86_956_522.0,
                    "SSHPRNAMTTYPE": "SH",
                    "PUTCALL": None,
                },
                {
                    "ACCESSION_NUMBER": "dollar-summary",
                    "NAMEOFISSUER": "OTHER HOLDINGS",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": "000000002",
                    "VALUE": 7_572_921_679.0,
                    "SSHPRNAMT": 1.0,
                    "SSHPRNAMTTYPE": "SH",
                    "PUTCALL": None,
                },
            ]
        )
        securities = pd.DataFrame(
            [
                {
                    "NAMEOFISSUER": "TEST SECURITY",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": "123456789",
                    "PUTCALL": "",
                    "ISSUER_NORMALIZED": "TEST SECURITY",
                }
            ]
        )
        filings_path = root / "filings.parquet"
        holdings_path = root / "holdings.parquet"
        securities_path = root / "securities.parquet"
        filings.to_parquet(filings_path, index=False)
        holdings.to_parquet(holdings_path, index=False)
        securities.to_parquet(securities_path, index=False)
        metadata_path = root / "metadata.json"
        metadata_path.write_text("{}", encoding="utf-8")
        return PreparedDataset(
            slug="fixture",
            label="Fixture",
            source_url="https://www.sec.gov/fixture.zip",
            cache_dir=root,
            filings_path=filings_path,
            holdings_path=holdings_path,
            securities_path=securities_path,
            metadata_path=metadata_path,
            prepared_at="2026-08-17T00:00:00+00:00",
            holdings_rows=len(holdings),
        )

    def test_weights_reconcile_when_summary_units_are_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepared_fixture(Path(temporary))
            ranking = rank_fund_exposure(
                prepared,
                ["123456789"],
                report_period="2026-03-31",
                minimum_portfolio_millions=0,
            )

        legacy = ranking.loc[ranking["CIK"].eq("0001537191")].iloc[0]
        dollar = ranking.loc[ranking["CIK"].eq("0000000002")].iloc[0]
        self.assertAlmostEqual(legacy["PORTFOLIO_VALUE_USD"], 5_704_793_000.0)
        self.assertAlmostEqual(legacy["POSITION_VALUE_USD"], 25_039_362.0)
        self.assertAlmostEqual(
            legacy["PORTFOLIO_WEIGHT_PCT"], 25_039_362 / 5_704_793_000 * 100
        )
        self.assertAlmostEqual(dollar["PORTFOLIO_VALUE_USD"], 11_410_312_995.0)
        self.assertAlmostEqual(dollar["PORTFOLIO_WEIGHT_PCT"], 33.63, places=2)
        self.assertTrue((ranking["PORTFOLIO_WEIGHT_PCT"] <= 100.0).all())

    def test_manager_portfolio_sums_to_one_hundred_percent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepared_fixture(Path(temporary))
            summary, portfolio = manager_portfolio(
                prepared, "0001537191", "2026-03-31"
            )

        self.assertAlmostEqual(summary["PORTFOLIO_VALUE_USD"], 5_704_793_000.0)
        self.assertAlmostEqual(portfolio["PORTFOLIO_WEIGHT_PCT"].sum(), 100.0)

    def test_manager_search_resolves_name_and_exact_cik(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self.prepared_fixture(Path(temporary))
            by_name = search_manager_candidates(
                prepared,
                "Legacy Summary Manager",
                report_period="2026-03-31",
            )
            by_cik = search_manager_candidates(
                prepared,
                "Legacy Summary Manager (CIK: 0001537191)",
                report_period="2026-03-31",
            )

        self.assertEqual(by_name.iloc[0]["CIK"], "0001537191")
        self.assertEqual(by_cik.iloc[0]["CIK"], "0001537191")
        self.assertEqual(by_cik.iloc[0]["MANAGER"], "LEGACY SUMMARY MANAGER")


if __name__ == "__main__":
    unittest.main()
