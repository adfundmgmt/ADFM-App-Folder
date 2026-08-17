"""Deterministic tests for SEC 13F discovery, preparation, and exposure math."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from adfm_core.sec_13f import (
    PreparedDataset,
    QuarterDataset,
    discover_quarter_datasets,
    prepare_dataset,
    rank_fund_exposure,
    search_security_candidates,
    select_effective_filing_components,
)


class FakeResponse:
    def __init__(self, payload: bytes, *, content_type: str = "text/html") -> None:
        self.payload = payload
        self.headers = {
            "content-length": str(len(payload)),
            "content-type": content_type,
        }
        self.text = payload.decode("utf-8") if content_type == "text/html" else ""

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses

    def get(self, url: str, **_: object) -> FakeResponse:
        return self.responses[url]


def archive_bytes() -> bytes:
    files = {
        "SUBMISSION.tsv": (
            "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
            "0000000001-26-000001\t15-MAY-2026\t13F-HR\t0000000001\t31-MAR-2026\n"
        ),
        "COVERPAGE.tsv": (
            "ACCESSION_NUMBER\tREPORTCALENDARORQUARTER\tISAMENDMENT\tAMENDMENTNO\tAMENDMENTTYPE\tFILINGMANAGER_NAME\n"
            "0000000001-26-000001\t31-MAR-2026\tN\t\t\tALPHA CAPITAL\n"
        ),
        "SUMMARYPAGE.tsv": (
            "ACCESSION_NUMBER\tTABLEENTRYTOTAL\tTABLEVALUETOTAL\n"
            "0000000001-26-000001\t2\t1000\n"
        ),
        "INFOTABLE.tsv": (
            "ACCESSION_NUMBER\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
            "0000000001-26-000001\tINTEL CORP\tCOM\t458140100\t100\t5000\tSH\t\n"
            "0000000001-26-000001\tOTHER CO\tCOM\t000000001\t900\t9000\tSH\t\n"
        ),
    }
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def prepared_fixture(root: Path) -> PreparedDataset:
    filings = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": "a-original",
                "FILING_DATE": pd.Timestamp("2026-05-01"),
                "SUBMISSIONTYPE": "13F-HR",
                "CIK": "0000000001",
                "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                "AMENDMENTNO": None,
                "AMENDMENTTYPE": None,
                "FILINGMANAGER_NAME": "ALPHA CAPITAL",
                "TABLEVALUETOTAL": 1000.0,
            },
            {
                "ACCESSION_NUMBER": "a-add-before",
                "FILING_DATE": pd.Timestamp("2026-05-04"),
                "SUBMISSIONTYPE": "13F-HR/A",
                "CIK": "0000000001",
                "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                "AMENDMENTNO": 1,
                "AMENDMENTTYPE": "NEW HOLDINGS",
                "FILINGMANAGER_NAME": "ALPHA CAPITAL",
                "TABLEVALUETOTAL": 100.0,
            },
            {
                "ACCESSION_NUMBER": "a-restatement",
                "FILING_DATE": pd.Timestamp("2026-05-06"),
                "SUBMISSIONTYPE": "13F-HR/A",
                "CIK": "0000000001",
                "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                "AMENDMENTNO": 2,
                "AMENDMENTTYPE": "RESTATEMENT",
                "FILINGMANAGER_NAME": "ALPHA CAPITAL",
                "TABLEVALUETOTAL": 900.0,
            },
            {
                "ACCESSION_NUMBER": "a-add-after",
                "FILING_DATE": pd.Timestamp("2026-05-08"),
                "SUBMISSIONTYPE": "13F-HR/A",
                "CIK": "0000000001",
                "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                "AMENDMENTNO": 3,
                "AMENDMENTTYPE": "NEW HOLDINGS",
                "FILINGMANAGER_NAME": "ALPHA CAPITAL",
                "TABLEVALUETOTAL": 50.0,
            },
            {
                "ACCESSION_NUMBER": "b-original",
                "FILING_DATE": pd.Timestamp("2026-05-10"),
                "SUBMISSIONTYPE": "13F-HR",
                "CIK": "0000000002",
                "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
                "REPORTCALENDARORQUARTER": pd.Timestamp("2026-03-31"),
                "AMENDMENTNO": None,
                "AMENDMENTTYPE": None,
                "FILINGMANAGER_NAME": "BETA PARTNERS",
                "TABLEVALUETOTAL": 100.0,
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": "a-restatement",
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "VALUE": 90.0,
                "SSHPRNAMT": 4500.0,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": None,
            },
            {
                "ACCESSION_NUMBER": "a-add-after",
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "VALUE": 10.0,
                "SSHPRNAMT": 500.0,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": None,
            },
            {
                "ACCESSION_NUMBER": "a-restatement",
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "VALUE": 40.0,
                "SSHPRNAMT": 2000.0,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": "CALL",
            },
            {
                "ACCESSION_NUMBER": "b-original",
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "VALUE": 30.0,
                "SSHPRNAMT": 1500.0,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": None,
            },
        ]
    )
    securities = pd.DataFrame(
        [
            {
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "PUTCALL": "",
                "ISSUER_NORMALIZED": "INTEL",
            },
            {
                "NAMEOFISSUER": "INTEL CORP",
                "TITLEOFCLASS": "COM",
                "CUSIP": "458140100",
                "PUTCALL": "CALL",
                "ISSUER_NORMALIZED": "INTEL",
            },
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
        prepared_at="2026-08-16T00:00:00+00:00",
        holdings_rows=len(holdings),
    )


class Sec13FTests(unittest.TestCase):
    def test_discovers_official_archive_links(self) -> None:
        html = b"""
        <table><tr><td><a href="/files/structureddata/data/form-13f-data-sets/test_form13f.zip">2026 March April May 13F</a></td><td>ZIP</td><td>94 MB</td></tr></table>
        """
        session = FakeSession(
            {
                "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets": FakeResponse(html)
            }
        )

        releases = discover_quarter_datasets(session=session)

        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0].label, "2026 March April May 13F")
        self.assertEqual(releases[0].size_label, "94 MB")

    def test_prepares_archive_and_resolves_intc(self) -> None:
        release = QuarterDataset(
            slug="test-release",
            label="Test release",
            url="https://www.sec.gov/test_form13f.zip",
        )
        session = FakeSession(
            {
                release.url: FakeResponse(
                    archive_bytes(), content_type="application/octet-stream"
                )
            }
        )
        ticker_directory = pd.DataFrame(
            [{"TICKER": "INTC", "COMPANY_NAME": "Intel Corp", "CIK": 50863}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepare_dataset(
                release,
                cache_root=temporary,
                session=session,
            )
            catalog = pd.read_parquet(prepared.securities_path)
            matches = search_security_candidates(catalog, ticker_directory, "INTC")
            ranking = rank_fund_exposure(
                prepared,
                ["458140100"],
                report_period="2026-03-31",
            )

        self.assertEqual(prepared.holdings_rows, 2)
        self.assertEqual(matches.iloc[0]["CUSIP"], "458140100")
        self.assertAlmostEqual(ranking.iloc[0]["PORTFOLIO_WEIGHT_PCT"], 10.0)

    def test_restatement_supersedes_prior_base_and_additions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepared_fixture(Path(temporary))
            filings = pd.read_parquet(prepared.filings_path)

            components = select_effective_filing_components(filings, "2026-03-31")

        alpha = components.loc[components["CIK"].eq("0000000001")]
        self.assertEqual(
            alpha["ACCESSION_NUMBER"].tolist(),
            ["a-restatement", "a-add-after"],
        )
        self.assertEqual(alpha["TABLEVALUETOTAL"].sum(), 950.0)

    def test_ranks_long_exposure_by_portfolio_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepared_fixture(Path(temporary))

            ranking = rank_fund_exposure(
                prepared,
                ["458140100"],
                report_period="2026-03-31",
                position_kind="Long holdings",
            )

        self.assertEqual(ranking["MANAGER"].tolist(), ["BETA PARTNERS", "ALPHA CAPITAL"])
        self.assertAlmostEqual(ranking.iloc[0]["PORTFOLIO_WEIGHT_PCT"], 30.0)
        self.assertAlmostEqual(ranking.iloc[1]["PORTFOLIO_WEIGHT_PCT"], 100 / 950 * 100)
        self.assertEqual(ranking.iloc[1]["POSITION_VALUE_USD"], 100_000.0)

    def test_separates_calls_from_long_holdings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepared_fixture(Path(temporary))

            calls = rank_fund_exposure(
                prepared,
                ["458140100"],
                report_period="2026-03-31",
                position_kind="Call options",
            )

        self.assertEqual(calls["MANAGER"].tolist(), ["ALPHA CAPITAL"])
        self.assertEqual(calls.iloc[0]["POSITION_VALUE_USD"], 40_000.0)


if __name__ == "__main__":
    unittest.main()
