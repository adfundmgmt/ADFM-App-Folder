from __future__ import annotations

import unittest

import pandas as pd

from adfm_core.sec_fundamentals import (
    annual_cagr,
    build_valuation_snapshot,
    extract_metric,
    extract_metrics,
    filing_index_url,
    ltm_value,
    maturity_table,
    recent_filings,
    resolve_company,
)


def duration_fact(
    value: float,
    start: str,
    end: str,
    *,
    accession: str,
    fiscal_period: str,
    form: str = "10-Q",
) -> dict[str, object]:
    return {
        "val": value,
        "start": start,
        "end": end,
        "filed": "2026-05-01",
        "form": form,
        "accn": accession,
        "fy": 2025,
        "fp": fiscal_period,
    }


def instant_fact(
    value: float,
    end: str = "2025-12-31",
    *,
    accession: str = "0000000001-26-000001",
) -> dict[str, object]:
    return {
        "val": value,
        "end": end,
        "filed": "2026-02-15",
        "form": "10-K",
        "accn": accession,
        "fy": 2025,
        "fp": "FY",
    }


def company_facts_payload() -> dict[str, object]:
    quarter_ends = ("2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31")
    quarter_starts = ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01")

    def quarterly(
        values: tuple[float, ...], concept: str, unit: str = "USD"
    ) -> dict[str, object]:
        return {
            "label": concept,
            "units": {
                unit: [
                    duration_fact(
                        value,
                        start,
                        end,
                        accession=f"0000000001-26-00000{index}",
                        fiscal_period=("Q1", "Q2", "Q3", "FY")[index - 1],
                        form="10-K" if index == 4 else "10-Q",
                    )
                    for index, (value, start, end) in enumerate(
                        zip(values, quarter_starts, quarter_ends, strict=True), start=1
                    )
                ]
            },
        }

    return {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [instant_fact(100.0)]}
                }
            },
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": quarterly(
                    (100.0, 100.0, 100.0, 100.0), "Revenue"
                ),
                "GrossProfit": quarterly(
                    (40.0, 40.0, 40.0, 40.0), "Gross Profit"
                ),
                "OperatingIncomeLoss": quarterly(
                    (20.0, 20.0, 20.0, 20.0), "Operating Income"
                ),
                "NetIncomeLoss": quarterly((10.0, 10.0, 10.0, 10.0), "Net Income"),
                "NetCashProvidedByUsedInOperatingActivities": quarterly(
                    (15.0, 15.0, 15.0, 15.0), "CFO"
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": quarterly(
                    (5.0, 5.0, 5.0, 5.0), "Capex"
                ),
                "DepreciationDepletionAndAmortization": quarterly(
                    (5.0, 5.0, 5.0, 5.0), "D&A"
                ),
                "InterestExpenseNonOperating": quarterly(
                    (5.0, 5.0, 5.0, 5.0), "Interest"
                ),
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": quarterly(
                    (12.5, 12.5, 12.5, 12.5), "Pre-Tax Income"
                ),
                "IncomeTaxExpenseBenefit": quarterly(
                    (2.5, 2.5, 2.5, 2.5), "Income Tax"
                ),
                "PaymentsOfDividendsCommonStock": quarterly(
                    (1.0, 1.0, 1.0, 1.0), "Dividends"
                ),
                "EarningsPerShareDiluted": quarterly(
                    (0.1, 0.1, 0.1, 0.1), "Diluted EPS", "USD/shares"
                ),
                "WeightedAverageNumberOfDilutedSharesOutstanding": quarterly(
                    (100.0, 100.0, 100.0, 100.0), "Diluted Shares", "shares"
                ),
                "CashAndCashEquivalentsAtCarryingValue": {
                    "units": {"USD": [instant_fact(100.0)]}
                },
                "AccountsReceivableNetCurrent": {
                    "units": {"USD": [instant_fact(50.0)]}
                },
                "AssetsCurrent": {"units": {"USD": [instant_fact(400.0)]}},
                "LiabilitiesCurrent": {"units": {"USD": [instant_fact(200.0)]}},
                "Assets": {
                    "units": {
                        "USD": [
                            instant_fact(900.0, end="2024-12-31", accession="0000000001-25-000001"),
                            instant_fact(1_000.0),
                        ]
                    }
                },
                "LongTermDebt": {"units": {"USD": [instant_fact(300.0)]}},
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            instant_fact(400.0, end="2024-12-31", accession="0000000001-25-000001"),
                            instant_fact(500.0),
                        ]
                    }
                },
            },
        }
    }


class SecFundamentalsTests(unittest.TestCase):
    def test_resolves_ticker_cik_and_unambiguous_company_name(self) -> None:
        payload = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        }

        self.assertEqual(resolve_company("aapl", payload).cik, 320193)
        self.assertEqual(resolve_company("0000320193", payload).ticker, "AAPL")
        self.assertEqual(resolve_company("Microsoft Corp", payload).ticker, "MSFT")

    def test_reconstructs_cash_flow_quarters_from_ytd_disclosures(self) -> None:
        payload = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                duration_fact(30.0, "2025-01-01", "2025-03-31", accession="0000000001-25-000001", fiscal_period="Q1"),
                                duration_fact(70.0, "2025-01-01", "2025-06-30", accession="0000000001-25-000002", fiscal_period="Q2"),
                                duration_fact(120.0, "2025-01-01", "2025-09-30", accession="0000000001-25-000003", fiscal_period="Q3"),
                                duration_fact(170.0, "2025-01-01", "2025-12-31", accession="0000000001-26-000001", fiscal_period="FY", form="10-K"),
                            ]
                        }
                    }
                }
            }
        }

        metric = extract_metric(payload, "cfo")

        self.assertEqual([item.value for item in metric.quarterly], [30.0, 40.0, 50.0, 50.0])
        self.assertEqual([item.derived for item in metric.quarterly], [False, True, True, True])
        self.assertEqual(ltm_value(metric), 170.0)

    def test_builds_current_equity_and_credit_ratios(self) -> None:
        metrics = extract_metrics(company_facts_payload())

        snapshot = build_valuation_snapshot(
            metrics,
            price=10.0,
            price_date=pd.Timestamp("2026-08-11"),
        )

        self.assertEqual(snapshot.market_cap, 1_000.0)
        self.assertEqual(snapshot.enterprise_value, 1_200.0)
        self.assertEqual(snapshot.ltm_revenue, 400.0)
        self.assertEqual(snapshot.ltm_ebitda, 100.0)
        self.assertEqual(snapshot.ltm_fcf, 40.0)
        self.assertAlmostEqual(snapshot.pe or 0.0, 25.0)
        self.assertAlmostEqual(snapshot.ev_revenue or 0.0, 3.0)
        self.assertAlmostEqual(snapshot.ev_ebitda or 0.0, 12.0)
        self.assertAlmostEqual(snapshot.fcf_yield or 0.0, 0.04)
        self.assertAlmostEqual(snapshot.price_sales or 0.0, 2.5)
        self.assertAlmostEqual(snapshot.price_book or 0.0, 2.0)
        self.assertAlmostEqual(snapshot.price_cash or 0.0, 10.0)
        self.assertAlmostEqual(snapshot.price_fcf or 0.0, 25.0)
        self.assertAlmostEqual(snapshot.gross_margin or 0.0, 0.4)
        self.assertAlmostEqual(snapshot.profit_margin or 0.0, 0.1)
        self.assertAlmostEqual(snapshot.current_ratio or 0.0, 2.0)
        self.assertAlmostEqual(snapshot.quick_ratio or 0.0, 0.75)
        self.assertAlmostEqual(snapshot.debt_equity or 0.0, 0.6)
        self.assertAlmostEqual(snapshot.roa or 0.0, 40.0 / 950.0)
        self.assertAlmostEqual(snapshot.roe or 0.0, 40.0 / 450.0)
        self.assertAlmostEqual(snapshot.roic or 0.0, 64.0 / 700.0)
        self.assertAlmostEqual(snapshot.dividend_yield or 0.0, 0.004)
        self.assertAlmostEqual(snapshot.payout_ratio or 0.0, 0.1)
        self.assertAlmostEqual(snapshot.eps or 0.0, 0.4)
        self.assertAlmostEqual(snapshot.sales_per_share or 0.0, 4.0)
        self.assertAlmostEqual(snapshot.book_per_share or 0.0, 5.0)
        self.assertAlmostEqual(snapshot.cash_per_share or 0.0, 1.0)
        self.assertAlmostEqual(snapshot.net_debt_ebitda or 0.0, 2.0)
        self.assertAlmostEqual(snapshot.interest_coverage or 0.0, 5.0)

    def test_calculates_filing_based_annual_cagr(self) -> None:
        payload = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                duration_fact(100.0, "2020-01-01", "2020-12-31", accession="0000000001-21-000001", fiscal_period="FY", form="10-K"),
                                duration_fact(121.0, "2022-01-01", "2022-12-31", accession="0000000001-23-000001", fiscal_period="FY", form="10-K"),
                                duration_fact(161.05, "2025-01-01", "2025-12-31", accession="0000000001-26-000001", fiscal_period="FY", form="10-K"),
                            ]
                        }
                    }
                }
            }
        }

        metric = extract_metric(payload, "revenue")

        self.assertAlmostEqual(annual_cagr(metric, 3) or 0.0, 0.10, places=2)
        self.assertAlmostEqual(annual_cagr(metric, 5) or 0.0, 0.10, places=2)

    def test_extracts_standardized_maturity_buckets(self) -> None:
        payload = {
            "facts": {
                "us-gaap": {
                    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": {
                        "units": {"USD": [instant_fact(125.0)]}
                    },
                    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": {
                        "units": {"USD": [instant_fact(250.0)]}
                    },
                }
            }
        }

        result = maturity_table(payload)

        self.assertEqual(result["Maturity Bucket"].tolist(), ["Within 1 year", "Year 2"])
        self.assertEqual(result["Principal"].tolist(), [125.0, 250.0])

    def test_builds_recent_filing_links(self) -> None:
        submissions = {
            "cik": 320193,
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-26-000001"],
                    "filingDate": ["2026-05-01"],
                    "reportDate": ["2026-03-31"],
                    "form": ["10-Q"],
                    "primaryDocument": ["aapl-20260331.htm"],
                    "primaryDocDescription": ["Quarterly report"],
                }
            },
        }

        result = recent_filings(submissions)

        self.assertEqual(result.loc[0, "Form"], "10-Q")
        self.assertIn("/320193/000032019326000001/aapl-20260331.htm", result.loc[0, "Document"])
        self.assertEqual(
            filing_index_url("0000320193-26-000001"),
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/0000320193-26-000001-index.html",
        )


if __name__ == "__main__":
    unittest.main()
