from __future__ import annotations
import pandas as pd
from adfm_engine.data.sec13f import QuarterDataset
TITLE = "SEC 13F Exposure Browser"

SEARCH_MODES = ("Security", "Manager")

DEFAULT_MANAGER_QUERY = "Duquesne Family Office LLC (CIK: 0001536411)"

POSITION_KINDS = ("Long holdings", "Call options", "Put options", "All reported")

SORT_OPTIONS = {
    "Portfolio weight": "PORTFOLIO_WEIGHT_PCT",
    "Reported market value": "POSITION_VALUE_USD",
    "Reported shares": "REPORTED_SHARES",
}

DETAIL_COLUMN_LABELS = {
    "PORTFOLIO_WEIGHT_PCT": "Portfolio weight",
    "POSITION_VALUE_USD": "Position value",
    "REPORTED_SHARES": "Reported shares",
    "PORTFOLIO_VALUE_USD": "13F portfolio",
    "LATEST_FILING_DATE": "Latest filing",
    "CIK": "Manager CIK",
    "COMPONENT_COUNT": "Filing components",
    "FILING_URL": "EDGAR filing",
}

DEFAULT_DETAIL_COLUMNS = [
    "PORTFOLIO_WEIGHT_PCT",
    "POSITION_VALUE_USD",
    "REPORTED_SHARES",
    "PORTFOLIO_VALUE_USD",
    "LATEST_FILING_DATE",
    "FILING_URL",
]

OFFICIAL_RELEASE_FALLBACKS = (
    QuarterDataset(
        slug="01mar2026-31may2026_form13f",
        label="2026 March April May 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2026-31may2026_form13f.zip",
        size_label="94.81 MB",
    ),
    QuarterDataset(
        slug="01dec2025-28feb2026_form13f",
        label="2025 December 2026 January February 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2025-28feb2026_form13f.zip",
        size_label="86.08 MB",
    ),
    QuarterDataset(
        slug="01sep2025-30nov2025_form13f",
        label="2025 September October November 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01sep2025-30nov2025_form13f.zip",
        size_label="81.65 MB",
    ),
    QuarterDataset(
        slug="01jun2025-31aug2025_form13f",
        label="2025 June July August 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01jun2025-31aug2025_form13f.zip",
        size_label="82.3 MB",
    ),
    QuarterDataset(
        slug="01mar2025-31may2025_form13f",
        label="2025 March April May 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2025-31may2025_form13f.zip",
        size_label="84.18 MB",
    ),
    QuarterDataset(
        slug="01dec2024-28feb2025_form13f",
        label="2024 December 2025 January February 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2024-28feb2025_form13f.zip",
        size_label="82.53 MB",
    ),
)

def money_label(value: float) -> str:
    if not pd.notna(value):
        return "N/A"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"

def candidate_label(row: pd.Series) -> str:
    put_call = row.get("PUTCALL", "")
    instrument = "Long" if pd.isna(put_call) or not str(put_call).strip() else str(put_call)
    return (
        f"{row['NAMEOFISSUER']} | {row['TITLEOFCLASS']} | "
        f"CUSIP {row['CUSIP']} | {instrument.title()}"
    )

def manager_candidate_label(row: pd.Series) -> str:
    filing_date = pd.Timestamp(row["LATEST_FILING_DATE"])
    return f"{row['MANAGER']} | CIK {row['CIK']} | Filed {filing_date:%b. %d, %Y}"

