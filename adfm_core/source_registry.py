"""Institutional primary-source registry and adapter capability contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True)
class SourceSystem:
    """One official source system available to the research platform."""

    key: str
    institution: str
    coverage: str
    endpoint: str
    format: str
    authentication: str
    adapter_status: str
    revision_policy: str


SOURCE_SYSTEMS: Final[tuple[SourceSystem, ...]] = (
    SourceSystem(
        "treasury",
        "US Treasury",
        "Fiscal data, auctions, debt, cash balance, and yield curve",
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/",
        "JSON",
        "None",
        "Registered",
        "Use observation date and API record-date metadata.",
    ),
    SourceSystem(
        "bls",
        "US Bureau of Labor Statistics",
        "CPI, PPI, employment, wages, productivity, and labor flows",
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        "JSON",
        "Optional API key",
        "Registered",
        "Preserve release period; vintage history requires ALFRED or archive.",
    ),
    SourceSystem(
        "bea",
        "US Bureau of Economic Analysis",
        "GDP, income, consumption, profits, trade, and industry accounts",
        "https://apps.bea.gov/api/data/",
        "JSON",
        "API key",
        "Registered",
        "Store result vintage because national accounts are revised.",
    ),
    SourceSystem(
        "fed",
        "Federal Reserve",
        "Rates, balance sheet, liquidity, credit, industrial production, and FRED",
        "https://fred.stlouisfed.org/",
        "CSV/JSON",
        "None for FRED CSV",
        "Live",
        "Missing values remain missing; revised series require vintage capture.",
    ),
    SourceSystem(
        "ecb",
        "European Central Bank",
        "Euro-area rates, money, credit, inflation expectations, FX, and balance sheet",
        "https://data-api.ecb.europa.eu/service/data/",
        "SDMX-CSV",
        "None",
        "Registered",
        "Retain observation status and last-update fields.",
    ),
    SourceSystem(
        "boj",
        "Bank of Japan",
        "Policy, money, balance sheet, rates, inflation expectations, and Tankan",
        "https://www.stat-search.boj.or.jp/api/v1/",
        "JSON/CSV",
        "None",
        "Registered",
        "Keep release timestamps and preliminary/final flags.",
    ),
    SourceSystem(
        "boe",
        "Bank of England",
        "Bank Rate, yield curves, money, credit, inflation, and financial conditions",
        "https://www.bankofengland.co.uk/boeapps/database/",
        "CSV",
        "None",
        "Registered",
        "Retain series code, observation date, and retrieval timestamp.",
    ),
    SourceSystem(
        "cftc",
        "US Commodity Futures Trading Commission",
        "Commitments of Traders and market-participant positioning",
        "https://publicreporting.cftc.gov/resource/",
        "Socrata JSON/CSV",
        "None",
        "Registered",
        "Use report date; releases describe prior Tuesday positioning.",
    ),
    SourceSystem(
        "eia",
        "US Energy Information Administration",
        "Oil, gas, power, inventories, production, demand, and trade",
        "https://api.eia.gov/v2/",
        "JSON",
        "API key",
        "Registered",
        "Preserve period frequency and revision/retrieval timestamp.",
    ),
    SourceSystem(
        "sec",
        "US Securities and Exchange Commission",
        "Filings, submissions, XBRL company facts, ownership, and fund disclosures",
        "https://data.sec.gov/",
        "JSON/XBRL",
        "User-Agent required",
        "Registered",
        "Use filing acceptance time and accession number as immutable keys.",
    ),
)


def source_capability_table() -> pd.DataFrame:
    """Return the registry as a page-ready diagnostics table."""

    return pd.DataFrame(asdict(source) for source in SOURCE_SYSTEMS)


def source_by_key(key: str) -> SourceSystem | None:
    """Resolve one primary source by stable key."""

    return next((source for source in SOURCE_SYSTEMS if source.key == key), None)
