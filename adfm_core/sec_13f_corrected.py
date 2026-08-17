"""Corrected Form 13F value and portfolio-weight calculations.

SEC Form 13F information-table values changed from thousands of dollars to
nearest-dollar reporting for filings submitted on or after January 3, 2023.
Some filers still submit summary-page totals using the legacy thousands
convention.  These helpers therefore calculate portfolio denominators directly
from the effective information-table holdings rather than trusting the summary
TABLEVALUETOTAL field.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from adfm_core import sec_13f as base

PreparedDataset = base.PreparedDataset
QuarterDataset = base.QuarterDataset
Sec13FError = base.Sec13FError


def discover_quarter_datasets(*args, **kwargs):
    return base.discover_quarter_datasets(*args, **kwargs)


def prepare_dataset(*args, **kwargs):
    return base.prepare_dataset(*args, **kwargs)


def load_company_tickers(*args, **kwargs):
    return base.load_company_tickers(*args, **kwargs)


def load_security_catalog(*args, **kwargs):
    return base.load_security_catalog(*args, **kwargs)


def search_security_candidates(*args, **kwargs):
    return base.search_security_candidates(*args, **kwargs)


def available_report_periods(*args, **kwargs):
    return base.available_report_periods(*args, **kwargs)


def select_effective_filing_components(*args, **kwargs):
    return base.select_effective_filing_components(*args, **kwargs)


def filing_url(*args, **kwargs):
    return base.filing_url(*args, **kwargs)


def _value_multiplier(report_period: str | pd.Timestamp | None) -> float:
    """Convert the raw SEC information-table VALUE field to US dollars."""

    if report_period is None:
        return 1.0
    period = pd.Timestamp(report_period)
    # EDGAR Release 22.4.1 changed Form 13F VALUE from $000s to dollars.
    return 1_000.0 if period < pd.Timestamp("2023-01-03") else 1.0


def _position_kind_mask(holdings: pd.DataFrame, position_kind: str) -> pd.Series:
    put_call = holdings["PUTCALL"].fillna("").astype(str).str.upper().str.strip()
    if position_kind == "Long holdings":
        return put_call.eq("")
    if position_kind == "Call options":
        return put_call.eq("CALL")
    if position_kind == "Put options":
        return put_call.eq("PUT")
    return pd.Series(True, index=holdings.index)


def _effective_holdings(
    prepared: PreparedDataset,
    report_period: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return effective filing components and their complete information tables."""

    filings = pd.read_parquet(prepared.filings_path)
    components = base.select_effective_filing_components(filings, report_period)
    if components.empty:
        return components, pd.DataFrame()

    components = components.copy()
    components["CIK"] = components["CIK"].astype(str).str.zfill(10)
    accessions = components["ACCESSION_NUMBER"].astype(str).drop_duplicates().tolist()
    try:
        holdings = pd.read_parquet(
            prepared.holdings_path,
            filters=[("ACCESSION_NUMBER", "in", accessions)],
        )
    except (TypeError, ValueError):
        holdings = pd.read_parquet(prepared.holdings_path)
        holdings = holdings.loc[
            holdings["ACCESSION_NUMBER"].astype(str).isin(accessions)
        ].copy()
    if holdings.empty:
        return components, holdings

    accession_map = components[
        ["ACCESSION_NUMBER", "CIK", "FILING_DATE"]
    ].drop_duplicates()
    accession_map["ACCESSION_NUMBER"] = accession_map["ACCESSION_NUMBER"].astype(str)
    holdings = holdings.copy()
    holdings["ACCESSION_NUMBER"] = holdings["ACCESSION_NUMBER"].astype(str)
    holdings = holdings.merge(accession_map, on="ACCESSION_NUMBER", how="inner")
    holdings["VALUE"] = pd.to_numeric(holdings["VALUE"], errors="coerce")
    holdings["SSHPRNAMT"] = pd.to_numeric(holdings["SSHPRNAMT"], errors="coerce")
    holdings["VALUE_USD"] = holdings["VALUE"] * _value_multiplier(report_period)
    return components, holdings


def rank_fund_exposure(
    prepared: PreparedDataset,
    cusips: Sequence[str],
    *,
    report_period: str | pd.Timestamp | None = None,
    position_kind: str = "Long holdings",
    minimum_portfolio_millions: float = 0.0,
) -> pd.DataFrame:
    """Rank holders using a self-reconciling information-table denominator.

    The denominator is the sum of every effective information-table VALUE line
    for the manager.  This prevents legacy-style summary totals from producing
    impossible portfolio weights above 100%.
    """

    selected_cusips = sorted(
        {str(value).strip() for value in cusips if str(value).strip()}
    )
    if not selected_cusips:
        return pd.DataFrame()

    components, all_holdings = _effective_holdings(prepared, report_period)
    if components.empty or all_holdings.empty:
        return pd.DataFrame()

    totals = (
        all_holdings.groupby("CIK", as_index=False)["VALUE_USD"]
        .sum(min_count=1)
        .rename(columns={"VALUE_USD": "PORTFOLIO_VALUE_USD"})
    )
    totals = totals.loc[totals["PORTFOLIO_VALUE_USD"].gt(0)].copy()

    manager_rows: list[dict[str, object]] = []
    for cik, group in components.groupby("CIK", sort=False):
        base_rows = group.loc[group["COMPONENT_ROLE"].eq("Base")]
        base_row = base_rows.iloc[-1] if not base_rows.empty else group.iloc[-1]
        manager_rows.append(
            {
                "CIK": str(cik).zfill(10),
                "MANAGER": str(base_row.get("FILINGMANAGER_NAME", "")).strip()
                or str(cik).zfill(10),
                "REPORT_PERIOD": pd.Timestamp(base_row["PERIODOFREPORT"]),
                "LATEST_FILING_DATE": pd.to_datetime(
                    group["FILING_DATE"], errors="coerce"
                ).max(),
                "COMPONENT_COUNT": len(group),
                "PRIMARY_ACCESSION_NUMBER": str(
                    base_row.get("PRIMARY_ACCESSION_NUMBER", base_row["ACCESSION_NUMBER"])
                ),
            }
        )
    managers = pd.DataFrame(manager_rows).merge(totals, on="CIK", how="inner")
    managers = managers.loc[
        managers["PORTFOLIO_VALUE_USD"].ge(
            max(0.0, float(minimum_portfolio_millions)) * 1_000_000.0
        )
    ]
    if managers.empty:
        return pd.DataFrame()

    positions = all_holdings.loc[
        all_holdings["CUSIP"].astype(str).isin(selected_cusips)
        & _position_kind_mask(all_holdings, position_kind)
    ].copy()
    if positions.empty:
        return pd.DataFrame()

    positions["SHARES_ONLY"] = positions["SSHPRNAMT"].where(
        positions["SSHPRNAMTTYPE"].fillna("").astype(str).str.upper().eq("SH")
    )
    positions = positions.sort_values(["CIK", "FILING_DATE", "ACCESSION_NUMBER"])
    positions = (
        positions.groupby("CIK", as_index=False, sort=False)
        .agg(
            POSITION_VALUE_USD=("VALUE_USD", "sum"),
            REPORTED_SHARES=("SHARES_ONLY", lambda values: values.sum(min_count=1)),
            POSITION_LINES=("ACCESSION_NUMBER", "size"),
            POSITION_ACCESSION_NUMBER=("ACCESSION_NUMBER", "last"),
        )
    )

    ranking = managers.merge(positions, on="CIK", how="inner")
    if ranking.empty:
        return ranking
    ranking["PORTFOLIO_WEIGHT_PCT"] = (
        ranking["POSITION_VALUE_USD"] / ranking["PORTFOLIO_VALUE_USD"] * 100.0
    )
    ranking["FILING_URL"] = ranking.apply(
        lambda row: base.filing_url(row["CIK"], row["POSITION_ACCESSION_NUMBER"]),
        axis=1,
    )
    ranking = ranking.loc[
        ranking["PORTFOLIO_WEIGHT_PCT"].between(0.0, 100.000001, inclusive="both")
    ].copy()
    ranking = ranking.sort_values(
        ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "MANAGER"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "RANK", range(1, len(ranking) + 1))
    return ranking


def manager_portfolio(
    prepared: PreparedDataset,
    cik: str,
    report_period: str | pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Reconstruct a manager portfolio from effective information-table lines."""

    components, holdings = _effective_holdings(prepared, report_period)
    if components.empty or holdings.empty:
        return {}, pd.DataFrame()

    target_cik = str(cik).strip().zfill(10)
    manager_components = components.loc[components["CIK"].eq(target_cik)].copy()
    manager_holdings = holdings.loc[holdings["CIK"].eq(target_cik)].copy()
    if manager_components.empty or manager_holdings.empty:
        return {}, pd.DataFrame()

    total_usd = float(manager_holdings["VALUE_USD"].sum())
    if total_usd <= 0:
        return {}, pd.DataFrame()

    manager_holdings["PUTCALL"] = (
        manager_holdings["PUTCALL"].fillna("").astype(str).str.upper().str.strip()
    )
    manager_holdings["TITLEOFCLASS"] = (
        manager_holdings["TITLEOFCLASS"].fillna("").astype(str)
    )
    manager_holdings["SSHPRNAMTTYPE"] = (
        manager_holdings["SSHPRNAMTTYPE"].fillna("").astype(str)
    )
    manager_holdings = manager_holdings.sort_values(
        ["FILING_DATE", "ACCESSION_NUMBER"]
    )
    portfolio = (
        manager_holdings.groupby(
            ["NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "PUTCALL", "SSHPRNAMTTYPE"],
            as_index=False,
            dropna=False,
        )
        .agg(
            POSITION_VALUE_USD=("VALUE_USD", "sum"),
            REPORTED_AMOUNT=("SSHPRNAMT", lambda values: values.sum(min_count=1)),
            SOURCE_ACCESSION_NUMBER=("ACCESSION_NUMBER", "last"),
            SOURCE_FILING_DATE=("FILING_DATE", "max"),
            LINES=("ACCESSION_NUMBER", "size"),
        )
    )
    portfolio["PORTFOLIO_WEIGHT_PCT"] = (
        portfolio["POSITION_VALUE_USD"] / total_usd * 100.0
    )
    portfolio["POSITION_TYPE"] = portfolio["PUTCALL"].replace(
        {"": "Long", "CALL": "Call", "PUT": "Put"}
    )
    portfolio["FILING_URL"] = portfolio["SOURCE_ACCESSION_NUMBER"].map(
        lambda accession: base.filing_url(target_cik, accession)
    )
    portfolio = portfolio.sort_values(
        ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "NAMEOFISSUER"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    portfolio.insert(0, "RANK", range(1, len(portfolio) + 1))

    base_rows = manager_components.loc[manager_components["COMPONENT_ROLE"].eq("Base")]
    base_row = base_rows.iloc[-1] if not base_rows.empty else manager_components.iloc[-1]
    summary = {
        "CIK": target_cik,
        "MANAGER": str(base_row.get("FILINGMANAGER_NAME", "")).strip() or target_cik,
        "REPORT_PERIOD": pd.Timestamp(base_row["PERIODOFREPORT"]),
        "LATEST_FILING_DATE": pd.to_datetime(
            manager_components["FILING_DATE"], errors="coerce"
        ).max(),
        "PORTFOLIO_VALUE_USD": total_usd,
        "POSITION_COUNT": len(portfolio),
        "TOP_TEN_PCT": float(portfolio.head(10)["PORTFOLIO_WEIGHT_PCT"].sum()),
        "COMPONENT_COUNT": len(manager_components),
        "FILER_URL": (
            "https://www.sec.gov/edgar/browse/?CIK="
            f"{target_cik}&owner=exclude&action=getcompany&type=13F-HR"
        ),
    }
    return summary, portfolio
