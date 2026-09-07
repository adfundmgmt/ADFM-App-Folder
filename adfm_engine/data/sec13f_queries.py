from __future__ import annotations
import re
from typing import Sequence
import pandas as pd
from adfm_engine.data import sec13f as base
from adfm_engine.data.sec13f import *
from adfm_engine.analytics.sec13f import _value_multiplier,find_managers,rank_holdings,summarize_portfolio
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
    holdings = _holdings_for_components(prepared, components, report_period)
    return components, holdings

def _holdings_for_components(
    prepared: PreparedDataset,
    components: pd.DataFrame,
    report_period: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Load and normalize only the information tables owned by components."""

    if components.empty:
        return pd.DataFrame()
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
        return holdings

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
    return holdings

def search_manager_candidates(prepared,query,**kwargs):
    return find_managers(pd.read_parquet(prepared.filings_path),query,**kwargs)
def rank_fund_exposure(prepared,cusips,**kwargs):
    # Stream effective holdings; keep only the selected security in memory.
    # Every effective line still contributes to the same portfolio denominator.
    import pyarrow.parquet as parquet
    components=base.select_effective_filing_components(pd.read_parquet(prepared.filings_path),kwargs.get('report_period'))
    if components.empty:return pd.DataFrame()
    components=components.copy();components['CIK']=components['CIK'].astype(str).str.zfill(10)
    mapping=components[['ACCESSION_NUMBER','CIK','FILING_DATE']].drop_duplicates()
    mapping['ACCESSION_NUMBER']=mapping['ACCESSION_NUMBER'].astype(str)
    selected=[];totals=[];wanted={str(c).strip() for c in cusips}
    for batch in parquet.ParquetFile(prepared.holdings_path).iter_batches(batch_size=32768):
        frame=batch.to_pandas();frame['ACCESSION_NUMBER']=frame['ACCESSION_NUMBER'].astype(str)
        frame=frame.merge(mapping,on='ACCESSION_NUMBER',how='inner')
        if frame.empty:continue
        frame['VALUE']=pd.to_numeric(frame['VALUE'],errors='coerce')
        frame['SSHPRNAMT']=pd.to_numeric(frame['SSHPRNAMT'],errors='coerce')
        frame['VALUE_USD']=frame['VALUE']*_value_multiplier(kwargs.get('report_period'))
        totals.append(frame.groupby('CIK',as_index=False)['VALUE_USD'].sum(min_count=1))
        subset=frame.loc[frame['CUSIP'].astype(str).isin(wanted)]
        if not subset.empty:selected.append(subset)
    if not totals or not selected:return pd.DataFrame()
    portfolio_totals=pd.concat(totals,ignore_index=True).groupby('CIK',as_index=False)['VALUE_USD'].sum(min_count=1).rename(columns={'VALUE_USD':'PORTFOLIO_VALUE_USD'})
    return rank_holdings(components,pd.concat(selected,ignore_index=True),cusips,portfolio_totals=portfolio_totals,**kwargs)
def manager_portfolio(prepared,cik,report_period):
    filings=pd.read_parquet(prepared.filings_path)
    components=base.select_effective_filing_components(filings,report_period)
    if components.empty:return {},pd.DataFrame()
    components=components.copy()
    components['CIK']=components['CIK'].astype(str).str.zfill(10)
    components=components.loc[components['CIK'].eq(str(cik).zfill(10))]
    holdings=_holdings_for_components(prepared,components,report_period)
    return summarize_portfolio(filings,holdings,cik,report_period)
