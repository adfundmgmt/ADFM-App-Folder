"""13F screening and manager dossiers using official filed holdings."""
from dataclasses import asdict
import re
import pandas as pd
from adfm_engine.data.sec13f import *
from adfm_engine.data.sec13f_queries import search_manager_candidates,rank_fund_exposure,manager_portfolio
from adfm_engine.analytics.sec13f_controls import *
from adfm_engine.charts.sec13f import exposure_chart
from adfm_engine.cache import ttl_cache
from adfm_engine.serialization import records,figure_json
from adfm_engine.services import DataUnavailable

@ttl_cache(seconds=21600)
def releases():
    try:return discover_quarter_datasets() or list(OFFICIAL_RELEASE_FALLBACKS)
    except Sec13FError:return list(OFFICIAL_RELEASE_FALLBACKS)

@ttl_cache(seconds=86400)
def ticker_directory():return load_company_tickers()

def release_list():return [asdict(r) for r in releases()]

def profile_result(summary,portfolio,period,portfolio_filter='',portfolio_kind='All'):
    if not summary or portfolio.empty:raise DataUnavailable('The selected manager portfolio could not be reconstructed.')
    filtered=portfolio.copy()
    if portfolio_filter.strip():
        mask=pd.Series(False,index=filtered.index)
        for col in ['NAMEOFISSUER','CUSIP','TITLEOFCLASS']:mask|=filtered[col].astype(str).str.contains(portfolio_filter.strip(),case=False,na=False,regex=False)
        filtered=filtered.loc[mask]
    if portfolio_kind!='All':filtered=filtered.loc[filtered['POSITION_TYPE'].eq(portfolio_kind)]
    cols=['RANK','NAMEOFISSUER','TITLEOFCLASS','POSITION_TYPE','CUSIP','PORTFOLIO_WEIGHT_PCT','POSITION_VALUE_USD','REPORTED_AMOUNT','SSHPRNAMTTYPE','SOURCE_FILING_DATE','FILING_URL']
    return {'profile':records(pd.DataFrame([summary]))[0],'profile_facts':[['CIK',summary['CIK']],['13F portfolio',money_label(float(summary['PORTFOLIO_VALUE_USD']))],['Positions',f"{int(summary['POSITION_COUNT']):,}"],['Top 10 concentration',f"{float(summary['TOP_TEN_PCT']):.1f}%"]],
      'portfolio':records(filtered[cols].head(750)),'portfolio_types':['All',*sorted(portfolio['POSITION_TYPE'].dropna().unique())],
      'csv':filtered.to_csv(index=False),'filename':f"sec_13f_manager_{re.sub(r'[^A-Za-z0-9_-]+','_',str(summary['MANAGER']))}_{period}.csv"}

def screen_result(ranking,selected,sort_label,top_n,manager_filter,detail_columns,query,period):
    if ranking.empty:return {'message':'No effective 13F filings met the current filters.','holdings':[],'cards':[]}
    ranked=ranking.sort_values(SORT_OPTIONS[sort_label],ascending=False).reset_index(drop=True);ranked['RANK']=range(1,len(ranked)+1)
    highest=ranked.iloc[0]
    filtered=ranked.loc[ranked['MANAGER'].str.contains(manager_filter.strip(),case=False,na=False,regex=False)] if manager_filter.strip() else ranked
    optional=DEFAULT_DETAIL_COLUMNS if detail_columns is None else list(dict.fromkeys(detail_columns))
    return {'selected_security':records(pd.DataFrame([selected]))[0],
      'cards':[['Managers reporting',f'{len(ranked):,}','After active filters'],['Highest allocation',f"{highest['PORTFOLIO_WEIGHT_PCT']:.2f}%",str(highest['MANAGER'])],['Largest position',money_label(ranked['POSITION_VALUE_USD'].max()),'Reported market value'],['Aggregate reported value',money_label(ranked['POSITION_VALUE_USD'].sum()),'Across matching managers'],['Median allocation',f"{ranked['PORTFOLIO_WEIGHT_PCT'].median():.2f}%",'Across matching managers']],
      'figure':figure_json(exposure_chart(ranked,sort_label,top_n)), 'holdings':records(filtered[['RANK','MANAGER',*optional]].head(500)),
      'managers':records(filtered[['RANK','MANAGER','CIK']].head(500)),'count':len(filtered),'total':len(ranked),
      'csv':filtered.to_csv(index=False),'filename':f"sec_13f_exposure_{re.sub(r'[^A-Za-z0-9_-]+','_',query)}_{period}.csv"}

def load_sec13f(search_mode='Security',query='INTC',release_slug='',position_kind='Long holdings',minimum_portfolio_millions=1000.,sort_label='Portfolio weight',top_n=25,candidate=0,manager_cik='',manager_filter='',detail_columns=None,portfolio_filter='',portfolio_kind='All'):
    try:
        options=releases();release=next((r for r in options if r.slug==release_slug),None) if release_slug else options[0]
        if release is None:raise DataUnavailable('Select a current SEC data release.')
        prepared=prepare_dataset(release);periods=available_report_periods(prepared)
        if not periods:raise DataUnavailable('The selected release has no usable report period.')
        period=periods[0].date().isoformat(); common={'report_period':period,'release':release.label}
        if search_mode=='Manager':
            candidates=search_manager_candidates(prepared,query,report_period=period)
            if candidates.empty:raise DataUnavailable('No matching effective 13F manager was found in this release.')
            selected=candidates.iloc[min(candidate,len(candidates)-1)]
            return {**common,'candidates':[manager_candidate_label(row) for _,row in candidates.iterrows()],**profile_result(*manager_portfolio(prepared,str(selected['CIK']),period),period,portfolio_filter,portfolio_kind)}
        candidates=search_security_candidates(load_security_catalog(prepared),ticker_directory(),query,position_kind=position_kind)
        if candidates.empty:raise DataUnavailable('No matching filed security was found.')
        selected=candidates.iloc[min(candidate,len(candidates)-1)]; common['candidates']=[candidate_label(row) for _,row in candidates.iterrows()]
        if manager_cik:return {**common,**profile_result(*manager_portfolio(prepared,manager_cik,period),period,portfolio_filter,portfolio_kind)}
        ranking=rank_fund_exposure(prepared,(str(selected['CUSIP']),),report_period=period,position_kind=position_kind,minimum_portfolio_millions=minimum_portfolio_millions)
        return {**common,**screen_result(ranking,selected,sort_label,top_n,manager_filter,detail_columns,query,period)}
    except Sec13FError as exc:raise DataUnavailable(str(exc)) from exc
