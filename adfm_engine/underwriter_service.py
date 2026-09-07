"""Filing-driven underwriting reusable by the API and reporting jobs."""
from dataclasses import asdict
import pandas as pd
from adfm_engine.analytics.sec_fundamentals import *
from adfm_engine.analytics.underwriter import *
from adfm_engine.analytics.underwriter import _signed_currency
from adfm_engine.charts.underwriter import quarterly_chart,price_history_chart
from adfm_engine.data.underwriter import load_ticker_map,load_company_facts,load_submissions,market_history
from adfm_engine.serialization import records,figure_json
from adfm_engine.services import DataUnavailable

def underwriter(identity,company_facts,submissions,close_history):
    metrics=extract_metrics(company_facts)
    price=float(close_history.iloc[-1]) if not close_history.empty else None
    price_date=pd.Timestamp(close_history.index[-1]).normalize() if not close_history.empty else None
    currency=statement_currency(metrics)
    valuation=build_valuation_snapshot(metrics,price=price if currency=='USD' else None,price_date=price_date)
    keys=('revenue','gross_profit','operating_income','net_income','cfo','capex')
    quarterly=financial_table(metrics,keys,frequency='quarterly',periods=12)
    annual=financial_table(metrics,keys,frequency='annual',periods=8)
    balance=balance_sheet_table(metrics,('cash','short_term_investments','receivables','current_assets','current_liabilities','debt_current','debt_noncurrent','short_term_borrowings','equity','assets'),periods=12)
    maturities=maturity_table(company_facts)
    if not maturities.empty:
        maturities['Principal']=pd.to_numeric(maturities['Principal'],errors='coerce').div(1e6).map(lambda v:_signed_currency(v,currency) if pd.notna(v) else 'Unavailable')
        maturities=maturities.rename(columns={'Principal':f'Principal ({currency_prefix(currency)} millions)'})
    audit=source_audit_table(metrics)
    events=recent_filings(submissions,forms=('8-K','6-K'),limit=8)
    if not events.empty:events=events[['Filed','Period','Form','Description','Document']]
    tables={'valuation_rows':valuation_table(valuation,currency=currency),'snapshot_rows':sec_snapshot_table(valuation,currency=currency),'growth':growth_table(metrics),'events':events,'quarterly':scale_financial_table(quarterly,currency),'annual':scale_financial_table(annual,currency),'balance':scale_financial_table(balance,currency),'credit':credit_table(valuation,currency=currency),'maturities':maturities,'filings':recent_filings(submissions,limit=35),'audit':format_source_audit(audit)}
    return {**{k:records(v) for k,v in tables.items()},'identity':asdict(identity),'currency':currency,
      'issuer_note':f"CIK {identity.padded_cik} · {submissions.get('sicDescription','Unavailable')} · Fiscal year end {submissions.get('fiscalYearEnd','Unavailable')} · Latest filing {first_recent_value(submissions,'form')} on {first_recent_value(submissions,'filingDate')}",
      'warnings':[] if currency=='USD' else [f'This issuer reports primarily in {currency}. Current US-dollar market multiples are suppressed until a filing-currency FX conversion is available.'],
      'cards':[['Price',format_money(price),f'Close through {period_label(price_date)}'],['Market Cap',format_money(valuation.market_cap),'Price × SEC shares'],['Enterprise Value',format_money(valuation.enterprise_value),'Calculated capital value'],['LTM Revenue',format_money(valuation.ltm_revenue,currency=currency),'Latest four quarters'],['LTM Free Cash Flow',format_money(valuation.ltm_fcf,currency=currency),'CFO less capex'],['Net Debt / EBITDA',format_multiple(valuation.net_debt_ebitda),'Calculated issuer leverage']],
      'valuation_cards':valuation_cards(valuation,currency=currency) if currency=='USD' else [],'snapshot_cards':sec_snapshot_cards(valuation,currency=currency),'reads':underwrite_read(metrics,valuation,currency=currency),
      'price':figure_json(price_history_chart(close_history,identity.ticker,'USD')) if not close_history.empty else None,
      'quarterly_chart':figure_json(quarterly_chart(quarterly,currency)) if {'Revenue','Operating Income'}.issubset(quarterly.columns) else None,
      'quarterly_csv':quarterly.to_csv(index=False),'audit_csv':audit.drop(columns=['Source'],errors='ignore').to_csv(index=False)}

def load_underwriter(query='AAPL'):
    try:
        identity=resolve_company(query,load_ticker_map())
        close,_,_=market_history(identity.ticker)
        return underwriter(identity,load_company_facts(identity.cik),load_submissions(identity.cik),close)
    except SecDataError as exc: raise DataUnavailable(str(exc)) from exc
