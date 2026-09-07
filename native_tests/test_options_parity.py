"""Frozen original page replay: formulas, figures, tables, and exact CSVs."""
import ast,json
from datetime import date,datetime,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reference_roc import Capture,StopPage
from adfm_engine.options_service import options
from adfm_engine.serialization import records,figure_json
SOURCES=json.loads((Path(__file__).parent/'fixtures/options_baseline.json').read_text())['sources']
AS_OF=date(2026,8,28)
class OptionsCapture(Capture):
    def __init__(self,p):super().__init__(p);self.figures=[];self.tables=[];self.downloads={};self.cards=[]
    def header(self,*a,**k):pass
    def caption(self,*a,**k):pass
    def info(self,*a,**k):pass
    def spinner(self,*a,**k):return self
    def tabs(self,labels):return [self]*len(labels)
    def text_area(self,label,*a,**k):return self.controls[label]
    def slider(self,label,*a,**k):return self.controls[label]
    def number_input(self,label,*a,**k):return self.controls[label]
    def dataframe(self,frame,**k):self.tables.append(frame.data if hasattr(frame,'data') else frame)
    def plotly_chart(self,fig,**k):self.figures.append(fig)
class Clock(datetime):
    @classmethod
    def now(cls,tz=None):return cls(2026,8,28,17,tzinfo=tz)

def fixture():
    raw,calendars,chains={},{},{}
    for j,symbol in enumerate(['QQQ','SPY','TLT']):
        ix=pd.bdate_range(end=AS_OF,periods=260);v=100*np.exp(np.cumsum(np.sin(np.arange(260)*.7+j)*.008+.0005))
        raw[symbol]=pd.DataFrame({c:v for c in ['Open','High','Low','Close','Adj Close']},index=ix).assign(Volume=1000000)
        calendars[symbol]=tuple((AS_OF+timedelta(days=d)).isoformat() for d in [1,7,14,30,45,60,90,120,180,270,365,400])
        for expiry in calendars[symbol]:
            pair=[]
            for kind in ['C','P']:
                strikes=np.arange(70,151,5,dtype=float)
                f=pd.DataFrame({'contractSymbol':[f'{symbol}{expiry}{kind}{s}' for s in strikes],'lastTradeDate':pd.Timestamp(AS_OF,tz='UTC'),'strike':strikes,'lastPrice':6.0,'bid':5.5,'ask':6.5,'volume':np.arange(len(strikes))*13+4,'openInterest':np.arange(len(strikes))*170+1,'impliedVolatility':.18+j*.05+np.abs(strikes-v[-1])*.001+(kind=='P')*.025})
                f.loc[1,'impliedVolatility']=.001;f.loc[2,['bid','ask']]=np.nan
                pair.append(f)
            chains[(symbol,expiry)]=(*pair,{'regularMarketPrice':float(v[-1])},None,'Cboe delayed quotes' if j else 'Yahoo Finance','2026-08-28 20:00:00' if j else '')
    return raw,pd.DataFrame(),calendars,chains

def baseline(inputs,**p):
    from adfm_engine.palette import PASTEL
    from adfm_engine.data.market import adjusted_ohlcv,unique_tickers
    from adfm_engine.analytics.relative_volatility import annualized_realized_volatility
    raw,failures,calendars,chains=inputs
    controls={'Focus ticker':p.get('selected','QQQ'),'Comparison universe':p.get('universe_text','SPY, QQQ, TLT'),'Target expiration':p.get('target_dte',45),'Term-structure expirations':p.get('term_count',6),'Risk-free rate':p.get('risk_free_rate',.04)}
    cap=OptionsCapture(controls);ns={'__name__':__name__,'st':cap,'PASTEL':PASTEL,'adjusted_ohlcv':adjusted_ohlcv,'unique_tickers':unique_tickers,'annualized_realized_volatility':annualized_realized_volatility,'clock':Clock}
    exec(compile(SOURCES['adfm_core/options_positioning.py'],'original-options-model','exec'),ns)
    for name in ['configure_yfinance_cache','inject_explorer_style','render_sidebar_about','render_page_header','PageHeader','render_footer','render_section_header','render_selection_note','render_status_line']:ns[name]=lambda *a,**k:None
    ns['render_kpi_cards']=lambda cards:cap.cards.extend(cards)
    ns['dataframe_download']=lambda label,frame,filename:cap.downloads.update({filename:frame.to_csv(index=False)})
    ns['fetch_daily_ohlcv']=lambda *a,**k:(raw,failures)
    ns['calendar_input']=lambda symbol:calendars.get(symbol,())
    ns['chain_input']=lambda symbol,expiry:chains[(symbol,expiry)]
    page=ast.parse(SOURCES['pages/16_Options_Positioning_Compass.py'])
    page.body=[n for n in page.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('adfm_core'))]
    for i,n in enumerate(page.body):
        if isinstance(n,ast.FunctionDef) and n.name in ['available_expirations','fetch_chain']:page.body[i]=ast.parse(n.name+' = '+('calendar_input' if n.name=='available_expirations' else 'chain_input')).body[0]
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='as_of_date' for t in n.targets):page.body[i]=ast.parse('as_of_date=clock.now(NY_TZ).date()').body[0]
    exec(compile(ast.fix_missing_locations(page),'original-options-page','exec'),ns)
    return ns,cap

@pytest.mark.parametrize('target_dte',[14,45,120])
@pytest.mark.parametrize('term_count',[3,6,10])
@pytest.mark.parametrize('risk_free_rate',[0,.04,.20])
def test_original_outputs_for_all_control_boundaries(target_dte,term_count,risk_free_rate):
    inputs=fixture();p=dict(universe_text='SPY, QQQ, TLT',target_dte=target_dte,term_count=term_count,risk_free_rate=risk_free_rate)
    ns,cap=baseline(inputs,**p);actual=options(*inputs,as_of_date=AS_OF,**p)
    assert actual['cards']==cap.cards
    assert [actual['compass'],actual['structure'],actual['surface']]==[figure_json(f) for f in cap.figures]
    assert [actual['compass_rows'],actual['term_rows'],actual['activity_rows']]==[records(f) for f in cap.tables]
    assert actual['compass_csv']==cap.downloads['options_positioning_compass.csv']
    assert actual['term_csv']==cap.downloads['QQQ_options_term_structure.csv']
    assert actual['activity_csv']==cap.downloads[f"QQQ_{actual['expiry']}_premium_activity.csv"]


def test_chain_spot_survives_missing_price_history():
    inputs=list(fixture());inputs[0].pop('QQQ');inputs[1]=pd.DataFrame([{'Ticker':'QQQ','Reason':'unavailable'}])
    actual=options(*inputs,as_of_date=AS_OF,universe_text='SPY, QQQ, TLT')
    assert actual['activity_rows'] and actual['structure']
    assert all(row['moneyness'] is not None for row in actual['activity_rows'])
    assert actual['diagnostics']==[{'Ticker':'QQQ','Issue':'unavailable'}]


def test_missing_peer_and_terms_remain_explicit():
    inputs=list(fixture());inputs[2]['TLT']=();inputs[2]['QQQ']=(inputs[2]['QQQ'][-1],)
    actual=options(*inputs,as_of_date=AS_OF,universe_text='SPY, QQQ, TLT')
    assert actual['chain_count']==2 and actual['term_csv'] is None
    assert actual['structure'] is None and actual['diagnostics'][0]['Ticker']=='TLT'


def test_focus_chain_failure_keeps_provider_diagnostics():
    from adfm_engine.services import DataUnavailable
    inputs=list(fixture());inputs[2]['QQQ']=()
    with pytest.raises(DataUnavailable) as exc:
        options(*inputs,as_of_date=AS_OF,universe_text='QQQ, SPY')
    assert exc.value.diagnostics[0]['Ticker']=='QQQ'
