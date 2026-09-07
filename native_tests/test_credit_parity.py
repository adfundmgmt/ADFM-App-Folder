"""Original Credit page replay plus independent sovereign transform checks."""
import ast,json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reference_roc import Capture,StopPage
from adfm_engine.analytics.credit_definitions import MARKET_TICKERS,SOVEREIGN_UNIVERSE,FOCUS_WINDOWS,HISTORY_DAYS,GLOBAL_WINDOWS
from adfm_engine.credit_service import credit
from adfm_engine.serialization import records,figure_json
SOURCE=json.loads((Path(__file__).parent/'fixtures/credit_baseline.json').read_text())['sources']['pages/5_Credit_Conditions_Monitor.py']
class CreditCapture(Capture):
    def __init__(self,p):super().__init__(p);self.figures=[];self.tables=[];self.cards=[]
    def header(self,*a,**k):pass
    def selectbox(self,label,*a,**k):return self.controls[label]
    def checkbox(self,label,*a,**k):return self.controls[label]
    def spinner(self,*a,**k):return self
    def columns(self,n):return [self]*(n if isinstance(n,int) else len(n))
    def info(self,*a,**k):pass
    def plotly_chart(self,fig,**k):self.figures.append(fig)
    def dataframe(self,frame,**k):self.tables.append(frame)

def fixture(missing=False):
    rng=np.random.default_rng(5);ix=pd.bdate_range(end=date.today(),periods=2800)
    fred=pd.DataFrame(rng.normal(0,.02,(len(ix),5)).cumsum(axis=0)+[4,1,2,4,4],index=ix,columns=['hy_oas','ig_oas','bbb_oas','dgs10','dgs30'])
    market=pd.DataFrame(100*np.exp(rng.normal(.0003,.015,(len(ix),len(MARKET_TICKERS))).cumsum(axis=0)),index=ix,columns=MARKET_TICKERS)
    status=pd.DataFrame({'status':['OK']*5,'key':list(fred)})
    sovereign={row['country']:pd.Series(4+rng.normal(0,.03,len(ix)).cumsum(),index=ix) for row in SOVEREIGN_UNIVERSE}
    if missing:fred=fred.drop(columns=['ig_oas','dgs30']);market=market.drop(columns=['KRE','LQD']);sovereign={}
    return fred,market,status,sovereign

def baseline(inputs,**p):
    from adfm_engine.palette import PASTEL,PASTEL_20
    from adfm_engine.data.registry import PRIMARY_MACRO_SERIES,SeriesDefinition
    fred,market,status,sovereign=inputs
    cap=CreditCapture({'Credit move window':p.get('focus_window','1M'),'Global 10Y move window':p.get('global_window','1Y'),'Chart history':p.get('history','3 Years'),'Show data audit':True})
    ns={'__name__':__name__,'st':cap,'PASTEL':PASTEL,'PASTEL_20':PASTEL_20,'PRIMARY_MACRO_SERIES':PRIMARY_MACRO_SERIES,'SeriesDefinition':SeriesDefinition,'fetch_fred_series':lambda *a,**k:(fred,status),'market_input':lambda *a,**k:market}
    for name in ['configure_yfinance_cache','PageHeader','inject_explorer_style','render_page_header','render_footer','render_sidebar_about','render_selection_note']:ns[name]=lambda *a,**k:None
    ns['render_kpi_cards']=lambda cards:cap.cards.extend(cards)
    ns['global_input']=lambda horizon:(ns['sovereign_move_rows'](sovereign,horizon,'Trading Economics API'),'Trading Economics API','fixture')
    tree=ast.parse(SOURCE);tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and ((n.module or '').startswith('adfm_core') or n.module=='pandas_datareader'))]
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name in {'fetch_market_prices','load_global_sovereign_moves'}:tree.body[i]=ast.parse(n.name+' = '+('market_input' if n.name=='fetch_market_prices' else 'global_input')).body[0]
    try:exec(compile(ast.fix_missing_locations(tree),'original-credit','exec'),ns)
    except StopPage:pass
    return ns,cap

@pytest.mark.parametrize('focus_window',FOCUS_WINDOWS)
@pytest.mark.parametrize('history',HISTORY_DAYS)
def test_all_credit_controls_charts_and_tape(focus_window,history):
    inputs=fixture();p=dict(focus_window=focus_window,history=history);ns,cap=baseline(inputs,**p)
    actual=credit(*inputs[:3],ns['sovereign_moves'],'Trading Economics API','fixture',**p)
    assert actual['cards']==cap.cards
    assert actual['narrative']==ns['active_read']
    assert [actual['spread'],actual['funding'],*actual['sovereign'],actual['appetite']]==[figure_json(fig) for fig in cap.figures]
    for key,table in zip(['rows','fred_status','market_status'],cap.tables):assert actual[key]==records(table)

@pytest.mark.parametrize('horizon',GLOBAL_WINDOWS)
def test_sovereign_horizons_and_missing_providers(horizon):
    from adfm_engine.analytics.credit import sovereign_move_rows
    inputs=fixture(True);ns,cap=baseline(inputs,global_window=horizon)
    actual=credit(*inputs[:3],ns['sovereign_moves'],global_window=horizon)
    assert [f for f in [actual['spread'],actual['funding'],*actual['sovereign'],actual['appetite']] if f is not None]==[figure_json(fig) for fig in cap.figures]
    raw=fixture()[3]
    pd.testing.assert_frame_equal(sovereign_move_rows(raw,horizon,'Trading Economics API'),ns['sovereign_move_rows'](raw,horizon,'Trading Economics API'))


def test_public_quote_with_no_verified_date_is_rejected():
    from adfm_engine.data.credit import _parse_te_public_page
    assert _parse_te_public_page('United States','10Y Bond Yield Actual 4.25 Daily Change Over the past month, down 0.2 points and is 0.5 points higher than a year ago') is None
