import ast,json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reference_roc import Capture,StopPage
from adfm_engine.analytics.yields import PERIODS,CURVE_OPTIONS,YAHOO_YIELD_TICKERS
from adfm_engine.yield_service import yields,LOOKBACKS
from adfm_engine.serialization import records,figure_json
from adfm_engine.services import DataUnavailable
SOURCE=json.loads((Path(__file__).parent/'fixtures/yields_baseline.json').read_text())['sources']['pages/4_Yield_Curve_Rates_Regime_Monitor.py']
class YieldCapture(Capture):
    def __init__(self,p):super().__init__(p);self.figures=[];self.tables=[]
    def header(self,*a,**k):pass
    def caption(self,*a,**k):pass
    def spinner(self,*a,**k):return self
    def expander(self,*a,**k):return self
    def code(self,*a,**k):pass
    def info(self,*a,**k):pass
    def columns(self,widths):return [self]*(widths if isinstance(widths,int) else len(widths))
    def selectbox(self,label,*a,**k):return self.controls[label]
    def radio(self,label,*a,**k):return self.controls[label]
    def checkbox(self,label,*a,**k):return self.controls[label]
    def plotly_chart(self,fig,**k):self.figures.append(fig)
    def dataframe(self,frame,**k):self.tables.append(frame)

def fixture(missing=False,scaled=False):
    rng=np.random.default_rng(4);frame=pd.DataFrame(rng.normal(0,.03,(1100,4)).cumsum(axis=0)+[3.,3.5,4.,4.5],index=pd.bdate_range(end=date.today(),periods=1100),columns=YAHOO_YIELD_TICKERS)
    frame.iloc[30:34,1]=np.nan
    if missing:frame=frame.drop(columns=['^IRX'])
    if scaled:frame=frame*10
    return frame

def baseline(frame,**p):
    from adfm_engine.palette import PASTEL,PASTEL_RATES_SCALE
    cap=YieldCapture({'History':p.get('history','5Y'),'Regime window':p.get('regime_period','1M'),'Curve gauge':p.get('selected_curve','3m10y'),'Curve comparison':p.get('curve_compare','1M'),'Show raw Yahoo table':True,'Show Yahoo download status':True})
    ns={'__name__':__name__,'st':cap,'PASTEL':PASTEL,'PASTEL_RATES_SCALE':PASTEL_RATES_SCALE,'supplied':lambda *a,**k:(frame.copy(),())}
    for n in ['PageHeader','inject_institutional_tool_finish','render_footer','render_page_header','render_sidebar_about']:ns[n]=lambda *a,**k:None
    tree=ast.parse(SOURCE);tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('adfm_core'))]
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name=='fetch_yahoo_close':tree.body[i]=ast.parse('fetch_yahoo_close = supplied').body[0]
    try:exec(compile(ast.fix_missing_locations(tree),'original-yields','exec'),ns)
    except StopPage:pass
    return ns,cap

@pytest.mark.parametrize('period',PERIODS)
@pytest.mark.parametrize('curve',CURVE_OPTIONS)
@pytest.mark.parametrize('compare',['1W','1M','3M','YTD'])
def test_every_regime_curve_comparison(period,curve,compare):
    frame=fixture();p=dict(regime_period=period,selected_curve=curve,curve_compare=compare);ns,cap=baseline(frame,**p);actual=yields(frame,**p)
    assert actual['cards']==ns['cards']
    assert [actual['snapshot'],actual['pressure'],actual['history']]==[figure_json(f) for f in cap.figures]
    assert actual['rows']==records(cap.tables[0].rename_axis('Date').reset_index())
    assert actual['warnings']==cap.warnings

@pytest.mark.parametrize('history',LOOKBACKS)
def test_all_history_bounds_and_missing_tenor_fallback(history):
    frame=fixture(True,True);ns,cap=baseline(frame,history=history);actual=yields(frame,history=history)
    assert (date.today()-ns['start']).days==LOOKBACKS[history]+10
    assert actual['selected_curve']=='5s10s'
    assert actual['cards']==ns['cards']
    assert actual['warnings']==cap.warnings
    assert [actual['snapshot'],actual['pressure'],actual['history']]==[figure_json(f) for f in cap.figures]

@pytest.mark.parametrize('frame',[pd.DataFrame(),fixture()[['^IRX']],fixture()[['^TNX']]])
def test_missing_required_data_is_explicit(frame):
    ns,cap=baseline(frame)
    with pytest.raises(DataUnavailable) as error:yields(frame)
    assert str(error.value)==cap.warnings[-1]
