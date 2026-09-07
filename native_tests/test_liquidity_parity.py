"""Replay the actually bound liquidity engine and its current page, never the obsolete base UI."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reference_roc import Capture,StopPage
from adfm_engine.analytics.liquidity import market_tickers
from adfm_engine.analytics.liquidity_definitions import FRED_IDS
from adfm_engine.liquidity_service import liquidity,LOOKBACKS
from adfm_engine.serialization import records,figure_json
SOURCES=json.loads((Path(__file__).parent/'fixtures/liquidity_baseline.json').read_text())['sources']
class LiquidityCapture(Capture):
    def __init__(self,p):super().__init__(p);self.figures=[];self.tables=[];self.csv=None
    def selectbox(self,label,*a,**k):return self.controls[label]
    def number_input(self,label,*a,**k):return self.controls[label]
    def checkbox(self,label,*a,**k):return self.controls[label]
    def caption(self,*a,**k):pass
    def spinner(self,*a,**k):return self
    def expander(self,*a,**k):return self
    def write(self,*a,**k):pass
    def info(self,*a,**k):pass
    def tabs(self,names):return [self]*len(names)
    def dataframe(self,frame,**k):self.tables.append(frame)
    def plotly_chart(self,fig,**k):self.figures.append(fig)
    def download_button(self,label,data,*a,**k):self.csv=data.decode()

def fixture(missing=False):
    rng=np.random.default_rng(3);ix=pd.bdate_range('2019-01-01',periods=1800)
    raw=rng.normal(0,.025,(len(ix),len(FRED_IDS))).cumsum(axis=0)
    fred=pd.DataFrame(raw+np.array([3,8,1,1,4,4,4,4,1,100,2]),index=ix,columns=FRED_IDS)
    fred[['WRESBAL','WALCL','WTREGEN']]*=1e6;fred['RRPONTSYD']*=1e3
    tickers=market_tickers();prices=pd.DataFrame(100*np.exp(rng.normal(.0003,.02,(len(ix),len(tickers))).cumsum(axis=0)),index=ix,columns=tickers)
    fcig=pd.DataFrame(rng.normal(0,.1,(80,2)).cumsum(axis=0),index=pd.date_range('2019-01-01',periods=80,freq='MS'),columns=['FCI-G Baseline','FCI-G 1Y Lookback'])
    errors={}
    if missing:fred=fred.drop(columns=['WRESBAL','EFFR']);errors={'WRESBAL':'fixture outage','EFFR':'fixture outage'};prices=pd.DataFrame();fcig=pd.DataFrame()
    return fred,prices,fcig,errors

def baseline(inputs,**p):
    from adfm_engine.palette import PASTEL
    fred,prices,fcig,errors=inputs
    cap=LiquidityCapture({'Display lookback':p.get('lookback','5y'),'Score lookback, business days':p.get('z_window',756),'Minimum score observations':p.get('min_periods',252),'Composite smoothing, business days':p.get('smoothing',3),'Show primary liquidity drivers':True,'Show component scorecards':True,'Show Fed financial conditions':p.get('show_fcig',True),'Show download':True})
    ns={'__name__':__name__,'st':cap,'PASTEL':PASTEL}
    for n in ['configure_yfinance_cache','PageHeader','inject_explorer_style','render_footer','render_kpi_cards','render_page_header','render_section_header','render_selection_note','render_sidebar_about']:ns[n]=lambda *a,**k:None
    tree=ast.parse(SOURCES['adfm_core/_liquidity_tracker_base.py'].split('\nrender_page_header(\n',1)[0])
    tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and ((n.module or '').startswith('adfm_core') or n.module=='pandas_datareader'))]
    exec(compile(tree,'original-liquidity-engine','exec'),ns)
    ns.update(load_fred=lambda *a:(fred,errors),load_market=lambda *a:prices,load_fcig=lambda:(fcig,{}))
    for k,key in [('BLUE','blue'),('GREEN','sage'),('ORANGE','coral'),('PURPLE','plum')]:ns[k]=PASTEL[key]
    page=ast.parse(SOURCES['pages/3_Liquidity_Conditions_Monitor.py']);page.body=[n for n in page.body if n.lineno>=76]
    exec(compile(page,'original-liquidity-page','exec'),ns)
    return ns,cap

@pytest.mark.parametrize('lookback',LOOKBACKS)
def test_every_display_lookback_preserves_scores_figures_tables_csv(lookback):
    inputs=fixture();ns,cap=baseline(inputs,lookback=lookback);actual=liquidity(*inputs,lookback=lookback)
    assert [actual['main'],actual['fcig'],*actual['drivers']]==[figure_json(fig) for fig in cap.figures]
    assert actual['csv']==cap.csv
    for key,table in zip(['primary','market','diagnostics'],cap.tables):assert actual[key]==records(table.data if hasattr(table,'data') else table)
    assert actual['warnings']==cap.warnings
    for key in ['primary','market']:
        styler=cap.tables[0 if key=='primary' else 1]._compute()
        for (row,col),css in styler.ctx.items():
            column=styler.data.columns[col];assert actual[key+'_colors']['backgrounds'][row][column]==dict(css)['background-color']
            assert actual[key+'_colors']['foregrounds'][row][column]==dict(css)['color']

@pytest.mark.parametrize('z_window,min_periods,smoothing,missing',[(252,126,1,False),(1260,756,21,False),(756,252,3,True)])
def test_scoring_boundaries_and_missing_sleeves(z_window,min_periods,smoothing,missing):
    inputs=fixture(missing);p=dict(z_window=z_window,min_periods=min_periods,smoothing=smoothing);ns,cap=baseline(inputs,**p);actual=liquidity(*inputs,**p)
    assert [actual['main']]+([actual['fcig']] if actual['fcig'] else [])+actual['drivers']==[figure_json(fig) for fig in cap.figures]
    assert actual['csv']==cap.csv
    assert actual['warnings']==cap.warnings

def test_invalid_window_combination_is_rejected():
    with pytest.raises(ValueError,match='Minimum observations'):liquidity(*fixture(),z_window=252,min_periods=756)
