"""Full original page replay, with only provider input replaced by fixtures."""
import ast
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from adfm_engine.analytics.ratio_universe import RATIO_FAMILIES, CORE_RATIO_SPECS, MA_DEFAULTS
from adfm_engine.ratio_service import ratios, SPANS, date_bounds
from adfm_engine.serialization import figure_json
from reference_roc import Capture, StopPage

SOURCE=json.loads((Path(__file__).parent/'fixtures/ratios_baseline.json').read_text())['sources']['pages/11_Cross-Asset_Ratio_Chartbook.py']
class RatioCapture(Capture):
    def __init__(self, controls):
        super().__init__(controls); self.figures=[];self.captions=[]
    def header(self,*a,**k):pass
    def subheader(self,*a,**k):pass
    def caption(self,message):self.captions.append(message)
    def spinner(self,*a,**k):return self
    def expander(self,*a,**k):return self
    def columns(self,count,**k):return [self]*count
    def multiselect(self,label,*a,**k):return self.controls[label]
    def selectbox(self,label,*a,**k):return self.controls[label]
    def checkbox(self,label,*a,**k):return self.controls[label]
    def slider(self,label,*a,**k):return self.controls[label]
    def text_area(self,label,*a,**k):return self.controls[label]
    def plotly_chart(self,fig,**k):self.figures.append(fig)

def fixture(missing=False):
    symbols=sorted({t for s in CORE_RATIO_SPECS for t in (s.ticker_1,s.ticker_2)}|{'AAA','BBB'})
    rng=np.random.default_rng(11)
    frame=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0002,.013,(5600,len(symbols))),axis=0)),index=pd.bdate_range(end=date.today(),periods=5600),columns=symbols)
    if missing:
        frame=frame.drop(columns=['JAAA','AAA']);frame.loc[frame.index[:-80],'PBDC']=np.nan;frame.loc[frame.index[-20:],'SHY']=0
    return frame

def baseline(frame,**p):
    families=p.get('families',list(RATIO_FAMILIES));mas=p.get('moving_averages',[21,50,200])
    capture=RatioCapture({'Chart families':families,'Period':p.get('history','3 Years'),'Show signal strip':p.get('show_signal_strip',True),'RSI window':p.get('rsi_window',14),'Show RSI pane':p.get('show_rsi',False),'Enter one or more ratios':p.get('custom',''),**{f'{n} DMA':n in mas for n in MA_DEFAULTS}})
    from adfm_engine.palette import PASTEL
    namespace={'__name__':__name__,'st':capture,'PASTEL':PASTEL,'supplied_closes':lambda *a,**k:frame.copy()}
    for name in ('PageHeader','render_page_header','render_sidebar_about','render_footer'):namespace[name]=lambda *a,**k:None
    page=ast.parse(SOURCE)
    page.body=[n for n in page.body if not (isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('adfm_core'))]
    for i,n in enumerate(page.body):
        if isinstance(n,ast.FunctionDef) and n.name=='fetch_closes':page.body[i]=ast.parse('fetch_closes = supplied_closes').body[0]
    try:exec(compile(ast.fix_missing_locations(page),'original-ratios','exec'),namespace)
    except StopPage:pass
    return namespace,capture

@pytest.mark.parametrize('history',SPANS)
@pytest.mark.parametrize('show_rsi,mas',[(False,[]),(True,[8,21,50,100,200])])
def test_all_lookbacks_charts_signals_and_custom(history,show_rsi,mas):
    frame=fixture();p=dict(history=history,show_rsi=show_rsi,moving_averages=mas,custom='AAA/BBB; GLD/SPY\nAAA BBB; invalid; SPY/SPY',rsi_window=30 if show_rsi else 5)
    original,cap=baseline(frame,**p);actual=ratios(frame,frame,**p)
    assert [c['figure'] for c in actual['charts']]==[figure_json(f) for f in cap.figures]
    assert [c['signal'] for c in actual['charts']]==[c for c in cap.captions if c.startswith('Last ')]
    assert actual['warnings']==cap.warnings
    assert len(actual['charts'])==40
    assert date_bounds(history,date.today())[0]==original['hist_start']
    assert date_bounds(history,date.today())[2]==original['disp_start']

@pytest.mark.parametrize('families',[[],['Financial Intermediaries'],['Credit / Funding','Duration / Crisis Hedges']])
def test_filtered_missing_and_disabled_signals(families):
    frame=fixture(True);p=dict(families=families,show_signal_strip=False,custom='AAA/BBB')
    original,cap=baseline(frame,**p);actual=ratios(frame,frame,**p)
    assert [c['figure'] for c in actual['charts']]==[figure_json(f) for f in cap.figures]
    assert all(c['signal'] is None for c in actual['charts'])
    assert actual['unavailable']==sorted(set(original.get('failed_pairs',[])))
    assert actual['unavailable_custom']==sorted(set(original.get('custom_failed',[])))

def test_rebase_uses_prior_valid_close_and_not_first_visible():
    from adfm_engine.analytics.ratios import compute_price_ratio
    s=pd.Series([1.,2.,4.],index=pd.to_datetime(['2026-01-02','2026-01-05','2026-01-06']))
    out=compute_price_ratio(s,pd.Series(1.,index=s.index),pd.Timestamp('2026-01-04'))
    assert out.tolist()==[100.,200.,400.]
