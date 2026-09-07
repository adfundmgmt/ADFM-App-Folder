import ast,subprocess
from datetime import timedelta
from pathlib import Path
from typing import Dict,List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest
from native_tests.test_calendar_parity import Capture,FixedDate
from adfm_engine.analytics.stress import ALL_TICKERS
from adfm_engine.stress_service import stress
from adfm_engine.serialization import records,figure_json
from adfm_engine.palette import PASTEL
ROOT=Path(__file__).resolve().parents[1]
class StressCapture(Capture):
    def selectbox(self,label,*args,**kwargs):return self.controls[label]
    def radio(self,label,*args,**kwargs):return self.controls[label]
    def slider(self,label,*args,**kwargs):return self.controls[label]
    def dataframe(self,frame,**kwargs):self.tables.append(records(frame if isinstance(frame,pd.DataFrame) else frame.data))
@pytest.fixture(scope='module')
def prices():
    rng=np.random.default_rng(51);n=1800;values={}
    for i,ticker in enumerate(ALL_TICKERS):
        returns=rng.normal(.0002,.01,n);returns[-55:-30]-=(i%4)*.002
        values[ticker]=100*np.exp(returns.cumsum())
    return pd.DataFrame(values,index=pd.bdate_range(end='2026-09-04',periods=n))
def original(prices,lookback,target,z,speed):
    source=subprocess.check_output(['git','show','ac2bd39d8371e959c778437519d390e1891c1d08:pages/19_Market_Stress_Composite.py'],cwd=ROOT,text=True)
    controls={'Chart lookback':lookback,'U.S. overlay':target,'Normalization window':z,'Signal speed':speed};capture=StressCapture(controls)
    ns=dict(date=FixedDate,timedelta=timedelta,Dict=Dict,List=List,np=np,pd=pd,go=go,make_subplots=make_subplots,PASTEL=PASTEL,st=capture,load_prices=lambda *args:prices.copy())
    for name in ['PageHeader','inject_explorer_style','render_footer','render_page_header','render_sidebar_about']:ns[name]=lambda *a,**k:None
    tree=ast.parse(source);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.FunctionDef) and n.name=='load_prices')]
    exec(compile(tree,'original-stress','exec'),ns);return ns,capture
@pytest.mark.parametrize('lookback',[1,2,3,5,10,25,50])
@pytest.mark.parametrize('target',['Auto','S&P 500','Nasdaq Composite'])
def test_full_page(prices,lookback,target):
    expected,cap=original(prices,lookback,target,3,'Slow - 10D');actual=stress(prices,today=FixedDate.today(),lookback_years=lookback,target_mode=target)
    assert actual['figure']==cap.figures[0]
    assert actual['moves']==records(expected['moves'])
    assert actual['health']==cap.tables[0]
    assert actual['regime']==expected['regime'] and actual['action']==expected['action']
@pytest.mark.parametrize('speed',['Fast - 3D','Base - 5D','Slow - 10D','21D','63D'])
@pytest.mark.parametrize('z',[1,5])
def test_speed_missing_series(prices,speed,z):
    prices=prices.drop(columns=['^IXIC','^HSI','^FTSE','AUDJPY=X'])
    expected,cap=original(prices,5,'Auto',z,speed);actual=stress(prices,today=FixedDate.today(),z_window_years=z,smoothing_mode=speed)
    assert actual['figure']==cap.figures[0]
    assert actual['moves']==records(expected['moves'])
