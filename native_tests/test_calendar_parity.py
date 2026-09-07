import ast,subprocess,types,json
from datetime import date,timedelta
from io import StringIO
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from adfm_engine.palette import PASTEL,PASTEL_DIVERGING_SCALE
from adfm_engine.data.registry import SeriesDefinition
from adfm_engine.calendar_service import calendar
from adfm_engine.serialization import records,figure_json
ROOT=Path(__file__).resolve().parents[1]
class FixedDate(date):
    @classmethod
    def today(cls):return cls(2026,9,7)
class Capture:
    def __init__(self,controls):self.controls=controls;self.tables=[];self.figures=[];self.cards=[];self.sidebar=self
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def __getattr__(self,name):return lambda *a,**k:None
    def columns(self,n):return [self]*(n if isinstance(n,int) else len(n))
    def checkbox(self,label,**kwargs):return self.controls[label]
    def select_slider(self,label,**kwargs):return self.controls[label]
    def text_area(self,label,**kwargs):return self.controls[label]
    def expander(self,*args,**kwargs):return self
    def plotly_chart(self,f,**kwargs):self.figures.append(figure_json(f))
    def dataframe(self,f,**kwargs):self.tables.append(records(f))
def original(market,panel,status,controls):
    capture=Capture(controls);base=types.SimpleNamespace()
    ns={'date':FixedDate,'timedelta':timedelta,'StringIO':StringIO,'Dict':Dict,'List':List,'Tuple':Tuple,'np':np,'pd':pd,'go':go,'st':capture,'PASTEL':PASTEL,'PASTEL_DIVERGING_SCALE':PASTEL_DIVERGING_SCALE,'SeriesDefinition':SeriesDefinition,'base':base}
    for file in ['catalyst_calendar_page','catalyst_calendar_official_page','catalyst_calendar_exact_page']:
        source=subprocess.check_output(['git','show',f'ac2bd39d8371e959c778437519d390e1891c1d08:adfm_core/{file}.py'],cwd=ROOT,text=True)
        nodes=[]
        for n in ast.parse(source).body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)) and all(isinstance(t,ast.Name) for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) and not isinstance(getattr(n,'value',None),ast.Attribute):nodes.append(n)
            if isinstance(n,ast.FunctionDef) and not (file.endswith('_page') and n.name in ['_fetch_market','_fetch_macro','_metric_card','_dated_calendar']):
                n.decorator_list=[];nodes.append(n)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),ns)
        if file=='catalyst_calendar_page':base.__dict__.update(ns)
    for name in ['PageHeader','render_page_header','render_footer','inject_institutional_tool_finish']:setattr(base,name,lambda *a,**k:None)
    base._fetch_market=lambda *a:market.copy();base._fetch_macro=lambda *a:(panel.copy(),status.copy());base._metric_card=lambda *args:capture.cards.append(list(args))
    ns['_dated_calendar']=ns['_official_dated_calendar'];ns['render_sidebar_about']=lambda *a:None
    ns['render_catalyst_calendar']();return capture
@pytest.mark.parametrize('horizon',[14,30,60,90,120,180])
@pytest.mark.parametrize('vix',[None,18,20,25,31])
def test_page_matches_original(horizon,vix):
    idx=pd.bdate_range(end='2026-09-04',periods=500)
    market=pd.DataFrame({ticker:np.linspace(100,150,500) for ticker in ['SPY','QQQ','IWM','TLT','UUP','GLD']},index=idx)
    if vix is not None:market['^VIX']=vix
    panel=pd.DataFrame({'cpi':np.linspace(100,120,36),'gdp':np.linspace(100,140,36)},index=pd.date_range('2023-09-01',periods=36,freq='MS'));status=pd.DataFrame()
    fed=horizon!=30;hide=vix==31
    controls={'Event horizon':horizon,'Include recurring macro catalysts':True,'Include FOMC dates':fed,'Hide low-risk rows':hide,'Paste custom event CSV':''}
    expected=original(market,panel,status,controls);actual=calendar(market,panel,status,today=FixedDate.today(),horizon_days=horizon,include_fed=fed,hide_low=hide)
    assert actual['cards']==expected.cards
    assert [actual['timeline'],actual['heatmap']]==expected.figures
    assert [actual['macro'],actual['decision'],actual['details']]==expected.tables
    json.dumps(actual,allow_nan=False)
def test_empty_custom_outside_horizon():
    result=calendar(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),today=date(2026,9,7),include_macro=False,hide_low=True,custom_text='Date,Event\n2020-01-01,Expired')
    assert result['decision']==[]
