import ast,subprocess
from dataclasses import dataclass
from datetime import date,timedelta
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
import pytest
from adfm_engine.analytics import hedge as native
from adfm_engine.hedge_service import hedge
from adfm_engine.charts.hedge import chart_data
from adfm_engine.serialization import records
pytestmark=pytest.mark.filterwarnings("ignore:Bitwise inversion '~' on bool is deprecated:DeprecationWarning")
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def oracle():
    source=subprocess.check_output(['git','show','ac2bd39d8371e959c778437519d390e1891c1d08:pages/21_Hedge_Timer.py'],cwd=ROOT,text=True)
    nodes=[]
    for n in ast.parse(source).body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)) and 52<=n.lineno<=325:nodes.append(n)
        if isinstance(n,ast.ClassDef):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and (n.lineno<654 or n.name in ['_display_name','summarize_eps']):
            if n.name=='yf_download':continue
            nodes.append(n)
    ns={'__name__':__name__,'date':date,'timedelta':timedelta,'Dict':Dict,'List':List,'Tuple':Tuple,'np':np,'pd':pd,'dataclass':dataclass}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-hedge','exec'),ns);return ns
@pytest.fixture(scope='module')
def prices():
    rng=np.random.default_rng(68);n=2800
    returns=rng.normal(.0002,.012,(n,len(native.TICKERS)))
    returns[1600:1620,:]-=.022;returns[2200:2230,:]-=.011
    return pd.DataFrame(100*np.exp(returns.cumsum(axis=0)),index=pd.bdate_range(end='2026-09-04',periods=n),columns=native.TICKERS)
@pytest.mark.parametrize('years',[1,2,3,5,10])
def test_original_model_and_chart_data(oracle,prices,years):
    df=prices.reindex(prices[native.SPX_TICKER].dropna().index.intersection(prices[native.NDX_TICKER].dropna().index)).ffill()
    a,ma=oracle['compute_score_and_meta'](df,native.SPX_TICKER);b,mb=oracle['compute_score_and_meta'](df,native.NDX_TICKER)
    threshold=oracle['calibrate_threshold'](a,ma,df[native.SPX_TICKER],b,mb,df[native.NDX_TICKER]);target=oracle['pick_target_today'](df);oracle['t_short']=threshold
    expected=pd.concat([oracle['summarize_eps']('^SPX',df[native.SPX_TICKER],a,ma),oracle['summarize_eps']('^NDX',df[native.NDX_TICKER],b,mb)],ignore_index=True)
    result=hedge(prices,years)
    assert result['threshold']==threshold;assert result['target']==oracle['_display_name'](target);assert result['episodes']==records(expected)
    score,meta=(b,mb) if target==native.NDX_TICKER else (a,ma)
    data=chart_data(df[target],score,meta,threshold,years);idx=df[target].dropna().index[-252*years:]
    pd.testing.assert_series_equal(data.price,df[target].reindex(idx),check_names=False)
    pd.testing.assert_series_equal(data.score,score.reindex(idx),check_names=False)
    pd.testing.assert_series_equal(data.MA50,meta['ma50'].reindex(idx),check_names=False)
    pd.testing.assert_series_equal(data.MA200,meta['ma200'].reindex(idx),check_names=False)
    pd.testing.assert_series_equal(data.onset,oracle['signal_onset'](score.reindex(idx),{k:v.reindex(idx) for k,v in meta.items()},threshold),check_names=False)
    for ticker,sc,mt,actual in [(native.SPX_TICKER,a,ma,result['stats'][0]),(native.NDX_TICKER,b,mb,result['stats'][1])]:
        stats=oracle['forward_stats'](sc[sc.index>='2020-01-01'],df[ticker][df.index>='2020-01-01'],{k:v[v.index>='2020-01-01'] for k,v in mt.items()},threshold)
        assert actual==records(pd.DataFrame([{'Index':oracle['_display_name'](ticker),**stats}]))[0]
def test_missing_risk_inputs(oracle,prices):
    df=prices.drop(columns=['^VIX9D','^VVIX','HYG'])
    expected,meta=oracle['compute_score_and_meta'](df,native.SPX_TICKER);actual,actual_meta=native.compute_score_and_meta(df,native.SPX_TICKER)
    pd.testing.assert_series_equal(actual,expected)
    for key in meta:pd.testing.assert_series_equal(meta[key],actual_meta[key])
