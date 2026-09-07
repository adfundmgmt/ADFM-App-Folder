import ast,subprocess,tempfile,time,json
from pathlib import Path
import pandas as pd
import pytest
from native_tests.test_sec13f_corrected_native import Corrected13FValueTests
from adfm_engine.data import sec13f as base
from adfm_engine.data import sec13f_queries as native
from adfm_engine.sec13f_service import profile_result,screen_result
from adfm_engine.jobs import JobQueue
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def reference():
    source=subprocess.check_output(['git','show','ac2bd39d8371e959c778437519d390e1891c1d08:adfm_core/sec_13f_corrected.py'],cwd=ROOT,text=True)
    module=ast.parse(source);functions=[n for n in module.body if isinstance(n,ast.FunctionDef)]
    ns={'pd':pd,'base':base,'PreparedDataset':base.PreparedDataset,'Sequence':list,'re':__import__('re')}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'original-corrected-13f','exec'),ns)
    return ns
@pytest.mark.parametrize('kind',['Long holdings','Call options','Put options','All reported'])
@pytest.mark.parametrize('cutoff',[0,1000])
def test_original_ranking(reference,tmp_path,kind,cutoff):
    prepared=Corrected13FValueTests().prepared_fixture(tmp_path)
    holdings=pd.read_parquet(prepared.holdings_path);cusips=tuple(holdings.CUSIP.unique())
    args={'report_period':'2026-03-31','position_kind':kind,'minimum_portfolio_millions':cutoff}
    pd.testing.assert_frame_equal(native.rank_fund_exposure(prepared,cusips,**args),reference['rank_fund_exposure'](prepared,cusips,**args))
@pytest.mark.parametrize('query',['LEGACY','0001537191','DOLLAR'])
def test_original_manager(reference,tmp_path,query):
    prepared=Corrected13FValueTests().prepared_fixture(tmp_path)
    candidates=native.search_manager_candidates(prepared,query,report_period='2026-03-31')
    pd.testing.assert_frame_equal(candidates,reference['search_manager_candidates'](prepared,query,report_period='2026-03-31'))
    cik=str(candidates.iloc[0].CIK)
    a,b=native.manager_portfolio(prepared,cik,'2026-03-31');x,y=reference['manager_portfolio'](prepared,cik,'2026-03-31')
    assert a==x;pd.testing.assert_frame_equal(b,y)
    result=profile_result(a,b,'2026-03-31');assert len(result['portfolio'])==len(b);json.dumps(result,allow_nan=False)
def test_queue_recovery_and_coalescing(tmp_path):
    calls=[]
    def handler(value):calls.append(value);return {'value':value}
    path=tmp_path/'jobs.sqlite';q=JobQueue(path,{'test':handler});job=q.submit('test',{'value':5})
    deadline=time.monotonic()+3
    while q.get(job['id'])['status']!='completed' and time.monotonic()<deadline:time.sleep(.01)
    assert q.get(job['id'])['result']=={'value':5}
    assert q.submit('test',{'value':5})['id']==job['id'];q.close()
    q2=JobQueue(path,{'test':handler});assert q2.get(job['id'])['result']=={'value':5};q2.close();assert calls==[5]
