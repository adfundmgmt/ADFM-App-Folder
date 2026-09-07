"""Replay the original macro page and compare all states, tables and styling."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from adfm_engine.analytics.macro import TICKERS
from adfm_engine.macro_service import macro_regime
from adfm_engine.serialization import records
from reference_roc import Capture

SOURCE=json.loads((Path(__file__).parent/'fixtures/macro_baseline.json').read_text())['sources']['pages/2_Global_Macro_Regime.py']
class MacroCapture(Capture):
    def __init__(self):super().__init__({});self.tables=[]
    def dataframe(self,frame,**kw):self.tables.append(frame)
    def expander(self,*a,**k):return self
    def info(self,*a,**k):pass

def fixture(seed,missing):
    rng=np.random.default_rng(seed); index=pd.bdate_range('2022-01-03',periods=1200)
    prices=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(rng.uniform(-.002,.002,len(TICKERS)),.015,(len(index),len(TICKERS))),axis=0)),index=index,columns=TICKERS)
    columns=['dgs2','dgs10','dgs30','dfii10','t10yie','hy_oas','walcl','tga','rrp']
    macro=pd.DataFrame(rng.normal(0,.025,(len(index),len(columns))).cumsum(axis=0),index=index,columns=columns)
    macro=macro.add([4,4,4,2,2,4,8,1,1]).mul([1,1,1,1,1,1,1e6,1e6,1e3])
    macro.loc[index[::3],['walcl','tga']]=np.nan
    status=pd.DataFrame({'key':columns,'status':['OK']*len(columns)})
    failed=[]
    if missing==1:prices=prices.drop(columns=['DX-Y.NYB','RSP','CL=F']);macro=macro.drop(columns=['t10yie','rrp']);failed=['DX-Y.NYB','RSP','CL=F']
    if missing==2:macro=pd.DataFrame();status=pd.DataFrame()
    return prices,macro,status,failed

def baseline(prices,macro,status,failed):
    from adfm_engine.palette import PASTEL
    cap=MacroCapture();namespace={'__name__':__name__,'st':cap,'PASTEL':PASTEL,'market_input':lambda *a,**k:(prices,failed),'macro_input':lambda *a,**k:(macro,status)}
    for name in ('configure_yfinance_cache','PageHeader','inject_explorer_style','render_page_header','render_sidebar_about','render_footer'):namespace[name]=lambda *a,**k:None
    tree=ast.parse(SOURCE);tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('adfm_core'))]
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name in {'fetch_market_prices','fetch_macro_data'}:tree.body[i]=ast.parse(n.name+' = '+('market_input' if n.name=='fetch_market_prices' else 'macro_input')).body[0]
    exec(compile(ast.fix_missing_locations(tree),'original-macro','exec'),namespace)
    return namespace,cap

@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('missing',[0,1,2])
def test_all_states_tables_tensions_and_cell_colors(seed,missing):
    inputs=fixture(seed,missing);n,cap=baseline(*inputs);actual=macro_regime(*inputs)
    assert actual['current']=={k:n['current'][k] for k in actual['current']}
    assert actual['narrative']==n['narrative'](n['current'])
    assert actual['tensions']==n['tensions']
    for key,table in zip(['drivers','states','rates','performance','macro_status'],cap.tables):
        assert actual[key]==records(table.data if hasattr(table,'data') else table)
    styled=cap.tables[3]._compute()
    for (row,col),css in styled.ctx.items():
        assert dict(css)['background-color']==actual['performance_colors'][row][styled.data.columns[col]]
    assert actual['data_through']==n['asof'].date().isoformat()
    assert actual['failed']==inputs[3]


def test_fred_adapter_preserves_missing_and_isolates_failure(monkeypatch):
    from adfm_engine.data import primary
    from adfm_engine.data.registry import PRIMARY_MACRO_SERIES
    def read(symbol,start,end):
        if symbol=='DGS10':raise TimeoutError('fixture outage')
        return pd.DataFrame({symbol:[1.,np.nan,3.]},index=pd.date_range('2026-01-01',periods=3))
    monkeypatch.setattr(primary,'read_fred',read)
    panel,status=primary.fetch_fred_series(PRIMARY_MACRO_SERIES[:2],start='2026-01-01',end='2026-01-03')
    assert panel['dgs2'].isna().tolist()==[False,True,False]
    assert status['status'].tolist()==['OK','FAILED']
    assert status.iloc[0]['observations']==2
