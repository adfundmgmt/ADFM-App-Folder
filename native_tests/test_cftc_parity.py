"""Independent original-page replay for both CFTC report types and all cohorts."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reference_roc import Capture,StopPage
from adfm_engine.analytics.cftc import COHORTS,DEFAULT_COHORT,_all_position_fields
from adfm_engine.cftc_service import cftc,LOOKBACKS,SORTS
from adfm_engine.serialization import records,figure_json
SOURCES=json.loads((Path(__file__).parent/'fixtures/cftc_baseline.json').read_text())['sources']
class CFTCCapture(Capture):
    def __init__(self,p):super().__init__(p);self.figures=[];self.tables=[];self.downloads={};self.cards=[];self.selected=p.get('selected')
    def select_slider(self,label,*a,**k):return self.controls[label]
    def selectbox(self,label,options,*a,**k):
        if label=='Market to inspect':
            options=list(options)
            if self.selected:return next(x for x in options if x.startswith(self.selected))
            return options[k.get('index',0)]
        return self.controls[label]
    def multiselect(self,label,*a,**k):return self.controls[label]
    def expander(self,*a,**k):return self
    def spinner(self,*a,**k):return self
    def columns(self,n):return [self]*(n if isinstance(n,int) else len(n))
    def tabs(self,names):return [self]*len(names)
    def caption(self,*a,**k):pass
    def dataframe(self,frame,**k):self.tables.append(frame)
    def plotly_chart(self,fig,**k):self.figures.append(fig)

def fixture(report):
    rng=np.random.default_rng(18 if report=='TFF' else 19);frames=[]
    contracts=[('209742','NASDAQ-100'),('043602','10Y TREASURY'),('099741','EURO FX')] if report=='TFF' else [('088691','GOLD'),('067651','CRUDE OIL'),('XXXXX','UNMAPPED COFFEE')]
    for code,name in contracts:
        ix=pd.date_range('2018-01-02',periods=420,freq='W-TUE')
        frame=pd.DataFrame({'report_date':ix,'contract_code':code,'market_name':name,'commodity_name':name,'open_interest':rng.uniform(10000,30000,len(ix))})
        for field in _all_position_fields(report):frame[field]=rng.uniform(1000,5000,len(ix))
        frame.loc[frame.index[::13],_all_position_fields(report)[0]]=np.nan
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)

def baseline(tff,disagg,price,**p):
    from adfm_engine.palette import EXCEL,PASTEL_20
    controls={'Crowding lookback':p.get('lookback','3Y'),'Financial futures cohort':p.get('tff_cohort',DEFAULT_COHORT['TFF']),'Physical futures cohort':p.get('disagg_cohort',DEFAULT_COHORT['Disaggregated']),'Asset class':p.get('assets',[]),'Rank scanner':p.get('sort',SORTS[0]),'selected':p.get('selected_market')}
    cap=CFTCCapture(controls);ns={'__name__':__name__,'st':cap,'EXCEL':EXCEL,'PASTEL_20':PASTEL_20}
    exec(compile(SOURCES['adfm_core/cftc_positioning.py'],'original-cftc-engine','exec'),ns)
    for name in ['configure_yfinance_cache','PageHeader','inject_explorer_style','render_page_header','render_footer','render_section_header','render_sidebar_about','render_selection_note','render_status_line']:ns[name]=lambda *a,**k:None
    ns['render_kpi_cards']=lambda cards:cap.cards.extend(cards)
    ns['dataframe_download']=lambda label,frame,filename:cap.downloads.update({filename:frame.to_csv(index=False)})
    ns['report_input']=lambda report:((tff if report=='TFF' else disagg),'')
    ns['history_input']=lambda report,code:((tff if report=='TFF' else disagg).loc[lambda f:f['contract_code'].eq(code)],'')
    ns['price_input']=lambda ticker:(price,'')
    page=ast.parse(SOURCES['pages/18_CFTC_Positioning_Monitor.py']);page.body=[n for n in page.body if not(isinstance(n,ast.Import) and any(a.name=='streamlit' for a in n.names)) and not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('adfm_core'))]
    for i,n in enumerate(page.body):
        if isinstance(n,ast.FunctionDef) and n.name in {'load_report','load_history','load_price'}:page.body[i]=ast.parse(n.name+' = '+{'load_report':'report_input','load_history':'history_input','load_price':'price_input'}[n.name]).body[0]
    exec(compile(ast.fix_missing_locations(page),'original-cftc-page','exec'),ns)
    return ns,cap

@pytest.mark.parametrize('lookback',LOOKBACKS)
@pytest.mark.parametrize('cohort_index',range(5))
def test_all_lookbacks_cohorts_ranks_charts_and_downloads(lookback,cohort_index):
    tff,disagg=fixture('TFF'),fixture('Disaggregated');price=pd.Series(np.linspace(100,200,1000),index=pd.bdate_range('2021-01-01',periods=1000))
    p=dict(lookback=lookback,tff_cohort=list(COHORTS['TFF'])[cohort_index],disagg_cohort=list(COHORTS['Disaggregated'])[cohort_index%4],sort=SORTS[cohort_index],assets=['Energy','Equity / Vol'] if cohort_index%2 else [])
    ns,cap=baseline(tff,disagg,price,selected_market='GOLD' if cohort_index%2 else None,**p)
    actual=cftc(tff,disagg,ns['history_raw'],price,selected=ns['report_type']+'|'+ns['contract_code'],**p)
    assert actual['cards']==cap.cards
    assert [actual['main'],actual['cohorts']]==[figure_json(fig) for fig in cap.figures]
    for key,table in zip(['shorts','longs','shifts','scanner','history'],cap.tables):assert actual[key]==records(table.data if hasattr(table,'data') else table)
    assert actual['scanner_csv']==cap.downloads['adfm_cftc_positioning_scanner.csv']
    assert actual['history_csv']==cap.downloads[actual['history_filename']]


def test_report_missing_keeps_available_scanner_and_unmapped_history():
    disagg=fixture('Disaggregated');history=disagg.loc[disagg['contract_code'].eq('XXXXX')]
    result=cftc(pd.DataFrame(),disagg,history,selected='Disaggregated|XXXXX',report_errors=['TFF unavailable'])
    assert result['main'] and result['cohorts'] and len(result['selection'])==3
    assert len(result['warnings'])==2


def test_api_uses_validated_cohorts_and_rejects_unknown_fields(monkeypatch):
    from fastapi.testclient import TestClient
    from adfm_api import main
    monkeypatch.setenv('ADFM_ENV','development')
    monkeypatch.delenv('ADFM_GATEWAY_TOKEN',raising=False)
    monkeypatch.setattr(main,'load_cftc',lambda **p:p)
    with TestClient(main.create_app()) as client:
        assert client.post('/v1/cftc',json={}).json()['lookback']=='3Y'
        assert client.post('/v1/cftc',json={'tff_cohort':'invented'}).status_code==422
        assert client.post('/v1/cftc',json={'selected':'https://example.com'}).status_code==422
        assert client.post('/v1/cftc',json={'lookback':'3Y','unknown':1}).status_code==422
