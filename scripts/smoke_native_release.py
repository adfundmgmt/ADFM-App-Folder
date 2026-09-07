"""Live service smoke: report availability without printing tokens or data payloads."""
import concurrent.futures,json,os,time
from pathlib import Path
import requests
origin=os.environ.get('ADFM_SMOKE_ORIGIN','https://adfm-python-api.onrender.com')
headers={'Authorization':'Bearer '+os.environ['ADFM_SMOKE_TOKEN']}
paths=[('rate-of-change','GET'),('overview','GET'),('leadership','POST'),('relative-volatility','POST'),('ratios','POST'),('macro-regime','GET'),('yields','POST'),('liquidity','POST'),('credit','POST'),('cftc','POST'),('options','POST'),('underwriter','POST'),('sec13f-releases','GET'),('calendar','POST'),('stress','POST'),('hedge','POST')]
def check(pair):
    path,method=pair;start=time.monotonic()
    try:
        response=requests.request(method,origin+'/v1/'+path,headers=headers,json={} if method=='POST' else None,timeout=110)
        try:body=response.json()
        except ValueError:body={}
        result={'endpoint':path,'status':response.status_code,'seconds':round(time.monotonic()-start,1),'detail':body.get('detail','') if isinstance(body,dict) else '', 'fields':list(body) if isinstance(body,dict) else [],'rows':len(body) if isinstance(body,list) else {k:len(v) for k,v in body.items() if isinstance(v,list)}}
    except Exception as exc:result={'endpoint':path,'status':'error','seconds':round(time.monotonic()-start,1),'detail':str(exc)}
    print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(check,paths))
Path(os.environ.get('ADFM_SMOKE_REPORT','/tmp/adfm-smoke.json')).write_text(json.dumps(results,indent=2))
