import ast,textwrap
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'adfm_core/catalyst_calendar_page.py').read_text();header='from __future__ import annotations\nfrom datetime import date,timedelta\nfrom io import StringIO\nfrom typing import Dict,List,Tuple\nimport numpy as np\nimport pandas as pd\nfrom adfm_engine.data.registry import SeriesDefinition\nfrom adfm_engine.palette import PASTEL,PASTEL_DIVERGING_SCALE\n'
model=header;chart=header+'import plotly.graph_objects as go\nfrom adfm_engine.analytics.calendar import TYPE_COLORS,ASSET_LABELS,RISK_COLORS\n'
provider=header+'import yfinance as yf\nfrom adfm_engine.cache import ttl_cache\nfrom adfm_engine.data.primary import fetch_fred_series\nfrom adfm_engine.analytics.calendar import MARKET_TICKERS,MACRO_SERIES,_close_from_yfinance\n'
for n in ast.parse(s).body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)):model+=ast.get_source_segment(s,n)+'\n'
 elif isinstance(n,ast.FunctionDef) and n.name not in ['_metric_card','render_catalyst_calendar']:
  code=ast.get_source_segment(s,n)+'\n\n'
  if n.name in ['_timeline','_heatmap']:chart+=code
  elif n.name.startswith('_fetch_'):provider+=('@ttl_cache(seconds=900)\n' if n.name=='_fetch_market' else '@ttl_cache(seconds=3600)\n')+code.replace('threads=False)','threads=False, timeout=15)')
  else:
   if n.name=='_parse_custom_events':code=code.replace('text: str)', 'text: str, warnings: list | None = None)').replace('    cols =','    warnings = warnings if warnings is not None else []\n    cols =').replace('st.warning(', 'warnings.append(')
   model+=code
s=(r/'adfm_core/catalyst_calendar_official_page.py').read_text()
for n in ast.parse(s).body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.lineno<118 or isinstance(n,ast.FunctionDef):model+=ast.get_source_segment(s,n).replace('base.','')+'\n\n'
s=(r/'adfm_core/catalyst_calendar_exact_page.py').read_text()
for n in ast.parse(s).body:
 if isinstance(n,ast.FunctionDef) and n.name.startswith('_format_'):model+=ast.get_source_segment(s,n)+'\n\n'
for path,code in [('analytics/calendar.py',model),('data/calendar.py',provider),('charts/calendar.py',chart)]: (r/'adfm_engine'/path).write_text(code)
# Copy original composition calculations as ordinary Python; no UI interpreter.
body=s[s.index('    frames: List'):s.index('    base.render_page_header')].replace('base.','').replace('_dated_calendar(', '_official_dated_calendar(').replace('_parse_custom_events(custom_text)', '_parse_custom_events(custom_text,warnings)')
service='''from datetime import date,timedelta
from typing import List
import pandas as pd
from adfm_engine.analytics import calendar as model
from adfm_engine.analytics.calendar import *
from adfm_engine.data.calendar import _fetch_market,_fetch_macro
from adfm_engine.charts.calendar import _timeline,_heatmap
from adfm_engine.serialization import records,figure_json
'''
# Underscore helpers are imported explicitly, never runtime star discovery.
funcs=[n.name for n in ast.parse(model).body if isinstance(n,ast.FunctionDef) and n.name.startswith('_')]
service+='from adfm_engine.analytics.calendar import '+','.join(funcs)+'\n\n'
service+='def calendar(market,macro_panel,macro_status,today=None,horizon_days=90,include_macro=True,include_fed=True,hide_low=False,custom_text=""):\n    today=today or date.today()\n    warnings=[]\n    stress_bonus,stress_label=_market_stress(market)\n'+body
body=body.replace('if hide_low:', 'if hide_low and not calendar.empty:')
service+='''    if include_macro and (today < date(2026,9,1) or today+timedelta(days=horizon_days)>date(2026,12,31)):
        warnings.append('The original confirmed macro-date catalog covers September–December 2026, plus the January 27, 2027 FOMC meeting. Other scheduled macro dates require an update; calendar rules and custom events remain available.')
    result={'asof':today.isoformat(),'warnings':warnings,'cards':[],'timeline':None,'heatmap':None,'macro':records(_macro_prints(macro_panel)),'decision':[],'details':[],'status':records(macro_status[[c for c in ['key','symbol','provider','data_through','status'] if c in macro_status]]),'csv':''}
    if calendar.empty:return result
'''
start=s.index('    next_event =');end=s.index('    cols =');service+=s[start:end].replace('base.','')
fn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='render_catalyst_calendar')
calls=sorted([n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_metric_card'],key=lambda n:n.lineno)
service+='    result["cards"]=[\n'+',\n'.join('        ['+','.join(ast.get_source_segment(s,a).replace('base.','') for a in n.args)+']' for n in calls)+'\n    ]\n'
start=s.index('    decision =');end=s.index('    st.dataframe(decision');service+=s[start:end].replace('base.','')
start=s.index('        details =');end=s.index('        st.dataframe(',start);service+=textwrap.indent(textwrap.dedent(s[start:end]),'    ').replace('base.','')
service+='''    result.update(timeline=figure_json(_timeline(calendar,today)),heatmap=figure_json(_heatmap(perf)) if not perf.empty else None,decision=records(decision),details=records(details[['Date','When','Event','Type','Status','Source','Region','Risk Score','Cluster','Why It Matters','Exposure','Action']]),csv=calendar.to_csv(index=False))
    return result

def load_calendar(**kwargs):
    today=date.today()
    market=_fetch_market(min(date(today.year,1,1)-timedelta(days=10),today-timedelta(days=460)).isoformat())
    panel,status=_fetch_macro(date(today.year-3,1,1).isoformat(),today.isoformat())
    return calendar(market,panel,status,today=today,**kwargs)
'''
(r/'adfm_engine/calendar_service.py').write_text(service)
