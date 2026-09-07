"""Development extraction of CFTC analytics and presentation-free loaders."""
import ast,json,textwrap,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];cp='adfm_core/cftc_positioning.py';pp='pages/18_CFTC_Positioning_Monitor.py';core=(r/cp).read_text();page=(r/pp).read_text()
def src(text,n):
 lines=text.splitlines(keepends=True);start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
 return ''.join(lines[start:n.end_lineno])+'\n\n'
imports='from __future__ import annotations\nfrom datetime import date,timedelta\nfrom typing import Final,Mapping\nimport numpy as np\nimport pandas as pd\n'
model='"""Original CFTC positioning metrics and universes."""\n'+imports
provider='"""CFTC public reporting data retrieval and cached market overlays."""\n'+imports+'import requests\nfrom adfm_engine.analytics.cftc import *\nfrom adfm_engine.analytics.cftc import _all_position_fields\nfrom adfm_engine.cache import ttl_cache\nfrom adfm_engine.data.market import fetch_daily_ohlcv, adjusted_ohlcv\n\n'
for n in ast.parse(core).body:
 if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)):
  if isinstance(n,ast.FunctionDef) and n.name in {'_request','fetch_recent','fetch_contract_history'}:provider+=src(core,n)
  else:model+=src(core,n)
chart='"""Original CFTC chart transformations and hover behavior."""\n'+imports+'import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\nfrom adfm_engine.palette import EXCEL,PASTEL_20\nfrom adfm_engine.analytics.cftc import add_metrics,COHORTS\nPRICE_COLOR="#111111"\nPOSITION_COLOR=EXCEL["rose"]\nGRID_COLOR="rgba(127,140,141,0.20)"\n\n'
for n in ast.parse(page).body:
 if isinstance(n,ast.FunctionDef):
  if n.name.startswith('load_'):provider+=src(page,n).replace('@st.cache_data(ttl=21_600, show_spinner=False)','@ttl_cache(seconds=21600)').replace('@st.cache_data(ttl=3_600, show_spinner=False)','@ttl_cache(seconds=3600)')
  elif n.name in {'positioning_chart','cohort_chart'}:chart+=src(page,n)
  elif n.name!='compact_signal_table':model+=src(page,n)
model+='''def compact_signal_rows(frame,lookback_label):
    view=frame[["market","net_pct_oi","percentile","one_week_oi_shift","signal"]].copy()
    view.columns=["Market","Net % OI",f"{lookback_label} %ile","1W shift","Signal"]
    return view
'''
for path,content in [('adfm_engine/analytics/cftc.py',model),('adfm_engine/data/cftc.py',provider),('adfm_engine/charts/cftc.py',chart)]: (r/path).write_text(content)
revision='ac2bd39d8371e959c778437519d390e1891c1d08';(r/'native_tests/fixtures/cftc_baseline.json').write_text(json.dumps({'commit':revision,'sources':{p:subprocess.check_output(['git','show',f'{revision}:{p}'],cwd=r,text=True) for p in [cp,pp]}},indent=2))
