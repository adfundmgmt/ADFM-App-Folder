"""Development-only extraction; production contains ordinary Python functions."""
import ast
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'adfm_core/sec_fundamentals.py').read_text(); nodes=ast.parse(s).body
client=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name=='SecClient')
lines=s.splitlines(True); core=''.join(lines[:client.lineno-1]+lines[client.end_lineno:])
core=core.replace('return pd.DataFrame(rows).sort_values("Metric").reset_index(drop=True)','return pd.DataFrame(rows).sort_values("Metric").reset_index(drop=True) if rows else pd.DataFrame()')
core=core.replace('import os\n','').replace('import time\n','').replace('import requests\n','')
(r/'adfm_engine/analytics/sec_fundamentals.py').write_text(core)
(r/'adfm_engine/data/sec.py').write_text('from __future__ import annotations\nimport os,time,requests\nfrom typing import Any,Mapping\nfrom adfm_engine.analytics.sec_fundamentals import SecDataError,SEC_DATA_BASE,SEC_TICKER_URL,DEFAULT_SEC_USER_AGENT\n'+ast.get_source_segment(s,client)+'\n')
p=(r/'pages/9_ADFM_Underwriter.py').read_text(); header='from __future__ import annotations\nfrom typing import Any,Mapping,Optional\nimport pandas as pd\nfrom adfm_engine.analytics.sec_fundamentals import *\n'
model=header; chart=header+'import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\nfrom adfm_engine.palette import PASTEL\nfrom adfm_engine.analytics.underwriter import currency_prefix\n'
provider=header+'from adfm_engine.data.sec import SecClient\nfrom adfm_engine.cache import ttl_cache\nfrom adfm_engine.data.market import fetch_daily_ohlcv\n'
for n in ast.parse(p).body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.lineno<837: model+=ast.get_source_segment(p,n)+'\n'
 if isinstance(n,ast.FunctionDef) and not n.name.startswith('render_'):
  code=ast.get_source_segment(p,n)+'\n\n'
  if n.name in ['quarterly_chart','price_history_chart']:chart+=code
  elif n.name.startswith('load_') or n.name=='market_history': provider+=('@ttl_cache(seconds=86400)\n' if n.name=='load_ticker_map' else '@ttl_cache(seconds=900)\n')+code
  else:model+=code
for path,code in [('analytics/underwriter.py',model),('charts/underwriter.py',chart),('data/underwriter.py',provider)]: (r/'adfm_engine'/path).write_text(code)
(r/'native_tests/test_sec_fundamentals_native.py').write_text((r/'tests/test_sec_fundamentals.py').read_text().replace('adfm_core.sec_fundamentals','adfm_engine.analytics.sec_fundamentals'))
