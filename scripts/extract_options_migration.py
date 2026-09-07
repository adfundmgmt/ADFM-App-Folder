"""One-time extraction of unchanged options models, loaders, and figures."""
import ast,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1]
page_path='pages/16_Options_Positioning_Compass.py';page=(r/page_path).read_text();definitions=page.split('st.set_page_config(')[0]
for a,b in [('adfm_core/options_positioning.py','adfm_engine/analytics/options_positioning.py'),('adfm_core/options_sources.py','adfm_engine/data/options_sources.py')]: (r/b).write_text((r/a).read_text())
header='from __future__ import annotations\nfrom datetime import date,datetime\nfrom typing import Mapping\nfrom zoneinfo import ZoneInfo\nimport numpy as np\nimport pandas as pd\n'
model='"""Original option-compass page transformations."""\n'+header+'from adfm_engine.data.market import adjusted_ohlcv,unique_tickers\n'
provider='"""Option calendars and chains, preserving Yahoo then Cboe fallback."""\n'+header+'import yfinance as yf\nfrom adfm_engine.cache import ttl_cache\nfrom adfm_engine.data.options_sources import expirations_from_cboe,fetch_cboe_delayed_options,select_cboe_expiry\n'
chart='"""Original compass, term structure, and fixed-moneyness surface."""\n'+header+'import plotly.graph_objects as go\nfrom adfm_engine.palette import PASTEL\nfrom adfm_engine.analytics.options_positioning import prepare_chain\n'
for n in ast.parse(definitions).body:
 if isinstance(n,ast.Assign):
  text=ast.get_source_segment(page,n)+'\n'
  if any(x in text for x in ['COLOR =','PASTEL[']):chart+=text
  else:model+=text
 elif isinstance(n,ast.FunctionDef):
  text=ast.get_source_segment(page,n)+'\n\n'
  if n.name in ['compass_chart','term_structure_chart','iv_surface_chart']:chart+=text
  elif n.name.startswith('fetch_') or n.name=='available_expirations':provider+=('@ttl_cache(seconds=900)\n' if n.decorator_list else '')+text
  else:model+=text
for p,s in [('adfm_engine/analytics/options.py',model),('adfm_engine/data/options.py',provider),('adfm_engine/charts/options.py',chart)]: (r/p).write_text(s)
revision='ac2bd39d8371e959c778437519d390e1891c1d08'
(r/'native_tests/fixtures/options_baseline.json').write_text(json.dumps({'commit':revision,'sources':{p:subprocess.check_output(['git','show',f'{revision}:{p}'],cwd=r,text=True) for p in [page_path,'adfm_core/options_positioning.py','adfm_core/options_sources.py']}},indent=2))
