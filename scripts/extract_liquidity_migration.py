"""Development-only extraction of audited legacy functions; not a runtime adapter."""
import ast,json,textwrap,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
basepath='adfm_core/_liquidity_tracker_base.py';pagepath='pages/3_Liquidity_Conditions_Monitor.py'
base=(root/basepath).read_text();page=(root/pagepath).read_text();bt=ast.parse(base);pt=ast.parse(page)
def source(text,n):
 lines=text.splitlines(keepends=True);start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
 return ''.join(lines[start:n.end_lineno])+'\n\n'
imports='from __future__ import annotations\nfrom typing import Dict, List, Mapping, Optional, Sequence, Tuple\nimport numpy as np\nimport pandas as pd\n'
constants='"""Original liquidity source definitions, weights and thresholds."""\n'+imports
for n in bt.body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.lineno<110:constants+=source(base,n)
(root/'adfm_engine/analytics/liquidity_definitions.py').write_text(constants)
model='"""Original liquidity formulas and scoring; no dynamic loading or UI."""\n'+imports+'from adfm_engine.analytics.liquidity_definitions import *\n\n'
data='"""Independent cached FRED, Fed FCI-G and market retrieval."""\n'+imports+'from io import BytesIO\nimport requests\nfrom adfm_engine.cache import ttl_cache\nfrom adfm_engine.data.primary import read_fred\nfrom adfm_engine.data.market import close_panel, fetch_daily_ohlcv\nfrom adfm_engine.analytics.liquidity_definitions import *\n\n'
data_names={'_normalize_fred_frame','fetch_fred_one','load_fred','load_market','fcig_column','load_fcig'}
charts='"""Original liquidity Plotly transformations, isolated from presentation."""\n'+imports+'import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\nfrom adfm_engine.palette import PASTEL\nfrom adfm_engine.analytics.liquidity_definitions import *\nfrom adfm_engine.analytics.liquidity import latest, filter_lookback\nBLUE=PASTEL["blue"]\nGREEN=PASTEL["sage"]\nORANGE=PASTEL["coral"]\nPURPLE=PASTEL["plum"]\n\n'
for n in bt.body:
 if isinstance(n,ast.FunctionDef) and n.lineno<539:
  if n.name in data_names:
   chunk=source(base,n).replace('@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)','@ttl_cache(seconds=21600)').replace('@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)','@ttl_cache(seconds=14400)').replace('@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)','@ttl_cache(seconds=86400)')
   chunk=chunk.replace('pdr.DataReader(series_id, "fred", start, end)','read_fred(series_id, start, end)').replace('pandas_datareader:','FRED CSV:').replace('Failed calls are not cached by Streamlit.','Failed calls are not cached.').replace('raw = pd.read_csv(BytesIO(response.content))','raw = pd.read_csv(BytesIO(response.content), index_col=0, parse_dates=True)' if n.name=='fetch_fred_one' else 'raw = pd.read_csv(BytesIO(response.content))')
   data+=chunk
  elif n.name=='plot_layout':charts+=source(base,n)
  else:model+=source(base,n)
for n in pt.body:
 if isinstance(n,ast.FunctionDef) and n.name in {'_read_bucket','_color_score'}:model+=source(page,n)
 if isinstance(n,ast.FunctionDef) and n.name=='_add_regime_bands':charts+=source(page,n)
model+='def compute_liquidity(fred, prices, z_window=756, min_periods=252, smoothing=3, lookback="5y"):\n'
model+='    primary, primary_specs = build_primary(fred)\n'
for n in pt.body:
 if 249<=n.lineno<=318:model+=textwrap.indent(source(page,n),'    ')
model+='    return locals()\n'
# Isolate original figure statements into explicit chart functions.
charts+='def main_chart(display_level, display_impulse):\n'
for n in pt.body:
 if 345<=n.lineno<=398:charts+=textwrap.indent(source(page,n),'    ')
charts+='    return fig_main\n\n'
fcig=next(n for n in pt.body if n.lineno==401);conditional=next(n for n in fcig.body if isinstance(n,ast.If))
charts+='def financial_conditions_chart(fcig, lookback):\n'
for n in conditional.orelse:
 if n.lineno<472:charts+=textwrap.indent(textwrap.dedent(source(page,n)),'    ')
charts+='    return fig_fcig\n\n'
drivers=next(n for n in pt.body if n.lineno==474)
class StripCharts(ast.NodeTransformer):
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='st':return None
  return n
nodes=[n for n in drivers.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='render_section_header')]
code=ast.unparse(ast.fix_missing_locations(StripCharts().visit(ast.Module(body=nodes,type_ignores=[]))))
charts+='def driver_charts(display_sleeve_impulses, primary_impulses):\n    fig_sleeves = fig_components = None\n'+textwrap.indent(code,'    ')+'\n    return fig_sleeves, fig_components\n'
for path,content in [('adfm_engine/analytics/liquidity.py',model),('adfm_engine/data/liquidity.py',data),('adfm_engine/charts/liquidity.py',charts)]: (root/path).write_text(content)
revision='ac2bd39d8371e959c778437519d390e1891c1d08'
(root/'native_tests/fixtures/liquidity_baseline.json').write_text(json.dumps({'commit':revision,'sources':{p:subprocess.check_output(['git','show',f'{revision}:{p}'],cwd=root,text=True) for p in [basepath,pagepath]}},indent=2))
