import ast,textwrap
from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'pages/19_Market_Stress_Composite.py').read_text();tree=ast.parse(s)
header='from __future__ import annotations\nfrom datetime import date,timedelta\nfrom typing import Dict,List\nimport numpy as np\nimport pandas as pd\nfrom adfm_engine.palette import PASTEL\n'
model=header
for n in tree.body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.lineno<85:model+=ast.get_source_segment(s,n)+'\n'
 elif isinstance(n,ast.FunctionDef) and n.name!='load_prices':model+=ast.get_source_segment(s,n).replace('def choose_target()', 'def choose_target(target_mode,px,risk_score,dislocation_score)')+'\n\n'
chunk=s[s.index('calendar = px[SPX]'):s.index('with st.sidebar:',s.index('signal_age ='))]
t=ast.parse(chunk);t.body=[n for n in t.body if not isinstance(n,ast.FunctionDef)];chunk=ast.unparse(t).replace('st.warning(', 'warnings.append(').replace('choose_target()', 'choose_target(target_mode,px,risk_score,dislocation_score)')
model+='def compute_stress(px,z_window_years=3,smooth_days=10,target_mode="Auto"):\n    warnings=[]\n'+textwrap.indent(chunk,'    ')+'\n    return {key:value for key,value in locals().items() if key not in ("key","value")}\n'
chunk=s[s.index('rows = []',s.index('# ---------------- Main table')):s.index('    styled_moves =')]
model+='\ndef market_moves(px,eq_cols,carry_cols,haven_cols,bond_cols,z_window):\n'+textwrap.indent(chunk,'    ')+'    return moves\n'
(r/'adfm_engine/analytics/stress.py').write_text(model)
chart=header+'import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\nfrom adfm_engine.analytics.stress import *\n'
chunk=s[s.index('cutoff = pd.Timestamp'):s.index('st.plotly_chart(')]
chart+='\ndef stress_chart(target_px,target_label,risk_score,dislocation_score,onset_dates,lookback_years,today):\n'+textwrap.indent(chunk.replace('date.today()', 'today'),'    ')+'    return fig\n'
(r/'adfm_engine/charts/stress.py').write_text(chart)
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_prices')
(r/'adfm_engine/data/stress.py').write_text(header+'import yfinance as yf\nfrom adfm_engine.cache import ttl_cache\n@ttl_cache(seconds=900)\n'+ast.get_source_segment(s,n).replace('threads=True,','threads=4,\n        timeout=15,')+'\n')
