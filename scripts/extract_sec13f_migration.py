"""Extract SEC data access and original corrected calculation functions."""
import ast
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'adfm_core/sec_13f.py').read_text()
s=s.replace('Path(__file__).resolve().parents[1] / "data" / "13f"','Path(os.getenv("ADFM_DATA_DIR", "/tmp/adfm-data")) / "13f"').replace('ADFM Analytics 13F Browser/1.0 (public-data research)','AD Fund Management LP aryadeniz@adfundmgmt.com')
s=s.replace('import json','import fcntl\nimport json').replace('descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n        os.close(descriptor)','descriptor = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)\n        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)').replace('except FileExistsError as exc:','except BlockingIOError as exc:\n        os.close(descriptor)').replace('lock_path.unlink(missing_ok=True)','fcntl.flock(descriptor, fcntl.LOCK_UN)\n        os.close(descriptor)')
# Drop obsolete uncorrected ranking; preserve the official archive and amendment model.
t=ast.parse(s);lines=s.splitlines(True)
for n in reversed(t.body):
 if isinstance(n,ast.FunctionDef) and n.name in ['rank_fund_exposure','_position_kind_mask']:del lines[n.lineno-1:n.end_lineno]
(r/'adfm_engine/data/sec13f.py').write_text(''.join(lines))
s=(r/'adfm_core/sec_13f_corrected.py').read_text();nodes=ast.parse(s).body
header='from __future__ import annotations\nimport re\nfrom typing import Sequence\nimport pandas as pd\nfrom adfm_engine.data import sec13f as base\n'
model=header; data=header+'from adfm_engine.data.sec13f import *\nfrom adfm_engine.analytics.sec13f import _value_multiplier,find_managers,rank_holdings,summarize_portfolio\n'
for n in nodes:
 if not isinstance(n,ast.FunctionDef) or n.lineno<57:continue
 code=ast.get_source_segment(s,n)+'\n\n'
 if n.name in ['_effective_holdings','_holdings_for_components']:
  if n.name=='_holdings_for_components':code=code.replace('return components, holdings','return holdings')
  data+=code
 elif n.name=='search_manager_candidates':
  code=code.replace('search_manager_candidates(', 'find_managers(').replace('prepared: PreparedDataset','filings: pd.DataFrame').replace('    filings = pd.read_parquet(prepared.filings_path)\n','');model+=code
 elif n.name=='rank_fund_exposure':
  code=code.replace('rank_fund_exposure(', 'rank_holdings(').replace('prepared: PreparedDataset,','components: pd.DataFrame,\n    all_holdings: pd.DataFrame,').replace('    components, all_holdings = _effective_holdings(prepared, report_period)\n','');model+=code
 elif n.name=='manager_portfolio':
  code=code.replace('manager_portfolio(', 'summarize_portfolio(').replace('prepared: PreparedDataset,','filings: pd.DataFrame,\n    holdings: pd.DataFrame,').replace('    filings = pd.read_parquet(prepared.filings_path)\n','').replace('manager_holdings = _holdings_for_components(\n        prepared, manager_components, report_period\n    )','manager_holdings = holdings.copy()');model+=code
 else:model+=code
(r/'adfm_engine/analytics/sec13f.py').write_text(model)
data+='''def search_manager_candidates(prepared,query,**kwargs):
    return find_managers(pd.read_parquet(prepared.filings_path),query,**kwargs)
def rank_fund_exposure(prepared,cusips,**kwargs):
    components,holdings=_effective_holdings(prepared,kwargs.get('report_period'))
    return rank_holdings(components,holdings,cusips,**kwargs)
def manager_portfolio(prepared,cik,report_period):
    filings=pd.read_parquet(prepared.filings_path)
    components=base.select_effective_filing_components(filings,report_period)
    if components.empty:return {},pd.DataFrame()
    components=components.copy()
    components['CIK']=components['CIK'].astype(str).str.zfill(10)
    components=components.loc[components['CIK'].eq(str(cik).zfill(10))]
    holdings=_holdings_for_components(prepared,components,report_period)
    return summarize_portfolio(filings,holdings,cik,report_period)
'''
(r/'adfm_engine/data/sec13f_queries.py').write_text(data)
s=(r/'adfm_core/sec_13f_browser.py').read_text();controls='from __future__ import annotations\nimport pandas as pd\nfrom adfm_engine.data.sec13f import QuarterDataset\n'
for n in ast.parse(s).body:
 if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in ['money_label','candidate_label','manager_candidate_label']:controls+=ast.get_source_segment(s,n)+'\n\n'
(r/'adfm_engine/analytics/sec13f_controls.py').write_text(controls)
n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='exposure_chart')
(r/'adfm_engine/charts/sec13f.py').write_text('import pandas as pd\nimport plotly.graph_objects as go\nfrom adfm_engine.palette import PASTEL\nfrom adfm_engine.analytics.sec13f_controls import SORT_OPTIONS\n'+ast.get_source_segment(s,n)+'\n')
s=(r/'tests/test_sec_13f_corrected.py').read_text().replace('adfm_core.sec_13f_corrected','adfm_engine.data.sec13f_queries').replace('adfm_core.sec_13f','adfm_engine.data.sec13f')
(r/'native_tests/test_sec13f_corrected_native.py').write_text(s)
p=r/'requirements-native.txt';s=p.read_text();p.write_text(s+'pyarrow==21.0.0\nbeautifulsoup4==4.13.5\n' if 'pyarrow' not in s else s)
