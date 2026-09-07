"""Frozen original formulas/figures and complete native issuer output regression."""
import ast,json,subprocess
from pathlib import Path
import pandas as pd
import pytest
from adfm_engine.underwriter_service import underwriter
from adfm_engine.analytics.sec_fundamentals import CompanyIdentity,source_audit_table
from native_tests.test_sec_fundamentals_native import company_facts_payload
ROOT=Path(__file__).resolve().parents[1]
BASE='ac2bd39d8371e959c778437519d390e1891c1d08'

def test_original_functions_preserved():
    original=ast.parse(subprocess.check_output(['git','show',BASE+':pages/9_ADFM_Underwriter.py'],cwd=ROOT,text=True))
    functions={n.name:ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef)}
    for path in ['analytics/underwriter.py','charts/underwriter.py']:
        for n in ast.parse((ROOT/'adfm_engine'/path).read_text()).body:
            if isinstance(n,ast.FunctionDef):assert ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False)==functions[n.name]

@pytest.mark.parametrize('currency',['USD','EUR','JPY'])
@pytest.mark.parametrize('has_price',[True,False])
def test_complete_underwrite(currency,has_price):
    facts=company_facts_payload()
    for taxonomy in facts['facts'].values():
        for concept in taxonomy.values():
            for unit in list(concept['units']):
                if unit.startswith('USD'):concept['units'][unit.replace('USD',currency)]=concept['units'].pop(unit)
    close=pd.Series(range(1,501),index=pd.bdate_range('2024-01-01',periods=500)) if has_price else pd.Series(dtype=float)
    result=underwriter(CompanyIdentity(1,'TEST','Test'),facts,{},close)
    assert len(result['snapshot_cards'])==17
    assert len(result['valuation_cards'])==(12 if currency=='USD' else 0)
    assert bool(result['price'])==has_price
    assert len(result['quarterly'])==4 and len(result['audit'])==21
    assert len(result['cards'])==6
    json.dumps(result,allow_nan=False)

def test_empty_audit():assert source_audit_table({}).empty
