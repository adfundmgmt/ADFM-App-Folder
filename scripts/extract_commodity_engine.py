"""Extract unchanged event-study calculations from the Streamlit page for the native API."""
import ast
from pathlib import Path

source = Path('adfm_core/commodity_event_study_page.py').read_text()
module = ast.parse(source)
functions = {'_flatten_yfinance_columns', 'load_contract_history', '_rsi', '_window_label',
             'build_signal', 'detect_events', 'build_event_observations',
             'summarize_forward_performance', '_format_signal_value', 'make_price_chart',
             '_history_display'}
constants = {'COMMODITY_GROUPS', 'CONTRACT_LABEL_TO_SYMBOL', 'CONTRACT_SYMBOL_TO_NAME',
             'CONTRACT_OPTIONS', 'RETURN_WINDOWS', 'FORWARD_HORIZONS',
             'SPACING_OPTIONS', 'LOOKBACK_OPTIONS'}
selected = []
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        node.decorator_list = []
        selected.append(node)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)] if isinstance(node, ast.Assign) else [node.target.id] if isinstance(node.target, ast.Name) else []
        if any(name in constants for name in names):
            selected.append(node)
    elif isinstance(node, ast.For) and '_group, _contracts' in ast.unparse(node.target):
        selected.append(node)
output = '''"""Commodity event study calculations extracted from adfm_core/commodity_event_study_page.py.
Regenerate with python scripts/extract_commodity_engine.py after changing source math."""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

'''+ '\n\n'.join(ast.unparse(node) for node in selected)+'\n'
output = output.replace('"#2f7fd1"', '"#8ba9d5"').replace('"#e52822"', '"#d99d9c"')
Path('adfm_engine/commodity_legacy_math.py').write_text(output)
