"""Copy the Streamlit volume signal mathematics into the isolated native runtime."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'pages/15_Volume_Based_Sentiment_Indicator.py').read_text()
tree = ast.parse(source)
names = {'safe_numeric', 'normalize_dt_index', 'validate_ohlcv',
         'classify_state', 'classify_setup', 'setup_color',
         'compute_forward_outcomes', 'compute_volume_framework',
         'volume_bar_color', 'build_chart', 'build_recent_events',
         'build_setup_outcomes'}
functions = [ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
regime = (root / 'adfm_core/regime_math.py').read_text()
regime_tree = ast.parse(regime)
rolling = next(ast.get_source_segment(regime, node) for node in regime_tree.body
               if isinstance(node, ast.FunctionDef) and node.name == 'rolling_percentile_previous')
header = '''"""Extracted causal volume math from the Streamlit tool. Regenerate with scripts/extract_volume_sentiment_engine.py."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
PASTEL_GREEN = '#88bca0'
PASTEL_RED = '#e8aaaa'
PASTEL_GREY = '#a7b6c7'
AMBER = '#e3c391'
BLUE = '#a4bee6'
'''
(root / 'adfm_engine/volume_sentiment_legacy_math.py').write_text(header + '\n\n' + rolling + '\n\n' + '\n\n'.join(functions) + '\n')
