"""Extract the rotation math from the Streamlit page without its UI runtime."""
import ast
from pathlib import Path

page = Path('pages/7_Sector_Breadth_and_Rotation.py')
module = ast.parse(page.read_text())
sections = []
for node in module.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != 'style_snapshot_table':
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        sections.append(node)
pastels = {
    'blue': '#9db8e0', 'coral': '#e8b79e', 'sage': '#a7cda8',
    'rose': '#dba8ae', 'teal': '#9accca', 'amber': '#e8d39b',
    'lavender': '#b8addd', 'salmon': '#e5b2a9', 'clay': '#c9b4a4',
    'cornflower': '#a2bce4', 'slate_blue': '#adbaca',
}
config = Path('adfm_sector_rotation_config.py').read_text().replace(
    'from adfm_core.palette import PASTEL', f'PASTEL = {pastels!r}')
Path('adfm_engine/sector_rotation_config.py').write_text(config)
header = '''"""Rotation formulas extracted from pages/7_Sector_Breadth_and_Rotation.py.
Regenerate with python scripts/extract_sector_rotation_engine.py after editing source math."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from adfm_engine.sector_rotation_config import *

'''
body = '\n\n'.join(ast.unparse(node) for node in sections) + '\n'
for old,new in {'#1f77b4':'#8ba9d5','#2ca02c':'#a6d9bd','#9467bd':'#b7a6d5'}.items():
    body = body.replace(old,new)
Path('adfm_engine/sector_rotation_legacy_math.py').write_text(header+body)
