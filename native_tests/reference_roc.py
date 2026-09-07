"""TEST ONLY: execute the frozen original page as an independent oracle.

Production never loads this file, the baseline source, or a UI compatibility
layer. The stub captures the original page's figure and injects fixture data.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SOURCE = json.loads((Path(__file__).parent / "fixtures/roc_baseline.json").read_text())["sources"]


class StopPage(Exception):
    pass


class Capture:
    def __init__(self, controls):
        self.controls = controls
        self.sidebar = self
        self.figure = None
        self.warnings = []

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def set_page_config(self, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def cache_data(self, **kwargs): return lambda fn: fn
    def columns(self, widths): return [self] * len(widths)
    def text_input(self, label, value): return self.controls[label]
    def toggle(self, label, value): return self.controls[label]
    def selectbox(self, label, options, index): return self.controls[label]
    def warning(self, message): self.warnings.append(message)
    def error(self, message): self.warnings.append(message)
    def stop(self): raise StopPage()
    def plotly_chart(self, figure, **kwargs): self.figure = figure


def original(frame, symbol="^SPX", window="3Y", roc="63D", view="Candlestick", inflections=True):
    capture = Capture({"Ticker symbol": symbol, "Analysis window": window, "ROC period": roc, "Chart view": view, "Show inflection markers": inflections})
    namespace = dict(st=capture, np=np, pd=pd, go=go, make_subplots=make_subplots, __name__="reference")
    noop = lambda *args, **kwargs: None
    for name in ("PageHeader", "render_footer", "render_page_header", "render_sidebar_about", "render_status_line"):
        namespace[name] = noop
    for source_name in ("adfm_core/palette.py", "adfm_core/rate_of_change.py"):
        exec(compile(SOURCE[source_name], source_name, "exec"), namespace)
    # Extract the ORIGINAL calendar helpers required by the quality report.
    market = ast.parse(SOURCE["adfm_core/market_data.py"])
    keep = {"unique_tickers", "benchmark_calendar", "stale_session_count"}
    tree = ast.Module(body=[n for n in market.body if isinstance(n, ast.FunctionDef) and n.name in keep], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "reference-market", "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    integrity = ast.parse(SOURCE["adfm_core/data_integrity.py"])
    integrity.body = [n for n in integrity.body if not isinstance(n, ast.ImportFrom) or not n.level]
    # Dataclasses inspect sys.modules; run reference objects under a real module.
    namespace["__name__"] = __name__
    exec(compile(integrity, "reference-integrity", "exec"), namespace)
    namespace["fetch_daily_ohlcv"] = lambda *args: ({symbol: frame.copy()}, pd.DataFrame())
    page = ast.parse(SOURCE["pages/12_Rate_of_Change_Regime_Explorer.py"])
    page.body = [n for n in page.body if not (isinstance(n, ast.Import) and any(a.name == "streamlit" for a in n.names)) and not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("adfm_core"))]
    try:
        exec(compile(page, "reference-roc-page", "exec"), namespace)
    except StopPage:
        pass
    return namespace, capture
