"""Copy the Streamlit ETF universe and pressure mathematics into native runtime."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'pages/14_ETF_Flow_Pressure_Proxy.py').read_text()
tree = ast.parse(source)
constants = {'etf_info', 'etf_tickers', 'US_EQUITY_TICKERS', 'INTERNATIONAL_EQUITY_TICKERS',
             'RATES_CREDIT_TICKERS', 'COMMODITY_TICKERS', 'FX_TICKERS', 'CRYPTO_VOL_TICKERS'}
functions = {'strip_tz_from_index', 'calc_start_date', 'last_friday', 'monday_of_week',
             'week_start_monday', 'business_day_gap', 'infer_asset_class', 'normalize_ohlcv',
             'compute_money_flow_proxy', 'compute_traded_value', 'compute_pressure_score',
             'compute_window_sum', 'classify_data_status', 'build_table'}
parts = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
        parts.append(ast.get_source_segment(source, node))
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        parts.append(ast.get_source_segment(source, node))
assert len(parts) == len(constants) + len(functions)
header = '''"""Extracted ETF coverage and price-volume flow mathematics from the Streamlit page."""
from datetime import date
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from adfm_engine.data.market import fetch_daily_ohlcv

def fetch_prices(tickers, start_date, end_date, cache_key):
    # The native market loader batches, retries, caches and retains missing symbols.
    days = (end_date - start_date).days
    period = "2y" if days > 370 else "1y"
    frames, _ = fetch_daily_ohlcv(tickers, period=period)
    return {ticker: normalize_ohlcv(frame.loc[frame.index >= pd.Timestamp(start_date)]
            if not frame.empty else frame) for ticker in tickers
            for frame in (frames.get(ticker, pd.DataFrame()),)}
'''
(root / 'adfm_engine/etf_flow_legacy_math.py').write_text(header + '\n\n' + '\n\n'.join(parts) + '\n')
