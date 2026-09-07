from __future__ import annotations
from datetime import date,timedelta
from io import StringIO
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from adfm_engine.data.registry import SeriesDefinition
from adfm_engine.palette import PASTEL,PASTEL_DIVERGING_SCALE
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.data.primary import fetch_fred_series
from adfm_engine.analytics.calendar import MARKET_TICKERS,MACRO_SERIES,_close_from_yfinance
@ttl_cache(seconds=900)
def _fetch_market(start_iso: str) -> pd.DataFrame:
    try:
        raw = yf.download(list(MARKET_TICKERS), start=start_iso, interval="1d", auto_adjust=True, progress=False, threads=False, timeout=15)
    except Exception:
        return pd.DataFrame()
    return _close_from_yfinance(raw)

@ttl_cache(seconds=3600)
def _fetch_macro(start_iso: str, end_iso: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_fred_series(MACRO_SERIES, start=start_iso, end=end_iso)

