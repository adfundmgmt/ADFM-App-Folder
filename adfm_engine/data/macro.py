"""Macro page data retrieval with the original cache lifetimes."""
from __future__ import annotations
from datetime import timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
from adfm_engine.cache import ttl_cache
from adfm_engine.data.market import close_panel, fetch_daily_ohlcv
from adfm_engine.data.primary import fetch_fred_series

@ttl_cache(seconds=900)
def fetch_market_prices(tickers: Tuple[str, ...]) -> Tuple[pd.DataFrame, List[str]]:
    frames, diagnostics = fetch_daily_ohlcv(tickers, period="5y")
    close = close_panel(frames, tickers, adjusted=True)
    if close.empty:
        return pd.DataFrame(), list(tickers)
    ordered = [ticker for ticker in tickers if ticker in close.columns]
    close = close.reindex(columns=ordered).dropna(axis=1, how="all").dropna(how="all")
    failed = diagnostics["Ticker"].astype(str).tolist() if not diagnostics.empty else []
    return close, failed


@ttl_cache(seconds=1800)
def fetch_macro_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_fred_series(start="2015-01-01")


