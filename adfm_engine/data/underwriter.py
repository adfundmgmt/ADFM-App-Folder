from __future__ import annotations
from typing import Any,Mapping,Optional
import pandas as pd
from adfm_engine.analytics.sec_fundamentals import *
from adfm_engine.data.sec import SecClient
from adfm_engine.cache import ttl_cache
from adfm_engine.data.market import fetch_daily_ohlcv
@ttl_cache(seconds=86400)
def load_ticker_map() -> Mapping[str, Any]:
    return SecClient().company_tickers()

@ttl_cache(seconds=900)
def load_company_facts(cik: int) -> Mapping[str, Any]:
    return SecClient().company_facts(cik)

@ttl_cache(seconds=900)
def load_submissions(cik: int) -> Mapping[str, Any]:
    return SecClient().submissions(cik)

@ttl_cache(seconds=900)
def market_history(
    ticker: str,
) -> tuple[pd.Series, Optional[float], Optional[pd.Timestamp]]:
    frames, _ = fetch_daily_ohlcv((ticker,), period="2y")
    frame = frames.get(ticker)
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype="float64"), None, None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return close, None, None
    return close, float(close.iloc[-1]), pd.Timestamp(close.index[-1]).normalize()

