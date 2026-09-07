from __future__ import annotations
from datetime import date,timedelta
from typing import Dict,List
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
import yfinance as yf
from adfm_engine.cache import ttl_cache
@ttl_cache(seconds=900)
def load_prices(tickers: List[str], start: date) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=4,
        timeout=15,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
            elif (ticker, "Adj Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Adj Close")], errors="coerce")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if col and tickers:
            out[tickers[0]] = pd.to_numeric(raw[col], errors="coerce")

    df = pd.DataFrame(out)
    if df.empty:
        return df

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx.normalize()
    return df.sort_index().groupby(level=0).last()
