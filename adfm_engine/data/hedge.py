from __future__ import annotations
from dataclasses import dataclass
from datetime import date,timedelta
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
import yfinance as yf
from adfm_engine.cache import ttl_cache
@ttl_cache(seconds=900)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=4,
        timeout=15,
    )

