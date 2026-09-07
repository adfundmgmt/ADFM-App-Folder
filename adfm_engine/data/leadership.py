from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.palette import PASTEL
def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()

def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = clean_ticker(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output

def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])

@ttl_cache(seconds=3600)
def fetch_closes(tickers: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    ticker_list = unique_keep_order(tickers)
    if not ticker_list:
        return pd.DataFrame()

    def extract_close(block: pd.DataFrame) -> Optional[pd.Series]:
        if block is None or block.empty:
            return None
        for field in ("Close", "Adj Close"):
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce")
        numeric = block.select_dtypes(include=[np.number])
        return None if numeric.empty else pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    def normalize(raw: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        output = pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            level_0 = set(raw.columns.get_level_values(0).astype(str)).intersection(requested)
            level_1 = set(raw.columns.get_level_values(1).astype(str)).intersection(requested)
            if level_0:
                for ticker in requested:
                    if ticker not in level_0:
                        continue
                    try:
                        close = extract_close(raw[ticker])
                        if close is not None:
                            output[ticker] = close
                    except Exception:
                        continue
            elif level_1:
                for ticker in requested:
                    for field in ("Close", "Adj Close"):
                        try:
                            if (field, ticker) in raw.columns:
                                output[ticker] = pd.to_numeric(raw[(field, ticker)], errors="coerce")
                                break
                        except Exception:
                            continue
        elif len(requested) == 1:
            close = extract_close(raw)
            if close is not None:
                output[requested[0]] = close

        if output.empty:
            return output
        output.index = pd.to_datetime(output.index).tz_localize(None)
        output = output.sort_index()
        output = output.loc[~output.index.duplicated(keep="last")]
        return output.ffill().dropna(how="all")

    frames = []
    for batch in chunked(ticker_list, 30):
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = normalize(raw, batch)
            if not normalized.empty:
                frames.append(normalized)
                continue
        except Exception:
            pass
        try:
            raw = yf.download(
                tickers=batch,
                period="max",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = normalize(raw, batch)
            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, axis=1)
    return output.loc[:, ~output.columns.duplicated()].sort_index().ffill().dropna(how="all")
