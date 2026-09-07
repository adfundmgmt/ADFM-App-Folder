"""Yahoo Treasury-yield data retrieval; preserves raw-yield conventions."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL, PASTEL_RATES_SCALE
from adfm_engine.cache import ttl_cache

def extract_close_frame(
    raw: pd.DataFrame, tickers: Tuple[str, ...]
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    if raw is None or raw.empty:
        return pd.DataFrame(), ("Yahoo download returned an empty frame.",)

    data = pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))

        if "Close" in level0:
            data = raw["Close"].copy()
        elif "Adj Close" in level0:
            data = raw["Adj Close"].copy()
        elif "Close" in level1:
            data = raw.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in level1:
            data = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame(), (
                "Yahoo frame did not include Close or Adj Close columns.",
            )
    else:
        close_col = (
            "Close"
            if "Close" in raw.columns
            else "Adj Close"
            if "Adj Close" in raw.columns
            else None
        )
        if close_col is None:
            return pd.DataFrame(), ("Yahoo frame did not include a Close column.",)
        data = raw[[close_col]].copy()
        if len(tickers) == 1:
            data.columns = [tickers[0]]

    data.columns = [str(c) for c in data.columns]
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()]
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.apply(pd.to_numeric, errors="coerce")

    for ticker in tickers:
        if ticker not in data.columns:
            diagnostics.append(f"{ticker}: missing from Yahoo close frame")

    present_cols = [ticker for ticker in tickers if ticker in data.columns]
    if not present_cols:
        return pd.DataFrame(), tuple(
            diagnostics + ["No requested tickers were present in the Yahoo close frame."]
        )

    return data[present_cols].dropna(how="all"), tuple(diagnostics)


@ttl_cache(seconds=900)
def fetch_yahoo_close(
    tickers: Tuple[str, ...], start_date: date, end_date: date
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), (
            f"yfinance import failed: {type(exc).__name__}: {exc}",
        )

    try:
        raw = yf.download(
            list(tickers),
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=12,
        )
    except Exception as exc:
        return pd.DataFrame(), (
            f"Yahoo download failed: {type(exc).__name__}: {exc}",
        )

    data, extract_diag = extract_close_frame(raw, tickers)
    diagnostics.extend(extract_diag)
    return data, tuple(diagnostics)


