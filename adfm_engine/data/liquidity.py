"""Independent cached FRED, Fed FCI-G and market retrieval."""
from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from io import BytesIO
import requests
from adfm_engine.cache import ttl_cache
from adfm_engine.data.primary import read_fred
from adfm_engine.data.market import close_panel, fetch_daily_ohlcv
from adfm_engine.analytics.liquidity_definitions import *

def _normalize_fred_frame(raw: pd.DataFrame, series_id: str) -> pd.Series:
    if raw is None or raw.empty:
        raise ValueError("empty response")
    frame = raw.copy()
    if series_id in frame.columns:
        series = frame[series_id]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        date_col = frame.columns[0]
        value_col = frame.columns[-1]
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).set_index(date_col)
        series = frame[value_col]
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[series.index.notna()].sort_index()
    try:
        if series.index.tz is not None:
            series.index = series.index.tz_convert(None)
    except Exception:
        pass
    series = series[~series.index.duplicated(keep="last")].dropna().rename(series_id)
    if series.empty:
        raise ValueError("no numeric observations")
    return series


@ttl_cache(seconds=21600)
def fetch_fred_one(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch one FRED series sequentially. Failed calls are not cached."""
    errors: List[str] = []

    try:
        raw = read_fred(series_id, start, end)
        return _normalize_fred_frame(raw, series_id)
    except Exception as exc:
        errors.append(f"FRED CSV: {type(exc).__name__}: {exc}")

    try:
        response = requests.get(
            FRED_CSV_URL.format(series_id=series_id, start=start, end=end),
            headers={
                "User-Agent": "Mozilla/5.0 ADFM-Liquidity-Monitor/3.1",
                "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
            },
            timeout=(8, 45),
        )
        response.raise_for_status()
        raw = pd.read_csv(BytesIO(response.content), index_col=0, parse_dates=True)
        return _normalize_fred_frame(raw, series_id)
    except Exception as exc:
        errors.append(f"direct CSV: {type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors))


def load_fred(ids: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load series one at a time to avoid FRED throttling and retry transient failures on rerun."""
    end = pd.Timestamp.utcnow().date().isoformat()
    data: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    for series_id in ids:
        try:
            data[series_id] = fetch_fred_one(series_id, FRED_START, end)
        except Exception as exc:
            errors[series_id] = str(exc)

    if not data:
        return pd.DataFrame(), errors

    panel = pd.concat(data.values(), axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    business_index = pd.date_range(panel.index.min(), panel.index.max(), freq="B")
    panel = panel.reindex(business_index).ffill(limit=10)
    return panel.dropna(how="all"), errors


@ttl_cache(seconds=14400)
def load_market(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    frames, _ = fetch_daily_ohlcv(tickers, period=period)
    close = close_panel(frames, tickers, adjusted=True)
    if close.empty:
        return pd.DataFrame()
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.loc[close.index.notna()].sort_index()
    close = close.loc[:, ~close.columns.duplicated(keep="last")]
    valid = [column for column in close.columns if pd.to_numeric(close[column], errors="coerce").notna().sum() >= 90]
    return close[valid].apply(pd.to_numeric, errors="coerce")


def fcig_column(frame: pd.DataFrame) -> Optional[str]:
    numeric = []
    for column in frame:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() >= 12:
            numeric.append(column)
    for term in ("fci-g index", "fci-g", "fcig", "fci_g", "fci g"):
        for column in numeric:
            lower = str(column).lower()
            if term in lower and "cont" not in lower:
                return column
    return numeric[0] if numeric else None


@ttl_cache(seconds=86400)
def load_fcig() -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}
    for label, url in FCIG_URLS.items():
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 ADFM-Liquidity-Monitor/3.1"}, timeout=(8, 45))
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content))
            date_col = next((column for column in frame if any(term in str(column).lower() for term in ("date", "month", "time"))), frame.columns[0])
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
            value_col = fcig_column(frame)
            if value_col is None:
                errors[label] = "No numeric FCI-G column."
            else:
                frames.append(frame[[value_col]].rename(columns={value_col: label}))
        except Exception as exc:
            errors[label] = str(exc)
    return (pd.concat(frames, axis=1).sort_index().dropna(how="all"), errors) if frames else (pd.DataFrame(), errors)


