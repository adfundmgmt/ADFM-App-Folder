"""Reusable, deterministic daily market-data helpers for ADFM Streamlit pages.

The module preserves raw provider observations, reports non-fatal failures,
and keeps benchmark-calendar alignment separate from any optional filling.
It is suitable for both price-only dashboards and OHLCV technical analysis.
"""

from __future__ import annotations

import time
import tempfile
from dataclasses import dataclass
from datetime import datetime
from datetime import time as clock_time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from .observability import record_data_load


@dataclass(frozen=True)
class MarketDataConfig:
    """Provider settings shared by daily-equity pages."""

    cache_ttl_seconds: int = 3_600
    chunk_size: int = 40
    retries: int = 3
    retry_base_seconds: float = 0.75
    completed_session_cutoff: clock_time = clock_time(16, 15)


DEFAULT_CONFIG = MarketDataConfig()
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def configure_yfinance_cache() -> None:
    """Point yfinance's timezone database at a writable shared temp folder."""
    try:
        cache_dir = Path(tempfile.gettempdir()) / "adfm-yfinance-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def unique_tickers(tickers: Iterable[str]) -> Tuple[str, ...]:
    """Normalize, de-duplicate, and preserve the input ticker order."""
    return tuple(
        dict.fromkeys(
            ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()
        )
    )


def canonicalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, timezone-naive daily index with duplicate dates removed."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index().loc[~out.index.duplicated(keep="last")]


def drop_unfinished_daily_session(
    frame: pd.DataFrame,
    now: Optional[datetime] = None,
    cutoff: clock_time = DEFAULT_CONFIG.completed_session_cutoff,
) -> pd.DataFrame:
    """Exclude today's bar before the US cash-session close is settled.

    Passing ``now`` makes this policy easy to test.  The function never
    modifies historical data and does not infer missing sessions.
    """
    if frame.empty:
        return frame.copy()
    current = now or datetime.now(ZoneInfo("America/New_York"))
    latest = pd.Timestamp(frame.index[-1]).date()
    if latest == current.date() and current.timetz().replace(tzinfo=None) < cutoff:
        return frame.iloc[:-1].copy()
    return frame.copy()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide aligned series without generating infinite values."""
    return numerator.div(denominator.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def percent_change(series: pd.Series, periods: int) -> float:
    """Latest point-to-point return, or NaN when the requested window is thin."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-(periods + 1)] - 1.0)


def benchmark_calendar(
    raw_frames: Mapping[str, pd.DataFrame], benchmark: str
) -> pd.DatetimeIndex:
    """Return actual observed benchmark sessions, not a synthetic business calendar."""
    frame = raw_frames.get(benchmark)
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DatetimeIndex([])
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    return pd.DatetimeIndex(close.index).sort_values().unique()


def stale_session_count(
    frame: pd.DataFrame, benchmark_sessions: pd.DatetimeIndex
) -> int:
    """Number of observed benchmark sessions after a security's latest close."""
    if frame.empty or "Close" not in frame or len(benchmark_sessions) == 0:
        return len(benchmark_sessions)
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return len(benchmark_sessions)
    return int((benchmark_sessions > close.index.max()).sum())


def align_to_benchmark_calendar(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    forward_fill_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Align a frame to benchmark sessions.

    Filling is opt-in and should only be used for ratio-style analysis.  Do
    not use it for OHLCV patterns, volume, gaps, ATR, or breakout signals.
    """
    if frame.empty:
        return pd.DataFrame(index=sessions)
    out = canonicalize_date_index(frame).reindex(sessions)
    if forward_fill_limit is not None:
        out = out.ffill(limit=forward_fill_limit)
    return out


def adjusted_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive split/dividend-adjusted OHLCV while retaining raw data upstream."""
    out = raw.copy()
    if out.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(
        out.columns
    ):
        return out
    close = pd.to_numeric(out["Close"], errors="coerce")
    adjusted_close = pd.to_numeric(out.get("Adj Close", close), errors="coerce")
    factor = safe_divide(adjusted_close, close)
    if factor.notna().any():
        for column in ("Open", "High", "Low", "Close"):
            out[column] = pd.to_numeric(out[column], errors="coerce") * factor
        out["Volume"] = safe_divide(
            pd.to_numeric(out["Volume"], errors="coerce"), factor
        )
    out["Adj Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _extract_ohlcv(
    raw: pd.DataFrame, tickers: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """Normalize both yfinance single- and multi-ticker response layouts."""
    frames: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return frames
    for ticker in tickers:
        columns: Dict[str, pd.Series] = {}
        for field in OHLCV_COLUMNS:
            series: Optional[pd.Series] = None
            if isinstance(raw.columns, pd.MultiIndex):
                if (field, ticker) in raw.columns:
                    series = raw[(field, ticker)]
                elif (ticker, field) in raw.columns:
                    series = raw[(ticker, field)]
            elif len(tickers) == 1 and field in raw.columns:
                series = raw[field]
            if series is not None:
                columns[field] = pd.to_numeric(series, errors="coerce")
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(columns):
            frame = canonicalize_date_index(pd.DataFrame(columns))
            frame = drop_unfinished_daily_session(frame)
            if not frame.empty and frame["Close"].notna().any():
                frames[ticker] = frame
    return frames


def _download_chunk(
    tickers: Sequence[str], period: str, config: MarketDataConfig
) -> Dict[str, pd.DataFrame]:
    for attempt in range(config.retries):
        try:
            raw = yf.download(
                tickers=list(tickers),
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                group_by="column",
                threads=True,
            )
            frames = _extract_ohlcv(raw, tickers)
            if frames:
                return frames
        except Exception:
            # Pages receive an explicit per-symbol failure instead of a hard stop.
            pass
        if attempt + 1 < config.retries:
            time.sleep(config.retry_base_seconds * (attempt + 1))
    return {}


@st.cache_data(ttl=DEFAULT_CONFIG.cache_ttl_seconds, show_spinner=False)
def fetch_daily_ohlcv(
    tickers: Tuple[str, ...], period: str = "3y"
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch raw daily OHLCV in batches with retries and individual fallback.

    The returned frames are raw provider observations.  The diagnostics table
    permits a page to continue rendering even when individual symbols fail.
    """
    symbols = unique_tickers(tickers)
    frames: Dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), DEFAULT_CONFIG.chunk_size):
        frames.update(
            _download_chunk(
                symbols[start : start + DEFAULT_CONFIG.chunk_size],
                period,
                DEFAULT_CONFIG,
            )
        )
    for ticker in (symbol for symbol in symbols if symbol not in frames):
        frames.update(_download_chunk([ticker], period, DEFAULT_CONFIG))
    missing = pd.DataFrame(
        {
            "Ticker": [ticker for ticker in symbols if ticker not in frames],
            "Reason": "No valid OHLCV data returned",
        }
    )
    record_data_load("Yahoo Finance", frames, symbols)
    return frames, missing


def close_panel(
    raw_frames: Mapping[str, pd.DataFrame],
    tickers: Sequence[str],
    *,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Build a wide close panel without filling observations."""
    series: Dict[str, pd.Series] = {}
    for ticker in unique_tickers(tickers):
        frame = raw_frames.get(ticker)
        if frame is None or frame.empty:
            continue
        source = adjusted_ohlcv(frame) if adjusted else frame
        if "Close" in source:
            series[ticker] = pd.to_numeric(source["Close"], errors="coerce")
    return canonicalize_date_index(pd.DataFrame(series)) if series else pd.DataFrame()
