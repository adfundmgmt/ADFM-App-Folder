"""Reusable, deterministic daily market-data helpers for ADFM Streamlit pages.

The module preserves raw provider observations, reports non-fatal failures,
and keeps benchmark-calendar alignment separate from any optional filling.
It is suitable for both price-only dashboards and OHLCV technical analysis.
"""

from __future__ import annotations

import hashlib
import pickle
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from datetime import time as clock_time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
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
    request_timeout_seconds: float = 10.0
    completed_session_cutoff: clock_time = clock_time(16, 15)


DEFAULT_CONFIG = MarketDataConfig()
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def configure_yfinance_cache() -> None:
    """Point yfinance's timezone and cookie databases at a writable folder."""
    try:
        cache_dir = Path(tempfile.gettempdir()) / "adfm-yfinance-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        # Newer yfinance releases also persist Yahoo cookies through a separate
        # SQLite cache. Redirect both stores so read-only deployments work.
        if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
            yf.cache.set_cache_location(str(cache_dir))
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
        # Daily labels represent the exchange's date, not a UTC instant.
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out.loc[out.index.notna()]
    return out.loc[~out.index.duplicated(keep="last")].sort_index()


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
    if current.tzinfo is not None:
        current = current.astimezone(ZoneInfo("America/New_York"))
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
    clean = pd.to_numeric(series, errors="coerce")
    if periods < 1 or len(clean) <= periods:
        return np.nan
    latest, previous = clean.iloc[-1], clean.iloc[-(periods + 1)]
    if pd.isna(latest) or pd.isna(previous) or not np.isfinite([latest, previous]).all() or previous == 0:
        return np.nan
    return float(latest / previous - 1.0)


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
        out = fill_short_calendar_gaps(out, limit=forward_fill_limit)
    return out


def adjusted_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Adjust Yahoo OHLC prices while preserving provider share volume.

    Yahoo already handles splits in its historical share volume. The adjusted
    close ratio also includes dividends and must not be used to rescale volume.
    """
    out = raw.copy()
    if out.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(
        out.columns
    ):
        return out
    close = pd.to_numeric(out["Close"], errors="coerce")
    adjusted_close = pd.to_numeric(out.get("Adj Close", close), errors="coerce")
    factor = safe_divide(adjusted_close, close)
    for column in ("Open", "High", "Low", "Close"):
        out[column] = pd.to_numeric(out[column], errors="coerce") * factor
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")
    out["Adj Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def fill_short_calendar_gaps(frame: pd.DataFrame, limit: int = 2) -> pd.DataFrame:
    """Bridge short interior close-price gaps without extending stale endpoints.

    This is for explicitly aligned ratio calendars, never OHLCV or volume.
    Each series retains its first and last actual provider observation dates.
    """
    clean = canonicalize_date_index(frame).replace([np.inf, -np.inf], np.nan)
    observed = clean.notna()
    interior = observed.cummax() & observed.iloc[::-1].cummax().iloc[::-1]
    return clean.ffill(limit=limit).where(interior)


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
            if not frame.empty and frame["Close"].notna().any():
                frames[ticker] = frame
    return frames


def is_us_cash_symbol(symbol: str) -> bool:
    """Yahoo unsuffixed listings trade in US cash sessions; indices are explicit.

    Foreign exchange suffixes, futures/FX '=...' and crypto '-USD' symbols
    have independent sessions. Index caret symbols require a known US map.
    """
    symbol = symbol.strip().upper()
    if symbol.startswith("^"):
        return symbol in {"^SPX", "^GSPC", "^NDX", "^DJI", "^RUT", "^VIX", "^VIX9D", "^VIX3M", "^VVIX", "^VXN"}
    return bool(symbol) and "." not in symbol and "=" not in symbol and not symbol.endswith(("-USD", "-EUR", "-BTC"))


def _last_completed_us_session(now: Optional[datetime] = None) -> Optional[pd.Timestamp]:
    """Calendar epoch shared by daily filtering and provider-cache freshness."""
    import exchange_calendars as xcals

    current = pd.Timestamp(now or datetime.now(ZoneInfo("America/New_York")))
    current = current.tz_localize("America/New_York") if current.tzinfo is None else current.tz_convert("America/New_York")
    calendar = xcals.get_calendar("XNYS")
    # Only recent scheduling is necessary to prune provisional daily endpoints.
    schedule = calendar.schedule.loc[str((current - pd.Timedelta(days=14)).date()):str(current.date())]
    completed = schedule.loc[schedule["close"] + pd.Timedelta(minutes=15) <= current.tz_convert("UTC")]
    if completed.empty:
        return None
    return pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()


def completed_daily_observations(frame: pd.DataFrame, symbol: str, *, now: Optional[datetime] = None) -> pd.DataFrame:
    """Known US cash symbols use actual NYSE closes plus a 15-minute buffer.

    Unknown exchange/session conventions retain provider bars with an explicit
    completion limitation. No US clock is applied to foreign or futures bars.
    """
    out = frame.copy()
    out.attrs["completion_policy"] = "unverified_exchange_close"
    if not is_us_cash_symbol(symbol) or out.empty:
        return out
    last_complete = _last_completed_us_session(now)
    if last_complete is None:
        return out.iloc[:0]
    out = out.loc[out.index <= last_complete].copy()
    out.attrs["completion_policy"] = "XNYS_actual_close_plus_15_minutes"
    return out


def fetch_daily_ohlcv(
    tickers: Tuple[str, ...], period: str = "3y", *, completed_only: bool = True, now: Optional[datetime] = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Shared daily data; complete known US cash bars, retain other exchanges.

    completed_only=False returns raw observations for caller-owned exchange
    policies. Unknown markets carry an explicit per-frame completion label.
    """
    symbols = unique_tickers(tickers)
    raw = download_market_data(symbols, period=period, interval="1d", auto_adjust=False, group_by="column", threads=True, now=now)
    frames = _extract_ohlcv(raw, symbols)
    for symbol, frame in frames.items():
        frames[symbol] = completed_daily_observations(frame, symbol, now=now) if completed_only else frame
        health = dict(raw.attrs.get("market_data_health", {}).get(symbol, {}))
        close = frames[symbol]["Close"].dropna()
        health["as_of"] = pd.Timestamp(close.index[-1]).date().isoformat() if len(close) else None
        if not len(close):
            health["status"] = "unavailable"
        frames[symbol].attrs["market_data_health"] = health
        frames[symbol].attrs["auto_adjust"] = False
        if not completed_only:
            frames[symbol].attrs["completion_policy"] = "raw_observations"
    frames = {symbol: frame for symbol, frame in frames.items() if _usable(frame)}
    missing = pd.DataFrame({"Ticker": [symbol for symbol in symbols if symbol not in frames], "Reason": "No valid OHLCV data returned"})
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


def required_inputs_fresh(
    observed: pd.DataFrame,
    required: Sequence[str],
    *,
    now: Optional[datetime] = None,
    max_age_days: int = 4,
) -> bool:
    """Require actual values at the panel endpoint and bounded calendar age.

    Four calendar days permits ordinary weekends. Exchange-specific closures
    may require an explicit policy override; missing endpoints never pass.
    Call this on raw observations, before ratio-only interior alignment.
    """
    if observed.empty or not required or any(column not in observed for column in required):
        return False
    endpoint = pd.Timestamp(observed.index[-1]).tz_localize(None)
    current = pd.Timestamp(now or datetime.now()).tz_localize(None)
    age = (current.normalize() - endpoint.normalize()).days
    if age < 0 or age > max_age_days:
        return False
    values = pd.to_numeric(observed.loc[observed.index[-1], list(required)], errors="coerce")
    return bool(np.isfinite(values.to_numpy(dtype=float)).all())


def _symbol_frame(raw: pd.DataFrame, symbol: str, symbols: Sequence[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for level in (1, 0):
            if symbol in raw.columns.get_level_values(level):
                return raw.xs(symbol, level=level, axis=1).copy()
        return pd.DataFrame(index=raw.index)
    return raw.copy() if len(symbols) == 1 else pd.DataFrame(index=raw.index)


def _usable(frame: pd.DataFrame) -> bool:
    return "Close" in frame and pd.to_numeric(frame["Close"], errors="coerce").notna().any()


# Exact-request last-good observations survive TTL expiry, remain process-local,
# and are capped by both requests and bytes. These are never current substitutes.
_LAST_GOOD = OrderedDict()
_LAST_GOOD_LOCK = threading.RLock()
_LAST_GOOD_MAX_BYTES = 64 * 1024 * 1024


def _last_good_key(symbols: Tuple[str, ...], kwargs: dict, completion_epoch: tuple = ()) -> str:
    return hashlib.sha256(pickle.dumps((symbols, sorted(kwargs.items()), completion_epoch))).hexdigest()


class _UncachedDownload(Exception):
    """Deliver recoverable failures without committing them to cache_data."""

    def __init__(self, result: pd.DataFrame, created: float):
        super().__init__("Provider request contains unavailable or last-good symbols")
        self.result = result
        self.created = created


_PROVIDER_CALL_LOCK = threading.Lock()


class _ProviderCallPending(TimeoutError):
    """A timed-out provider call is still running; do not enqueue more work."""


def _download_before_deadline(symbols: Sequence[str], kwargs: dict, deadline: float):
    """Bound total provider wait even when Yahoo's HTTP timeout overruns.

    Only one daemon worker can run at a time. Late results remain local to the
    abandoned call and never update cached or last-good observations. The
    worker releases the slot when it finishes, so later reruns can recover.
    """
    remaining = deadline - time.perf_counter()
    if remaining <= 0 or not _PROVIDER_CALL_LOCK.acquire(timeout=remaining):
        raise _ProviderCallPending("Provider deadline elapsed or previous request is pending")
    done = threading.Event()
    outcome = {}
    provider = yf.download

    def run():
        try:
            outcome["result"] = provider(tickers=list(symbols), **kwargs)
        except Exception as error:
            outcome["error"] = error
        finally:
            _PROVIDER_CALL_LOCK.release()
            done.set()

    worker = threading.Thread(target=run, name="adfm-yahoo-download", daemon=True)
    try:
        worker.start()
    except Exception:
        _PROVIDER_CALL_LOCK.release()
        raise
    if not done.wait(timeout=max(0.0, deadline - time.perf_counter())):
        raise _ProviderCallPending("Provider did not complete before the recovery deadline")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


_SUCCESS_CACHE = OrderedDict()
_SUCCESS_CACHE_LOCK = threading.Lock()
_SUCCESS_CACHE_MAX_ENTRIES = 64  # Retain one large-universe sweep within the existing byte cap.


@st.cache_data(show_spinner=False)
def _transport_cache_generation() -> int:
    """Make Streamlit's normal cache clearing invalidate delivery entries."""
    return time.monotonic_ns()


def _clear_success_cache() -> None:
    with _SUCCESS_CACHE_LOCK:
        _SUCCESS_CACHE.clear()


def _cached_download(symbols: Tuple[str, ...], provider_kwargs: dict, _retries: int, _recovery_budget_seconds: float = 25.0, completion_epoch: tuple = (), _request_timeout_seconds: float = 10.0, _recover_missing: bool = True) -> tuple:
    """Cache successes without holding a cache lock during provider I/O.

    Streamlit's per-key function lock cannot wrap a network request: waiting
    for another caller would bypass this caller's delivery deadline.
    """
    from .observability import record_performance

    started = time.perf_counter()
    cache_key = (_transport_cache_generation(), _last_good_key(symbols, provider_kwargs, completion_epoch))
    with _SUCCESS_CACHE_LOCK:
        cached = _SUCCESS_CACHE.get(cache_key)
        if cached is not None:
            result, created = cached
            if started - created < DEFAULT_CONFIG.cache_ttl_seconds:
                _SUCCESS_CACHE.move_to_end(cache_key)
                return result, created
            del _SUCCESS_CACHE[cache_key]
    if not symbols:
        return pd.DataFrame(), time.perf_counter()
    deadline = started + _recovery_budget_seconds
    retries = _retries
    configure_yfinance_cache()
    frames: Dict[str, pd.DataFrame] = {}
    raw = pd.DataFrame()
    for attempt in range(max(1, retries)):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        request_kwargs = {**provider_kwargs, "timeout": min(_request_timeout_seconds, remaining)}
        try:
            raw = _download_before_deadline(symbols, request_kwargs, deadline)
            if not isinstance(raw, pd.DataFrame):
                raw = pd.DataFrame()
            frames = {symbol: _symbol_frame(raw, symbol, symbols) for symbol in symbols}
            if any(_usable(frame) for frame in frames.values()):
                break
        except _ProviderCallPending:
            raw = pd.DataFrame()
            break
        except Exception:
            raw = pd.DataFrame()
        if attempt + 1 < retries:
            time.sleep(min(DEFAULT_CONFIG.retry_base_seconds * (attempt + 1), max(0.0, deadline - time.perf_counter())))
    # Multi-symbol partial failure should not discard successful observations.
    provider_returned_any = any(_usable(frame) for frame in frames.values())
    for symbol in symbols:
        if not _recover_missing or not provider_returned_any or time.perf_counter() >= deadline:
            break
        if _usable(frames.get(symbol, pd.DataFrame())):
            continue
        if len(symbols) == 1:
            continue
        for attempt in range(max(1, retries)):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            request_kwargs = {**provider_kwargs, "timeout": min(_request_timeout_seconds, remaining)}
            try:
                single = _download_before_deadline([symbol], request_kwargs, deadline)
                recovered = _symbol_frame(single, symbol, [symbol])
                if _usable(recovered):
                    frames[symbol] = recovered
                    break
            except _ProviderCallPending:
                break
            except Exception:
                pass
            if attempt + 1 < retries:
                time.sleep(min(DEFAULT_CONFIG.retry_base_seconds * (attempt + 1), max(0.0, deadline - time.perf_counter())))
    last_good_symbols = []
    request_key = _last_good_key(symbols, provider_kwargs, completion_epoch)
    with _LAST_GOOD_LOCK:
        previous = _LAST_GOOD.get(request_key, {})
        for symbol in symbols:
            if not _usable(frames.get(symbol, pd.DataFrame())) and symbol in previous:
                frames[symbol] = previous[symbol].copy()
                last_good_symbols.append(symbol)
        genuine = {symbol: frame.copy() for symbol, frame in frames.items() if _usable(frame) and symbol not in last_good_symbols}
        if genuine:
            saved = {**previous, **genuine}
            _LAST_GOOD[request_key] = saved
            _LAST_GOOD.move_to_end(request_key)
            while len(_LAST_GOOD) > 16 or sum(int(frame.memory_usage(deep=True).sum()) for request in _LAST_GOOD.values() for frame in request.values()) > _LAST_GOOD_MAX_BYTES:
                _LAST_GOOD.popitem(last=False)
    fields = list(dict.fromkeys(field for frame in frames.values() for field in frame.columns)) or list(OHLCV_COLUMNS)
    # Union includes missing provider endpoints; only recovery can add observations.
    index = raw.index if raw is not None else pd.DatetimeIndex([])
    for frame in frames.values():
        index = index.union(frame.index)
    normalized = {symbol: frames.get(symbol, pd.DataFrame()).reindex(index=index, columns=fields) for symbol in symbols}
    if not symbols:
        result = pd.DataFrame()
    elif len(symbols) == 1 and (not isinstance(raw.columns, pd.MultiIndex) or provider_kwargs.get("multi_level_index") is False):
        result = normalized[symbols[0]]
    else:
        result = pd.concat(normalized, axis=1)
        result.columns.names = ["Ticker", "Price"]
        if provider_kwargs.get("group_by", "column") != "ticker":
            result = result.swaplevel(axis=1)
    health = {}
    for symbol, frame in normalized.items():
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        health[symbol] = {"status": ("last_good" if symbol in last_good_symbols else "ok") if len(close) else "unavailable", "as_of": pd.Timestamp(close.index[-1]).date().isoformat() if len(close) else None, "source": "Yahoo Finance"}
    result.attrs["market_data_health"] = health
    result.attrs["auto_adjust"] = provider_kwargs.get("auto_adjust", True)
    result.attrs["last_good_symbols"] = last_good_symbols
    record_performance("provider", time.perf_counter() - started, requested_count=len(symbols), failed_count=sum(item["status"] != "ok" for item in health.values()))
    created = time.perf_counter()
    if any(item["status"] != "ok" for item in health.values()):
        # Partial/last-good observations are delivered without caching the
        # outage, so the next rerun can retry Yahoo.
        raise _UncachedDownload(result, created)
    with _SUCCESS_CACHE_LOCK:
        _SUCCESS_CACHE[cache_key] = (result, created)
        _SUCCESS_CACHE.move_to_end(cache_key)
        while len(_SUCCESS_CACHE) > _SUCCESS_CACHE_MAX_ENTRIES or sum(int(frame.memory_usage(deep=True).sum()) for frame, _ in _SUCCESS_CACHE.values()) > _LAST_GOOD_MAX_BYTES:
            _SUCCESS_CACHE.popitem(last=False)
    return result, created


_cached_download.clear = _clear_success_cache


def download_market_data(tickers, **kwargs) -> pd.DataFrame:
    """Yahoo-compatible bounded daily/date-range/intraday transport.

    All Yahoo arguments (including actions, repair, rounding, keepna, timeout,
    prepost, back_adjust and multi_level_index) pass through. Additional options
    are retries, recover_missing (True by default), recovery_budget_seconds (25 by default), completed_only,
    session_timezone, session_cutoff and now. The recovery budget stops new
    requests/retries; provider timeout bounds each underlying request. An
    entirely failed universe does not trigger individual-symbol fanout.
    Completion filtering is opt-in and requires a caller-specified exchange
    timezone. Unknown markets retain raw bars and carry an unverified label.
    Failed symbols remain missing. No recovery value is invented or extended.
    """
    from .observability import record_performance

    started = time.perf_counter()
    symbols = unique_tickers(tickers.replace(",", " ").split() if isinstance(tickers, str) else tickers)
    retries = min(5, max(1, int(kwargs.pop("retries", DEFAULT_CONFIG.retries))))
    recover_missing = bool(kwargs.pop("recover_missing", True))
    recovery_budget_seconds = min(60.0, max(0.01, float(kwargs.pop("recovery_budget_seconds", 25.0))))
    completed_only = kwargs.pop("completed_only", False)
    session_timezone = kwargs.pop("session_timezone", None)
    custom_cutoff = "session_cutoff" in kwargs
    cutoff = kwargs.pop("session_cutoff", DEFAULT_CONFIG.completed_session_cutoff)
    now = kwargs.pop("now", None)
    kwargs.setdefault("progress", False)
    request_timeout_seconds = float(kwargs.pop("timeout", DEFAULT_CONFIG.request_timeout_seconds))
    completion_epoch = ()
    if kwargs.get("interval", "1d") == "1d":
        if any(is_us_cash_symbol(symbol) for symbol in symbols):
            last_complete = _last_completed_us_session(now)
            completion_epoch = ("XNYS", last_complete.date().isoformat() if last_complete is not None else None)
        if completed_only and session_timezone and (custom_cutoff or session_timezone != "America/New_York"):
            current = pd.Timestamp(now or datetime.now(ZoneInfo(session_timezone)))
            current = current.tz_localize(session_timezone) if current.tzinfo is None else current.tz_convert(session_timezone)
            completion_epoch += (session_timezone, current.date().isoformat(), current.time() >= cutoff)
    try:
        # Deadline/retry limits control delivery, not the identity of successful
        # observations. Keep varying remaining budgets out of the cache key.
        result, created = _cached_download(symbols, kwargs, retries, recovery_budget_seconds, completion_epoch, request_timeout_seconds, recover_missing)
    except _UncachedDownload as failure:
        result, created = failure.result, failure.created
    result = result.copy()
    if kwargs.get("interval", "1d") == "1d":
        result = canonicalize_date_index(result)
    policy = "raw_observations"
    if completed_only and kwargs.get("interval", "1d") == "1d":
        policy = "unverified_exchange_close"
        if session_timezone == "America/New_York" and not custom_cutoff and symbols and all(is_us_cash_symbol(symbol) for symbol in symbols):
            result = completed_daily_observations(result, symbols[0], now=now)
            policy = result.attrs["completion_policy"]
        elif session_timezone:
            current = pd.Timestamp(now or datetime.now(ZoneInfo(session_timezone)))
            current = current.tz_localize(session_timezone) if current.tzinfo is None else current.tz_convert(session_timezone)
            if current.time() < cutoff:
                result = result.loc[result.index.date < current.date()].copy()
            policy = f"explicit_{session_timezone}_cutoff"
    result.attrs["completion_policy"] = policy
    # Recompute observed dates after an explicit completed-session filter.
    health = {}
    for symbol in symbols:
        frame = _symbol_frame(result, symbol, symbols)
        close = pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        health[symbol] = {"status": ("last_good" if symbol in result.attrs.get("last_good_symbols", []) else "ok") if len(close) else "unavailable", "as_of": pd.Timestamp(close.index[-1]).date().isoformat() if len(close) else None, "source": "Yahoo Finance"}
    result.attrs["market_data_health"] = health
    record_performance("delivery", time.perf_counter() - started, cache_hit=created < started, requested_count=len(symbols), failed_count=sum(item["status"] != "ok" for item in health.values()))
    record_data_load("Yahoo Finance", {symbol: _symbol_frame(result, symbol, symbols) for symbol in symbols if health[symbol]["status"] != "unavailable"}, symbols)
    return result


# Preserve the existing cache invalidation interface used by page callers/tests.
fetch_daily_ohlcv.clear = _cached_download.clear
