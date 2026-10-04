"""Completed-session inputs and targeted endpoint recovery for Hedge Timer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from . import market_data


def callout_session_inputs(observed: pd.DataFrame) -> pd.DataFrame:
    """Keep unknown NYSE sessions in recovery windows, excluding market holidays.

    Calendar alignment adds missing rows only; it never fills a price.
    """
    if observed.empty:
        return observed.copy()
    first, last = observed.index.min(), observed.index.max()
    calendar = xcals.get_calendar("XNYS", start=first - pd.Timedelta(days=7), end=last + pd.Timedelta(days=7))
    sessions = calendar.sessions_in_range(first, last).tz_localize(None)
    sessions.name = observed.index.name
    return observed.reindex(sessions)


def extract_close(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Accept either Yahoo column order without filling absent observations."""
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    columns = {}
    for ticker in tickers:
        for field in ("Close", "Adj Close"):
            keys = ((ticker, field), (field, ticker)) if isinstance(raw.columns, pd.MultiIndex) else (field,)
            for key in keys:
                if key in raw.columns and (isinstance(raw.columns, pd.MultiIndex) or len(tickers) == 1):
                    columns[ticker] = pd.to_numeric(raw[key], errors="coerce")
                    break
            if ticker in columns:
                break
    for ticker in ("^GSPC", "^NDX"):
        if ticker not in tickers:
            continue
        for field in ("High", "Low"):
            keys = ((ticker, field), (field, ticker)) if isinstance(raw.columns, pd.MultiIndex) else (field,)
            for key in keys:
                if key in raw.columns and (isinstance(raw.columns, pd.MultiIndex) or len(tickers) == 1):
                    columns[f"{ticker} {field}"] = pd.to_numeric(raw[key], errors="coerce")
                    break
    return market_data.canonicalize_date_index(pd.DataFrame(columns, index=raw.index)).replace([np.inf, -np.inf], np.nan)


def load_hedge_inputs(
    tickers: Sequence[str], start: date, *, now: datetime | None = None,
    research_snapshot_path: Path | None = None
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.DataFrame]:
    """Retry lagging endpoints, then report actual dates against the NYSE close.

    Endpoint gaps are retried even when a symbol has usable older history.
    Recovery is bounded and never substitutes an ETF or extends an old close.
    """
    expected = market_data._last_completed_us_session(now)
    options = dict(
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
        completed_only=True, session_timezone="America/New_York", now=now,
    )
    raw = market_data.download_market_data(list(tickers), start=start.isoformat(), **options)
    observed = extract_close(raw, tickers)
    observed = observed.reindex(columns=[*tickers, *[key for key in observed if key not in tickers]])
    if expected is not None:
        observed = observed.loc[observed.index <= expected]
    health = raw.attrs.get("market_data_health", {})
    last_good = {ticker for ticker in tickers if health.get(ticker, {}).get("status") == "last_good"}
    endpoint = observed.reindex(index=[expected]).iloc[0] if expected is not None else pd.Series(dtype=float)
    lagging = [ticker for ticker in tickers if pd.isna(endpoint.get(ticker)) or ticker in last_good]
    if lagging and expected is not None:
        retry = market_data.download_market_data(
            lagging, start=start.isoformat(), end=(expected + pd.Timedelta(days=1)).date().isoformat(),
            retries=1, recovery_budget_seconds=10,
            timeout=5, recover_missing=False, force_refresh=True, **options,
        )
        recent = extract_close(retry, lagging)
        recent = recent.loc[recent.index <= expected]
        observed = recent.combine_first(observed).sort_index()
        retry_health = retry.attrs.get("market_data_health", {})
        for ticker in lagging:
            if retry_health.get(ticker, {}).get("status") == "last_good":
                last_good.add(ticker)
            elif ticker in recent and pd.notna(recent[ticker].get(expected)):
                last_good.discard(ticker)
    rows = []
    for ticker in tickers:
        values = observed[ticker].dropna()
        as_of = values.index[-1] if len(values) else None
        status = "Missing" if as_of is None else "Current" if as_of == expected else "Lagging"
        if ticker in last_good:
            status = "Last-good cache"
        rows.append({"Input": ticker, "Last observation": as_of.date().isoformat() if as_of is not None else "Unavailable", "Status": status})
    if not observed.reindex(columns=list(tickers)).notna().any().any() and research_snapshot_path is not None and expected is not None:
        close_file = research_snapshot_path / "research_inputs.csv"
        ranges_file = research_snapshot_path / "research_indices_ohlc.csv"
        if close_file.is_file() and ranges_file.is_file():
            observed = pd.read_csv(close_file, index_col="Date", parse_dates=True).reindex(columns=list(tickers))
            raw_ranges = pd.read_csv(ranges_file, header=[0, 1], index_col=0, parse_dates=True)
            for ticker in ("^GSPC", "^NDX"):
                if ticker in tickers:
                    for field in ("High", "Low"):
                        observed[f"{ticker} {field}"] = raw_ranges[(field, ticker)]
            observed = observed.loc[(observed.index >= pd.Timestamp(start)) & (observed.index <= expected)]
            rows = [{"Input": ticker, "Last observation": values.index[-1].date().isoformat() if len(values := observed[ticker].dropna()) else "Unavailable", "Status": "Research snapshot"} for ticker in tickers]
    return observed, expected, pd.DataFrame(rows)
