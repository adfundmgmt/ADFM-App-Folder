"""Option calendars and chains, preserving Yahoo then Cboe fallback."""
from __future__ import annotations
from datetime import date,datetime
from typing import Mapping
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from adfm_engine.cache import ttl_cache
from adfm_engine.data.options_sources import expirations_from_cboe,fetch_cboe_delayed_options,select_cboe_expiry
@ttl_cache(seconds=900)
def fetch_expirations(symbol: str) -> tuple[str, ...]:
    try:
        return tuple(yf.Ticker(symbol).options)
    except Exception:
        return ()

@ttl_cache(seconds=900)
def fetch_cboe_snapshot(
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    return fetch_cboe_delayed_options(symbol)

@ttl_cache(seconds=900)
def fetch_chain(
    symbol: str, expiry: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    str | None,
    str,
    str,
]:
    try:
        chain = yf.Ticker(symbol).option_chain(expiry)
        underlying = chain.underlying if isinstance(chain.underlying, Mapping) else {}
        if not chain.calls.empty and not chain.puts.empty:
            return (
                chain.calls.copy(),
                chain.puts.copy(),
                dict(underlying),
                None,
                "Yahoo Finance",
                "",
            )
        yahoo_error = "Empty Yahoo option chain"
    except Exception as exc:
        yahoo_error = str(exc)

    try:
        cboe_frame, underlying, timestamp = fetch_cboe_snapshot(symbol)
        calls, puts = select_cboe_expiry(cboe_frame, expiry)
        if calls.empty or puts.empty:
            raise ValueError("No matching Cboe expiration")
        return calls, puts, underlying, None, "Cboe delayed quotes", timestamp
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            f"Yahoo: {yahoo_error}; Cboe: {exc}",
            "",
            "",
        )

@ttl_cache(seconds=900)
def fetch_cboe_expirations(symbol: str) -> tuple[str, ...]:
    try:
        frame, _, _ = fetch_cboe_snapshot(symbol)
        return expirations_from_cboe(frame)
    except Exception:
        return ()

def available_expirations(symbol: str) -> tuple[str, ...]:
    """Use Yahoo's calendar when available and Cboe's when Yahoo is blocked."""
    return fetch_expirations(symbol) or fetch_cboe_expirations(symbol)

