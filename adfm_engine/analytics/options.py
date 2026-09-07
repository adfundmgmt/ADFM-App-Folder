"""Original option-compass page transformations."""
from __future__ import annotations
from datetime import date,datetime
from typing import Mapping
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from adfm_engine.data.market import adjusted_ohlcv,unique_tickers
TITLE = "Options Positioning Compass"
DEFAULT_UNIVERSE = "SPY, QQQ, IWM, DIA, TLT, GLD, USO, SMH, EEM, HYG, LQD"
NY_TZ = ZoneInfo("America/New_York")
def normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()

def parse_universe(value: str, selected: str) -> tuple[str, ...]:
    raw = str(value or "").replace("\n", ",").split(",")
    return unique_tickers([selected, *raw])

def nearest_expiry(expirations: tuple[str, ...], target_dte: int, as_of: date) -> str | None:
    eligible = []
    for expiry in expirations:
        try:
            dte = (pd.Timestamp(expiry).date() - as_of).days
        except Exception:
            continue
        if dte >= 2:
            eligible.append((abs(dte - target_dte), dte, expiry))
    return min(eligible)[2] if eligible else None

def close_series(raw_frames: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    frame = raw_frames.get(ticker)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    adjusted = adjusted_ohlcv(frame)
    return pd.to_numeric(adjusted.get("Close"), errors="coerce").dropna()

def latest_value(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan

def fmt(value: float, suffix: str = "", digits: int = 1) -> str:
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "N/A"

def money(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"

