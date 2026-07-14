"""Deterministic data and analytics engine for the ADFM Momentum Expansion Scanner.

All indicator functions use only observations available at their requested end
date.  The Streamlit page is deliberately kept separate from this module so
the calculations can be tested without rendering the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from adfm_sector_rotation_config import MAJOR_SECTORS, SUBSECTOR_ROWS


# Signal thresholds.  Keep these together so research users can audit or alter
# the deterministic rules without rewriting the calculation engine.
MIN_PRICE = 5.0
MIN_HISTORY = 252
MIN_DOLLAR_VOLUME = 20_000_000.0
MAX_STALE_SESSIONS = 3
DOWNLOAD_CHUNK_SIZE = 40
DOWNLOAD_RETRIES = 3
VOLUME_RATIO = 1.25
VOLUME_FIVE_DAY_RATIO = 1.10
VDU_RATIO = 0.75
EXTREME_DOWN_VOLUME_RATIO = 1.50
ATR_PERCENTILE_WINDOW = 126
ATR_PERCENTILE_CUTOFF = 0.35
RS_SLOPE_WINDOW = 10
RVOL_AVOID_CHASING = 2.50

SIGNAL_COLUMNS = [
    "VOL", "MA FAN", "ATR SQZ", "CLOSE HI", "52W HI", "VDU", "HH/HL",
    "W.EMA", "OBV+", "NR7", "U/D VOL", "SQUEEZE",
]
SIGNAL_WEIGHTS = {"Trend": 30.0, "Compression": 25.0, "Participation": 25.0, "Relative Strength": 20.0}
SECTOR_WEIGHTS = {
    "ETF RS 1M": 15.0, "ETF RS 3M": 15.0, "ETF RS 6M": 10.0,
    "Median Subsector RS 1M": 10.0, "Median Subsector RS 3M": 10.0,
    "Positive RS 1M": 10.0, "Above EMA50": 10.0, "Above SMA200": 5.0,
    "ETF RS Slope 10D": 10.0, "ETF RS Acceleration": 5.0,
}

GICS_TO_ADFM = {
    "Communication Services": "Communication Services", "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Cyclical": "Consumer Discretionary", "Consumer Staples": "Consumer Staples",
    "Consumer Defensive": "Consumer Staples", "Energy": "Energy", "Financials": "Financials",
    "Financial Services": "Financials", "Health Care": "Health Care", "Healthcare": "Health Care",
    "Industrials": "Industrials", "Basic Materials": "Materials", "Materials": "Materials",
    "Real Estate": "Real Estate", "Information Technology": "Technology", "Technology": "Technology",
    "Utilities": "Utilities",
}


@dataclass(frozen=True)
class DownloadResult:
    """Raw daily OHLCV observations and non-fatal download diagnostics."""

    frames: Dict[str, pd.DataFrame]
    dropped: pd.DataFrame


def _clean_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index()[~out.index.duplicated(keep="last")]


def _drop_unfinished_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude today's daily bar before the US cash-market close is settled."""
    if frame.empty:
        return frame
    now = datetime.now(ZoneInfo("America/New_York"))
    last_date = pd.Timestamp(frame.index[-1]).date()
    if last_date == now.date() and (now.hour, now.minute) < (16, 15):
        return frame.iloc[:-1].copy()
    return frame


def _normalise_ohlcv(raw: pd.DataFrame, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frames: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return frames
    for ticker in tickers:
        data: Dict[str, pd.Series] = {}
        for field in fields:
            series: Optional[pd.Series] = None
            if isinstance(raw.columns, pd.MultiIndex):
                if (field, ticker) in raw.columns:
                    series = raw[(field, ticker)]
                elif (ticker, field) in raw.columns:
                    series = raw[(ticker, field)]
            elif len(tickers) == 1 and field in raw.columns:
                series = raw[field]
            if series is not None:
                data[field] = pd.to_numeric(series, errors="coerce")
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(data):
            frame = _drop_unfinished_session(_clean_index(pd.DataFrame(data)))
            if not frame.empty and frame["Close"].notna().any():
                frames[ticker] = frame
    return frames


def _download_ohlcv_batch(tickers: Sequence[str], period: str) -> Dict[str, pd.DataFrame]:
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            raw = yf.download(
                tickers=list(tickers), period=period, interval="1d", progress=False,
                auto_adjust=False, group_by="column", threads=True,
            )
            frames = _normalise_ohlcv(raw, tickers)
            if frames:
                return frames
        except Exception:
            pass
        if attempt + 1 < DOWNLOAD_RETRIES:
            time.sleep(0.75 * (attempt + 1))
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(tickers: Tuple[str, ...], period: str = "3y") -> DownloadResult:
    """Chunked, retried OHLCV loader with individual fallback; never fills data."""
    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    frames: Dict[str, pd.DataFrame] = {}
    for start in range(0, len(symbols), DOWNLOAD_CHUNK_SIZE):
        frames.update(_download_ohlcv_batch(symbols[start:start + DOWNLOAD_CHUNK_SIZE], period))
    for ticker in [symbol for symbol in symbols if symbol not in frames]:
        frames.update(_download_ohlcv_batch([ticker], period))
    dropped = pd.DataFrame({"Ticker": [s for s in symbols if s not in frames], "Reason": "No valid OHLCV data returned"})
    return DownloadResult(frames=frames, dropped=dropped)


def adjusted_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Adjust OHLC and volume with Yahoo's split/dividend adjustment factor.

    This preserves raw observations in the loader while avoiding split-created
    ranges, ATR spikes, and false breakouts in the derived pattern series.
    """
    out = raw.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    adj_close = pd.to_numeric(out.get("Adj Close", close), errors="coerce")
    factor = (adj_close / close).replace([np.inf, -np.inf, 0], np.nan)
    if factor.notna().any():
        for col in ["Open", "High", "Low", "Close"]:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce") / factor
    out["Adj Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def benchmark_calendar(frames: Mapping[str, pd.DataFrame], benchmark: str) -> pd.DatetimeIndex:
    frame = frames.get(benchmark)
    if frame is None or frame.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(frame.dropna(subset=["Close"]).index).sort_values().unique()


def stale_sessions(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> int:
    if frame.empty or len(calendar) == 0:
        return len(calendar)
    last = pd.Timestamp(frame.dropna(subset=["Close"]).index.max())
    return int((calendar > last).sum())


def valid_ohlcv_reason(frame: Optional[pd.DataFrame], calendar: pd.DatetimeIndex) -> Optional[str]:
    if frame is None or frame.empty:
        return "Missing OHLCV history"
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(frame.columns) or frame[required].dropna(how="any").empty:
        return "Invalid OHLCV history"
    if len(frame.dropna(subset=required)) < MIN_HISTORY:
        return f"Thin history (<{MIN_HISTORY} valid sessions)"
    if stale_sessions(frame, calendar) > MAX_STALE_SESSIONS:
        return f"Stale (> {MAX_STALE_SESSIONS} benchmark sessions)"
    if (frame["High"] < frame["Low"]).any() or (frame["Volume"] < 0).any():
        return "Invalid high/low or volume observations"
    return None


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series.dropna()) <= periods:
        return np.nan
    return float(series.pct_change(periods, fill_method=None).iloc[-1])


def _slope(series: pd.Series, periods: int) -> float:
    clean = series.dropna().tail(periods)
    if len(clean) < periods or np.isclose(clean.iloc[0], 0):
        return np.nan
    return float((clean.iloc[-1] / clean.iloc[0]) - 1.0)


def _percentile_of_latest(series: pd.Series, window: int) -> float:
    values = series.dropna().tail(window)
    if len(values) < window:
        return np.nan
    return float(values.rank(pct=True).iloc[-1])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = frame["High"], frame["Low"], frame["Close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def signal_snapshot(frame: pd.DataFrame, end: Optional[int] = None) -> Dict[str, bool]:
    """Return the 12 core signal flags plus SPRING as of one completed date."""
    data = frame if end is None else frame.iloc[:end + 1]
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    flags = {name: False for name in SIGNAL_COLUMNS + ["SPRING"]}
    if len(data) < MIN_HISTORY:
        return flags
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    ema20, ema50, sma200 = _ema(close, 20), _ema(close, 50), close.rolling(200, min_periods=200).mean()
    vol20, vol5 = volume.rolling(20, min_periods=20).mean(), volume.rolling(5, min_periods=5).mean()
    atr_pct = _atr(data, 14) / close
    prior_high_20 = high.shift(20).rolling(20, min_periods=20).max()
    prior_low_20 = low.shift(20).rolling(20, min_periods=20).min()
    weekly = close.resample("W-FRI").last().dropna()
    w10, w30 = _ema(weekly, 10), _ema(weekly, 30)
    obv = _obv(close, volume)
    rolling_std = close.rolling(20, min_periods=20).std()
    bb_upper, bb_lower = close.rolling(20, min_periods=20).mean() + 2 * rolling_std, close.rolling(20, min_periods=20).mean() - 2 * rolling_std
    kc_mid = _ema(close, 20)
    atr20 = _atr(data, 20)
    kc_upper, kc_lower = kc_mid + 1.5 * atr20, kc_mid - 1.5 * atr20
    down_days = close.diff().tail(5) < 0
    extreme_down = (volume.tail(5) > 1.5 * vol20.iloc[-1]) & down_days
    up_volume = volume.where(close.diff() > 0, 0.0).rolling(20, min_periods=20).sum()
    down_volume = volume.where(close.diff() < 0, 0.0).rolling(20, min_periods=20).sum()
    current_range = (high - low)
    flags["VOL"] = bool(volume.iloc[-1] >= VOLUME_RATIO * vol20.iloc[-1] or vol5.iloc[-1] >= VOLUME_FIVE_DAY_RATIO * vol20.iloc[-1])
    flags["MA FAN"] = bool(close.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > sma200.iloc[-1] and ema20.iloc[-1] > ema20.iloc[-6] and ema50.iloc[-1] > ema50.iloc[-6])
    flags["ATR SQZ"] = bool(atr_pct.iloc[-1] <= atr_pct.dropna().tail(ATR_PERCENTILE_WINDOW).quantile(ATR_PERCENTILE_CUTOFF))
    flags["CLOSE HI"] = bool(close.iloc[-1] >= close.tail(20).max() * 0.98)
    flags["52W HI"] = bool(close.iloc[-1] >= close.tail(252).max() * 0.95)
    flags["VDU"] = bool(vol5.iloc[-1] / vol20.iloc[-1] < VDU_RATIO and close.iloc[-1] > ema20.iloc[-1] and not extreme_down.any())
    flags["HH/HL"] = bool(high.tail(20).max() > prior_high_20.iloc[-1] and low.tail(20).min() > prior_low_20.iloc[-1])
    flags["W.EMA"] = bool(len(weekly) >= 35 and weekly.iloc[-1] > w10.iloc[-1] > w30.iloc[-1] and w10.iloc[-1] > w10.iloc[-2] and w30.iloc[-1] > w30.iloc[-2])
    flags["OBV+"] = bool(obv.iloc[-1] > obv.rolling(20, min_periods=20).mean().iloc[-1] and _slope(obv, 5) > 0)
    flags["NR7"] = bool(current_range.iloc[-1] <= current_range.tail(7).min())
    flags["U/D VOL"] = bool(down_volume.iloc[-1] > 0 and up_volume.iloc[-1] / down_volume.iloc[-1] >= 1.25)
    flags["SQUEEZE"] = bool(bb_upper.iloc[-1] < kc_upper.iloc[-1] and bb_lower.iloc[-1] > kc_lower.iloc[-1])
    prior_low = low.shift(1).rolling(20, min_periods=20).min()
    candidates = pd.DataFrame({"low": low, "prior": prior_low}).tail(10).dropna()
    spring = False
    for date, row in candidates.iterrows():
        if row["low"] < row["prior"] and row["low"] >= row["prior"] * 0.97:
            later = close.loc[close.index > date]
            if not later.empty and (later > row["prior"]).any():
                spring = True
                break
    flags["SPRING"] = bool(spring and close.iloc[-1] > ema20.iloc[-1])
    return flags


def security_features(ticker: str, raw_frame: pd.DataFrame, benchmark_close: pd.Series, sector_close: Optional[pd.Series] = None) -> Dict[str, object]:
    """Compute all unformatted card/table fields for a valid security."""
    frame = adjusted_ohlcv(raw_frame).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    close, volume = frame["Close"], frame["Volume"]
    flags = signal_snapshot(frame)
    old_flags = signal_snapshot(frame, len(frame) - 6)
    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    sma200 = close.rolling(200, min_periods=200).mean()
    atr14 = _atr(frame, 14)
    vol20 = volume.rolling(20, min_periods=20).mean()
    dollar_volume = (close * volume).rolling(20, min_periods=20).median()
    rs = close.div(benchmark_close.reindex(close.index)).replace([np.inf, -np.inf], np.nan).dropna()
    bw = (4 * close.rolling(20, min_periods=20).std()).div(close.rolling(20, min_periods=20).mean())
    sector_rs = pd.Series(dtype=float) if sector_close is None else close.div(sector_close.reindex(close.index)).replace([np.inf, -np.inf], np.nan).dropna()
    row: Dict[str, object] = {"Ticker": ticker, **flags}
    row.update({
        "Price": float(close.iloc[-1]), "Return 1D": _pct_change(close, 1), "Return 5D": _pct_change(close, 5),
        "Return 21D": _pct_change(close, 21), "Return 63D": _pct_change(close, 63), "Return 126D": _pct_change(close, 126),
        "RS 5D": _pct_change(rs, 5), "RS 21D": _pct_change(rs, 21), "RS 63D": _pct_change(rs, 63), "RS 126D": _pct_change(rs, 126),
        "Sector RS 5D": _pct_change(sector_rs, 5), "Sector RS 21D": _pct_change(sector_rs, 21), "Sector RS 63D": _pct_change(sector_rs, 63),
        "RS Slope 10D": _slope(rs, 10), "RS Acceleration 5D": _pct_change(rs, 5) - _pct_change(rs.iloc[:-5], 5) if len(rs) > 10 else np.nan,
        "Distance EMA20": float(close.iloc[-1] / ema20.iloc[-1] - 1), "Distance EMA50": float(close.iloc[-1] / ema50.iloc[-1] - 1),
        "Distance SMA200": float(close.iloc[-1] / sma200.iloc[-1] - 1), "Distance 20D High": float(close.iloc[-1] / close.tail(20).max() - 1),
        "Distance 52W High": float(close.iloc[-1] / close.tail(252).max() - 1), "ATR %": float(atr14.iloc[-1] / close.iloc[-1]),
        "RVOL": float(volume.iloc[-1] / vol20.iloc[-1]), "Median Dollar Volume": float(dollar_volume.iloc[-1]),
        "Realized Vol 20D": float(close.pct_change().rolling(20, min_periods=20).std().iloc[-1] * np.sqrt(252)),
        "BB Width Percentile": _percentile_of_latest(bw, 126), "Above SMA200": bool(close.iloc[-1] > sma200.iloc[-1]),
    })
    row["Signal Count"] = int(sum(bool(flags[k]) for k in SIGNAL_COLUMNS))
    row["Signal Count Change 5D"] = row["Signal Count"] - int(sum(bool(old_flags[k]) for k in SIGNAL_COLUMNS))
    return row


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).rank(pct=True, method="average") * 100.0


def rank_change(current: pd.DataFrame, prior: pd.DataFrame, key: str = "Ticker") -> pd.DataFrame:
    """Return current ranking with positive values denoting an improved rank."""
    out = current.copy()
    if prior.empty or "Rank" not in prior or key not in prior:
        out["Prior Rank"] = np.nan
        out["Rank change over five sessions"] = np.nan
        return out
    prior_ranks = prior[[key, "Rank"]].rename(columns={"Rank": "Prior Rank"})
    out = out.merge(prior_ranks, on=key, how="left")
    out["Rank change over five sessions"] = out["Prior Rank"] - out["Rank"]
    return out


def _weighted_components(frame: pd.DataFrame, components: Sequence[Tuple[str, float]]) -> pd.Series:
    pieces = pd.DataFrame({name: frame[name] for name, _ in components if name in frame}, index=frame.index)
    weights = pd.Series({name: weight for name, weight in components if name in pieces})
    present = pieces.notna().astype(float)
    denominator = present.mul(weights, axis=1).sum(axis=1)
    # Components are expressed on a 0â€“100 scale before entering the blend.
    score = pieces.fillna(0).mul(weights, axis=1).sum(axis=1).div(denominator)
    return score.where(denominator >= 0.60 * weights.sum())


def score_scanner(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent component scores, cross-sectional ranks and setup labels."""
    out = frame.copy()
    for column in ["RS 21D", "RS 63D", "RS Slope 10D", "RS Acceleration 5D", "Sector RS 21D"]:
        out[f"Pct {column}"] = percentile_rank(out[column])
    out["Trend Score"] = out[["MA FAN", "CLOSE HI", "52W HI", "HH/HL", "W.EMA"]].astype(float).mean(axis=1) * 100
    out["Compression Score"] = pd.concat([
        out[["ATR SQZ", "NR7", "SQUEEZE", "SPRING"]].astype(float),
        (1 - out["BB Width Percentile"]).rename("BB Compression"),
    ], axis=1).mean(axis=1) * 100
    out["Participation Score"] = pd.concat([
        out[["VOL", "VDU", "OBV+", "U/D VOL"]].astype(float),
        out["RVOL"].clip(upper=1.5).div(1.5).rename("RVOL Score"),
    ], axis=1).mean(axis=1) * 100
    out["Relative Strength Score"] = _weighted_components(out, [
        ("Pct RS 21D", 1), ("Pct RS 63D", 1), ("Pct RS Slope 10D", 1),
        ("Pct RS Acceleration 5D", 1), ("Pct Sector RS 21D", 1),
    ])
    out["Expansion Score"] = _weighted_components(out, [
        ("Trend Score", .30), ("Compression Score", .25), ("Participation Score", .25), ("Relative Strength Score", .20),
    ])
    out["RS Percentile"] = percentile_rank(out["Relative Strength Score"])
    out["Rank"] = out["Expansion Score"].rank(ascending=False, method="min")
    out["Setup State"] = "Watch"
    positive_trend = out[["MA FAN", "CLOSE HI", "HH/HL", "W.EMA"]].any(axis=1)
    out.loc[(out["Signal Count"] >= 7) & positive_trend, "Setup State"] = "Constructive"
    out.loc[(out["Signal Count"] >= 8) & (out["RS 21D"] > 0) & (out["RS 63D"] > 0) & (out["RS Slope 10D"] > 0), "Setup State"] = "Momentum Building"
    out.loc[(out["Signal Count Change 5D"] >= 2) & (out["RS Acceleration 5D"] > 0) & (out["Setup State"] == "Watch"), "Setup State"] = "Early Turn"
    out.loc[(out["Signal Count Change 5D"] <= -2) & (out["RS Slope 10D"] < 0), "Setup State"] = "Deteriorating"
    extended = (out["Distance EMA20"] > 2.5 * out["ATR %"]) | (out["Distance EMA50"] > .15)
    out["Risk Warning"] = np.where(extended & (out["RVOL"] > RVOL_AVOID_CHASING), "Avoid Chasing", np.where(extended, "Extended", ""))
    out["Trend Label"] = np.where(out["Risk Warning"] != "", "Extended", np.where(out["MA FAN"], "Trending up", np.where((out["Distance EMA20"] > 0) & (out["RS Acceleration 5D"] > 0), "Turning higher", np.where(out["Distance EMA50"] < 0, "Trending down", "Range-bound"))))
    out["RS Label"] = np.where(out["RS Percentile"] >= 80, "Top RS", np.where((out["RS Slope 10D"] > 0) & (out["RS Acceleration 5D"] > 0), "Improving RS", "RS weakening"))
    out["Setup Description"] = out.apply(setup_description, axis=1)
    return out.sort_values(["Expansion Score", "Signal Count", "RS Percentile", "Median Dollar Volume"], ascending=False, na_position="last").reset_index(drop=True)


def setup_description(row: pd.Series) -> str:
    """A deterministic explanation assembled only from calculated features."""
    parts: List[str] = []
    if row.get("MA FAN", False): parts.append("trend structure aligned")
    if row.get("ATR SQZ", False) or row.get("SQUEEZE", False) or row.get("NR7", False): parts.append("volatility compressed")
    if row.get("VOL", False) or row.get("OBV+", False) or row.get("U/D VOL", False): parts.append("participation expanding")
    if row.get("CLOSE HI", False) or row.get("52W HI", False): parts.append("price pressing the upper end of its range")
    if not parts: parts.append("the setup is incomplete and needs confirmation")
    lead = "Constructive base" if row.get("Signal Count", 0) >= 7 else "Early technical setup"
    sentence = f"{lead}, " + ", ".join(parts[:3]) + "."
    if row.get("Risk Warning") == "Avoid Chasing": return sentence + " Price is extended on elevated relative volume."
    if row.get("Risk Warning") == "Extended": return sentence + " Price is extended from its short-term trend."
    if row.get("RS Label") == "RS weakening": return sentence + " Relative strength remains below its intermediate trend."
    return sentence


def market_cap_bucket(value: object) -> str:
    if value is None or pd.isna(value): return "â€”"
    cap = float(value)
    if cap >= 200e9: return "Mega"
    if cap >= 10e9: return "Large"
    if cap >= 2e9: return "Mid"
    return "Small"


def sector_etf(sector: object) -> Optional[str]:
    if not isinstance(sector, str): return None
    normalized = GICS_TO_ADFM.get(sector, sector)
    return next((ticker for ticker, name in MAJOR_SECTORS.items() if name == normalized), None)


def _normalise_symbol(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _read_html_table(url: str, required_columns: Sequence[str]) -> pd.DataFrame:
    response = requests.get(url, timeout=20, headers={"User-Agent": "ADFM Analytics/1.0"})
    response.raise_for_status()
    for table in pd.read_html(StringIO(response.text)):
        if set(required_columns).issubset(table.columns):
            return table
    return pd.DataFrame()


@st.cache_data(ttl=86_400, show_spinner=False)
def load_equity_universe(choice: str, custom_watchlist: str = "") -> Tuple[pd.DataFrame, List[str]]:
    """Load public constituent lists; failed sources are reported, never invented."""
    errors: List[str] = []
    pieces: List[pd.DataFrame] = []
    if choice in {"S&P 500", "Combined liquid indices"}:
        try:
            table = _read_html_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"])
            pieces.append(pd.DataFrame({"Ticker": table["Symbol"].map(_normalise_symbol), "Name": table.get("Security", ""), "Sector": table.get("GICS Sector", ""), "Industry": table.get("GICS Sub-Industry", ""), "Source Universe": "S&P 500"}))
        except Exception as exc:
            errors.append(f"S&P 500 constituents unavailable: {type(exc).__name__}")
    if choice in {"Nasdaq 100", "Combined liquid indices"}:
        try:
            table = _read_html_table("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker"])
            pieces.append(pd.DataFrame({"Ticker": table["Ticker"].map(_normalise_symbol), "Name": table.get("Company", table.get("Company Name", "")), "Sector": table.get("GICS Sector", ""), "Industry": table.get("Industry", ""), "Source Universe": "Nasdaq 100"}))
        except Exception as exc:
            errors.append(f"Nasdaq 100 constituents unavailable: {type(exc).__name__}")
    if choice in {"Russell 1000 liquid", "Combined liquid indices"}:
        try:
            url = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
            response = requests.get(url, timeout=30, headers={"User-Agent": "ADFM Analytics/1.0"})
            response.raise_for_status()
            lines = response.text.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("Ticker,"))
            holdings = pd.read_csv(StringIO("\n".join(lines[start:])))
            ticker_col = "Ticker" if "Ticker" in holdings else "Ticker "
            pieces.append(pd.DataFrame({"Ticker": holdings[ticker_col].map(_normalise_symbol), "Name": holdings.get("Name", ""), "Sector": holdings.get("Sector", ""), "Industry": holdings.get("Industry", ""), "Source Universe": "Russell 1000 liquid"}))
        except Exception as exc:
            errors.append(f"Russell 1000 liquid constituents unavailable: {type(exc).__name__}")
    if choice == "Custom watchlist" or custom_watchlist.strip():
        tickers = [_normalise_symbol(t) for t in custom_watchlist.replace("\n", ",").split(",") if t.strip()]
        if tickers:
            pieces.append(pd.DataFrame({"Ticker": tickers, "Name": "", "Sector": "", "Industry": "", "Source Universe": "Custom watchlist"}))
        elif choice == "Custom watchlist":
            errors.append("Enter at least one ticker in the custom watchlist.")
    if not pieces:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Industry", "Source Universe"]), errors
    universe = pd.concat(pieces, ignore_index=True).replace({"â€”": np.nan, "-": np.nan})
    universe = universe[universe["Ticker"].str.match(r"^[A-Z0-9^=\-]+$", na=False)].drop_duplicates("Ticker", keep="first")
    universe["ADFM Sector"] = universe["Sector"].map(GICS_TO_ADFM)
    return universe.reset_index(drop=True), errors


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_market_metadata(tickers: Tuple[str, ...]) -> pd.DataFrame:
    """Best-effort metadata. Missing values stay missing and do not become zero."""
    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_fast_info()
            rows.append({"Ticker": ticker, "Market Cap": info.get("market_cap", np.nan), "Quote Type": info.get("quote_type", ""), "Last Price Meta": info.get("last_price", np.nan)})
        except Exception:
            rows.append({"Ticker": ticker, "Market Cap": np.nan, "Quote Type": "", "Last Price Meta": np.nan})
    return pd.DataFrame(rows)


def _returns_row(ticker: str, close: pd.Series, benchmark: pd.Series) -> Dict[str, object]:
    rs = close.div(benchmark.reindex(close.index)).replace([np.inf, -np.inf], np.nan).dropna()
    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    sma200 = close.rolling(200, min_periods=200).mean()
    return {"Ticker": ticker, "Abs 1W": _pct_change(close, 5), "Abs 1M": _pct_change(close, 21), "Abs 3M": _pct_change(close, 63), "Abs 6M": _pct_change(close, 126), "RS 1W": _pct_change(rs, 5), "RS 1M": _pct_change(rs, 21), "RS 3M": _pct_change(rs, 63), "RS 6M": _pct_change(rs, 126), "RS Slope 10D": _slope(rs, 10), "RS Acceleration": _pct_change(rs, 5) - _pct_change(rs.iloc[:-5], 5) if len(rs) > 10 else np.nan, "Above EMA20": bool(close.iloc[-1] > ema20.iloc[-1]), "Above EMA50": bool(close.iloc[-1] > ema50.iloc[-1]), "Above SMA200": bool(close.iloc[-1] > sma200.iloc[-1])}


def sector_leadership(frames: Mapping[str, pd.DataFrame], benchmark: str, include_thematic: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute equal-sector-weight leadership and unweighted subsector medians."""
    adjusted = {ticker: adjusted_ohlcv(frame) for ticker, frame in frames.items()}
    if benchmark not in adjusted: return pd.DataFrame(), pd.DataFrame()
    bench = adjusted[benchmark]["Close"]
    subs = pd.DataFrame(SUBSECTOR_ROWS)
    if not include_thematic: subs = subs[subs["Tier"] == "Core"]
    sub_rows: List[Dict[str, object]] = []
    for item in subs.to_dict("records"):
        ticker = item["Ticker"]
        if ticker in adjusted and len(adjusted[ticker]) >= MIN_HISTORY:
            sub_rows.append({**item, **_returns_row(ticker, adjusted[ticker]["Close"], bench)})
    sub_df = pd.DataFrame(sub_rows)
    if not sub_df.empty:
        sub_df["Composite Score"] = _weighted_components(pd.DataFrame({"a": percentile_rank(sub_df["RS 1M"]), "b": percentile_rank(sub_df["RS 3M"]), "c": percentile_rank(sub_df["RS Slope 10D"]), "d": percentile_rank(sub_df["RS Acceleration"])}), [("a", .4), ("b", .3), ("c", .2), ("d", .1)])
        sub_df["Rank"] = sub_df["Composite Score"].rank(ascending=False, method="min")
        sub_df["Rotation Quadrant"] = np.select([(sub_df["RS 1M"] >= 0) & (sub_df["RS Acceleration"] >= 0), (sub_df["RS 1M"] < 0) & (sub_df["RS Acceleration"] >= 0), (sub_df["RS 1M"] < 0) & (sub_df["RS Acceleration"] < 0)], ["Leading", "Improving", "Lagging"], default="Weakening")
    sector_rows: List[Dict[str, object]] = []
    for etf, name in MAJOR_SECTORS.items():
        if etf in adjusted and len(adjusted[etf]) >= MIN_HISTORY:
            etf_metrics = _returns_row(etf, adjusted[etf]["Close"], bench)
        else:
            etf_metrics = {"Ticker": etf, "Abs 1W": np.nan, "Abs 1M": np.nan, "Abs 3M": np.nan, "Abs 6M": np.nan, "RS 1W": np.nan, "RS 1M": np.nan, "RS 3M": np.nan, "RS 6M": np.nan, "RS Slope 10D": np.nan, "RS Acceleration": np.nan, "Above EMA20": np.nan, "Above EMA50": np.nan, "Above SMA200": np.nan}
        group = sub_df[sub_df["Sector Group"] == name] if not sub_df.empty else pd.DataFrame()
        sector_rows.append({"Sector": name, "Major ETF": etf, **etf_metrics, "Median Subsector RS 1M": group["RS 1M"].median() if not group.empty else np.nan, "Median Subsector RS 3M": group["RS 3M"].median() if not group.empty else np.nan, "Positive RS 1M": (group["RS 1M"] > 0).mean() if not group.empty else np.nan, "Positive RS 3M": (group["RS 3M"] > 0).mean() if not group.empty else np.nan, "Above EMA20 Breadth": group["Above EMA20"].mean() if not group.empty else np.nan, "Above EMA50 Breadth": group["Above EMA50"].mean() if not group.empty else np.nan, "Above SMA200 Breadth": group["Above SMA200"].mean() if not group.empty else np.nan, "Positive RS Slope": (group["RS Slope 10D"] > 0).mean() if not group.empty else np.nan, "Dispersion": group["RS 1M"].std() if not group.empty else np.nan, "Strongest Subsector": group.loc[group["RS 1M"].idxmax(), "Ticker"] if not group.empty and group["RS 1M"].notna().any() else "", "Weakest Subsector": group.loc[group["RS 1M"].idxmin(), "Ticker"] if not group.empty and group["RS 1M"].notna().any() else ""})
    sector_df = pd.DataFrame(sector_rows)
    if sector_df.empty: return sector_df, sub_df
    scored = pd.DataFrame({"ETF RS 1M": percentile_rank(sector_df["RS 1M"]), "ETF RS 3M": percentile_rank(sector_df["RS 3M"]), "ETF RS 6M": percentile_rank(sector_df["RS 6M"]), "Median Subsector RS 1M": percentile_rank(sector_df["Median Subsector RS 1M"]), "Median Subsector RS 3M": percentile_rank(sector_df["Median Subsector RS 3M"]), "Positive RS 1M": percentile_rank(sector_df["Positive RS 1M"]), "Above EMA50": percentile_rank(sector_df["Above EMA50 Breadth"]), "Above SMA200": percentile_rank(sector_df["Above SMA200 Breadth"]), "ETF RS Slope 10D": percentile_rank(sector_df["RS Slope 10D"]), "ETF RS Acceleration": percentile_rank(sector_df["RS Acceleration"])})
    sector_df["Leadership Score"] = _weighted_components(scored, list(SECTOR_WEIGHTS.items()))
    sector_df["Rank"] = sector_df["Leadership Score"].rank(ascending=False, method="min")
    sector_df["Leadership State"] = np.select([(sector_df["RS 1M"] > 0) & (sector_df["RS 3M"] > 0) & (sector_df["RS Slope 10D"] > 0), (sector_df["RS 3M"] <= 0) & (sector_df["RS 1M"] > 0) & (sector_df["RS Slope 10D"] > 0), (sector_df["RS 3M"] > 0) & ((sector_df["RS 1M"] < 0) | (sector_df["RS Slope 10D"] < 0)), (sector_df["RS 1M"] < 0) & (sector_df["RS 3M"] < 0)], ["Leading", "Improving", "Weakening", "Lagging"], default="Mixed")
    sector_df["Participation State"] = np.select([(sector_df["Positive RS 1M"] >= .65) & (sector_df["Above EMA50 Breadth"] >= .60), (sector_df["Leadership State"] == "Leading") & (sector_df["Positive RS 1M"] < .50), sector_df["Positive RS Slope"] >= .60, (sector_df["RS 3M"] > 0) & (sector_df["Positive RS 1M"] < .40)], ["Broad Leadership", "Narrow Leadership", "Broad Improvement", "Internal Deterioration"], default="Mixed")
    return sector_df.sort_values("Rank").reset_index(drop=True), sub_df.sort_values("Rank").reset_index(drop=True)


def diagnostic_table(
    frames: Mapping[str, pd.DataFrame],
    benchmark: str,
    max_sessions: int = 504,
    min_price: float = MIN_PRICE,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
) -> pd.DataFrame:
    """Walk-forward, no-lookahead signal diagnostics for the trailing two years."""
    if benchmark not in frames: return pd.DataFrame()
    bench = adjusted_ohlcv(frames[benchmark])["Close"]
    observations: List[Dict[str, object]] = []
    for ticker, raw in frames.items():
        if ticker == benchmark or len(raw) < MIN_HISTORY + 6: continue
        data = adjusted_ohlcv(raw).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        for end in range(max(MIN_HISTORY - 1, len(data) - max_sessions), len(data) - 5):
            # Historical liquidity and price eligibility are evaluated only as
            # of the signal date, rather than with today's values.
            history = data.iloc[:end + 1]
            median_dollar_volume = (history["Close"] * history["Volume"]).tail(20).median()
            if history["Close"].iloc[-1] < min_price or pd.isna(median_dollar_volume) or median_dollar_volume < min_dollar_volume:
                continue
            flags = signal_snapshot(data, end)
            count = sum(flags[s] for s in SIGNAL_COLUMNS)
            close = data["Close"].iloc[end]
            next_open, next_high, next_close = data[["Open", "High", "Close"]].iloc[end + 1]
            future = data.iloc[end + 1:end + 6]
            rs = data["Close"].div(bench.reindex(data.index))
            observations.append({"Signal Bucket": "10+" if count >= 10 else str(count), "Positive RS": bool(_pct_change(rs.iloc[:end + 1].dropna(), 21) > 0), "Gap Close-to-Open": next_open / close - 1, "Opened Higher": next_open > close, "Open-to-High": next_high / next_open - 1, "Close-to-Close": next_close / close - 1, "Forward 5D": future["Close"].iloc[-1] / close - 1, "MFE 5D": future["High"].max() / close - 1, "MAE 5D": future["Low"].min() / close - 1})
    obs = pd.DataFrame(observations)
    if obs.empty: return obs
    obs = obs[obs["Signal Bucket"].isin(["5", "6", "7", "8", "9", "10+"])].copy()
    obs["Signal Bucket"] = obs["Signal Bucket"].replace({"5": "5 to 6", "6": "5 to 6"})
    return obs.groupby(["Signal Bucket", "Positive RS"], observed=True).agg(**{"Next-session close-to-open gap": ("Gap Close-to-Open", "mean"), "Opened higher": ("Opened Higher", "mean"), "Next-session open-to-high": ("Open-to-High", "mean"), "Next-session close-to-close": ("Close-to-Close", "mean"), "Five-session forward return": ("Forward 5D", "mean"), "MFE over 5 sessions": ("MFE 5D", "mean"), "MAE over 5 sessions": ("MAE 5D", "mean"), "Observations": ("Gap Close-to-Open", "size")}).reset_index()
