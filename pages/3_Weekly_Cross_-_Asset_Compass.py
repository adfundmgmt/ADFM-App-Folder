# Weekly_Cross_Asset_Compass_Rebuilt.py
# ADFM | Weekly Cross-Asset Compass
# Clean rebuild: data validation, explicit source hierarchy, regime sleeves,
# shock detection, hedge quality, transmission, action matrix, and diagnostics.

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Weekly Cross-Asset Compass", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 1.5rem;
    max-width: 1580px;
}
.main-title {
    font-size: 2.15rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.12;
    margin-bottom: 0.2rem;
}
.subtle {
    color: #6b7280;
    font-size: 0.96rem;
    margin-bottom: 1rem;
}
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 15px 16px 13px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.035);
    height: 100%;
}
.card-title {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.055em;
    margin-bottom: 0.38rem;
}
.card-value {
    font-size: 1.36rem;
    font-weight: 760;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.card-sub {
    font-size: 0.86rem;
    color: #4b5563;
    line-height: 1.38;
}
.section-label {
    font-size: 1.12rem;
    font-weight: 760;
    margin-top: 0.65rem;
    margin-bottom: 0.65rem;
}
.commentary-box {
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 17px 19px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.035);
    margin-bottom: 12px;
}
.small-note {
    color: #6b7280;
    font-size: 0.84rem;
    line-height: 1.45;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="main-title">Weekly Cross-Asset Compass</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Regime, transmission, hedge quality, and action bias across equities, credit, rates, FX, commodities, liquidity, and volatility.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Configuration
# ============================================================
CACHE_DIR = Path(".adfm_cross_asset_cache")
CACHE_DIR.mkdir(exist_ok=True)
YAHOO_CACHE = CACHE_DIR / "yahoo_prices_last_good.pkl"
YAHOO_META = CACHE_DIR / "yahoo_prices_last_good_meta.json"
FRED_CACHE_DIR = CACHE_DIR / "fred"
FRED_CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_YEARS = 6
TRADING_DAYS = 252

# measurement controls formatting and move logic.
# price_return: percentage move
# fx_price: percentage move
# vol_level: level move
# yield_proxy_tnx: Yahoo ^TNX quote, normalized to yield percent by dividing by 10
YAHOO_ASSETS: Dict[str, Dict[str, str]] = {
    "SPY": {"label": "SPX", "group": "Equities", "measurement": "price_return"},
    "QQQ": {"label": "Nasdaq 100", "group": "Equities", "measurement": "price_return"},
    "IWM": {"label": "Russell 2000", "group": "Equities", "measurement": "price_return"},
    "FEZ": {"label": "Eurozone Equities", "group": "Equities", "measurement": "price_return"},
    "EWJ": {"label": "Japan Equities", "group": "Equities", "measurement": "price_return"},
    "EEM": {"label": "EM Equities", "group": "Equities", "measurement": "price_return"},
    "HYG": {"label": "HY Credit ETF", "group": "Credit", "measurement": "price_return"},
    "LQD": {"label": "IG Credit ETF", "group": "Credit", "measurement": "price_return"},
    "TLT": {"label": "Long Duration", "group": "Rates", "measurement": "price_return"},
    "IEF": {"label": "7-10Y Duration", "group": "Rates", "measurement": "price_return"},
    "UUP": {"label": "Dollar ETF", "group": "FX", "measurement": "price_return"},
    "USDJPY=X": {"label": "USDJPY", "group": "FX", "measurement": "fx_price"},
    "FXE": {"label": "Euro ETF", "group": "FX", "measurement": "price_return"},
    "GLD": {"label": "Gold", "group": "Commodities", "measurement": "price_return"},
    "SLV": {"label": "Silver", "group": "Commodities", "measurement": "price_return"},
    "USO": {"label": "Oil ETF", "group": "Commodities", "measurement": "price_return"},
    "CPER": {"label": "Copper ETF", "group": "Commodities", "measurement": "price_return"},
    "^VIX": {"label": "VIX", "group": "Volatility", "measurement": "vol_level"},
    "^VIX3M": {"label": "VIX 3M", "group": "Volatility", "measurement": "vol_level"},
    "^TNX": {"label": "10Y Yield Proxy", "group": "Rates", "measurement": "yield_proxy_tnx"},
}

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "DGS10": {"label": "10Y Treasury", "group": "Rates", "unit": "pct"},
    "DGS2": {"label": "2Y Treasury", "group": "Rates", "unit": "pct"},
    "T10Y2Y": {"label": "10Y-2Y Curve", "group": "Rates", "unit": "pct"},
    "T10Y3M": {"label": "10Y-3M Curve", "group": "Rates", "unit": "pct"},
    "DFII10": {"label": "10Y Real Yield", "group": "Rates", "unit": "pct"},
    "BAMLH0A0HYM2": {"label": "HY OAS", "group": "Credit", "unit": "pct"},
    "BAMLC0A0CM": {"label": "IG OAS", "group": "Credit", "unit": "pct"},
    "SOFR": {"label": "SOFR", "group": "Policy", "unit": "pct"},
    "WALCL": {"label": "Fed Balance Sheet", "group": "Liquidity", "unit": "usd_mm"},
    "RRPONTSYD": {"label": "ON RRP", "group": "Liquidity", "unit": "usd_bn"},
    "WRESBAL": {"label": "Bank Reserves", "group": "Liquidity", "unit": "usd_bn"},
}

PAIR_SPECS: List[Dict[str, str]] = [
    {"a": "SPY", "b": "TLT", "pair": "SPX vs TLT"},
    {"a": "SPY", "b": "HYG", "pair": "SPX vs HY"},
    {"a": "SPY", "b": "LQD", "pair": "SPX vs IG"},
    {"a": "SPY", "b": "UUP", "pair": "SPX vs Dollar"},
    {"a": "SPY", "b": "USO", "pair": "SPX vs Oil"},
    {"a": "SPY", "b": "GLD", "pair": "SPX vs Gold"},
    {"a": "IWM", "b": "HYG", "pair": "RTY vs HY"},
    {"a": "IWM", "b": "SPY", "pair": "RTY vs SPX"},
    {"a": "EEM", "b": "UUP", "pair": "EM vs Dollar"},
    {"a": "EWJ", "b": "USDJPY=X", "pair": "Japan vs USDJPY"},
    {"a": "GLD", "b": "TLT", "pair": "Gold vs TLT"},
    {"a": "HYG", "b": "USO", "pair": "HY vs Oil"},
]

# ============================================================
# Data classes
# ============================================================
@dataclass
class SignalResult:
    sleeve: str
    state: str
    direction: str
    score: int
    confidence: str
    drivers: List[str]
    implication: str
    coverage: int
    max_coverage: int

    @property
    def driver_text(self) -> str:
        return " ".join(self.drivers) if self.drivers else "Insufficient data."


@dataclass
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    fred: pd.DataFrame
    yahoo_meta: dict
    fred_meta: Dict[str, dict]

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Controls")
    lookback_years = st.slider("History window, years", 3, 15, DEFAULT_YEARS)
    compare_weeks = st.selectbox("Compare against", ["1 week ago", "2 weeks ago", "1 month ago"], index=0)
    week_back = {"1 week ago": 5, "2 weeks ago": 10, "1 month ago": 21}[compare_weeks]
    corr_window = st.selectbox("Rolling correlation window", [21, 42, 63], index=1)
    chart_history_months = st.slider("Chart history, months", 3, 36, 12)
    top_n_assets = st.slider("Assets in shock table", 8, 30, 18)
    top_pair_signals = st.slider("Cross-asset shifts", 4, 14, 8)
    use_fred = st.toggle("Use FRED macro series", value=True)
    show_portfolio_overlay = st.toggle("Show portfolio overlay", value=True)
    show_raw_data = st.toggle("Show data diagnostics", value=False)

    st.header("Purpose")
    st.markdown(
        """
        This rebuild separates data quality, regime scoring, shock detection, hedge quality, and action bias.

        Missing data is treated as lower confidence, not as a directional market signal.
        """
    )

# ============================================================
# Generic helpers
# ============================================================
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()


def safe_last(s: pd.Series) -> float:
    ss = safe_series(s)
    return float(ss.iloc[-1]) if not ss.empty else np.nan


def value_days_back(s: pd.Series, days: int) -> float:
    ss = safe_series(s)
    if ss.empty:
        return np.nan
    if len(ss) <= days:
        return float(ss.iloc[0])
    return float(ss.iloc[-days - 1])


def pct_change_days(s: pd.Series, days: int) -> float:
    ss = safe_series(s)
    if len(ss) <= days:
        return np.nan
    base = ss.iloc[-days - 1]
    if pd.isna(base) or base == 0:
        return np.nan
    return float((ss.iloc[-1] / base - 1.0) * 100.0)


def level_change_days(s: pd.Series, days: int) -> float:
    ss = safe_series(s)
    if len(ss) <= days:
        return np.nan
    return float(ss.iloc[-1] - ss.iloc[-days - 1])


def bps_change_days(s: pd.Series, days: int) -> float:
    chg = level_change_days(s, days)
    return float(chg * 100.0) if pd.notna(chg) else np.nan


def pct_rank_last(s: pd.Series, window: int = 252) -> float:
    ss = safe_series(s)
    if ss.empty:
        return np.nan
    if len(ss) > window:
        ss = ss.iloc[-window:]
    return float(ss.rank(pct=True).iloc[-1] * 100.0)


def zscore_last(s: pd.Series, window: int = 252) -> float:
    ss = safe_series(s)
    if len(ss) > window:
        ss = ss.iloc[-window:]
    if len(ss) < 20:
        return np.nan
    sd = ss.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return np.nan
    return float((ss.iloc[-1] - ss.mean()) / sd)


def realized_vol(ret_series: pd.Series, window: int = 21, annualization: int = 252) -> pd.Series:
    return safe_series(ret_series).rolling(window).std() * math.sqrt(annualization) * 100.0


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([safe_series(a), safe_series(b)], axis=1, join="inner").dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def ratio_series(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.concat([safe_series(a), safe_series(b)], axis=1, join="inner").dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df.iloc[:, 0] / df.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).dropna()


def score_clip(raw: float) -> int:
    """Round half away from zero and clip to [-2, +2]."""
    if pd.isna(raw):
        return 0
    if raw > 0:
        val = math.floor(raw + 0.5)
    elif raw < 0:
        val = math.ceil(raw - 0.5)
    else:
        val = 0
    return int(max(-2, min(2, val)))


def confidence_from_coverage(coverage: int, max_coverage: int) -> str:
    if max_coverage <= 0 or coverage <= 0:
        return "Low"
    ratio = coverage / max_coverage
    if ratio >= 0.75:
        return "High"
    if ratio >= 0.45:
        return "Medium"
    return "Low"


def direction_from_score(score: int) -> str:
    if score > 0:
        return "Improving/supportive"
    if score < 0:
        return "Deteriorating/tightening"
    return "Mixed/unchanged"


def fmt_level(x: float, suffix: str = "") -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:,.2f}{suffix}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.2f}%"


def fmt_bps(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.0f} bps"


def fmt_corr(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.2f}"


def color_for_state(state: str) -> str:
    positive = {"Supportive", "Constructive", "Contained", "Loose", "Working", "Disinflationary"}
    negative = {"Fragile", "Defensive", "Tightening", "Elevated", "Broken", "Hot", "Stress"}
    neutral = {"Mixed", "Neutral", "Watchful", "Balanced", "Choppy", "Unknown"}
    if state in positive:
        return "#166534"
    if state in negative:
        return "#991b1b"
    if state in neutral:
        return "#92400e"
    return "#111827"


def card(title: str, value: str, sub: str, state: str = "") -> str:
    color = color_for_state(state)
    return f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value" style="color:{color};">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>
    """

# ============================================================
# Data fetching
# ============================================================
def extract_yahoo_close(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Robustly extract close prices from yfinance output.

    yfinance can return MultiIndex columns as either:
    - level 0 = ticker, level 1 = field, usually with group_by='ticker'
    - level 0 = field, level 1 = ticker
    This function handles both instead of assuming one shape.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    tickers = list(tickers)
    frames: List[pd.Series] = []

    if isinstance(raw.columns, pd.MultiIndex):
        levels = [list(raw.columns.get_level_values(i).unique()) for i in range(raw.columns.nlevels)]
        for ticker in tickers:
            series = None

            if ticker in levels[0]:
                try:
                    sub = raw.xs(ticker, axis=1, level=0)
                    col = "Adj Close" if "Adj Close" in sub.columns else "Close" if "Close" in sub.columns else None
                    if col:
                        series = pd.to_numeric(sub[col], errors="coerce").rename(ticker)
                except Exception:
                    series = None

            if series is None and raw.columns.nlevels > 1 and ticker in levels[1]:
                try:
                    sub = raw.xs(ticker, axis=1, level=1)
                    col = "Adj Close" if "Adj Close" in sub.columns else "Close" if "Close" in sub.columns else None
                    if col:
                        series = pd.to_numeric(sub[col], errors="coerce").rename(ticker)
                except Exception:
                    series = None

            if series is not None and not series.dropna().empty:
                frames.append(series)
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close" if "Close" in raw.columns else None
        if col and len(tickers) == 1:
            frames.append(pd.to_numeric(raw[col], errors="coerce").rename(tickers[0]))

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).sort_index()
    prices = clean_index(prices)
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all")
    return prices


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_yahoo_prices(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, dict]:
    tickers = sorted(set(tickers))
    meta = {
        "source": "yahoo",
        "requested": len(tickers),
        "returned": 0,
        "missing": [],
        "status": "live_attempted",
    }

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
            ignore_tz=True,
        )
        prices = extract_yahoo_close(raw, tickers)
    except Exception as exc:
        prices = pd.DataFrame()
        meta["error"] = str(exc)

    if prices.empty:
        if YAHOO_CACHE.exists():
            try:
                cached = pd.read_pickle(YAHOO_CACHE)
                cached_meta = read_json(YAHOO_META)
                cached_meta["source"] = "last_good_yahoo_cache"
                cached_meta["status"] = "fallback_cache"
                return cached, cached_meta
            except Exception as exc:
                meta["cache_error"] = str(exc)
        meta["status"] = "failed"
        return pd.DataFrame(), meta

    meta["returned"] = int(prices.shape[1])
    meta["missing"] = sorted(set(tickers) - set(prices.columns))
    meta["last_date"] = str(prices.index.max().date()) if not prices.empty else None
    meta["status"] = "live_ok"

    min_good = max(4, int(len(tickers) * 0.55))
    if prices.shape[1] >= min_good:
        try:
            prices.to_pickle(YAHOO_CACHE)
            write_json(YAHOO_META, meta)
        except Exception:
            pass

    return prices, meta


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_fred_series(series_id: str, start: str) -> Tuple[pd.Series, dict]:
    cache_path = FRED_CACHE_DIR / f"{series_id}.pkl"
    meta_path = FRED_CACHE_DIR / f"{series_id}.json"
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    meta = {"series": series_id, "source": "fred", "status": "live_attempted"}

    try:
        response = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if "observation_date" not in df.columns or series_id not in df.columns:
            raise ValueError("Unexpected FRED CSV shape")

        df["observation_date"] = pd.to_datetime(df["observation_date"])
        vals = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
        s = pd.Series(vals.values, index=df["observation_date"], name=series_id).dropna()
        s = s[s.index >= pd.to_datetime(start)]
        if s.empty:
            raise ValueError("FRED series empty after start filter")

        s = safe_series(s)
        s.to_pickle(cache_path)
        meta["last_date"] = str(s.index.max().date())
        meta["status"] = "live_ok"
        write_json(meta_path, meta)
        return s, meta
    except Exception as exc:
        if cache_path.exists():
            try:
                s = pd.read_pickle(cache_path)
                cached_meta = read_json(meta_path)
                cached_meta["source"] = "last_good_fred_cache"
                cached_meta["status"] = "fallback_cache"
                cached_meta["live_error"] = str(exc)
                return safe_series(s), cached_meta
            except Exception as cache_exc:
                meta["cache_error"] = str(cache_exc)
        meta["status"] = "failed"
        meta["error"] = str(exc)
        return pd.Series(dtype=float, name=series_id), meta


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_fred_panel(series_ids: List[str], start: str) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    series: List[pd.Series] = []
    metas: Dict[str, dict] = {}
    for sid in series_ids:
        s, meta = fetch_fred_series(sid, start)
        metas[sid] = meta
        if not s.empty:
            series.append(s)
        time.sleep(0.03)

    if not series:
        return pd.DataFrame(), metas

    panel = pd.concat(series, axis=1).sort_index()
    panel = clean_index(panel).ffill()
    return panel, metas


def align_fred_to_prices(fred_panel: pd.DataFrame, price_index: pd.Index) -> pd.DataFrame:
    if fred_panel.empty or len(price_index) == 0:
        return pd.DataFrame(index=price_index)
    full_idx = pd.bdate_range(price_index.min(), price_index.max())
    fred_b = fred_panel.reindex(full_idx).ffill()
    return fred_b.reindex(price_index).ffill()

# ============================================================
# Market-data accessors and normalized series
# ============================================================
def p(md: MarketData, ticker: str) -> pd.Series:
    return safe_series(md.prices[ticker]) if ticker in md.prices.columns else pd.Series(dtype=float)


def r(md: MarketData, ticker: str) -> pd.Series:
    return safe_series(md.returns[ticker]) if ticker in md.returns.columns else pd.Series(dtype=float)


def f(md: MarketData, sid: str) -> pd.Series:
    return safe_series(md.fred[sid]) if sid in md.fred.columns else pd.Series(dtype=float)


def ten_year_yield(md: MarketData) -> pd.Series:
    fred_10y = f(md, "DGS10")
    if not fred_10y.empty:
        return fred_10y
    tnx = p(md, "^TNX")
    if not tnx.empty:
        return (tnx / 10.0).rename("DGS10_fallback_from_TNX")
    return pd.Series(dtype=float)


def two_year_yield(md: MarketData) -> pd.Series:
    return f(md, "DGS2")


def ten_year_real_yield(md: MarketData) -> pd.Series:
    return f(md, "DFII10")


def curve_10s2s(md: MarketData) -> pd.Series:
    direct = f(md, "T10Y2Y")
    if not direct.empty:
        return direct
    ten = ten_year_yield(md)
    two = two_year_yield(md)
    if ten.empty or two.empty:
        return pd.Series(dtype=float)
    df = pd.concat([ten, two], axis=1, join="inner").dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename("10s2s_fallback")

# ============================================================
# Regime model
# ============================================================
def make_signal(
    sleeve: str,
    state: str,
    raw_score: float,
    drivers: List[str],
    implication: str,
    coverage: int,
    max_coverage: int,
    direction: Optional[str] = None,
) -> SignalResult:
    score = score_clip(raw_score)
    confidence = confidence_from_coverage(coverage, max_coverage)
    return SignalResult(
        sleeve=sleeve,
        state=state if coverage > 0 else "Unknown",
        direction=direction or direction_from_score(score),
        score=score if coverage > 0 else 0,
        confidence=confidence,
        drivers=drivers,
        implication=implication,
        coverage=coverage,
        max_coverage=max_coverage,
    )


def compute_risk_appetite(md: MarketData, week: int) -> SignalResult:
    spy = p(md, "SPY")
    qqq = p(md, "QQQ")
    iwm = p(md, "IWM")
    qqq_spy = ratio_series(qqq, spy)
    iwm_spy = ratio_series(iwm, spy)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 5

    spy_1m = pct_change_days(spy, 21)
    spy_1w = pct_change_days(spy, week)
    if pd.notna(spy_1m):
        coverage += 1
        raw += 1.0 if spy_1m > 2.0 else -1.0 if spy_1m < -2.0 else 0.0
        drivers.append(f"SPX 1M {fmt_pct(spy_1m)} and {compare_weeks} {fmt_pct(spy_1w)}.")

    spy_last = safe_last(spy)
    ma50 = safe_last(spy.rolling(50).mean()) if len(spy) >= 50 else np.nan
    ma200 = safe_last(spy.rolling(200).mean()) if len(spy) >= 200 else np.nan
    if pd.notna(spy_last) and pd.notna(ma50):
        coverage += 1
        raw += 0.75 if spy_last > ma50 else -0.75
        drivers.append(f"SPX is {'above' if spy_last > ma50 else 'below'} its 50D moving average.")
    if pd.notna(spy_last) and pd.notna(ma200):
        coverage += 1
        raw += 0.75 if spy_last > ma200 else -0.75
        drivers.append(f"SPX is {'above' if spy_last > ma200 else 'below'} its 200D moving average.")

    qqq_rel = pct_change_days(qqq_spy, 21)
    if pd.notna(qqq_rel):
        coverage += 1
        raw += 0.5 if qqq_rel > 1.0 else -0.5 if qqq_rel < -1.0 else 0.0
        drivers.append(f"QQQ/SPY 1M relative return {fmt_pct(qqq_rel)}.")

    iwm_rel = pct_change_days(iwm_spy, 21)
    if pd.notna(iwm_rel):
        coverage += 1
        raw += 0.5 if iwm_rel > 1.0 else -0.5 if iwm_rel < -1.0 else 0.0
        drivers.append(f"IWM/SPY 1M relative return {fmt_pct(iwm_rel)}.")

    score = score_clip(raw)
    state = "Constructive" if score > 0 else "Defensive" if score < 0 else "Neutral"
    implication = "Press leadership only when breadth confirms. If index strength is narrow, treat beta as lower quality."
    return make_signal("Risk appetite", state, raw, drivers, implication, coverage, max_coverage)


def compute_leadership(md: MarketData, week: int) -> SignalResult:
    spy = p(md, "SPY")
    qqq = p(md, "QQQ")
    iwm = p(md, "IWM")
    eem = p(md, "EEM")
    fez = p(md, "FEZ")

    ratios = {
        "QQQ/SPY": ratio_series(qqq, spy),
        "IWM/SPY": ratio_series(iwm, spy),
        "EEM/SPY": ratio_series(eem, spy),
        "FEZ/SPY": ratio_series(fez, spy),
    }
    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = len(ratios)

    for name, ser in ratios.items():
        chg = pct_change_days(ser, 21)
        if pd.isna(chg):
            continue
        coverage += 1
        if name == "QQQ/SPY":
            raw += 0.75 if chg > 1 else -0.25 if chg < -1 else 0.0
        elif name == "IWM/SPY":
            raw += 0.5 if chg > 1 else -0.5 if chg < -1 else 0.0
        else:
            raw += 0.25 if chg > 1 else -0.25 if chg < -1 else 0.0
        drivers.append(f"{name} 1M {fmt_pct(chg)}.")

    score = score_clip(raw)
    if score > 0:
        state = "Constructive"
    elif score < 0:
        state = "Narrowing"
    else:
        state = "Mixed"
    implication = "Leadership quality matters more than index direction. Narrow mega-cap leadership supports concentration but not broad gross expansion."
    return make_signal("Leadership", state, raw, drivers, implication, coverage, max_coverage)


def compute_volatility(md: MarketData, week: int) -> SignalResult:
    vix = p(md, "^VIX")
    vix3m = p(md, "^VIX3M")
    spy_rv = realized_vol(r(md, "SPY"), 21)
    vix_term = ratio_series(vix3m, vix) - 1.0 if not vix.empty and not vix3m.empty else pd.Series(dtype=float)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 4

    vix_now = safe_last(vix)
    vix_w = level_change_days(vix, week)
    if pd.notna(vix_now):
        coverage += 1
        raw += 1.0 if vix_now < 16 else -1.0 if vix_now >= 22 else 0.0
        raw += -1.0 if vix_now >= 28 else 0.0
        drivers.append(f"VIX {fmt_level(vix_now)} with {compare_weeks} change {fmt_level(vix_w)}.")

    term_now = safe_last(vix_term)
    if pd.notna(term_now):
        coverage += 1
        raw += 1.0 if term_now > 0.08 else -1.0 if term_now < 0 else 0.0
        drivers.append(f"VIX3M/VIX - 1 at {fmt_pct(term_now * 100)}.")

    rv_now = safe_last(spy_rv)
    rv_pct = pct_rank_last(spy_rv, 252)
    if pd.notna(rv_pct):
        coverage += 1
        raw += 0.5 if rv_pct < 40 else -0.5 if rv_pct > 75 else 0.0
        drivers.append(f"SPX 21D realized vol {fmt_level(rv_now, '%')} at {rv_pct:.0f}th percentile.")

    if pd.notna(vix_now) and pd.notna(rv_now):
        coverage += 1
        spread = vix_now - rv_now
        drivers.append(f"VIX minus realized vol spread {fmt_level(spread)}.")

    score = score_clip(raw)
    state = "Constructive" if score > 0 else "Elevated" if score < 0 else "Watchful"
    implication = "Low vol supports carry and gross, but rising vol with poor term structure argues for defined-risk hedges."
    return make_signal("Volatility", state, raw, drivers, implication, coverage, max_coverage)


def compute_rates(md: MarketData, week: int) -> SignalResult:
    ten = ten_year_yield(md)
    real = ten_year_real_yield(md)
    curve = curve_10s2s(md)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 4

    ten_now = safe_last(ten)
    ten_w = bps_change_days(ten, week)
    if pd.notna(ten_w):
        coverage += 1
        raw += 1.0 if ten_w <= -12 else -1.0 if ten_w >= 12 else 0.0
        drivers.append(f"10Y yield {fmt_level(ten_now, '%')} moved {fmt_bps(ten_w)}.")

    real_now = safe_last(real)
    real_w = bps_change_days(real, week)
    if pd.notna(real_w):
        coverage += 1
        raw += 1.0 if real_w <= -10 else -1.0 if real_w >= 10 else 0.0
        drivers.append(f"10Y real yield {fmt_level(real_now, '%')} moved {fmt_bps(real_w)}.")

    curve_now = safe_last(curve)
    curve_w = bps_change_days(curve, week)
    if pd.notna(curve_now):
        coverage += 1
        raw += -0.5 if curve_now < -0.50 else 0.0
        drivers.append(f"10s2s curve {fmt_level(curve_now, '%')} moved {fmt_bps(curve_w)}.")

    if pd.notna(curve_w) and pd.notna(ten_w):
        coverage += 1
        if curve_w > 15 and ten_w > 0:
            raw -= 0.5
            drivers.append("Bear steepening pressure is present.")
        elif curve_w > 15 and ten_w <= 0:
            raw += 0.25
            drivers.append("Bull steepening is less hostile to risk than a bear steepener.")

    score = score_clip(raw)
    state = "Supportive" if score > 0 else "Tightening" if score < 0 else "Neutral"
    implication = "Rates are supportive when nominal and real yields fall together. Rising real yields should tighten equity duration and speculative beta."
    return make_signal("Rates", state, raw, drivers, implication, coverage, max_coverage)


def compute_credit(md: MarketData, week: int) -> SignalResult:
    hy_oas = f(md, "BAMLH0A0HYM2")
    ig_oas = f(md, "BAMLC0A0CM")
    hyg = p(md, "HYG")
    lqd = p(md, "LQD")
    hyg_lqd = ratio_series(hyg, lqd)
    spy_hyg_corr = rolling_corr(r(md, "SPY"), r(md, "HYG"), corr_window)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 5

    hy_now = safe_last(hy_oas)
    hy_w = bps_change_days(hy_oas, week)
    if pd.notna(hy_w):
        coverage += 1
        raw += 1.0 if hy_w <= -15 else -1.0 if hy_w >= 15 else 0.0
        drivers.append(f"HY OAS {fmt_level(hy_now, '%')} moved {fmt_bps(hy_w)}.")

    ig_now = safe_last(ig_oas)
    ig_w = bps_change_days(ig_oas, week)
    if pd.notna(ig_w):
        coverage += 1
        raw += 0.5 if ig_w <= -5 else -0.5 if ig_w >= 5 else 0.0
        drivers.append(f"IG OAS {fmt_level(ig_now, '%')} moved {fmt_bps(ig_w)}.")

    hyg_1w = pct_change_days(hyg, week)
    if pd.notna(hyg_1w):
        coverage += 1
        raw += 0.5 if hyg_1w > 0.5 else -0.5 if hyg_1w < -0.75 else 0.0
        drivers.append(f"HYG {compare_weeks} return {fmt_pct(hyg_1w)}.")

    rel_1m = pct_change_days(hyg_lqd, 21)
    if pd.notna(rel_1m):
        coverage += 1
        raw += 0.5 if rel_1m > 0 else -0.5 if rel_1m < -1.0 else 0.0
        drivers.append(f"HYG/LQD 1M {fmt_pct(rel_1m)}.")

    corr_now = safe_last(spy_hyg_corr)
    if pd.notna(corr_now):
        coverage += 1
        drivers.append(f"SPX-HYG rolling correlation {fmt_corr(corr_now)}.")

    score = score_clip(raw)
    state = "Contained" if score > 0 else "Tightening" if score < 0 else "Watchful"
    implication = "Credit is the confirmation layer. Spread widening should cut tolerance for small caps, lower-quality cyclicals, and levered beta."
    return make_signal("Credit", state, raw, drivers, implication, coverage, max_coverage)


def compute_dollar_liquidity(md: MarketData, week: int) -> SignalResult:
    uup = p(md, "UUP")
    usdjpy = p(md, "USDJPY=X")
    eem = p(md, "EEM")
    em_dollar_corr = rolling_corr(r(md, "EEM"), r(md, "UUP"), corr_window)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 5

    uup_1w = pct_change_days(uup, week)
    uup_1m = pct_change_days(uup, 21)
    if pd.notna(uup_1w):
        coverage += 1
        raw += 1.0 if uup_1w <= -0.75 else -1.0 if uup_1w >= 0.75 else 0.0
        drivers.append(f"UUP {compare_weeks} {fmt_pct(uup_1w)}.")
    if pd.notna(uup_1m):
        coverage += 1
        raw += 1.0 if uup_1m <= -1.5 else -1.0 if uup_1m >= 1.5 else 0.0
        drivers.append(f"UUP 1M {fmt_pct(uup_1m)}.")

    jpy_1w = pct_change_days(usdjpy, week)
    if pd.notna(jpy_1w):
        coverage += 1
        raw += -0.5 if jpy_1w >= 1.5 else 0.25 if jpy_1w <= -1.5 else 0.0
        drivers.append(f"USDJPY {compare_weeks} {fmt_pct(jpy_1w)}.")

    eem_1w = pct_change_days(eem, week)
    if pd.notna(eem_1w):
        coverage += 1
        drivers.append(f"EEM {compare_weeks} {fmt_pct(eem_1w)}.")

    corr_now = safe_last(em_dollar_corr)
    if pd.notna(corr_now):
        coverage += 1
        raw += -0.5 if corr_now <= -0.35 and pd.notna(uup_1w) and uup_1w > 0 else 0.0
        drivers.append(f"EM-dollar correlation {fmt_corr(corr_now)}.")

    score = score_clip(raw)
    state = "Loose" if score > 0 else "Tightening" if score < 0 else "Mixed"
    implication = "A stronger dollar tightens global financial conditions first through EM, commodities, and external debt channels."
    return make_signal("Dollar liquidity", state, raw, drivers, implication, coverage, max_coverage)


def compute_commodities(md: MarketData, week: int) -> SignalResult:
    oil = p(md, "USO")
    copper = p(md, "CPER")
    gold = p(md, "GLD")
    real = ten_year_real_yield(md)
    oil_rv = realized_vol(r(md, "USO"), 21)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 5

    oil_1w = pct_change_days(oil, week)
    oil_1m = pct_change_days(oil, 21)
    if pd.notna(oil_1w):
        coverage += 1
        raw += -1.0 if oil_1w >= 5 else 0.75 if oil_1w <= -5 else 0.0
        drivers.append(f"Oil ETF {compare_weeks} {fmt_pct(oil_1w)} and 1M {fmt_pct(oil_1m)}.")

    copper_1w = pct_change_days(copper, week)
    if pd.notna(copper_1w):
        coverage += 1
        raw += 0.25 if copper_1w > 2 and (pd.isna(oil_1w) or oil_1w < 3) else -0.25 if copper_1w < -3 else 0.0
        drivers.append(f"Copper ETF {compare_weeks} {fmt_pct(copper_1w)}.")

    gold_1w = pct_change_days(gold, week)
    real_w = bps_change_days(real, week)
    if pd.notna(gold_1w):
        coverage += 1
        raw += -0.5 if gold_1w > 2 and pd.notna(real_w) and real_w > 5 else 0.0
        drivers.append(f"Gold {compare_weeks} {fmt_pct(gold_1w)}.")

    if pd.notna(real_w):
        coverage += 1
        drivers.append(f"Real 10Y yield moved {fmt_bps(real_w)}.")

    oil_vol_z = zscore_last(oil_rv, 252)
    if pd.notna(oil_vol_z):
        coverage += 1
        raw += -0.75 if oil_vol_z >= 1.25 else 0.0
        drivers.append(f"Oil realized-vol z-score {fmt_level(oil_vol_z)}.")

    score = score_clip(raw)
    state = "Disinflationary" if score > 0 else "Hot" if score < 0 else "Balanced"
    implication = "Oil is the key macro commodity. A sharp oil rally can transmit into inflation expectations, margins, and rates."
    return make_signal("Commodities", state, raw, drivers, implication, coverage, max_coverage)


def compute_liquidity_plumbing(md: MarketData, week: int) -> SignalResult:
    walcl = f(md, "WALCL")
    rrp = f(md, "RRPONTSYD")
    reserves = f(md, "WRESBAL")

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 3

    walcl_1m = pct_change_days(walcl, 21)
    if pd.notna(walcl_1m):
        coverage += 1
        raw += 0.5 if walcl_1m > 0.25 else -0.5 if walcl_1m < -0.25 else 0.0
        drivers.append(f"Fed balance sheet 1M {fmt_pct(walcl_1m)}.")

    reserves_1m = pct_change_days(reserves, 21)
    if pd.notna(reserves_1m):
        coverage += 1
        raw += 0.75 if reserves_1m > 1.0 else -0.75 if reserves_1m < -1.0 else 0.0
        drivers.append(f"Bank reserves 1M {fmt_pct(reserves_1m)}.")

    rrp_1m = pct_change_days(rrp, 21)
    if pd.notna(rrp_1m):
        coverage += 1
        # Falling RRP can be liquidity-positive if reserves are stable, but signal is context-dependent.
        raw += 0.25 if rrp_1m < -10 and (pd.isna(reserves_1m) or reserves_1m >= -1.0) else 0.0
        drivers.append(f"ON RRP 1M {fmt_pct(rrp_1m)}.")

    score = score_clip(raw)
    state = "Loose" if score > 0 else "Tightening" if score < 0 else "Mixed"
    implication = "Liquidity plumbing is a slow-moving confirmation layer. Deterioration matters more when credit and dollar signals agree."
    return make_signal("Liquidity plumbing", state, raw, drivers, implication, coverage, max_coverage)


def compute_hedge_quality(md: MarketData, week: int) -> SignalResult:
    spy = p(md, "SPY")
    tlt = p(md, "TLT")
    vix = p(md, "^VIX")
    spy_tlt_corr = rolling_corr(r(md, "SPY"), r(md, "TLT"), corr_window)

    raw = 0.0
    drivers: List[str] = []
    coverage = 0
    max_coverage = 4

    corr_now = safe_last(spy_tlt_corr)
    corr_w = level_change_days(spy_tlt_corr, week)
    if pd.notna(corr_now):
        coverage += 1
        raw += 1.0 if corr_now <= -0.25 else -1.0 if corr_now >= 0.25 else 0.0
        drivers.append(f"SPX-TLT correlation {fmt_corr(corr_now)} with {compare_weeks} change {fmt_corr(corr_w)}.")

    spy_1w = pct_change_days(spy, week)
    tlt_1w = pct_change_days(tlt, week)
    if pd.notna(spy_1w) and pd.notna(tlt_1w):
        coverage += 1
        if spy_1w < -1.0:
            raw += 1.0 if tlt_1w > 0 else -1.0
        drivers.append(f"SPX {compare_weeks} {fmt_pct(spy_1w)}; TLT {compare_weeks} {fmt_pct(tlt_1w)}.")

    vix_w = level_change_days(vix, week)
    if pd.notna(vix_w):
        coverage += 1
        if vix_w > 2 and pd.notna(tlt_1w) and tlt_1w < 0:
            raw -= 1.0
        drivers.append(f"VIX changed {fmt_level(vix_w)} over {compare_weeks}.")

    if pd.notna(tlt_1w):
        coverage += 1
        drivers.append(f"Duration hedge realized return {fmt_pct(tlt_1w)} over the comparison window.")

    score = score_clip(raw)
    state = "Working" if score > 0 else "Broken" if score < 0 else "Mixed"
    implication = "When stock-bond correlation is positive, duration is a macro position, not automatic equity protection."
    return make_signal("Hedge quality", state, raw, drivers, implication, coverage, max_coverage)


def compute_all_signals(md: MarketData, week: int) -> List[SignalResult]:
    return [
        compute_risk_appetite(md, week),
        compute_leadership(md, week),
        compute_volatility(md, week),
        compute_rates(md, week),
        compute_credit(md, week),
        compute_dollar_liquidity(md, week),
        compute_commodities(md, week),
        compute_liquidity_plumbing(md, week),
        compute_hedge_quality(md, week),
    ]


def composite_regime(signals: List[SignalResult]) -> Tuple[str, float, str]:
    usable = [s for s in signals if s.confidence != "Low" or s.coverage > 0]
    if not usable:
        return "Unknown", 0.0, "Low"

    weights = {"High": 1.0, "Medium": 0.65, "Low": 0.35}
    total_weight = sum(weights[s.confidence] for s in usable)
    composite = sum(s.score * weights[s.confidence] for s in usable) / total_weight if total_weight else 0.0

    if composite >= 1.10:
        regime = "Supportive"
    elif composite >= 0.35:
        regime = "Constructive"
    elif composite <= -1.10:
        regime = "Fragile"
    elif composite <= -0.35:
        regime = "Choppy"
    else:
        regime = "Mixed"

    avg_coverage = np.mean([s.coverage / s.max_coverage for s in signals if s.max_coverage])
    confidence = "High" if avg_coverage >= 0.75 else "Medium" if avg_coverage >= 0.45 else "Low"
    return regime, float(composite), confidence


def signals_to_frame(signals: List[SignalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Sleeve": s.sleeve,
                "State": s.state,
                "Direction": s.direction,
                "Score": s.score,
                "Confidence": s.confidence,
                "Coverage": f"{s.coverage}/{s.max_coverage}",
                "Drivers": s.driver_text,
                "Portfolio implication": s.implication,
            }
            for s in signals
        ]
    )

# ============================================================
# Shock detector, transmission, action matrix
# ============================================================
def normalized_yahoo_series(md: MarketData, ticker: str) -> pd.Series:
    s = p(md, ticker)
    if ticker == "^TNX":
        return (s / 10.0).rename(ticker)
    return s


def build_market_tape(md: MarketData, week: int) -> pd.DataFrame:
    rows: List[dict] = []

    for ticker, meta in YAHOO_ASSETS.items():
        if ticker not in md.prices.columns:
            continue
        s = normalized_yahoo_series(md, ticker)
        if s.empty:
            continue
        measurement = meta["measurement"]
        last = safe_last(s)
        pctile = pct_rank_last(s, 252)

        if measurement in {"vol_level"}:
            move_1w = level_change_days(s, week)
            move_1m = level_change_days(s, 21)
            move_3m = level_change_days(s, 63)
            display_1w = fmt_level(move_1w)
            display_1m = fmt_level(move_1m)
            display_3m = fmt_level(move_3m)
            hist_move = s.diff(week)
            shock_z = zscore_last(hist_move, 252)
            abs_move = abs(shock_z) if pd.notna(shock_z) else abs(move_1w) if pd.notna(move_1w) else 0
            display_level = fmt_level(last)
        elif measurement == "yield_proxy_tnx":
            move_1w = bps_change_days(s, week)
            move_1m = bps_change_days(s, 21)
            move_3m = bps_change_days(s, 63)
            display_1w = fmt_bps(move_1w)
            display_1m = fmt_bps(move_1m)
            display_3m = fmt_bps(move_3m)
            hist_move = s.diff(week) * 100.0
            shock_z = zscore_last(hist_move, 252)
            abs_move = abs(shock_z) if pd.notna(shock_z) else abs(move_1w) if pd.notna(move_1w) else 0
            display_level = fmt_level(last, "%")
        else:
            move_1w = pct_change_days(s, week)
            move_1m = pct_change_days(s, 21)
            move_3m = pct_change_days(s, 63)
            display_1w = fmt_pct(move_1w)
            display_1m = fmt_pct(move_1m)
            display_3m = fmt_pct(move_3m)
            hist_move = s.pct_change(week) * 100.0
            shock_z = zscore_last(hist_move, 252)
            abs_move = abs(shock_z) if pd.notna(shock_z) else abs(move_1w) if pd.notna(move_1w) else 0
            display_level = fmt_level(last)

        rows.append(
            {
                "Asset": meta["label"],
                "Ticker": ticker,
                "Group": meta["group"],
                "Level": display_level,
                "Move": display_1w,
                "1M": display_1m,
                "3M": display_3m,
                "1Y pctile": f"{pctile:.0f}%" if pd.notna(pctile) else "NA",
                "Shock z": shock_z,
                "Sort": abs_move,
            }
        )

    for sid, meta in FRED_SERIES.items():
        s = f(md, sid)
        if s.empty:
            continue
        last = safe_last(s)
        pctile = pct_rank_last(s, 252)
        unit = meta["unit"]
        if unit == "pct":
            move_1w = bps_change_days(s, week)
            move_1m = bps_change_days(s, 21)
            move_3m = bps_change_days(s, 63)
            hist_move = s.diff(week) * 100.0
            level = fmt_level(last, "%")
            display_1w = fmt_bps(move_1w)
            display_1m = fmt_bps(move_1m)
            display_3m = fmt_bps(move_3m)
        else:
            move_1w = pct_change_days(s, week)
            move_1m = pct_change_days(s, 21)
            move_3m = pct_change_days(s, 63)
            hist_move = s.pct_change(week) * 100.0
            level = fmt_level(last)
            display_1w = fmt_pct(move_1w)
            display_1m = fmt_pct(move_1m)
            display_3m = fmt_pct(move_3m)

        shock_z = zscore_last(hist_move, 252)
        abs_move = abs(shock_z) if pd.notna(shock_z) else abs(move_1w) if pd.notna(move_1w) else 0
        rows.append(
            {
                "Asset": meta["label"],
                "Ticker": sid,
                "Group": meta["group"],
                "Level": level,
                "Move": display_1w,
                "1M": display_1m,
                "3M": display_3m,
                "1Y pctile": f"{pctile:.0f}%" if pd.notna(pctile) else "NA",
                "Shock z": shock_z,
                "Sort": abs_move,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Sort", ascending=False).reset_index(drop=True)


def pair_interpretation(pair: str, rho_now: float, rho_delta: float, pctile: float) -> Tuple[str, str]:
    shift = "rising" if pd.notna(rho_delta) and rho_delta > 0.10 else "falling" if pd.notna(rho_delta) and rho_delta < -0.10 else "stable"
    extreme = "high" if pd.notna(pctile) and pctile >= 80 else "low" if pd.notna(pctile) and pctile <= 20 else "normal"

    if pair == "SPX vs TLT":
        if pd.notna(rho_now) and rho_now >= 0.25:
            return "Duration hedge is impaired", f"Correlation is {shift} and {extreme}; do not rely on TLT as automatic equity protection."
        if pd.notna(rho_now) and rho_now <= -0.25:
            return "Duration is cushioning equity risk", f"Correlation is {shift}; duration can remain in the hedge stack."
        return "Duration hedge is mixed", f"Correlation is {shift}; validate hedge behavior before leaning on it."

    if pair == "SPX vs HY":
        if pd.notna(rho_now) and rho_now >= 0.50:
            return "Credit is tightly linked to equities", f"Correlation is {shift}; HY should guide cyclicals and gross exposure."
        return "Credit linkage is moderate", f"Correlation is {shift}; use credit as confirmation."

    if pair == "EM vs Dollar":
        if pd.notna(rho_now) and rho_now <= -0.35:
            return "Dollar is tightening global beta", f"Correlation is {shift}; respect pressure on EM and commodity-sensitive risk."
        return "Dollar linkage is less dominant", f"Correlation is {shift}; cross-check with credit and rates."

    if pair == "Japan vs USDJPY":
        if pd.notna(rho_now) and rho_now >= 0.35:
            return "Japan remains linked to yen weakness", f"Correlation is {shift}; do not separate Japan equity view from FX."
        return "Japan-FX transmission is weaker", f"Correlation is {shift}; Japan may trade more on equity beta."

    if pair == "Gold vs TLT":
        if pd.notna(rho_now) and rho_now <= -0.25:
            return "Gold is diversifying duration", f"Correlation is {shift}; gold may be carrying policy distrust or geopolitical premium."
        return "Gold-duration linkage is mixed", f"Correlation is {shift}; validate gold with real yields and dollar."

    return "Cross-asset linkage is moving", f"Correlation is {shift} and percentile is {extreme}; use as context, not a standalone signal."


def build_pair_table(md: MarketData, window: int, week: int) -> pd.DataFrame:
    rows: List[dict] = []
    for spec in PAIR_SPECS:
        a, b, pair = spec["a"], spec["b"], spec["pair"]
        if a not in md.returns.columns or b not in md.returns.columns:
            continue
        rc = rolling_corr(r(md, a), r(md, b), window)
        if len(rc) < week + 5:
            continue
        rho_now = safe_last(rc)
        rho_delta = level_change_days(rc, week)
        pctile = pct_rank_last(rc, 504)
        signal_score = 0.0
        signal_score += min(abs(rho_now) / 0.8, 1.0) * 45 if pd.notna(rho_now) else 0.0
        signal_score += min(abs(rho_delta) / 0.25, 1.0) * 40 if pd.notna(rho_delta) else 0.0
        signal_score += abs(pctile - 50) / 50 * 15 if pd.notna(pctile) else 0.0
        interp, action = pair_interpretation(pair, rho_now, rho_delta, pctile)
        rows.append(
            {
                "Pair": pair,
                "ρ now": rho_now,
                "Δρ": rho_delta,
                "1Y pctile": pctile,
                "Signal": round(signal_score, 1),
                "Interpretation": interp,
                "Action": action,
                "Series": rc,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Signal", "Δρ"], ascending=[False, False]).reset_index(drop=True)


def build_action_matrix(signals: List[SignalResult], regime: str) -> pd.DataFrame:
    state = {s.sleeve: s.state for s in signals}
    score = {s.sleeve: s.score for s in signals}

    risk_state = state.get("Risk appetite", "Unknown")
    leadership_state = state.get("Leadership", "Unknown")
    credit_state = state.get("Credit", "Unknown")
    rates_state = state.get("Rates", "Unknown")
    dollar_state = state.get("Dollar liquidity", "Unknown")
    commodity_state = state.get("Commodities", "Unknown")
    hedge_state = state.get("Hedge quality", "Unknown")
    vol_state = state.get("Volatility", "Unknown")

    rows: List[dict] = []

    if regime in {"Supportive", "Constructive"} and credit_state in {"Contained", "Watchful"}:
        gross = "Press selectively"
        gross_reason = "Regime and credit are not forcing broad de-risking, but leadership quality decides what deserves capital."
    elif regime in {"Fragile", "Choppy"}:
        gross = "Cut or keep tight"
        gross_reason = "The regime is not paying for broad gross. Positions need tighter invalidation and cleaner hedge math."
    else:
        gross = "Balanced"
        gross_reason = "The tape is mixed. Favor relative value and confirmed leadership over index-level conviction."

    if risk_state == "Constructive" and leadership_state == "Constructive" and credit_state == "Contained":
        net = "Modestly long / beta selective"
    elif regime in {"Fragile", "Choppy"} or credit_state == "Tightening":
        net = "Lower beta"
    else:
        net = "Neutral to tactical"

    if hedge_state == "Working":
        duration = "Useful hedge"
        duration_reason = "Stock-bond correlation and realized TLT behavior support keeping duration in the hedge stack."
    elif hedge_state == "Broken":
        duration = "Treat as macro risk"
        duration_reason = "Duration is not providing clean equity protection. Use equity optionality, credit overlays, or lower gross."
    else:
        duration = "Mixed hedge"
        duration_reason = "Duration can help, but sizing should reflect unstable correlation behavior."

    if credit_state == "Tightening":
        credit = "Add credit hedge / reduce low quality"
        credit_reason = "Spread or credit ETF behavior is moving against risk. Lower-quality cyclicals deserve tighter limits."
    else:
        credit = "Use as confirmation"
        credit_reason = "Credit is not forcing broad de-risking. Watch whether it confirms or rejects index moves."

    if dollar_state == "Tightening":
        fx = "Respect USD pressure"
        fx_reason = "Dollar strength tightens global beta and argues for caution in EM, commodity FX, and external-debt risk."
    elif dollar_state == "Loose":
        fx = "Liquidity tailwind"
        fx_reason = "Dollar weakness supports global beta if credit remains calm."
    else:
        fx = "Mixed"
        fx_reason = "FX is a cross-check rather than the primary regime signal."

    if commodity_state == "Hot":
        commodity = "Watch inflation shock"
        commodity_reason = "Oil or commodity volatility can move quickly into margins, rates, and inflation expectations."
    elif commodity_state == "Disinflationary":
        commodity = "Disinflationary impulse"
        commodity_reason = "Commodity tape is easing pressure on margins and rates, all else equal."
    else:
        commodity = "Contained"
        commodity_reason = "Commodities are not the dominant pressure point."

    if vol_state == "Constructive":
        option = "Finance hedges / avoid overpaying"
        option_reason = "Stress premium is contained. Use spreads, collars, and monetized convexity rather than naked crash premium."
    elif vol_state == "Elevated":
        option = "Defined-risk protection"
        option_reason = "Avoid chasing expensive vol unless credit, rates, and dollar signals are also deteriorating."
    else:
        option = "Opportunistic"
        option_reason = "Let VIX versus realized vol decide whether to buy or finance protection."

    rows.extend(
        [
            {"Decision area": "Gross", "Bias": gross, "Reason": gross_reason},
            {"Decision area": "Net beta", "Bias": net, "Reason": f"Regime {regime}; risk {risk_state}; leadership {leadership_state}; credit {credit_state}."},
            {"Decision area": "Duration hedge", "Bias": duration, "Reason": duration_reason},
            {"Decision area": "Credit hedge", "Bias": credit, "Reason": credit_reason},
            {"Decision area": "Dollar / JPY", "Bias": fx, "Reason": fx_reason},
            {"Decision area": "Commodity hedge", "Bias": commodity, "Reason": commodity_reason},
            {"Decision area": "Optionality", "Bias": option, "Reason": option_reason},
        ]
    )
    return pd.DataFrame(rows)


def build_dynamic_commentary(signals: List[SignalResult], regime: str, composite: float, confidence: str) -> Tuple[str, str, str]:
    df = signals_to_frame(signals)
    if df.empty:
        return "No regime read is available.", "No action bias is available.", "No invalidation rules are available."

    strongest = df.sort_values("Score", ascending=False).head(2)
    weakest = df.sort_values("Score", ascending=True).head(2)
    strong_text = ", ".join([f"{row['Sleeve']} ({row['State']}, {int(row['Score']):+d})" for _, row in strongest.iterrows()])
    weak_text = ", ".join([f"{row['Sleeve']} ({row['State']}, {int(row['Score']):+d})" for _, row in weakest.iterrows()])

    state = (
        f"The composite regime is {regime.lower()} with a weighted score of {composite:+.2f} and {confidence.lower()} confidence. "
        f"The supportive sleeves are {strong_text}. The pressure points are {weak_text}. "
        f"This is a regime read, not a price target; the signal is strongest when credit, dollar, rates, and hedge quality point in the same direction."
    )

    state_map = {s.sleeve: s.state for s in signals}
    hedge_state = state_map.get("Hedge quality", "Unknown")
    credit_state = state_map.get("Credit", "Unknown")
    dollar_state = state_map.get("Dollar liquidity", "Unknown")

    if regime in {"Supportive", "Constructive"}:
        action = "The book can press confirmed leadership, but the weakest sleeve should set gross limits."
    elif regime in {"Fragile", "Choppy"}:
        action = "The better posture is tighter gross, lower tolerance for crowded beta, and cleaner hedges."
    else:
        action = "The right posture is selective and relative-value driven, with less confidence in index-level conclusions."

    if hedge_state == "Broken":
        action += " Duration is not carrying the hedge book cleanly, so downside protection should lean more on equity optionality, credit hedges, or lower gross."
    elif hedge_state == "Working":
        action += " Duration can remain in the hedge stack, especially against growth shocks."

    if credit_state == "Tightening":
        action += " Credit deterioration raises the cost of being long lower-quality cyclicals."
    if dollar_state == "Tightening":
        action += " Dollar pressure argues for more caution in EM and commodity-sensitive risk."

    invalidation = (
        "Challenge the read if the weakest sleeves reverse for a full week, if credit stops confirming the current risk state, "
        "or if SPX-TLT correlation changes enough to alter hedge quality. A weekly compass is only useful if it updates when the transmission mechanism changes."
    )
    return state, action, invalidation


def build_regime_history(md: MarketData, week: int, months: int) -> pd.DataFrame:
    if md.prices.empty:
        return pd.DataFrame(columns=["Date", "Composite", "Regime"])

    cutoff = md.prices.index.max() - pd.DateOffset(months=months)
    sample_dates = md.prices.index[md.prices.index >= cutoff]
    # Weekly sampling keeps the chart honest without slowing the app.
    sample_dates = sample_dates[::5]
    if len(sample_dates) == 0 or sample_dates[-1] != md.prices.index.max():
        sample_dates = sample_dates.append(pd.Index([md.prices.index.max()]))

    rows: List[dict] = []
    for dt in sample_dates:
        p_slice = md.prices.loc[:dt]
        r_slice = p_slice.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        f_slice = md.fred.loc[:dt] if not md.fred.empty else md.fred
        if len(p_slice) < 220:
            continue
        snap = MarketData(p_slice, r_slice, f_slice, md.yahoo_meta, md.fred_meta)
        sigs = compute_all_signals(snap, week)
        regime, composite, conf = composite_regime(sigs)
        rows.append({"Date": dt, "Composite": composite, "Regime": regime, "Confidence": conf})

    return pd.DataFrame(rows)

# ============================================================
# Portfolio overlay helpers
# ============================================================
def parse_portfolio_overlay(text: str) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame(columns=["Exposure", "% NAV"])
    try:
        df = pd.read_csv(StringIO(text))
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        if "Exposure" not in df.columns or "% NAV" not in df.columns:
            return pd.DataFrame(columns=["Exposure", "% NAV"])
        df["% NAV"] = pd.to_numeric(df["% NAV"], errors="coerce")
        return df.dropna(subset=["Exposure", "% NAV"])
    except Exception:
        return pd.DataFrame(columns=["Exposure", "% NAV"])


def portfolio_commentary(portfolio_df: pd.DataFrame, signals: List[SignalResult], regime: str) -> str:
    if portfolio_df.empty:
        return "No portfolio overlay entered. Add exposures to translate the regime into book-level risk language."

    state = {s.sleeve: s.state for s in signals}
    notes: List[str] = []
    exp_text = ", ".join([f"{row['Exposure']} {row['% NAV']:+.0f}% NAV" for _, row in portfolio_df.iterrows()])
    notes.append(f"Entered exposures: {exp_text}.")

    exposure_lower = " ".join(portfolio_df["Exposure"].astype(str).str.lower().tolist())
    if "duration" in exposure_lower or "tlt" in exposure_lower or "treasury" in exposure_lower:
        if state.get("Hedge quality") == "Broken":
            notes.append("Duration exposure should be treated as macro risk because hedge quality is broken or impaired.")
        elif state.get("Hedge quality") == "Working":
            notes.append("Duration exposure has better hedge support this week because stock-bond behavior is working.")
    if "oil" in exposure_lower or "uso" in exposure_lower or "uco" in exposure_lower:
        if state.get("Commodities") == "Hot":
            notes.append("Commodity pressure is hot, so short oil exposure has adverse tape risk and should be sized against squeeze risk.")
        elif state.get("Commodities") == "Disinflationary":
            notes.append("Commodity pressure is disinflationary, which supports short oil or lower inflation-risk positioning.")
    if "usd" in exposure_lower or "jpy" in exposure_lower or "fx" in exposure_lower:
        if state.get("Dollar liquidity") == "Tightening":
            notes.append("Dollar pressure is tightening, which helps long-dollar risk but can pressure global beta. Watch correlation with equity shorts.")
        elif state.get("Dollar liquidity") == "Loose":
            notes.append("Dollar weakness is a liquidity tailwind, which can work against defensive dollar exposure.")
    if "spx" in exposure_lower or "equity" in exposure_lower or "beta" in exposure_lower or "ndx" in exposure_lower:
        notes.append(f"Equity beta should be judged against the {regime.lower()} regime and the credit state of {state.get('Credit', 'unknown').lower()}.")

    return " ".join(notes)

# ============================================================
# Load data
# ============================================================
end_date = datetime.now().date() + timedelta(days=1)
start_date = datetime.now().date() - timedelta(days=365 * lookback_years + 90)

with st.spinner("Downloading market and macro data..."):
    prices, yahoo_meta = fetch_yahoo_prices(list(YAHOO_ASSETS.keys()), str(start_date), str(end_date))
    if use_fred:
        fred_panel, fred_meta = fetch_fred_panel(list(FRED_SERIES.keys()), str(start_date))
    else:
        fred_panel, fred_meta = pd.DataFrame(), {}

if prices.empty:
    st.error("No Yahoo market data downloaded and no last-good cache was available.")
    st.stop()

prices = clean_index(prices).dropna(axis=1, how="all")
returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
fred_b = align_fred_to_prices(fred_panel, prices.index) if use_fred else pd.DataFrame(index=prices.index)
md = MarketData(prices=prices, returns=returns, fred=fred_b, yahoo_meta=yahoo_meta, fred_meta=fred_meta)

signals = compute_all_signals(md, week_back)
regime, composite_score, regime_confidence = composite_regime(signals)
signal_df = signals_to_frame(signals)
market_tape = build_market_tape(md, week_back)
pair_tbl = build_pair_table(md, corr_window, week_back)
action_df = build_action_matrix(signals, regime)
state_commentary, action_commentary, invalidation_commentary = build_dynamic_commentary(signals, regime, composite_score, regime_confidence)

# ============================================================
# Header cards
# ============================================================
state_map = {s.sleeve: s.state for s in signals}
score_map = {s.sleeve: s.score for s in signals}
confidence_map = {s.sleeve: s.confidence for s in signals}

pressure = signal_df.sort_values("Score", ascending=True).iloc[0] if not signal_df.empty else None
support = signal_df.sort_values("Score", ascending=False).iloc[0] if not signal_df.empty else None

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

with c1:
    st.markdown(card("Regime", f"{regime} ({composite_score:+.2f})", f"Composite confidence: {regime_confidence}", regime), unsafe_allow_html=True)
with c2:
    p_sleeve = str(pressure["Sleeve"]) if pressure is not None else "NA"
    p_state = str(pressure["State"]) if pressure is not None else "Unknown"
    st.markdown(card("Pressure point", p_sleeve, p_state, p_state), unsafe_allow_html=True)
with c3:
    s_sleeve = str(support["Sleeve"]) if support is not None else "NA"
    s_state = str(support["State"]) if support is not None else "Unknown"
    st.markdown(card("Supportive sleeve", s_sleeve, s_state, s_state), unsafe_allow_html=True)
with c4:
    h_state = state_map.get("Hedge quality", "Unknown")
    st.markdown(card("Hedge quality", h_state, confidence_map.get("Hedge quality", "Low"), h_state), unsafe_allow_html=True)
with c5:
    st.markdown(card("Credit", state_map.get("Credit", "Unknown"), f"Score {score_map.get('Credit', 0):+d}", state_map.get("Credit", "Unknown")), unsafe_allow_html=True)
with c6:
    st.markdown(card("Rates", state_map.get("Rates", "Unknown"), f"Score {score_map.get('Rates', 0):+d}", state_map.get("Rates", "Unknown")), unsafe_allow_html=True)
with c7:
    st.markdown(card("Dollar liquidity", state_map.get("Dollar liquidity", "Unknown"), f"Score {score_map.get('Dollar liquidity', 0):+d}", state_map.get("Dollar liquidity", "Unknown")), unsafe_allow_html=True)
with c8:
    st.markdown(card("Volatility", state_map.get("Volatility", "Unknown"), f"Score {score_map.get('Volatility', 0):+d}", state_map.get("Volatility", "Unknown")), unsafe_allow_html=True)

st.markdown("")

# ============================================================
# Commentary
# ============================================================
st.markdown('<div class="section-label">PM Read</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1.15, 1.0, 1.0])
with col_a:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:720; margin-bottom:0.5rem;">State</div>
            <div style="line-height:1.55; color:#1f2937;">{state_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:720; margin-bottom:0.5rem;">Action Bias</div>
            <div style="line-height:1.55; color:#1f2937;">{action_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:720; margin-bottom:0.5rem;">Invalidation</div>
            <div style="line-height:1.55; color:#1f2937;">{invalidation_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Charts
# ============================================================
left_chart, right_chart = st.columns([1.15, 0.85])
with left_chart:
    st.markdown('<div class="section-label">Regime History</div>', unsafe_allow_html=True)
    hist = build_regime_history(md, week_back, chart_history_months)
    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["Date"],
                y=hist["Composite"],
                mode="lines",
                name="Composite regime score",
                line=dict(width=2.2),
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:+.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=1.10, line_dash="dot", line_color="green", annotation_text="Supportive")
        fig.add_hline(y=0.35, line_dash="dot", line_color="gray", annotation_text="Constructive")
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.add_hline(y=-0.35, line_dash="dot", line_color="gray", annotation_text="Choppy")
        fig.add_hline(y=-1.10, line_dash="dot", line_color="red", annotation_text="Fragile")
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Weighted composite score",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to render regime history.")

with right_chart:
    st.markdown('<div class="section-label">Sleeve Contributions</div>', unsafe_allow_html=True)
    if not signal_df.empty:
        contrib = signal_df.sort_values("Score", ascending=True).copy()
        fig = px.bar(
            contrib,
            x="Score",
            y="Sleeve",
            orientation="h",
            text="Score",
            hover_data=["State", "Confidence", "Drivers"],
        )
        fig.update_traces(texttemplate="%{text:+d}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=30, t=20, b=10),
            xaxis=dict(range=[-2.8, 2.8], zeroline=True, zerolinewidth=1, zerolinecolor="gray"),
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sleeve scores available.")

# ============================================================
# What changed
# ============================================================
st.markdown('<div class="section-label">What Changed This Week</div>', unsafe_allow_html=True)
if not market_tape.empty:
    display_tape = market_tape.head(top_n_assets).drop(columns=["Sort"])
    display_tape["Shock z"] = display_tape["Shock z"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "NA")
    st.dataframe(display_tape, use_container_width=True, height=min(650, 38 * (len(display_tape) + 1)))
else:
    st.info("No market tape data available.")

# ============================================================
# Regime model table
# ============================================================
st.markdown('<div class="section-label">Regime Model Inputs</div>', unsafe_allow_html=True)
if not signal_df.empty:
    regime_display = signal_df[["Sleeve", "State", "Direction", "Score", "Confidence", "Coverage", "Drivers", "Portfolio implication"]].copy()
    st.dataframe(regime_display, use_container_width=True, height=min(760, 42 * (len(regime_display) + 1)))
else:
    st.info("No regime inputs available.")

# ============================================================
# Action matrix
# ============================================================
st.markdown('<div class="section-label">Positioning and Hedge Bias</div>', unsafe_allow_html=True)
st.dataframe(action_df, use_container_width=True, height=42 * (len(action_df) + 1))

# ============================================================
# Cross-asset shifts
# ============================================================
st.markdown('<div class="section-label">Top Cross-Asset Shifts</div>', unsafe_allow_html=True)
if not pair_tbl.empty:
    top_pairs = pair_tbl.head(top_pair_signals).copy()
    display_pairs = top_pairs.drop(columns=["Series"])
    display_pairs["ρ now"] = display_pairs["ρ now"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "NA")
    display_pairs["Δρ"] = display_pairs["Δρ"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "NA")
    display_pairs["1Y pctile"] = display_pairs["1Y pctile"].map(lambda x: f"{x:.0f}%" if pd.notna(x) else "NA")
    st.dataframe(display_pairs, use_container_width=True, height=min(560, 42 * (len(display_pairs) + 1)))
else:
    st.info("No cross-asset pair signals available.")

# ============================================================
# Correlation map and pair charts
# ============================================================
left, right = st.columns([1.0, 1.0])
with left:
    st.markdown('<div class="section-label">Current Correlation Map</div>', unsafe_allow_html=True)
    heat_tickers = [t for t in ["SPY", "QQQ", "IWM", "HYG", "LQD", "TLT", "UUP", "GLD", "USO", "USDJPY=X"] if t in md.returns.columns]
    hm = md.returns[heat_tickers].dropna().tail(corr_window)
    if len(hm) >= max(10, corr_window // 2):
        corr = hm.corr()
        labels = [YAHOO_ASSETS[t]["label"] for t in corr.columns]
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdYlGn",
                text=np.round(corr.values, 2),
                texttemplate="%{text:+.2f}",
                hovertemplate="%{y} vs %{x}<br>Corr: %{z:+.2f}<extra></extra>",
            )
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for correlation map.")

with right:
    st.markdown('<div class="section-label">Stress Map</div>', unsafe_allow_html=True)
    if not signal_df.empty:
        stress = signal_df[["Sleeve", "State", "Score", "Drivers"]].copy()
        stress["Stress"] = -stress["Score"]
        fig = px.bar(
            stress.sort_values("Stress", ascending=True),
            x="Stress",
            y="Sleeve",
            orientation="h",
            hover_data=["State", "Drivers"],
        )
        fig.add_vline(x=0, line_color="gray", line_width=1)
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=20, t=20, b=10),
            xaxis_title="Stress contribution, higher is worse",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stress score available.")

st.markdown('<div class="section-label">Detailed Pair Charts</div>', unsafe_allow_html=True)
if not pair_tbl.empty:
    chart_cutoff = md.prices.index.max() - pd.DateOffset(months=chart_history_months)
    chart_pairs = pair_tbl.head(min(4, len(pair_tbl))).copy()
    for i in range(0, len(chart_pairs), 2):
        cols = st.columns(2)
        subset = chart_pairs.iloc[i:i + 2]
        for col, (_, row) in zip(cols, subset.iterrows()):
            with col:
                rc = row["Series"].dropna()
                rc = rc[rc.index >= chart_cutoff]
                if rc.empty:
                    st.info(f"No data for {row['Pair']}.")
                    continue
                mean = rc.rolling(126, min_periods=20).mean()
                std = rc.rolling(126, min_periods=20).std()
                upper = mean + std
                lower = mean - std
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc.values, mode="lines", name=f"{corr_window}D corr", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines", name="6M mean", line=dict(width=1.5, dash="dash")))
                fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", line=dict(width=0), fill="tonexty", name="±1σ band", hoverinfo="skip"))
                fig.add_hline(y=0, line_color="gray", line_width=1)
                fig.update_layout(
                    title=row["Pair"],
                    height=330,
                    margin=dict(l=10, r=10, t=45, b=10),
                    yaxis=dict(range=[-1, 1]),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pair charts available.")

# ============================================================
# Portfolio overlay
# ============================================================
if show_portfolio_overlay:
    st.markdown('<div class="section-label">Portfolio Overlay</div>', unsafe_allow_html=True)
    default_overlay = """Exposure,% NAV
SPX / equity beta,0
Nasdaq / AI beta,0
Duration / TLT,0
Oil / energy,0
USDJPY / FX,0
Credit beta,0
Long volatility / optionality,0
"""
    overlay_text = st.text_area(
        "Enter exposures as CSV: Exposure,% NAV",
        value=default_overlay,
        height=180,
        help="Positive numbers are long exposure, negative numbers are short exposure. This is intentionally simple so it can be pasted from IBKR or a risk sheet.",
    )
    portfolio_df = parse_portfolio_overlay(overlay_text)
    col1, col2 = st.columns([0.85, 1.15])
    with col1:
        st.dataframe(portfolio_df, use_container_width=True, height=min(420, 38 * (len(portfolio_df) + 1)))
    with col2:
        overlay_comment = portfolio_commentary(portfolio_df, signals, regime)
        st.markdown(
            f"""
            <div class="commentary-box">
                <div style="font-weight:720; margin-bottom:0.5rem;">Book-Level Translation</div>
                <div style="line-height:1.55; color:#1f2937;">{overlay_comment}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Diagnostics
# ============================================================
if show_raw_data:
    st.markdown('<div class="section-label">Data Diagnostics</div>', unsafe_allow_html=True)
    with st.expander("Yahoo metadata", expanded=True):
        st.json(yahoo_meta)
    with st.expander("FRED metadata", expanded=False):
        st.json(fred_meta)
    with st.expander("Available columns", expanded=False):
        st.markdown("**Yahoo columns**")
        st.write(list(md.prices.columns))
        st.markdown("**FRED columns**")
        st.write(list(md.fred.columns))
    with st.expander("Data freshness", expanded=False):
        data_rows = []
        last_px_date = md.prices.index.max()
        for col in md.prices.columns:
            s = safe_series(md.prices[col])
            data_rows.append({"Series": col, "Source": "Yahoo", "Last date": s.index.max().date() if not s.empty else None, "Age vs price max": (last_px_date - s.index.max()).days if not s.empty else None})
        for col in md.fred.columns:
            s = safe_series(md.fred[col])
            data_rows.append({"Series": col, "Source": "FRED", "Last date": s.index.max().date() if not s.empty else None, "Age vs price max": (last_px_date - s.index.max()).days if not s.empty else None})
        st.dataframe(pd.DataFrame(data_rows), use_container_width=True)

last_date = md.prices.index.max()
footer_date = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A"
st.caption(f"As of {footer_date} | AD Fund Management LP | Weekly Cross-Asset Compass")
