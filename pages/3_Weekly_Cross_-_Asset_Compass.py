# 15_Weekly_Cross_Asset_Compass.py
# ADFM | Weekly Cross-Asset Compass
# Rebuilt around regime scoring, weekly change detection, hedge quality, and action bias.

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    padding-top: 2.35rem;
    padding-bottom: 1.6rem;
    max-width: 1580px;
}
.main-title {
    font-size: 2.15rem;
    font-weight: 720;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin-bottom: 0.20rem;
}
.subtle {
    color: #6b7280;
    font-size: 0.96rem;
    margin-bottom: 1.0rem;
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
    font-size: 0.76rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.055em;
    margin-bottom: 0.35rem;
}
.card-value {
    font-size: 1.42rem;
    font-weight: 720;
    letter-spacing: -0.02em;
    line-height: 1.08;
    margin-bottom: 0.18rem;
}
.card-sub {
    font-size: 0.88rem;
    color: #4b5563;
    line-height: 1.35;
}
.section-label {
    font-size: 1.14rem;
    font-weight: 720;
    margin-top: 0.55rem;
    margin-bottom: 0.65rem;
}
.commentary-box {
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.035);
    margin-bottom: 12px;
}
.signal-chip {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 650;
    margin-right: 0.35rem;
}
.chip-green { background: #e8f7ee; color: #166534; }
.chip-yellow { background: #fff7e6; color: #92400e; }
.chip-red { background: #fdecec; color: #991b1b; }
.chip-blue { background: #eaf2ff; color: #1d4ed8; }
.chip-gray { background: #f3f4f6; color: #374151; }
div[data-testid="stDataFrame"] div[role="table"] {
    border-radius: 14px;
    overflow: hidden;
}
.small-note {
    color: #6b7280;
    font-size: 0.86rem;
    line-height: 1.45;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="main-title">Weekly Cross-Asset Compass</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">A weekly PM dashboard for regime, transmission, hedge quality, and action bias across equities, credit, rates, FX, commodities, and volatility.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Constants and paths
# ============================================================
CACHE_DIR = Path(".adfm_cross_asset_cache")
CACHE_DIR.mkdir(exist_ok=True)
YAHOO_CACHE = CACHE_DIR / "yahoo_prices_last_good.pkl"
YAHOO_META = CACHE_DIR / "yahoo_prices_last_good_meta.json"
FRED_CACHE_DIR = CACHE_DIR / "fred"
FRED_CACHE_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
DEFAULT_YEARS = 6

YAHOO_ASSETS: Dict[str, Dict[str, str]] = {
    "SPY": {"label": "SPX", "group": "Equities", "kind": "price"},
    "QQQ": {"label": "Nasdaq", "group": "Equities", "kind": "price"},
    "IWM": {"label": "Russell 2000", "group": "Equities", "kind": "price"},
    "FEZ": {"label": "Europe", "group": "Equities", "kind": "price"},
    "EWJ": {"label": "Japan", "group": "Equities", "kind": "price"},
    "EEM": {"label": "EM", "group": "Equities", "kind": "price"},
    "HYG": {"label": "HY Credit", "group": "Credit", "kind": "price"},
    "LQD": {"label": "IG Credit", "group": "Credit", "kind": "price"},
    "TLT": {"label": "Long Duration", "group": "Rates", "kind": "price"},
    "IEF": {"label": "7-10Y Duration", "group": "Rates", "kind": "price"},
    "UUP": {"label": "Dollar ETF", "group": "FX", "kind": "price"},
    "USDJPY=X": {"label": "USDJPY", "group": "FX", "kind": "price"},
    "FXE": {"label": "Euro ETF", "group": "FX", "kind": "price"},
    "GLD": {"label": "Gold", "group": "Commodities", "kind": "price"},
    "SLV": {"label": "Silver", "group": "Commodities", "kind": "price"},
    "USO": {"label": "Oil", "group": "Commodities", "kind": "price"},
    "CPER": {"label": "Copper", "group": "Commodities", "kind": "price"},
    "^VIX": {"label": "VIX", "group": "Vol", "kind": "level"},
    "^VIX3M": {"label": "VIX3M", "group": "Vol", "kind": "level"},
    "^TNX": {"label": "10Y Proxy", "group": "Rates", "kind": "yield_proxy"},
}

FRED_SERIES: Dict[str, Dict[str, str]] = {
    "DGS10": {"label": "10Y Treasury", "unit": "pct", "group": "Rates"},
    "DGS2": {"label": "2Y Treasury", "unit": "pct", "group": "Rates"},
    "T10Y2Y": {"label": "10Y-2Y Curve", "unit": "pct", "group": "Rates"},
    "T10Y3M": {"label": "10Y-3M Curve", "unit": "pct", "group": "Rates"},
    "DFII10": {"label": "10Y Real Yield", "unit": "pct", "group": "Rates"},
    "BAMLH0A0HYM2": {"label": "HY OAS", "unit": "pct", "group": "Credit"},
    "BAMLC0A0CM": {"label": "IG OAS", "unit": "pct", "group": "Credit"},
    "SOFR": {"label": "SOFR", "unit": "pct", "group": "Policy"},
    "WALCL": {"label": "Fed Balance Sheet", "unit": "usd_mm", "group": "Liquidity"},
    "RRPONTSYD": {"label": "ON RRP", "unit": "usd_bn", "group": "Liquidity"},
    "WRESBAL": {"label": "Bank Reserves", "unit": "usd_bn", "group": "Liquidity"},
}

PAIR_SPECS = [
    {"a": "SPY", "b": "TLT", "pair": "SPX vs TLT", "kind": "returns"},
    {"a": "SPY", "b": "HYG", "pair": "SPX vs HY", "kind": "returns"},
    {"a": "SPY", "b": "LQD", "pair": "SPX vs IG", "kind": "returns"},
    {"a": "SPY", "b": "UUP", "pair": "SPX vs Dollar", "kind": "returns"},
    {"a": "SPY", "b": "USO", "pair": "SPX vs Oil", "kind": "returns"},
    {"a": "SPY", "b": "GLD", "pair": "SPX vs Gold", "kind": "returns"},
    {"a": "IWM", "b": "HYG", "pair": "RTY vs HY", "kind": "returns"},
    {"a": "IWM", "b": "SPY", "pair": "RTY vs SPX", "kind": "returns"},
    {"a": "EEM", "b": "UUP", "pair": "EM vs Dollar", "kind": "returns"},
    {"a": "EWJ", "b": "USDJPY=X", "pair": "Japan vs USDJPY", "kind": "returns"},
    {"a": "GLD", "b": "TLT", "pair": "Gold vs TLT", "kind": "returns"},
    {"a": "HYG", "b": "USO", "pair": "HY vs Oil", "kind": "returns"},
]

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Weekly cross-asset regime dashboard for risk-taking, hedge mix, and transmission.

        **What this tab shows**
        - A scored regime model across risk appetite, rates, credit, dollar liquidity, commodities, and hedge quality.
        - Weekly change tables that surface what moved and whether it matters.
        - A positioning matrix that translates the tape into gross, hedge, and optionality bias.

        **Data source**
        - Yahoo Finance for liquid market proxies.
        - FRED public CSV endpoints for rates, spreads, and liquidity where available.
        """
    )

    st.header("Controls")
    lookback_years = st.slider("History window, years", 3, 15, DEFAULT_YEARS)
    corr_window = st.selectbox("Rolling correlation window", [21, 42, 63], index=1)
    compare_weeks = st.selectbox("Compare against", ["1 week ago", "2 weeks ago", "1 month ago"], index=0)
    week_back = {"1 week ago": 5, "2 weeks ago": 10, "1 month ago": 21}[compare_weeks]
    top_n_assets = st.slider("Assets in weekly tape table", 8, 24, 18)
    top_pair_signals = st.slider("Top cross-asset shifts", 4, 14, 8)
    chart_history_months = st.slider("Chart history, months", 3, 36, 12)
    use_fred = st.toggle("Use FRED macro series", value=True)
    show_raw_data = st.toggle("Show data diagnostics", value=False)

# ============================================================
# Utility helpers
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


def safe_last(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def safe_value_at(series: pd.Series, days_back: int) -> float:
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    if len(s) <= days_back:
        return float(s.iloc[0])
    return float(s.iloc[-days_back - 1])


def pct_change_days(series: pd.Series, days: int) -> float:
    s = pd.Series(series).dropna()
    if len(s) <= days:
        return np.nan
    base = s.iloc[-days - 1]
    if pd.isna(base) or base == 0:
        return np.nan
    return float((s.iloc[-1] / base - 1.0) * 100.0)


def level_change_days(series: pd.Series, days: int) -> float:
    s = pd.Series(series).dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-days - 1])


def bps_change_days(series: pd.Series, days: int) -> float:
    val = level_change_days(series, days)
    return float(val * 100.0) if pd.notna(val) else np.nan


def pct_rank_last(series: pd.Series, window: int = 252) -> float:
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan
    if len(s) > window:
        s = s.iloc[-window:]
    return float(s.rank(pct=True).iloc[-1] * 100.0)


def zscore_last(series: pd.Series, window: int = 252) -> float:
    s = pd.Series(series).dropna()
    if len(s) > window:
        s = s.iloc[-window:]
    if len(s) < 20:
        return np.nan
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return np.nan
    return float((s.iloc[-1] - s.mean()) / sd)


def realized_vol(ret_series: pd.Series, window: int = 21, annualization: int = 252) -> pd.Series:
    return pd.Series(ret_series).rolling(window).std() * math.sqrt(annualization) * 100.0


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def score_clip(x: float) -> int:
    if pd.isna(x):
        return 0
    return int(max(-2, min(2, round(x))))


def format_level(x: float, suffix: str = "") -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:,.2f}{suffix}"


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.2f}%"


def format_bps(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:+.0f} bps"


def format_score(x: float) -> str:
    if pd.isna(x):
        return "0"
    return f"{int(x):+d}"


def chip_class(value: str) -> str:
    green = {"Supportive", "Constructive", "Working", "Contained", "Positive", "Press", "Loose"}
    yellow = {"Mixed", "Watchful", "Neutral", "Balanced", "Selective", "Choppy"}
    red = {"Fragile", "Elevated", "Broken", "Tightening", "Defensive", "Stress"}
    if value in green:
        return "chip-green"
    if value in yellow:
        return "chip-yellow"
    if value in red:
        return "chip-red"
    return "chip-blue"

# ============================================================
# Data fetching
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_yahoo_prices(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, dict]:
    tickers = sorted(set(tickers))
    meta = {"source": "yahoo", "requested": len(tickers), "returned": 0, "missing": []}
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
    except Exception as exc:
        raw = pd.DataFrame()
        meta["error"] = str(exc)

    if raw is None or len(raw) == 0:
        if YAHOO_CACHE.exists():
            try:
                cached = pd.read_pickle(YAHOO_CACHE)
                cached_meta = read_json(YAHOO_META)
                cached_meta["source"] = "last_good_yahoo_cache"
                return cached, cached_meta
            except Exception:
                pass
        return pd.DataFrame(), meta

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        top = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t not in top:
                continue
            df = raw[t].copy()
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if col in df.columns:
                frames.append(pd.to_numeric(df[col], errors="coerce").rename(t))
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col in raw.columns and len(tickers) == 1:
            frames.append(pd.to_numeric(raw[col], errors="coerce").rename(tickers[0]))

    if not frames:
        return pd.DataFrame(), meta

    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all")

    meta["returned"] = int(prices.shape[1])
    meta["missing"] = sorted(set(tickers) - set(prices.columns))
    meta["last_date"] = str(prices.index.max().date()) if not prices.empty else None

    if not prices.empty and prices.shape[1] >= max(4, int(len(tickers) * 0.65)):
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
    meta = {"series": series_id, "source": "fred", "status": "live"}

    try:
        r = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if "observation_date" not in df.columns or series_id not in df.columns:
            raise ValueError("Unexpected FRED CSV shape")
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        vals = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
        s = pd.Series(vals.values, index=df["observation_date"], name=series_id).dropna()
        s = s[s.index >= pd.to_datetime(start)]
        if s.empty:
            raise ValueError("FRED series empty after start filter")
        s.to_pickle(cache_path)
        meta["last_date"] = str(s.index.max().date())
        write_json(meta_path, meta)
        return s, meta
    except Exception as exc:
        if cache_path.exists():
            try:
                s = pd.read_pickle(cache_path)
                cached_meta = read_json(meta_path)
                cached_meta["source"] = "last_good_fred_cache"
                cached_meta["live_error"] = str(exc)
                return s, cached_meta
            except Exception:
                pass
        meta["status"] = "failed"
        meta["error"] = str(exc)
        return pd.Series(dtype=float, name=series_id), meta


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_fred_panel(series_ids: List[str], start: str) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    series = []
    metas: Dict[str, dict] = {}
    for sid in series_ids:
        s, meta = fetch_fred_series(sid, start)
        metas[sid] = meta
        if not s.empty:
            series.append(s)
        time.sleep(0.05)
    if not series:
        return pd.DataFrame(), metas
    panel = pd.concat(series, axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel = panel.ffill()
    return panel, metas

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

prices = prices.dropna(axis=1, how="all").copy()
returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

# Align FRED to business days for mixed charts and scoring.
if not fred_panel.empty:
    full_idx = pd.bdate_range(prices.index.min(), prices.index.max())
    fred_b = fred_panel.reindex(full_idx).ffill()
else:
    fred_b = pd.DataFrame(index=prices.index)

# ============================================================
# Derived series
# ============================================================
def get_price(ticker: str) -> pd.Series:
    return prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)


def get_ret(ticker: str) -> pd.Series:
    return returns[ticker].dropna() if ticker in returns.columns else pd.Series(dtype=float)


def get_fred(sid: str) -> pd.Series:
    return fred_b[sid].dropna() if sid in fred_b.columns else pd.Series(dtype=float)

spy = get_price("SPY")
qqq = get_price("QQQ")
iwm = get_price("IWM")
hyg = get_price("HYG")
lqd = get_price("LQD")
tlt = get_price("TLT")
uup = get_price("UUP")
usdjpy = get_price("USDJPY=X")
gld = get_price("GLD")
uso = get_price("USO")
cper = get_price("CPER")
vix = get_price("^VIX")
vix3m = get_price("^VIX3M")

ten_y = get_fred("DGS10")
two_y = get_fred("DGS2")
curve_10s2s = get_fred("T10Y2Y")
curve_10s3m = get_fred("T10Y3M")
real_10y = get_fred("DFII10")
hy_oas = get_fred("BAMLH0A0HYM2")
ig_oas = get_fred("BAMLC0A0CM")
walcl = get_fred("WALCL")
rrp = get_fred("RRPONTSYD")
reserves = get_fred("WRESBAL")

spy_rv = realized_vol(get_ret("SPY"), 21)
hyg_rv = realized_vol(get_ret("HYG"), 21)
oil_rv = realized_vol(get_ret("USO"), 21)
uup_rv = realized_vol(get_ret("UUP"), 21)

vix_term = ((vix3m / vix) - 1.0).replace([np.inf, -np.inf], np.nan).dropna() if not vix.empty and not vix3m.empty else pd.Series(dtype=float)
vix_rv_spread = pd.concat([vix.rename("vix"), spy_rv.rename("rv")], axis=1).dropna()
vix_rv_spread = (vix_rv_spread["vix"] - vix_rv_spread["rv"]) if not vix_rv_spread.empty else pd.Series(dtype=float)

hyg_lqd = (hyg / lqd).replace([np.inf, -np.inf], np.nan).dropna() if not hyg.empty and not lqd.empty else pd.Series(dtype=float)
iwm_spy = (iwm / spy).replace([np.inf, -np.inf], np.nan).dropna() if not iwm.empty and not spy.empty else pd.Series(dtype=float)
qqq_spy = (qqq / spy).replace([np.inf, -np.inf], np.nan).dropna() if not qqq.empty and not spy.empty else pd.Series(dtype=float)
gld_tlt = (gld / tlt).replace([np.inf, -np.inf], np.nan).dropna() if not gld.empty and not tlt.empty else pd.Series(dtype=float)

spy_tlt_corr = rolling_corr(get_ret("SPY"), get_ret("TLT"), corr_window)
spy_hyg_corr = rolling_corr(get_ret("SPY"), get_ret("HYG"), corr_window)
em_dollar_corr = rolling_corr(get_ret("EEM"), get_ret("UUP"), corr_window)
japan_fx_corr = rolling_corr(get_ret("EWJ"), get_ret("USDJPY=X"), corr_window)

# ============================================================
# Regime scoring
# ============================================================
def compute_risk_appetite_score() -> Tuple[int, str, str, List[dict]]:
    spy_1w = pct_change_days(spy, week_back)
    spy_1m = pct_change_days(spy, 21)
    qqq_rel_1m = pct_change_days(qqq_spy, 21)
    iwm_rel_1m = pct_change_days(iwm_spy, 21)
    spy_last = safe_last(spy)
    spy_50 = safe_last(spy.rolling(50).mean()) if len(spy) >= 50 else np.nan
    spy_200 = safe_last(spy.rolling(200).mean()) if len(spy) >= 200 else np.nan

    raw = 0
    if pd.notna(spy_1m) and spy_1m > 2.0:
        raw += 1
    elif pd.notna(spy_1m) and spy_1m < -2.0:
        raw -= 1
    if pd.notna(spy_last) and pd.notna(spy_50) and spy_last > spy_50:
        raw += 1
    else:
        raw -= 1
    if pd.notna(spy_last) and pd.notna(spy_200) and spy_last > spy_200:
        raw += 1
    else:
        raw -= 1
    if pd.notna(qqq_rel_1m) and qqq_rel_1m > 1.0:
        raw += 0.5
    if pd.notna(iwm_rel_1m) and iwm_rel_1m < -1.0:
        raw -= 0.5

    score = score_clip(raw / 1.5)
    state = "Constructive" if score > 0 else "Defensive" if score < 0 else "Neutral"
    read = f"SPX 1M {format_pct(spy_1m)}, QQQ/SPY 1M {format_pct(qqq_rel_1m)}, IWM/SPY 1M {format_pct(iwm_rel_1m)}."
    inputs = [
        {"Sleeve": "Risk appetite", "Input": "SPX 1W", "Latest": format_level(spy_last), "Weekly Move": format_pct(spy_1w), "Percentile": f"{pct_rank_last(spy.pct_change(21), 252):.0f}%" if not spy.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Risk appetite", "Input": "SPX vs 50D/200D", "Latest": "Above" if pd.notna(spy_last) and pd.notna(spy_50) and pd.notna(spy_200) and spy_last > spy_50 and spy_last > spy_200 else "Below/mixed", "Weekly Move": "", "Percentile": "", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_vol_score() -> Tuple[int, str, str, List[dict]]:
    vix_now = safe_last(vix)
    vix_1w = level_change_days(vix, week_back)
    term_now = safe_last(vix_term)
    rv_now = safe_last(spy_rv)
    rv_pct = pct_rank_last(spy_rv, 252)

    raw = 0
    if pd.notna(vix_now):
        if vix_now < 16:
            raw += 1
        elif vix_now >= 22:
            raw -= 1
        if vix_now >= 28:
            raw -= 1
    if pd.notna(term_now):
        if term_now > 0.08:
            raw += 1
        elif term_now < 0:
            raw -= 1
    if pd.notna(rv_pct):
        if rv_pct < 40:
            raw += 0.5
        elif rv_pct > 75:
            raw -= 0.5

    score = score_clip(raw)
    state = "Constructive" if score > 0 else "Elevated" if score < 0 else "Watchful"
    read = f"VIX {format_level(vix_now)}, VIX 1W {format_level(vix_1w)}, term structure {format_level(term_now)}, SPX RV percentile {rv_pct:.0f}%" if pd.notna(rv_pct) else f"VIX {format_level(vix_now)}."
    inputs = [
        {"Sleeve": "Volatility", "Input": "VIX", "Latest": format_level(vix_now), "Weekly Move": format_level(vix_1w), "Percentile": f"{pct_rank_last(vix, 252):.0f}%" if not vix.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Volatility", "Input": "VIX3M/VIX - 1", "Latest": format_level(term_now), "Weekly Move": format_level(level_change_days(vix_term, week_back)), "Percentile": f"{pct_rank_last(vix_term, 252):.0f}%" if not vix_term.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_rates_score() -> Tuple[int, str, str, List[dict]]:
    ten_now = safe_last(ten_y)
    ten_w = bps_change_days(ten_y, week_back)
    real_now = safe_last(real_10y)
    real_w = bps_change_days(real_10y, week_back)
    curve_now = safe_last(curve_10s2s)
    curve_w = bps_change_days(curve_10s2s, week_back)

    raw = 0
    if pd.notna(ten_w):
        if ten_w <= -12:
            raw += 1
        elif ten_w >= 12:
            raw -= 1
    if pd.notna(real_w):
        if real_w <= -10:
            raw += 1
        elif real_w >= 10:
            raw -= 1
    if pd.notna(curve_now) and curve_now < -0.50:
        raw -= 0.5
    if pd.notna(curve_w) and curve_w > 15 and pd.notna(ten_w) and ten_w > 0:
        raw -= 0.5

    score = score_clip(raw)
    state = "Supportive" if score > 0 else "Tightening" if score < 0 else "Neutral"
    read = f"10Y {format_level(ten_now, '%')} moved {format_bps(ten_w)}, real 10Y moved {format_bps(real_w)}, 10s2s {format_level(curve_now, '%')}."
    inputs = [
        {"Sleeve": "Rates", "Input": "10Y Treasury", "Latest": format_level(ten_now, "%"), "Weekly Move": format_bps(ten_w), "Percentile": f"{pct_rank_last(ten_y, 252):.0f}%" if not ten_y.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Rates", "Input": "10Y real yield", "Latest": format_level(real_now, "%"), "Weekly Move": format_bps(real_w), "Percentile": f"{pct_rank_last(real_10y, 252):.0f}%" if not real_10y.empty else "NA", "Score": score, "Read": read},
        {"Sleeve": "Rates", "Input": "10Y-2Y curve", "Latest": format_level(curve_now, "%"), "Weekly Move": format_bps(curve_w), "Percentile": f"{pct_rank_last(curve_10s2s, 252):.0f}%" if not curve_10s2s.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_credit_score() -> Tuple[int, str, str, List[dict]]:
    hy_now = safe_last(hy_oas)
    hy_w = bps_change_days(hy_oas, week_back)
    ig_now = safe_last(ig_oas)
    ig_w = bps_change_days(ig_oas, week_back)
    hyg_1w = pct_change_days(hyg, week_back)
    rel_1m = pct_change_days(hyg_lqd, 21)
    corr_now = safe_last(spy_hyg_corr)

    raw = 0
    if pd.notna(hy_w):
        if hy_w <= -15:
            raw += 1
        elif hy_w >= 15:
            raw -= 1
    if pd.notna(ig_w):
        if ig_w <= -5:
            raw += 0.5
        elif ig_w >= 5:
            raw -= 0.5
    if pd.notna(hyg_1w) and hyg_1w > 0.5:
        raw += 0.5
    elif pd.notna(hyg_1w) and hyg_1w < -0.75:
        raw -= 0.5
    if pd.notna(rel_1m) and rel_1m > 0:
        raw += 0.5
    elif pd.notna(rel_1m) and rel_1m < -1.0:
        raw -= 0.5

    score = score_clip(raw)
    state = "Contained" if score > 0 else "Tightening" if score < 0 else "Watchful"
    read = f"HY OAS {format_level(hy_now, '%')} moved {format_bps(hy_w)}, IG OAS moved {format_bps(ig_w)}, HYG/LQD 1M {format_pct(rel_1m)}, SPX-HY corr {format_level(corr_now)}."
    inputs = [
        {"Sleeve": "Credit", "Input": "HY OAS", "Latest": format_level(hy_now, "%"), "Weekly Move": format_bps(hy_w), "Percentile": f"{pct_rank_last(hy_oas, 252):.0f}%" if not hy_oas.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Credit", "Input": "IG OAS", "Latest": format_level(ig_now, "%"), "Weekly Move": format_bps(ig_w), "Percentile": f"{pct_rank_last(ig_oas, 252):.0f}%" if not ig_oas.empty else "NA", "Score": score, "Read": read},
        {"Sleeve": "Credit", "Input": "HYG/LQD", "Latest": format_level(safe_last(hyg_lqd)), "Weekly Move": format_pct(pct_change_days(hyg_lqd, week_back)), "Percentile": f"{pct_rank_last(hyg_lqd, 252):.0f}%" if not hyg_lqd.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_dollar_score() -> Tuple[int, str, str, List[dict]]:
    uup_1w = pct_change_days(uup, week_back)
    uup_1m = pct_change_days(uup, 21)
    jpy_1w = pct_change_days(usdjpy, week_back)
    em_corr_now = safe_last(em_dollar_corr)

    raw = 0
    if pd.notna(uup_1w):
        if uup_1w <= -0.75:
            raw += 1
        elif uup_1w >= 0.75:
            raw -= 1
    if pd.notna(uup_1m):
        if uup_1m >= 1.5:
            raw -= 1
        elif uup_1m <= -1.5:
            raw += 1
    if pd.notna(jpy_1w) and jpy_1w >= 1.5:
        raw -= 0.5
    if pd.notna(em_corr_now) and em_corr_now <= -0.35:
        raw -= 0.5

    score = score_clip(raw)
    state = "Loose" if score > 0 else "Tightening" if score < 0 else "Mixed"
    read = f"UUP 1W {format_pct(uup_1w)}, UUP 1M {format_pct(uup_1m)}, USDJPY 1W {format_pct(jpy_1w)}, EM-dollar corr {format_level(em_corr_now)}."
    inputs = [
        {"Sleeve": "Dollar liquidity", "Input": "Dollar ETF", "Latest": format_level(safe_last(uup)), "Weekly Move": format_pct(uup_1w), "Percentile": f"{pct_rank_last(uup, 252):.0f}%" if not uup.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Dollar liquidity", "Input": "USDJPY", "Latest": format_level(safe_last(usdjpy)), "Weekly Move": format_pct(jpy_1w), "Percentile": f"{pct_rank_last(usdjpy, 252):.0f}%" if not usdjpy.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_commodity_score() -> Tuple[int, str, str, List[dict]]:
    oil_1w = pct_change_days(uso, week_back)
    oil_1m = pct_change_days(uso, 21)
    copper_1w = pct_change_days(cper, week_back)
    gold_1w = pct_change_days(gld, week_back)
    oil_vol_z = zscore_last(oil_rv, 252)

    raw = 0
    if pd.notna(oil_1w):
        if oil_1w >= 5:
            raw -= 1
        elif oil_1w <= -5:
            raw += 0.5
    if pd.notna(oil_vol_z) and oil_vol_z >= 1.25:
        raw -= 1
    if pd.notna(copper_1w) and copper_1w > 2 and pd.notna(oil_1w) and oil_1w < 3:
        raw += 0.5
    if pd.notna(gold_1w) and gold_1w > 2 and pd.notna(real_10y) and bps_change_days(real_10y, week_back) > 5:
        raw -= 0.5

    score = score_clip(raw)
    state = "Balanced" if score >= 0 else "Hot"
    read = f"Oil 1W {format_pct(oil_1w)}, oil 1M {format_pct(oil_1m)}, copper 1W {format_pct(copper_1w)}, gold 1W {format_pct(gold_1w)}, oil vol z {format_level(oil_vol_z)}."
    inputs = [
        {"Sleeve": "Commodities", "Input": "Oil", "Latest": format_level(safe_last(uso)), "Weekly Move": format_pct(oil_1w), "Percentile": f"{pct_rank_last(uso, 252):.0f}%" if not uso.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Commodities", "Input": "Gold", "Latest": format_level(safe_last(gld)), "Weekly Move": format_pct(gold_1w), "Percentile": f"{pct_rank_last(gld, 252):.0f}%" if not gld.empty else "NA", "Score": score, "Read": read},
        {"Sleeve": "Commodities", "Input": "Copper", "Latest": format_level(safe_last(cper)), "Weekly Move": format_pct(copper_1w), "Percentile": f"{pct_rank_last(cper, 252):.0f}%" if not cper.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


def compute_hedge_score() -> Tuple[int, str, str, List[dict]]:
    corr_now = safe_last(spy_tlt_corr)
    corr_w = level_change_days(spy_tlt_corr, week_back)
    tlt_1w = pct_change_days(tlt, week_back)
    spy_1w = pct_change_days(spy, week_back)
    vix_1w = level_change_days(vix, week_back)

    raw = 0
    if pd.notna(corr_now):
        if corr_now <= -0.25:
            raw += 1
        elif corr_now >= 0.25:
            raw -= 1
    if pd.notna(spy_1w) and spy_1w < -1 and pd.notna(tlt_1w):
        if tlt_1w > 0:
            raw += 1
        else:
            raw -= 1
    if pd.notna(vix_1w) and vix_1w > 2 and pd.notna(tlt_1w) and tlt_1w < 0:
        raw -= 1

    score = score_clip(raw)
    state = "Working" if score > 0 else "Broken" if score < 0 else "Mixed"
    read = f"SPX-TLT corr {format_level(corr_now)}, corr change {format_level(corr_w)}, TLT 1W {format_pct(tlt_1w)}, SPX 1W {format_pct(spy_1w)}."
    inputs = [
        {"Sleeve": "Hedge quality", "Input": "SPX-TLT corr", "Latest": format_level(corr_now), "Weekly Move": format_level(corr_w), "Percentile": f"{pct_rank_last(spy_tlt_corr, 252):.0f}%" if not spy_tlt_corr.empty else "NA", "Score": score, "Read": state},
        {"Sleeve": "Hedge quality", "Input": "TLT", "Latest": format_level(safe_last(tlt)), "Weekly Move": format_pct(tlt_1w), "Percentile": f"{pct_rank_last(tlt, 252):.0f}%" if not tlt.empty else "NA", "Score": score, "Read": read},
    ]
    return score, state, read, inputs


score_functions = [
    compute_risk_appetite_score,
    compute_vol_score,
    compute_rates_score,
    compute_credit_score,
    compute_dollar_score,
    compute_commodity_score,
    compute_hedge_score,
]

score_rows: List[dict] = []
regime_inputs: List[dict] = []
state_reads: Dict[str, str] = {}

for fn in score_functions:
    score, state, read, inputs = fn()
    sleeve = inputs[0]["Sleeve"] if inputs else fn.__name__
    score_rows.append({"Sleeve": sleeve, "State": state, "Score": score, "Read": read})
    regime_inputs.extend(inputs)
    state_reads[sleeve] = read

score_df = pd.DataFrame(score_rows)
regime_input_df = pd.DataFrame(regime_inputs)
net_score = int(score_df["Score"].sum()) if not score_df.empty else 0

if net_score >= 5:
    master_regime = "Supportive"
elif net_score >= 2:
    master_regime = "Constructive"
elif net_score <= -5:
    master_regime = "Fragile"
elif net_score <= -2:
    master_regime = "Choppy"
else:
    master_regime = "Mixed"

# ============================================================
# Regime score history
# ============================================================
def rolling_regime_score_history() -> pd.Series:
    idx = prices.index
    out = pd.Series(index=idx, dtype=float)

    if "SPY" not in prices.columns:
        return out.dropna()

    spy_px = prices["SPY"]
    spy_ret_21 = spy_px.pct_change(21) * 100
    spy_ma50 = spy_px.rolling(50).mean()
    spy_ma200 = spy_px.rolling(200).mean()
    vix_px = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(index=idx, dtype=float)
    tlt_ret = returns["TLT"] if "TLT" in returns.columns else pd.Series(index=idx, dtype=float)
    spy_ret = returns["SPY"]
    spy_tlt = spy_ret.rolling(corr_window).corr(tlt_ret) if not tlt_ret.empty else pd.Series(index=idx, dtype=float)
    hyg_lqd_local = None
    if "HYG" in prices.columns and "LQD" in prices.columns:
        hyg_lqd_local = (prices["HYG"] / prices["LQD"]).replace([np.inf, -np.inf], np.nan)
    uup_ret_21 = prices["UUP"].pct_change(21) * 100 if "UUP" in prices.columns else pd.Series(index=idx, dtype=float)
    uso_ret_5 = prices["USO"].pct_change(5) * 100 if "USO" in prices.columns else pd.Series(index=idx, dtype=float)

    hy_oas_b = fred_b["BAMLH0A0HYM2"] if "BAMLH0A0HYM2" in fred_b.columns else pd.Series(index=idx, dtype=float)
    ten_b = fred_b["DGS10"] if "DGS10" in fred_b.columns else pd.Series(index=idx, dtype=float)
    real_b = fred_b["DFII10"] if "DFII10" in fred_b.columns else pd.Series(index=idx, dtype=float)

    for dt in idx:
        raw = 0.0
        if dt in spy_px.index:
            if pd.notna(spy_ret_21.loc[dt]):
                raw += 1 if spy_ret_21.loc[dt] > 2 else -1 if spy_ret_21.loc[dt] < -2 else 0
            if pd.notna(spy_ma50.loc[dt]):
                raw += 0.75 if spy_px.loc[dt] > spy_ma50.loc[dt] else -0.75
            if pd.notna(spy_ma200.loc[dt]):
                raw += 0.75 if spy_px.loc[dt] > spy_ma200.loc[dt] else -0.75
        if dt in vix_px.index and pd.notna(vix_px.loc[dt]):
            raw += 1 if vix_px.loc[dt] < 16 else -1 if vix_px.loc[dt] > 22 else 0
        if dt in spy_tlt.index and pd.notna(spy_tlt.loc[dt]):
            raw += 1 if spy_tlt.loc[dt] < -0.25 else -1 if spy_tlt.loc[dt] > 0.25 else 0
        if hyg_lqd_local is not None and dt in hyg_lqd_local.index:
            rel = hyg_lqd_local.pct_change(21).get(dt, np.nan) * 100
            raw += 0.75 if pd.notna(rel) and rel > 0 else -0.75 if pd.notna(rel) and rel < -1 else 0
        if dt in uup_ret_21.index and pd.notna(uup_ret_21.loc[dt]):
            raw += 0.75 if uup_ret_21.loc[dt] < -1.5 else -0.75 if uup_ret_21.loc[dt] > 1.5 else 0
        if dt in uso_ret_5.index and pd.notna(uso_ret_5.loc[dt]):
            raw += -0.75 if uso_ret_5.loc[dt] > 5 else 0.25 if uso_ret_5.loc[dt] < -5 else 0
        if dt in hy_oas_b.index and pd.notna(hy_oas_b.diff(5).loc[dt]):
            raw += 1 if hy_oas_b.diff(5).loc[dt] < -0.15 else -1 if hy_oas_b.diff(5).loc[dt] > 0.15 else 0
        if dt in ten_b.index and pd.notna(ten_b.diff(5).loc[dt]):
            raw += 0.5 if ten_b.diff(5).loc[dt] < -0.12 else -0.5 if ten_b.diff(5).loc[dt] > 0.12 else 0
        if dt in real_b.index and pd.notna(real_b.diff(5).loc[dt]):
            raw += 0.5 if real_b.diff(5).loc[dt] < -0.10 else -0.5 if real_b.diff(5).loc[dt] > 0.10 else 0
        out.loc[dt] = raw

    return out.dropna()

regime_history = rolling_regime_score_history()
chart_cutoff = prices.index.max() - pd.DateOffset(months=chart_history_months)
regime_history_chart = regime_history[regime_history.index >= chart_cutoff]

# ============================================================
# Top cards
# ============================================================
def card(title: str, value: str, sub: str, state: str = "") -> str:
    color = "#111827"
    if state in {"Supportive", "Constructive", "Working", "Contained", "Loose"}:
        color = "#166534"
    elif state in {"Fragile", "Elevated", "Broken", "Tightening", "Defensive", "Hot"}:
        color = "#991b1b"
    elif state in {"Mixed", "Watchful", "Neutral", "Balanced", "Choppy"}:
        color = "#92400e"
    return f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value" style="color:{color};">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>
    """

score_map = score_df.set_index("Sleeve").to_dict("index") if not score_df.empty else {}

risk_state = score_map.get("Risk appetite", {}).get("State", "NA")
vol_state = score_map.get("Volatility", {}).get("State", "NA")
rates_state = score_map.get("Rates", {}).get("State", "NA")
credit_state = score_map.get("Credit", {}).get("State", "NA")
dollar_state = score_map.get("Dollar liquidity", {}).get("State", "NA")
commodity_state = score_map.get("Commodities", {}).get("State", "NA")
hedge_state = score_map.get("Hedge quality", {}).get("State", "NA")

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

with c1:
    st.markdown(card("Regime", f"{master_regime} ({net_score:+d})", "Composite score across seven sleeves", master_regime), unsafe_allow_html=True)
with c2:
    st.markdown(card("Risk Appetite", f"{risk_state}", state_reads.get("Risk appetite", ""), risk_state), unsafe_allow_html=True)
with c3:
    st.markdown(card("Volatility", f"{vol_state}", state_reads.get("Volatility", ""), vol_state), unsafe_allow_html=True)
with c4:
    st.markdown(card("Rates Pressure", f"{rates_state}", state_reads.get("Rates", ""), rates_state), unsafe_allow_html=True)
with c5:
    st.markdown(card("Credit", f"{credit_state}", state_reads.get("Credit", ""), credit_state), unsafe_allow_html=True)
with c6:
    st.markdown(card("Dollar Liquidity", f"{dollar_state}", state_reads.get("Dollar liquidity", ""), dollar_state), unsafe_allow_html=True)
with c7:
    st.markdown(card("Commodities", f"{commodity_state}", state_reads.get("Commodities", ""), commodity_state), unsafe_allow_html=True)
with c8:
    st.markdown(card("Hedge Quality", f"{hedge_state}", state_reads.get("Hedge quality", ""), hedge_state), unsafe_allow_html=True)

st.markdown("")

# ============================================================
# Charts: regime history and score contribution
# ============================================================
left_chart, right_chart = st.columns([1.15, 0.85])

with left_chart:
    st.markdown('<div class="section-label">Regime Score Over Time</div>', unsafe_allow_html=True)
    if not regime_history_chart.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=regime_history_chart.index,
            y=regime_history_chart.values,
            mode="lines",
            line=dict(width=2.2),
            name="Regime score",
            hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
        ))
        fig.add_hline(y=5, line_dash="dot", line_color="green", annotation_text="Supportive")
        fig.add_hline(y=-5, line_dash="dot", line_color="red", annotation_text="Fragile")
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Composite score",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to render regime score history.")

with right_chart:
    st.markdown('<div class="section-label">Current Sleeve Contributions</div>', unsafe_allow_html=True)
    if not score_df.empty:
        contrib = score_df.sort_values("Score", ascending=True)
        fig = px.bar(
            contrib,
            x="Score",
            y="Sleeve",
            orientation="h",
            text="Score",
            hover_data=["State", "Read"],
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
# What changed this week
# ============================================================
st.markdown('<div class="section-label">What Changed This Week</div>', unsafe_allow_html=True)

asset_rows: List[dict] = []
for ticker, meta in YAHOO_ASSETS.items():
    if ticker not in prices.columns:
        continue
    s = prices[ticker].dropna()
    if s.empty:
        continue
    kind = meta["kind"]
    if kind in {"level", "yield_proxy"}:
        one_w = level_change_days(s, week_back)
        one_m = level_change_days(s, 21)
        three_m = level_change_days(s, 63)
        weekly_fmt = format_level(one_w)
        one_m_fmt = format_level(one_m)
        three_m_fmt = format_level(three_m)
        sort_abs = abs(one_w) if pd.notna(one_w) else 0
    else:
        one_w = pct_change_days(s, week_back)
        one_m = pct_change_days(s, 21)
        three_m = pct_change_days(s, 63)
        weekly_fmt = format_pct(one_w)
        one_m_fmt = format_pct(one_m)
        three_m_fmt = format_pct(three_m)
        sort_abs = abs(one_w) if pd.notna(one_w) else 0
    rv = realized_vol(returns[ticker], 21) if ticker in returns.columns else pd.Series(dtype=float)
    asset_rows.append({
        "Asset": meta["label"],
        "Ticker": ticker,
        "Group": meta["group"],
        "Level": safe_last(s),
        "1W": weekly_fmt,
        "1M": one_m_fmt,
        "3M": three_m_fmt,
        "1Y pctile": pct_rank_last(s, 252),
        "RV z": zscore_last(rv, 252),
        "Abs 1W Move": sort_abs,
    })

# Add FRED macro rows with bps moves.
for sid, meta in FRED_SERIES.items():
    if sid not in fred_b.columns:
        continue
    s = fred_b[sid].dropna()
    if s.empty:
        continue
    one_w_bps = bps_change_days(s, week_back) if meta["unit"] == "pct" else pct_change_days(s, week_back)
    one_m_bps = bps_change_days(s, 21) if meta["unit"] == "pct" else pct_change_days(s, 21)
    three_m_bps = bps_change_days(s, 63) if meta["unit"] == "pct" else pct_change_days(s, 63)
    if meta["unit"] == "pct":
        weekly_fmt = format_bps(one_w_bps)
        one_m_fmt = format_bps(one_m_bps)
        three_m_fmt = format_bps(three_m_bps)
        level = f"{safe_last(s):.2f}%"
        sort_abs = abs(one_w_bps) if pd.notna(one_w_bps) else 0
    else:
        weekly_fmt = format_pct(one_w_bps)
        one_m_fmt = format_pct(one_m_bps)
        three_m_fmt = format_pct(three_m_bps)
        level = format_level(safe_last(s))
        sort_abs = abs(one_w_bps) if pd.notna(one_w_bps) else 0
    asset_rows.append({
        "Asset": meta["label"],
        "Ticker": sid,
        "Group": meta["group"],
        "Level": level,
        "1W": weekly_fmt,
        "1M": one_m_fmt,
        "3M": three_m_fmt,
        "1Y pctile": pct_rank_last(s, 252),
        "RV z": np.nan,
        "Abs 1W Move": sort_abs,
    })

asset_tbl = pd.DataFrame(asset_rows)
if not asset_tbl.empty:
    display_tbl = asset_tbl.sort_values("Abs 1W Move", ascending=False).head(top_n_assets).drop(columns=["Abs 1W Move"])
    st.dataframe(
        display_tbl.style.format({"Level": lambda x: f"{x:,.2f}" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else str(x), "1Y pctile": "{:.0f}%", "RV z": "{:+.2f}"}),
        use_container_width=True,
        height=min(620, 38 * (len(display_tbl) + 1)),
    )
else:
    st.info("No weekly tape data available.")

# ============================================================
# Regime model table and action matrix
# ============================================================
st.markdown('<div class="section-label">Regime Model Inputs</div>', unsafe_allow_html=True)
if not regime_input_df.empty:
    regime_display = regime_input_df[["Sleeve", "Input", "Latest", "Weekly Move", "Percentile", "Score", "Read"]].copy()
    st.dataframe(
        regime_display.style.format({"Score": "{:+d}"}),
        use_container_width=True,
        height=min(650, 38 * (len(regime_display) + 1)),
    )
else:
    st.info("No regime inputs available.")

# ============================================================
# Action bias matrix
# ============================================================
def build_action_matrix() -> pd.DataFrame:
    rows = []
    if master_regime in {"Supportive", "Constructive"} and credit_state in {"Contained", "Watchful"}:
        gross = "Press selectively"
        gross_note = "Gross can stay elevated where leadership and credit confirmation agree. Avoid adding to areas that only work because vol is quiet."
    elif master_regime in {"Fragile", "Choppy"}:
        gross = "Cut or keep tight"
        gross_note = "Gross has to earn its place because cross-asset shock absorbers are weaker."
    else:
        gross = "Balanced"
        gross_note = "Favor relative value, quality leadership, and tighter stop discipline over index-level conviction."

    if risk_state in {"Constructive"} and credit_state == "Contained":
        net = "Modestly long"
    elif master_regime == "Fragile":
        net = "Lower beta"
    else:
        net = "Neutral to tactical"

    if hedge_state == "Working":
        duration = "Useful"
        duration_note = "Duration can remain in the hedge stack, especially against growth shocks."
    elif hedge_state == "Broken":
        duration = "Weak hedge"
        duration_note = "Treat duration as a macro position. Use equity optionality or credit hedges for cleaner downside protection."
    else:
        duration = "Mixed"
        duration_note = "Keep duration sizing honest and validate with the SPX-TLT correlation."

    if credit_state == "Tightening":
        credit = "Add credit lens"
        credit_note = "Lower-quality cyclicals and spread beta deserve tighter risk limits."
    else:
        credit = "Confirmation only"
        credit_note = "Credit is not yet forcing broad de-risking, so use it as a confirmation layer."

    if dollar_state == "Tightening":
        fx = "Respect USD"
        fx_note = "Dollar pressure argues for caution in EM, commodity FX, and externally sensitive risk."
    elif dollar_state == "Loose":
        fx = "Liquidity tailwind"
        fx_note = "Dollar softness supports EM and global beta if credit remains calm."
    else:
        fx = "Mixed"
        fx_note = "Keep FX as a cross-check rather than the main signal."

    if commodity_state == "Hot":
        commodity = "Watch inflation shock"
        commodity_note = "Oil volatility can turn into margin pressure and rates pressure quickly."
    else:
        commodity = "Contained"
        commodity_note = "Commodity tape is less likely to dominate the index regime right now."

    if vol_state == "Constructive":
        opt = "Own less naked protection"
        opt_note = "Spreads and overwrite structures make more sense when stress pricing is contained."
    elif vol_state == "Elevated":
        opt = "Use defined-risk hedges"
        opt_note = "Avoid chasing expensive crash premium unless the regime score is deteriorating."
    else:
        opt = "Be opportunistic"
        opt_note = "Use VIX versus realized to decide whether to buy or finance protection."

    rows.extend([
        {"Decision Area": "Gross", "Bias": gross, "Reason": gross_note},
        {"Decision Area": "Net beta", "Bias": net, "Reason": f"Regime {master_regime}, risk appetite {risk_state}, credit {credit_state}."},
        {"Decision Area": "Duration hedge", "Bias": duration, "Reason": duration_note},
        {"Decision Area": "Credit hedge", "Bias": credit, "Reason": credit_note},
        {"Decision Area": "Dollar / JPY", "Bias": fx, "Reason": fx_note},
        {"Decision Area": "Commodity hedge", "Bias": commodity, "Reason": commodity_note},
        {"Decision Area": "Optionality", "Bias": opt, "Reason": opt_note},
    ])
    return pd.DataFrame(rows)

st.markdown('<div class="section-label">Positioning and Hedge Bias</div>', unsafe_allow_html=True)
action_df = build_action_matrix()
st.dataframe(action_df, use_container_width=True, height=38 * (len(action_df) + 1))

# ============================================================
# Cross-asset signal ranking
# ============================================================
def pair_interpretation(pair: str, rho_now: float, rho_delta: float, pctile: float) -> Tuple[str, str]:
    if pair == "SPX vs TLT":
        if pd.notna(rho_now) and rho_now <= -0.25:
            return "Duration is cushioning equity risk", "Long duration can still work as part of hedge stack"
        if pd.notna(rho_now) and rho_now >= 0.25:
            return "Duration hedge is impaired", "Use equity optionality or credit hedges instead of relying on TLT"
        return "Duration hedge is mixed", "Treat TLT as tactical rather than automatic protection"
    if pair == "SPX vs HY":
        if pd.notna(rho_now) and rho_now >= 0.50:
            return "Credit is tightly linked to equities", "HY confirms the risk tape and should guide cyclicals"
        return "Credit linkage is moderate", "Use HY as confirmation rather than primary signal"
    if pair == "EM vs Dollar":
        if pd.notna(rho_now) and rho_now <= -0.35:
            return "Dollar is tightening global beta", "Respect pressure on EM and commodity-sensitive risk"
        return "Dollar linkage is present but less dominant", "Cross-check with credit and rates"
    if pair == "Japan vs USDJPY":
        if pd.notna(rho_now) and rho_now >= 0.35:
            return "Japan still benefits from yen weakness", "Do not fade Japan without a clean FX view"
        return "FX transmission into Japan is loosening", "Japan can trade more on equity beta"
    if pair == "Gold vs TLT":
        if pd.notna(rho_now) and rho_now <= -0.25:
            return "Gold is diversifying duration", "Gold may be carrying policy distrust or geopolitical premium"
        return "Gold and duration linkage is mixed", "Use real yields and dollar to validate gold"
    return "Cross-asset linkage is moving", "Use as context and confirm with price leadership"

pair_rows: List[dict] = []
for spec in PAIR_SPECS:
    a, b, pair = spec["a"], spec["b"], spec["pair"]
    if a not in returns.columns or b not in returns.columns:
        continue
    rc = rolling_corr(returns[a], returns[b], corr_window)
    if len(rc) < week_back + 5:
        continue
    rho_now = safe_last(rc)
    rho_delta = level_change_days(rc, week_back)
    pctile = pct_rank_last(rc, 504)
    signal_score = min(abs(rho_now) / 0.8, 1.0) * 45 if pd.notna(rho_now) else 0
    signal_score += min(abs(rho_delta) / 0.25, 1.0) * 40 if pd.notna(rho_delta) else 0
    signal_score += abs(pctile - 50) / 50 * 15 if pd.notna(pctile) else 0
    interp, action = pair_interpretation(pair, rho_now, rho_delta, pctile)
    pair_rows.append({
        "Pair": pair,
        "ρ now": rho_now,
        "Δρ": rho_delta,
        "1Y pctile": pctile,
        "Signal": round(signal_score, 1),
        "Interpretation": interp,
        "Action": action,
        "Series": rc,
    })

pair_tbl = pd.DataFrame(pair_rows)
if not pair_tbl.empty:
    pair_tbl = pair_tbl.sort_values(["Signal", "Δρ"], ascending=[False, False]).reset_index(drop=True)

st.markdown('<div class="section-label">Top Cross-Asset Shifts</div>', unsafe_allow_html=True)
if not pair_tbl.empty:
    top_pairs = pair_tbl.head(top_pair_signals).copy()
    st.dataframe(
        top_pairs.drop(columns=["Series"]).style.format({"ρ now": "{:+.2f}", "Δρ": "{:+.2f}", "1Y pctile": "{:.0f}%", "Signal": "{:.1f}"}),
        use_container_width=True,
        height=min(520, 38 * (len(top_pairs) + 1)),
    )
else:
    st.info("No cross-asset pair signals available.")

# ============================================================
# Correlation map and pair charts
# ============================================================
left, right = st.columns([1.0, 1.0])

with left:
    st.markdown('<div class="section-label">Current Correlation Map</div>', unsafe_allow_html=True)
    heat_tickers = [t for t in ["SPY", "QQQ", "IWM", "HYG", "LQD", "TLT", "UUP", "GLD", "USO", "USDJPY=X"] if t in returns.columns]
    hm = returns[heat_tickers].dropna().tail(corr_window)
    if len(hm) >= max(10, corr_window // 2):
        corr = hm.corr()
        labels = [YAHOO_ASSETS[t]["label"] for t in corr.columns]
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdYlGn",
            text=np.round(corr.values, 2),
            texttemplate="%{text:+.2f}",
            hovertemplate="%{y} vs %{x}<br>Corr: %{z:+.2f}<extra></extra>",
        ))
        fig.update_layout(height=530, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for correlation map.")

with right:
    st.markdown('<div class="section-label">Stress Map</div>', unsafe_allow_html=True)
    stress_df = score_df.copy()
    if not stress_df.empty:
        stress_df["Stress"] = -stress_df["Score"]
        fig = px.bar(
            stress_df.sort_values("Stress", ascending=True),
            x="Stress",
            y="Sleeve",
            orientation="h",
            hover_data=["State", "Read"],
        )
        fig.add_vline(x=0, line_color="gray", line_width=1)
        fig.update_layout(
            height=530,
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
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rc.index,
                    y=rc.values,
                    mode="lines",
                    name=f"{corr_window}D corr",
                    line=dict(width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=mean.index,
                    y=mean.values,
                    mode="lines",
                    name="6M mean",
                    line=dict(width=1.5, dash="dash"),
                ))
                upper = mean + std
                lower = mean - std
                fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", line=dict(width=0), fill="tonexty", name="6M band", hoverinfo="skip"))
                fig.add_hline(y=0, line_color="gray", line_width=1)
                fig.add_hline(y=0.30, line_dash="dot", line_color="gray")
                fig.add_hline(y=-0.30, line_dash="dot", line_color="gray")
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10), title=row["Pair"], yaxis=dict(range=[-1, 1]), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{row['Interpretation']}. {row['Action']}.")
else:
    st.info("No detailed pair charts available.")

# ============================================================
# Dynamic commentary
# ============================================================
def build_dynamic_commentary() -> Tuple[str, str, str]:
    strongest = score_df.sort_values("Score", ascending=False).head(2) if not score_df.empty else pd.DataFrame()
    weakest = score_df.sort_values("Score", ascending=True).head(2) if not score_df.empty else pd.DataFrame()

    strong_text = ", ".join([f"{r['Sleeve']} ({r['State']}, {int(r['Score']):+d})" for _, r in strongest.iterrows()]) if not strongest.empty else "none"
    weak_text = ", ".join([f"{r['Sleeve']} ({r['State']}, {int(r['Score']):+d})" for _, r in weakest.iterrows()]) if not weakest.empty else "none"

    state = (
        f"The composite regime is {master_regime.lower()} with a net score of {net_score:+d}. The supportive sleeves are {strong_text}; the pressure points are {weak_text}. "
        f"The read is driven by actual weekly changes rather than static labels: {state_reads.get('Rates', '')} {state_reads.get('Credit', '')} {state_reads.get('Dollar liquidity', '')}"
    )

    if master_regime in {"Supportive", "Constructive"}:
        action = "The book can keep pressing the parts of the market where leadership, credit, and volatility agree, but the sizing should still respect the weakest sleeve on the board."
    elif master_regime in {"Fragile", "Choppy"}:
        action = "The better posture is tighter gross, cleaner hedges, and less tolerance for positions that need benign correlations to work."
    else:
        action = "The right posture is selective and relative-value driven, with index-level conclusions subordinated to leadership, credit, and hedge behavior."

    if hedge_state == "Broken":
        action += " Duration is not carrying the hedge book cleanly, so downside protection should lean more on equity optionality, credit overlays, or lower gross."
    elif hedge_state == "Working":
        action += " Duration can still carry part of the hedge stack, which gives the portfolio more room to tolerate equity risk."

    invalidation = (
        "This read should be challenged if the weakest sleeves reverse for a full week, if credit spreads and HYG/LQD stop confirming the current credit state, "
        "or if SPX-TLT correlation changes enough to alter hedge quality. A weekly compass is useful only if it updates when the transmission mechanism changes."
    )
    return state, action, invalidation

state_commentary, action_commentary, invalidation_commentary = build_dynamic_commentary()

st.markdown('<div class="section-label">Dynamic Commentary</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1.2, 1.0, 1.0])
with col_a:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">State</div>
            <div style="line-height:1.55; color:#1f2937;">{state_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">Action Bias</div>
            <div style="line-height:1.55; color:#1f2937;">{action_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        f"""
        <div class="commentary-box">
            <div style="font-weight:700; margin-bottom:0.5rem;">Invalidation</div>
            <div style="line-height:1.55; color:#1f2937;">{invalidation_commentary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Diagnostics and footer
# ============================================================
if show_raw_data:
    with st.expander("Data Diagnostics", expanded=True):
        st.markdown("**Yahoo metadata**")
        st.json(yahoo_meta)
        if fred_meta:
            st.markdown("**FRED metadata**")
            st.json(fred_meta)
        st.markdown("**Available Yahoo columns**")
        st.write(list(prices.columns))
        st.markdown("**Available FRED columns**")
        st.write(list(fred_b.columns))

last_date = prices.index.max()
footer_date = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "N/A"
st.caption(f"As of {footer_date} | © 2026 AD Fund Management LP")
