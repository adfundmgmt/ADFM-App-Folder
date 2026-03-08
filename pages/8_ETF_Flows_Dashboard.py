import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="ETF Net Flows", layout="wide")

# =========================================================
# GLOBALS
# =========================================================
TZ = pytz.timezone("US/Eastern")

POS_COLOR = "#2E8B57"
NEG_COLOR = "#C0392B"
ZERO_COLOR = "#9E9E9E"
PROXY_POS = "#8BC4A8"
PROXY_NEG = "#E2A6A0"
GRID_COLOR = "#EAEAEA"
TEXT_COLOR = "#222222"
SUBTLE = "#6B7280"

LOOKBACK_OPTIONS = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": None,  # computed dynamically
}

# =========================================================
# ETF METADATA
# =========================================================
ETF_META = {
    "VTI": {"label": "US Total Market", "desc": "Total US equity market", "bucket": "US Equity", "group": "Core Beta"},
    "VUG": {"label": "US Growth", "desc": "Large-cap growth", "bucket": "US Equity", "group": "Style"},
    "VTV": {"label": "US Value", "desc": "Large-cap value", "bucket": "US Equity", "group": "Style"},
    "MTUM": {"label": "US Momentum", "desc": "Momentum factor", "bucket": "US Equity", "group": "Factor"},
    "QUAL": {"label": "US Quality", "desc": "Quality factor", "bucket": "US Equity", "group": "Factor"},
    "USMV": {"label": "US Min Vol", "desc": "Minimum volatility", "bucket": "US Equity", "group": "Factor"},
    "SCHD": {"label": "US Dividends", "desc": "Dividend growth and yield", "bucket": "US Equity", "group": "Income"},
    "XLB": {"label": "US Materials", "desc": "S&P 500 Materials", "bucket": "US Equity", "group": "Sector"},
    "XLC": {"label": "US Communication Services", "desc": "S&P 500 Communication Services", "bucket": "US Equity", "group": "Sector"},
    "XLE": {"label": "US Energy", "desc": "S&P 500 Energy", "bucket": "US Equity", "group": "Sector"},
    "XLF": {"label": "US Financials", "desc": "S&P 500 Financials", "bucket": "US Equity", "group": "Sector"},
    "XLI": {"label": "US Industrials", "desc": "S&P 500 Industrials", "bucket": "US Equity", "group": "Sector"},
    "XLK": {"label": "US Technology", "desc": "S&P 500 Technology", "bucket": "US Equity", "group": "Sector"},
    "XLP": {"label": "US Staples", "desc": "S&P 500 Consumer Staples", "bucket": "US Equity", "group": "Sector"},
    "XLRE": {"label": "US Real Estate", "desc": "S&P 500 Real Estate", "bucket": "US Equity", "group": "Sector"},
    "XLU": {"label": "US Utilities", "desc": "S&P 500 Utilities", "bucket": "US Equity", "group": "Sector"},
    "XLV": {"label": "US Healthcare", "desc": "S&P 500 Healthcare", "bucket": "US Equity", "group": "Sector"},
    "XLY": {"label": "US Discretionary", "desc": "S&P 500 Consumer Discretionary", "bucket": "US Equity", "group": "Sector"},
    "SMH": {"label": "Semiconductors", "desc": "Global semiconductor equities", "bucket": "Thematic Equity", "group": "Tech Theme"},
    "IGV": {"label": "Software", "desc": "US application software", "bucket": "Thematic Equity", "group": "Tech Theme"},
    "SKYY": {"label": "Cloud", "desc": "Cloud infrastructure and services", "bucket": "Thematic Equity", "group": "Tech Theme"},
    "VT": {"label": "Global Equity", "desc": "Total world equity market", "bucket": "Global Equity", "group": "Core Beta"},
    "ACWI": {"label": "Global Equity ex-Frontier", "desc": "All-country world equity", "bucket": "Global Equity", "group": "Core Beta"},
    "VEA": {"label": "Developed ex-US", "desc": "Developed markets ex-US", "bucket": "DM ex-US", "group": "Core Beta"},
    "VWO": {"label": "Emerging Markets", "desc": "Emerging markets equity", "bucket": "EM Equity", "group": "Core Beta"},
    "EXUS": {"label": "Non-US Equity", "desc": "Global equity ex-US", "bucket": "DM ex-US", "group": "Core Beta"},
    "EWG": {"label": "Germany", "desc": "Germany equities", "bucket": "Europe", "group": "Country"},
    "EWQ": {"label": "France", "desc": "France equities", "bucket": "Europe", "group": "Country"},
    "EWU": {"label": "United Kingdom", "desc": "UK equities", "bucket": "Europe", "group": "Country"},
    "EWI": {"label": "Italy", "desc": "Italy equities", "bucket": "Europe", "group": "Country"},
    "EWP": {"label": "Spain", "desc": "Spain equities", "bucket": "Europe", "group": "Country"},
    "EWL": {"label": "Switzerland", "desc": "Switzerland equities", "bucket": "Europe", "group": "Country"},
    "EWN": {"label": "Netherlands", "desc": "Netherlands equities", "bucket": "Europe", "group": "Country"},
    "EWD": {"label": "Sweden", "desc": "Sweden equities", "bucket": "Europe", "group": "Country"},
    "EWO": {"label": "Austria", "desc": "Austria equities", "bucket": "Europe", "group": "Country"},
    "EWK": {"label": "Belgium", "desc": "Belgium equities", "bucket": "Europe", "group": "Country"},
    "EWJ": {"label": "Japan", "desc": "Japan equities", "bucket": "Asia DM", "group": "Country"},
    "EWY": {"label": "South Korea", "desc": "Korea equities", "bucket": "Asia EM", "group": "Country"},
    "ASHR": {"label": "China A-Shares", "desc": "Onshore China equities", "bucket": "Asia EM", "group": "Country"},
    "FXI": {"label": "China Large-Cap", "desc": "China offshore large caps", "bucket": "Asia EM", "group": "Country"},
    "EWT": {"label": "Taiwan", "desc": "Taiwan equities", "bucket": "Asia EM", "group": "Country"},
    "INDA": {"label": "India", "desc": "India equities", "bucket": "Asia EM", "group": "Country"},
    "EWS": {"label": "Singapore", "desc": "Singapore equities", "bucket": "Asia DM", "group": "Country"},
    "EWA": {"label": "Australia", "desc": "Australia equities", "bucket": "Asia DM", "group": "Country"},
    "EWH": {"label": "Hong Kong", "desc": "Hong Kong equities", "bucket": "Asia EM", "group": "Country"},
    "EPHE": {"label": "Philippines", "desc": "Philippines equities", "bucket": "Asia EM", "group": "Country"},
    "EWM": {"label": "Malaysia", "desc": "Malaysia equities", "bucket": "Asia EM", "group": "Country"},
    "IDX": {"label": "Indonesia", "desc": "Indonesia equities", "bucket": "Asia EM", "group": "Country"},
    "THD": {"label": "Thailand", "desc": "Thailand equities", "bucket": "Asia EM", "group": "Country"},
    "VNM": {"label": "Vietnam", "desc": "Vietnam equities", "bucket": "Asia EM", "group": "Country"},
    "EWZ": {"label": "Brazil", "desc": "Brazil equities", "bucket": "LatAm", "group": "Country"},
    "EWW": {"label": "Mexico", "desc": "Mexico equities", "bucket": "LatAm", "group": "Country"},
    "EWC": {"label": "Canada", "desc": "Canada equities", "bucket": "DM ex-US", "group": "Country"},
    "EPU": {"label": "Peru", "desc": "Peru equities", "bucket": "LatAm", "group": "Country"},
    "ECH": {"label": "Chile", "desc": "Chile equities", "bucket": "LatAm", "group": "Country"},
    "ARGT": {"label": "Argentina", "desc": "Argentina equities", "bucket": "LatAm", "group": "Country"},
    "GXG": {"label": "Colombia", "desc": "Colombia equities", "bucket": "LatAm", "group": "Country"},
    "SGOV": {"label": "UST Bills", "desc": "0-3 month Treasuries", "bucket": "Rates", "group": "UST"},
    "SHY": {"label": "UST 1-3y", "desc": "Short-term Treasuries", "bucket": "Rates", "group": "UST"},
    "IEF": {"label": "UST 7-10y", "desc": "Intermediate Treasuries", "bucket": "Rates", "group": "UST"},
    "TLT": {"label": "UST 20y+", "desc": "Long-duration Treasuries", "bucket": "Rates", "group": "UST"},
    "TIP": {"label": "TIPS", "desc": "Inflation-linked Treasuries", "bucket": "Rates", "group": "UST"},
    "LQD": {"label": "IG Credit", "desc": "Investment-grade corporates", "bucket": "Credit", "group": "IG"},
    "VCIT": {"label": "IG Credit Duration", "desc": "Intermediate IG corporates", "bucket": "Credit", "group": "IG"},
    "HYG": {"label": "High Yield", "desc": "High-yield credit", "bucket": "Credit", "group": "HY"},
    "BKLN": {"label": "Floating-Rate Credit", "desc": "Senior loans", "bucket": "Credit", "group": "Loans"},
    "EMB": {"label": "EM Debt", "desc": "USD EM sovereign debt", "bucket": "Credit", "group": "EM Debt"},
    "BND": {"label": "US Aggregate", "desc": "Total US bond market", "bucket": "Rates", "group": "Aggregate"},
    "GLD": {"label": "Gold", "desc": "Gold bullion", "bucket": "Commodities", "group": "Precious Metals"},
    "SLV": {"label": "Silver", "desc": "Silver bullion", "bucket": "Commodities", "group": "Precious Metals"},
    "CPER": {"label": "Copper", "desc": "Industrial copper", "bucket": "Commodities", "group": "Industrial Metals"},
    "USO": {"label": "Crude Oil", "desc": "WTI crude oil", "bucket": "Commodities", "group": "Energy"},
    "DBC": {"label": "Broad Commodities", "desc": "Commodity basket", "bucket": "Commodities", "group": "Broad Basket"},
    "PDBC": {"label": "Broad Commodities Alt", "desc": "Rules-based commodities", "bucket": "Commodities", "group": "Broad Basket"},
    "URA": {"label": "Uranium", "desc": "Nuclear fuel cycle", "bucket": "Commodities", "group": "Energy Transition"},
    "VXX": {"label": "Equity Volatility", "desc": "Front-end VIX futures", "bucket": "Volatility", "group": "Vol"},
    "UUP": {"label": "USD", "desc": "US Dollar Index", "bucket": "FX", "group": "DM FX"},
    "FXE": {"label": "EURUSD", "desc": "Euro vs USD", "bucket": "FX", "group": "DM FX"},
    "FXY": {"label": "USDJPY", "desc": "Japanese Yen", "bucket": "FX", "group": "DM FX"},
    "FXF": {"label": "CHFUSD", "desc": "Swiss franc", "bucket": "FX", "group": "DM FX"},
    "CEW": {"label": "EM FX", "desc": "Emerging market currencies", "bucket": "FX", "group": "EM FX"},
    "IBIT": {"label": "Bitcoin", "desc": "Spot Bitcoin ETF", "bucket": "Crypto", "group": "Digital Assets"},
    "ETH": {"label": "Ethereum", "desc": "Spot Ethereum ETF", "bucket": "Crypto", "group": "Digital Assets"},
}

UNIVERSE_PRESETS = {
    "Core Macro": ["SPY"] if "SPY" in ETF_META else [],
    "US Equity": ["VTI", "VUG", "VTV", "MTUM", "QUAL", "USMV", "SCHD", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SMH", "IGV", "SKYY"],
    "Global Equity": ["VT", "ACWI", "VEA", "VWO", "EXUS", "EWG", "EWQ", "EWU", "EWI", "EWP", "EWL", "EWN", "EWD", "EWO", "EWK", "EWJ", "EWY", "ASHR", "FXI", "EWT", "INDA", "EWS", "EWA", "EWH", "EPHE", "EWM", "IDX", "THD", "VNM", "EWZ", "EWW", "EWC", "EPU", "ECH", "ARGT", "GXG"],
    "Rates & Credit": ["SGOV", "SHY", "IEF", "TLT", "TIP", "LQD", "VCIT", "HYG", "BKLN", "EMB", "BND"],
    "Commodities": ["GLD", "SLV", "CPER", "USO", "DBC", "PDBC", "URA"],
    "FX": ["UUP", "FXE", "FXY", "FXF", "CEW"],
    "Crypto": ["IBIT", "ETH"],
    "Full Cross-Asset": list(ETF_META.keys()),
}

# add SPY metadata if missing
if "SPY" not in ETF_META:
    ETF_META["SPY"] = {"label": "S&P 500", "desc": "S&P 500 ETF", "bucket": "US Equity", "group": "Core Beta"}
    UNIVERSE_PRESETS["US Equity"] = ["SPY"] + UNIVERSE_PRESETS["US Equity"]
    UNIVERSE_PRESETS["Full Cross-Asset"] = ["SPY"] + [x for x in UNIVERSE_PRESETS["Full Cross-Asset"] if x != "SPY"]

# =========================================================
# HELPERS
# =========================================================
def now_et() -> datetime:
    return datetime.now(TZ)


def ytd_days(as_of: datetime) -> int:
    start_ytd = TZ.localize(datetime(as_of.year, 1, 1))
    return max(1, (as_of - start_ytd).days)


def as_naive_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts.replace(tzinfo=None))


def fmt_compact_cur(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:,.2f}T"
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.1f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.0f}%"


def axis_fmt(x, _pos=None) -> str:
    return fmt_compact_cur(x)


def safe_num(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def retry(n=2, delay=0.5, exceptions=(Exception,)):
    def deco(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last = e
                    time.sleep(delay * (i + 1))
            raise last
        return wrap
    return deco


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx
    df = df[need].copy()
    return df.dropna(subset=["Close"]).sort_index()


def compute_start_date(days: int, extra_pad: int = 15) -> date:
    return (now_et() - pd.Timedelta(days=days + extra_pad)).date()


def get_week_bounds(view: str, as_of_ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp(as_of_ts).normalize()
    monday_this_week = today - pd.Timedelta(days=today.weekday())

    if view == "Week to date":
        return monday_this_week, today

    if view == "Last complete week":
        last_week_monday = monday_this_week - pd.Timedelta(days=7)
        last_week_friday = last_week_monday + pd.Timedelta(days=4)
        return last_week_monday, last_week_friday

    raise ValueError("Unsupported weekly view")


def classify_confidence(method: str, coverage_pct: float, shares_obs: int) -> str:
    if method != "share_flows":
        return "Low"
    if coverage_pct >= 90 and shares_obs >= 5:
        return "High"
    if coverage_pct >= 60 and shares_obs >= 2:
        return "Medium"
    return "Low"


def coverage_color_label(confidence: str) -> str:
    if confidence == "High":
        return "High"
    if confidence == "Medium":
        return "Medium"
    return "Low"


def build_label(ticker: str) -> str:
    meta = ETF_META.get(ticker, {})
    return f"{meta.get('label', ticker)} ({ticker})"


def get_universe(preset_name: str, extra_tickers: str = "") -> List[str]:
    base = list(UNIVERSE_PRESETS.get(preset_name, []))
    extra = []
    if extra_tickers.strip():
        extra = [x.strip().upper() for x in extra_tickers.split(",") if x.strip()]
    merged = []
    seen = set()
    for tk in base + extra:
        if tk not in seen:
            seen.add(tk)
            merged.append(tk)
    return merged


# =========================================================
# DATA FETCHING
# =========================================================
@retry(n=2, delay=0.5)
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prices(tickers: Tuple[str, ...], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date + timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out = {}
    for tk in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[tk].copy()
            else:
                df = data.copy()
        except Exception:
            df = pd.DataFrame()
        out[tk] = normalize_ohlcv(df)
    return out


@retry(n=2, delay=0.5)
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_one_shares_series_cached(ticker: str, start_date: date) -> pd.Series:
    t = yf.Ticker(ticker)
    try:
        s = t.get_shares_full(start=start_date)
    except Exception:
        s = None

    if s is None or (isinstance(s, pd.DataFrame) and s.empty):
        return pd.Series(dtype="float64")

    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series(dtype="float64")
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna()
    idx = pd.to_datetime(s.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    s.index = idx
    return s.sort_index()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_shares_series_parallel(
    tickers: Tuple[str, ...],
    start_date: date,
    timeout_sec: float = 12.0,
) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {tk: pd.Series(dtype="float64") for tk in tickers}
    status_map: Dict[str, str] = {tk: "not_requested" for tk in tickers}

    if not tickers:
        return series_map, status_map

    max_workers = min(12, max(1, len(tickers)))
    start_clock = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one_shares_series_cached, tk, start_date): tk for tk in tickers}

        for fut in as_completed(futures):
            tk = futures[fut]
            elapsed = time.time() - start_clock
            if elapsed > timeout_sec:
                status_map[tk] = "timed_out"
                continue
            try:
                s = fut.result(timeout=max(0.05, timeout_sec - elapsed))
                series_map[tk] = s
                if s is None or s.empty:
                    status_map[tk] = "missing"
                else:
                    status_map[tk] = "ok"
            except Exception:
                series_map[tk] = pd.Series(dtype="float64")
                status_map[tk] = "error"

    # mark any still untouched items as timed_out if needed
    for tk in tickers:
        if status_map[tk] == "not_requested":
            status_map[tk] = "timed_out"

    return series_map, status_map


# =========================================================
# ANALYTICS
# =========================================================
def compute_daily_share_flows(close: pd.Series, shares: pd.Series, trading_index: pd.DatetimeIndex) -> pd.Series:
    if close is None or shares is None or close.empty or shares.empty or len(trading_index) == 0:
        return pd.Series(dtype="float64")

    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    shares = pd.to_numeric(shares, errors="coerce").dropna().sort_index()

    close.index = pd.to_datetime(close.index, errors="coerce")
    shares.index = pd.to_datetime(shares.index, errors="coerce")

    try:
        close.index = close.index.tz_localize(None)
    except Exception:
        pass

    try:
        shares.index = shares.index.tz_localize(None)
    except Exception:
        pass

    idx = pd.DatetimeIndex(trading_index).sort_values()
    sh_aligned = shares.reindex(idx).ffill()
    close_aligned = close.reindex(idx).ffill()

    delta_shares = sh_aligned.diff().fillna(0.0)
    flows = (delta_shares * close_aligned).replace([np.inf, -np.inf], np.nan).dropna()

    return flows


def compute_turnover_pressure_proxy(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfm = mfm.fillna(0.0)
    typical = (high + low + close) / 3.0

    return float((mfm * typical * vol).sum())


def compute_period_value_from_daily_series(daily: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    if daily is None or daily.empty:
        return np.nan
    s = daily.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()
    out = s.loc[(s.index >= start_ts) & (s.index <= end_ts)]
    if out.empty:
        return np.nan
    return float(out.sum())


def slice_px(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    return out.loc[(out.index >= start_ts) & (out.index <= end_ts)]


def compute_analytics(
    tickers: List[str],
    start_date: date,
    end_date: date,
    period_days: int,
) -> Tuple[Dict[str, dict], Dict[str, pd.DataFrame]]:
    price_map = fetch_prices(tuple(tickers), start_date, end_date)
    shares_map, shares_status = fetch_shares_series_parallel(tuple(tickers), start_date, timeout_sec=12.0)

    analytics: Dict[str, dict] = {}
    cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=period_days)

    for tk in tickers:
        meta = ETF_META.get(tk, {})
        px = price_map.get(tk, pd.DataFrame()).copy()
        px = px.loc[px.index >= cutoff] if not px.empty else px
        shares = shares_map.get(tk, pd.Series(dtype="float64"))

        daily_share_flows = pd.Series(dtype="float64")
        total_value = np.nan
        method = "unavailable"

        price_obs = int(px.shape[0]) if px is not None else 0
        shares_obs = int(shares.dropna().shape[0]) if shares is not None else 0

        if px is not None and not px.empty:
            if shares is not None and not shares.empty:
                daily_share_flows = compute_daily_share_flows(px["Close"], shares, pd.DatetimeIndex(px.index))
                if not daily_share_flows.empty and daily_share_flows.abs().sum() > 0:
                    total_value = float(daily_share_flows.sum())
                    method = "share_flows"
                else:
                    total_value = compute_turnover_pressure_proxy(px)
                    method = "proxy_pressure"
            else:
                total_value = compute_turnover_pressure_proxy(px)
                method = "proxy_pressure"

        trading_days = max(1, price_obs)
        coverage_pct = float(min(100.0, 100.0 * shares_obs / trading_days)) if method == "share_flows" else 0.0
        confidence = classify_confidence(method, coverage_pct, shares_obs)

        analytics[tk] = {
            "Ticker": tk,
            "Label": build_label(tk),
            "Name": meta.get("label", tk),
            "Description": meta.get("desc", ""),
            "Bucket": meta.get("bucket", "Other"),
            "Group": meta.get("group", "Other"),
            "Lookback Value": total_value,
            "Method": method,
            "Confidence": confidence,
            "Coverage %": coverage_pct,
            "Shares Observations": shares_obs,
            "Price Observations": price_obs,
            "Shares Retrieval": shares_status.get(tk, "unknown"),
            "Daily Share Flows": daily_share_flows,
        }

    return analytics, price_map


def make_ticker_df(analytics: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for tk, a in analytics.items():
        rows.append({
            "Ticker": a["Ticker"],
            "Label": a["Label"],
            "Bucket": a["Bucket"],
            "Group": a["Group"],
            "Description": a["Description"],
            "Value": a["Lookback Value"],
            "Method": a["Method"],
            "Confidence": a["Confidence"],
            "Coverage %": a["Coverage %"],
            "Shares Obs": a["Shares Observations"],
            "Price Obs": a["Price Observations"],
            "Shares Retrieval": a["Shares Retrieval"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.sort_values("Value", ascending=False)


def make_bucket_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["Share-Based Value"] = np.where(tmp["Method"] == "share_flows", tmp["Value"], 0.0)
    tmp["Proxy Value"] = np.where(tmp["Method"] != "share_flows", tmp["Value"], 0.0)
    tmp["Share Count"] = (tmp["Method"] == "share_flows").astype(int)
    tmp["Proxy Count"] = (tmp["Method"] != "share_flows").astype(int)
    tmp["Coverage Weighted"] = np.where(tmp["Method"] == "share_flows", tmp["Coverage %"], np.nan)

    out = (
        tmp.groupby("Bucket", dropna=False)
        .agg(
            Total_Value=("Value", "sum"),
            Share_Based_Value=("Share-Based Value", "sum"),
            Proxy_Value=("Proxy Value", "sum"),
            Share_Based_Count=("Share Count", "sum"),
            Proxy_Count=("Proxy Count", "sum"),
            Avg_Coverage=("Coverage Weighted", "mean"),
        )
        .reset_index()
        .sort_values("Total_Value", ascending=False)
    )
    return out


def build_chart_df(
    ticker_df: pd.DataFrame,
    analytics: Dict[str, dict],
    price_map: Dict[str, pd.DataFrame],
    chart_view: str,
    period_mode: str,
    include_proxy_weekly: bool,
    as_of_ts: pd.Timestamp,
) -> pd.DataFrame:
    if ticker_df.empty:
        return pd.DataFrame(columns=["Label", "Value", "Method", "Bucket", "Ticker", "Confidence"])

    if period_mode == "Lookback total":
        out = ticker_df.copy()
        return out[["Ticker", "Label", "Bucket", "Value", "Method", "Confidence"]].sort_values("Value", ascending=False)

    start_ts, end_ts = get_week_bounds(period_mode, as_of_ts)

    rows = []
    for _, row in ticker_df.iterrows():
        tk = row["Ticker"]
        method = row["Method"]
        confidence = row["Confidence"]
        bucket = row["Bucket"]

        if method == "share_flows":
            daily = analytics[tk]["Daily Share Flows"]
            v = compute_period_value_from_daily_series(daily, start_ts, end_ts)
            if pd.notna(v):
                rows.append({
                    "Ticker": tk,
                    "Label": row["Label"],
                    "Bucket": bucket,
                    "Value": v,
                    "Method": "share_flows",
                    "Confidence": confidence,
                })
        elif include_proxy_weekly:
            px = price_map.get(tk, pd.DataFrame())
            px_slice = slice_px(px, start_ts, end_ts)
            if not px_slice.empty:
                v = compute_turnover_pressure_proxy(px_slice)
                if pd.notna(v):
                    rows.append({
                        "Ticker": tk,
                        "Label": row["Label"],
                        "Bucket": bucket,
                        "Value": v,
                        "Method": "proxy_pressure",
                        "Confidence": "Low",
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out.dropna(subset=["Value"]).sort_values("Value", ascending=False)


# =========================================================
# CHARTING
# =========================================================
def make_bar_colors(methods: pd.Series, values: pd.Series) -> List[str]:
    colors = []
    for m, v in zip(methods, values):
        if m == "share_flows":
            colors.append(POS_COLOR if v > 0 else NEG_COLOR if v < 0 else ZERO_COLOR)
        else:
            colors.append(PROXY_POS if v > 0 else PROXY_NEG if v < 0 else ZERO_COLOR)
    return colors


def plot_horizontal_bar(
    df: pd.DataFrame,
    title: str,
    value_col: str = "Value",
    label_col: str = "Label",
    method_col: str = "Method",
):
    if df.empty:
        st.info("No values available for this view.")
        return

    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    methods = df[method_col] if method_col in df.columns else pd.Series(["share_flows"] * len(df))
    labels = df[label_col].astype(str)

    x_min = min(float(vals.min()), 0.0)
    x_max = max(float(vals.max()), 0.0)
    span = (x_max - x_min) if (x_max - x_min) > 0 else 1.0
    pad = 0.08 * span

    n = len(df)
    fig_h = max(5.5, min(22, 0.34 * n + 2.0))
    colors = make_bar_colors(methods, vals)

    fig, ax = plt.subplots(figsize=(15.5, fig_h))
    bars = ax.barh(labels, vals, color=colors, alpha=0.95, height=0.82)

    ax.set_title(title, fontsize=14, color=TEXT_COLOR, pad=12)
    ax.set_xlabel("Estimated Value ($)", color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))
    ax.invert_yaxis()
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(n - 0.5, -0.5)
    ax.margins(y=0.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    text_pad = 0.012 * span
    for bar, raw in zip(bars, vals):
        txt = fmt_compact_cur(raw)
        if raw > 0:
            x_text, ha = raw + text_pad, "left"
        elif raw < 0:
            x_text, ha = raw - text_pad, "right"
        else:
            x_text, ha = 0.0, "center"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            txt,
            ha=ha,
            va="center",
            fontsize=9.5,
            color=TEXT_COLOR,
            clip_on=True,
        )

    legend_handles = [
        Patch(facecolor=POS_COLOR, label="Share-based inflow"),
        Patch(facecolor=NEG_COLOR, label="Share-based outflow"),
        Patch(facecolor=PROXY_POS, label="Proxy pressure positive"),
        Patch(facecolor=PROXY_NEG, label="Proxy pressure negative"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Estimate ETF primary market creations and redemptions where shares outstanding history is available, and clearly separate those estimates from a lower-confidence turnover pressure proxy when shares data are missing.

        What changed
        • Share-based estimates and proxy pressure are now explicitly separated
        • Confidence and coverage scoring are shown for every ticker
        • Universe presets reduce load time and improve workflow
        • Bucket aggregation answers the macro question first
        • Weekly views default to true share-based estimates, with proxy use optional
        """
    )
    st.markdown("---")

    as_of_dt = now_et()
    as_of_date = as_of_dt.date()
    as_of_dt_naive = as_naive_ts(as_of_dt)

    dynamic_lookbacks = LOOKBACK_OPTIONS.copy()
    dynamic_lookbacks["YTD"] = ytd_days(as_of_dt)

    preset_name = st.selectbox(
        "Universe preset",
        options=list(UNIVERSE_PRESETS.keys()),
        index=list(UNIVERSE_PRESETS.keys()).index("Full Cross-Asset"),
    )

    custom_tickers = st.text_input(
        "Add custom tickers (comma-separated)",
        value="",
        help="These are appended to the selected preset.",
    )

    period_label = st.radio(
        "Lookback period",
        options=list(dynamic_lookbacks.keys()),
        index=list(dynamic_lookbacks.keys()).index("1 Month"),
    )
    period_days = int(dynamic_lookbacks[period_label])

    st.markdown("---")
    st.subheader("Display")

    chart_mode = st.radio(
        "Chart layout",
        options=["Ranked All", "By Bucket", "Top/Bottom Only"],
        index=0,
    )

    weekly_mode = st.radio(
        "Time slice",
        options=["Lookback total", "Last complete week", "Week to date"],
        index=0,
        horizontal=False,
    )

    include_proxy_weekly = False
    if weekly_mode != "Lookback total":
        include_proxy_weekly = st.checkbox(
            "Include proxy values in weekly views",
            value=False,
            help="Weekly proxy values are derived from the turnover pressure proxy, not true share-based flow estimates.",
        )

    show_low_confidence = st.checkbox("Show low-confidence rows", value=True)
    min_abs_value_filter = st.number_input(
        "Minimum absolute value to show ($)",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
        format="%.0f",
    )

# =========================================================
# MAIN
# =========================================================
st.title("ETF Net Flows")
st.caption(f"As of {as_of_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} | Lookback: {period_label}")

tickers = get_universe(preset_name, custom_tickers)

if not tickers:
    st.warning("No tickers selected.")
    st.stop()

start_date = compute_start_date(period_days, extra_pad=20)
analytics, price_map = compute_analytics(
    tickers=tickers,
    start_date=start_date,
    end_date=as_of_date,
    period_days=period_days,
)

ticker_df = make_ticker_df(analytics)

if ticker_df.empty:
    st.warning("No data available for the selected universe.")
    st.stop()

if not show_low_confidence:
    ticker_df = ticker_df[ticker_df["Confidence"].isin(["High", "Medium"])].copy()

if min_abs_value_filter > 0:
    ticker_df = ticker_df[ticker_df["Value"].abs() >= min_abs_value_filter].copy()

bucket_df = make_bucket_df(ticker_df)

# =========================================================
# SUMMARY STRIP
# =========================================================
share_based_df = ticker_df[ticker_df["Method"] == "share_flows"].copy()
proxy_df = ticker_df[ticker_df["Method"] != "share_flows"].copy()

net_share_based = share_based_df["Value"].sum() if not share_based_df.empty else np.nan
avg_coverage = share_based_df["Coverage %"].mean() if not share_based_df.empty else np.nan
share_count = int((ticker_df["Method"] == "share_flows").sum())
proxy_count = int((ticker_df["Method"] != "share_flows").sum())

top_bucket = ""
if not bucket_df.empty:
    top_bucket_row = bucket_df.iloc[0]
    bottom_bucket_row = bucket_df.sort_values("Total_Value", ascending=True).iloc[0]
    top_bucket = f"{top_bucket_row['Bucket']} {fmt_compact_cur(top_bucket_row['Total_Value'])}"
    bottom_bucket = f"{bottom_bucket_row['Bucket']} {fmt_compact_cur(bottom_bucket_row['Total_Value'])}"
else:
    bottom_bucket = ""

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net share-based", fmt_compact_cur(net_share_based))
c2.metric("Share-based tickers", f"{share_count}")
c3.metric("Proxy tickers", f"{proxy_count}")
c4.metric("Avg coverage", fmt_pct(avg_coverage))
c5.metric("Strongest bucket", top_bucket if top_bucket else "N/A")

# =========================================================
# CHART VIEW
# =========================================================
st.markdown("### Flow View")

as_of_ts = pd.Timestamp(as_of_date)

chart_df = build_chart_df(
    ticker_df=ticker_df,
    analytics=analytics,
    price_map=price_map,
    chart_view=chart_mode,
    period_mode=weekly_mode,
    include_proxy_weekly=include_proxy_weekly,
    as_of_ts=as_of_ts,
)

if weekly_mode != "Lookback total":
    week_start, week_end = get_week_bounds(weekly_mode, as_of_ts)
    st.caption(f"{weekly_mode}: {week_start.date()} to {week_end.date()}")

if chart_mode == "Ranked All":
    plot_df = chart_df.copy()
    title = f"ETF Net Flows | Ranked All | {weekly_mode}"
    plot_horizontal_bar(plot_df, title)

elif chart_mode == "By Bucket":
    if chart_df.empty:
        st.info("No values available for this view.")
    else:
        grp = (
            chart_df.groupby("Bucket", dropna=False)
            .agg(
                Value=("Value", "sum"),
                Share_Based=("Method", lambda x: int((pd.Series(x) == "share_flows").sum())),
                Proxy=("Method", lambda x: int((pd.Series(x) != "share_flows").sum())),
            )
            .reset_index()
            .sort_values("Value", ascending=False)
        )
        grp["Label"] = grp["Bucket"]
        grp["Method"] = np.where(grp["Proxy"] > 0, "proxy_pressure", "share_flows")
        plot_horizontal_bar(grp[["Label", "Value", "Method"]], f"ETF Net Flows | By Bucket | {weekly_mode}")

        with st.expander("Bucket detail"):
            detail = grp.copy()
            detail["Value"] = detail["Value"].apply(fmt_compact_cur)
            st.dataframe(detail.rename(columns={"Share_Based": "Share-Based Count", "Proxy": "Proxy Count"}), use_container_width=True)

else:
    if chart_df.empty:
        st.info("No values available for this view.")
    else:
        top_n = 10
        top = chart_df.nlargest(top_n, "Value")
        bottom = chart_df.nsmallest(top_n, "Value")
        plot_df = pd.concat([top, bottom], axis=0).drop_duplicates(subset=["Ticker"]).sort_values("Value", ascending=False)
        plot_horizontal_bar(plot_df, f"ETF Net Flows | Top {top_n} Inflows / Outflows | {weekly_mode}")

# =========================================================
# TABLES
# =========================================================
st.markdown("---")
st.markdown("### Bucket Summary")

if bucket_df.empty:
    st.info("No bucket summary available.")
else:
    bucket_show = bucket_df.copy()
    bucket_show["Total Value"] = bucket_show["Total_Value"].apply(fmt_compact_cur)
    bucket_show["Share-Based Value"] = bucket_show["Share_Based_Value"].apply(fmt_compact_cur)
    bucket_show["Proxy Value"] = bucket_show["Proxy_Value"].apply(fmt_compact_cur)
    bucket_show["Avg Coverage"] = bucket_show["Avg_Coverage"].apply(fmt_pct)

    st.dataframe(
        bucket_show[
            ["Bucket", "Total Value", "Share-Based Value", "Proxy Value", "Share_Based_Count", "Proxy_Count", "Avg Coverage"]
        ].rename(
            columns={
                "Share_Based_Count": "Share-Based Count",
                "Proxy_Count": "Proxy Count",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.markdown("### Top Inflows and Outflows")

if chart_df.empty:
    st.info("No rows available for ranking tables.")
else:
    ranked = chart_df.copy()
    ranked["Display Value"] = ranked["Value"].apply(fmt_compact_cur)
    ranked["Method Label"] = ranked["Method"].map(
        {
            "share_flows": "Share-based",
            "proxy_pressure": "Proxy",
        }
    ).fillna(ranked["Method"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top Inflows**")
        inflows = ranked[ranked["Value"] > 0].nlargest(7, "Value")
        if inflows.empty:
            st.write("No positive values.")
        else:
            st.dataframe(
                inflows[["Label", "Bucket", "Display Value", "Method Label", "Confidence"]]
                .rename(columns={"Display Value": "Value", "Method Label": "Method"}),
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.write("**Top Outflows**")
        outflows = ranked[ranked["Value"] < 0].nsmallest(7, "Value")
        if outflows.empty:
            st.write("No negative values.")
        else:
            st.dataframe(
                outflows[["Label", "Bucket", "Display Value", "Method Label", "Confidence"]]
                .rename(columns={"Display Value": "Value", "Method Label": "Method"}),
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.markdown("### Ticker Detail")

detail_df = ticker_df.copy()

if detail_df.empty:
    st.info("No ticker detail available.")
else:
    method_filter = st.multiselect(
        "Filter methods",
        options=["share_flows", "proxy_pressure"],
        default=["share_flows", "proxy_pressure"],
    )
    bucket_filter = st.multiselect(
        "Filter buckets",
        options=sorted(detail_df["Bucket"].dropna().unique().tolist()),
        default=sorted(detail_df["Bucket"].dropna().unique().tolist()),
    )
    conf_filter = st.multiselect(
        "Filter confidence",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    filtered = detail_df[
        detail_df["Method"].isin(method_filter)
        & detail_df["Bucket"].isin(bucket_filter)
        & detail_df["Confidence"].isin(conf_filter)
    ].copy()

    filtered["Value"] = filtered["Value"].apply(fmt_compact_cur)
    filtered["Coverage %"] = filtered["Coverage %"].apply(fmt_pct)
    filtered["Method"] = filtered["Method"].map(
        {"share_flows": "Share-based", "proxy_pressure": "Proxy pressure"}
    ).fillna(filtered["Method"])

    st.dataframe(
        filtered[
            [
                "Label",
                "Bucket",
                "Group",
                "Value",
                "Method",
                "Confidence",
                "Coverage %",
                "Shares Obs",
                "Price Obs",
                "Shares Retrieval",
                "Description",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# FOOTER / NOTES
# =========================================================
st.markdown("---")
st.markdown(
    """
    **Method note:** Share-based values estimate primary market creations/redemptions using the change in shares outstanding multiplied by closing price.  
    **Proxy note:** Proxy pressure uses a CMF-style turnover construction and should be treated as directional pressure, not a true primary flow estimate.  
    **Interpretation:** When share-based coverage is low or absent, precision falls materially. Keep the confidence and coverage columns in the frame before making cross-ticker comparisons.
    """
)

st.caption(f"Last refresh: {as_of_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} | © 2026 AD Fund Management LP")
