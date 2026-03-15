############################################################
# Market Stress Composite
# AD Fund Management LP
#
# Revised / hardened version
# - Safer Yahoo + FRED loaders with retries and shorter timeouts
# - Safe secrets handling
# - Graceful degradation when individual series fail
# - Visible loading status so the app does not look blank
# - More robust calendar and price extraction logic
# - Export, diagnostics, contribution table preserved
############################################################

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Constants ----------------
NY_TZ = "America/New_York"
CACHE_DIR = Path(".cache_market_stress")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRED = {
    "vix": "VIXCLS",
    "yc_10y_3m": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",
    "cp_3m": "DCPF3M",
    "tbill_3m": "DTB3",
    "spx": "SP500",
}

DEFAULT_LOOKBACK = "3y"
PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1
REGIME_HI = 70
REGIME_LO = 30

STALE_LIMITS = {
    "VIX": 1,
    "HY_OAS": 5,
    "HY_LQD": 2,
    "T10Y3M": 3,
    "FUND": 3,
    "DD": 1,
    "RV21": 1,
    "SPY_RSP": 2,
    "SPX": 1,
}

W_VIX = 0.20
W_HY_OAS = 0.15
W_HY_LQD = 0.15
W_CURVE = 0.10
W_FUND = 0.10
W_DD = 0.15
W_RV = 0.10
W_BREADTH = 0.05

WEIGHTS = {
    "VIX_p": W_VIX,
    "HY_OAS_p": W_HY_OAS,
    "HY_LQD_p": W_HY_LQD,
    "CurveInv_p": W_CURVE,
    "Fund_p": W_FUND,
    "DD_p": W_DD,
    "RV21_p": W_RV,
    "Breadth_p": W_BREADTH,
}
WEIGHTS_VEC = np.array(list(WEIGHTS.values()), dtype=float)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

WEIGHTS_TEXT = (
    f"VIX {W_VIX:.2f}, HY_OAS {W_HY_OAS:.2f}, HY_LQD {W_HY_LQD:.2f}, "
    f"Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}, "
    f"RV21 {W_RV:.2f}, Breadth {W_BREADTH:.2f}"
)

REQUEST_TIMEOUT = 12
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 1.35
YF_TIMEOUT = 10
YF_RETRIES = 2

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox(
        "Lookback",
        ["1y", "2y", "3y", "5y", "10y"],
        index=["1y", "2y", "3y", "5y", "10y"].index(DEFAULT_LOOKBACK),
    )
    years = int(lookback[:-1])

    smooth_days = st.slider("Composite smoothing", 1, 10, DEFAULT_SMOOTH, 1)

    st.subheader("Weights (fixed)")
    st.caption(WEIGHTS_TEXT)

    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: A cross-asset stress gauge that blends volatility, credit, funding, curve shape, realized volatility, drawdown pressure, and equity breadth into one normalized composite.

        What to look at
        • Daily, weekly, and monthly regime reads  
        • Which factors are actively driving the score today  
        • Whether the composite is broad-based or being skewed by one or two inputs

        How to use it
        • High readings suggest stress is becoming systemic across assets  
        • Low readings suggest a calmer tape with less cross-asset spillover  
        • Neutral readings usually fit a market digesting shocks without full contagion

        Construction notes
        • Each factor is converted into a rolling percentile over a trailing trading-day window  
        • Missing or stale inputs are masked out and weights are re-normalized across active factors  
        • The diagnostics section shows freshness, activity, and raw inputs for auditability

        Data sources
        • FRED for macro and credit series  
        • Yahoo Finance for ETFs and market proxies
        """
    )

# ---------------- Helpers ----------------
def today_naive() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz=NY_TZ).date())

def as_naive_date(ts_like) -> pd.Timestamp:
    t = pd.to_datetime(ts_like)
    return pd.Timestamp(t.date())

def to_naive_date_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
        return out
    return obj

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def cache_path(name: str) -> Path:
    return CACHE_DIR / name

def save_series_cache(series_id: str, s: pd.Series) -> None:
    try:
        payload = pd.DataFrame(
            {"date": pd.to_datetime(s.index).astype(str), "value": pd.to_numeric(s.values, errors="coerce")}
        )
        payload.to_csv(cache_path(f"{series_id}.csv"), index=False)
    except Exception:
        pass

def load_series_cache(series_id: str) -> pd.Series:
    p = cache_path(f"{series_id}.csv")
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
        if {"date", "value"}.issubset(df.columns):
            s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"]))
            return to_naive_date_index(s.dropna())
    except Exception:
        pass
    return pd.Series(dtype=float)

def robust_request(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=headers)
            r.raise_for_status()
            return r
        except Exception:
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_BACKOFF ** attempt)
    return None

def get_fred_api_key() -> Optional[str]:
    try:
        return st.secrets.get("FRED_API_KEY", None)
    except Exception:
        return None

def parse_fred_text_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    cols = {str(c).lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("observation_date")
    val_col = cols.get(series_id.lower()) or cols.get("value")
    if date_col is None or val_col is None:
        raise ValueError(f"Unexpected FRED CSV format for {series_id}. Columns: {list(df.columns)}")
    vals = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    s = pd.Series(vals.values, index=pd.to_datetime(df[date_col]))
    return to_naive_date_index(s.dropna())

def parse_fred_json_observations(payload: dict) -> pd.Series:
    obs = payload.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(obs)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce").values, index=pd.to_datetime(df["date"]))
    return to_naive_date_index(s.dropna())

@st.cache_data(ttl=1800, show_spinner=False)
def fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start = as_naive_date(start)
    end = as_naive_date(end)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    csv_urls = [
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_str}&coed={end_str}",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]

    for url in csv_urls:
        try:
            r = robust_request(url)
            if r is None or not r.text.strip():
                continue
            s = parse_fred_text_csv(r.text, series_id)
            if not s.empty:
                s = s.loc[(s.index >= start) & (s.index <= end)].sort_index().ffill().dropna()
                save_series_cache(series_id, s)
                return s
        except Exception:
            continue

    try:
        params = {
            "series_id": series_id,
            "file_type": "json",
            "observation_start": start_str,
            "observation_end": end_str,
        }
        api_key = get_fred_api_key()
        if api_key:
            params["api_key"] = api_key
        r = robust_request("https://api.stlouisfed.org/fred/series/observations", params=params)
        if r is not None:
            payload = r.json()
            s = parse_fred_json_observations(payload)
            if not s.empty:
                s = s.loc[(s.index >= start) & (s.index <= end)].sort_index().ffill().dropna()
                save_series_cache(series_id, s)
                return s
    except Exception:
        pass

    cached = load_series_cache(series_id)
    if not cached.empty:
        cached = cached.loc[(cached.index >= start) & (cached.index <= end)].sort_index().ffill().dropna()
        if not cached.empty:
            return cached

    return pd.Series(dtype=float)

def _extract_preferred_price(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = [str(x) for x in data.columns.get_level_values(0)]
        level1 = [str(x) for x in data.columns.get_level_values(1)]

        if ticker in level1:
            sub = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        elif ticker in level0:
            sub = data[ticker].copy()
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(name=ticker)
        else:
            return pd.Series(dtype=float)
    else:
        sub = data.copy()

    if isinstance(sub, pd.Series):
        s = pd.to_numeric(sub, errors="coerce").dropna().rename(ticker)
        s.index = pd.to_datetime(s.index)
        return to_naive_date_index(s)

    sub.columns = [str(c) for c in sub.columns]

    preferred_cols = [c for c in ["Adj Close", "Close", ticker] if c in sub.columns]
    if preferred_cols:
        preferred = preferred_cols[0]
    else:
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return pd.Series(dtype=float)
        preferred = num_cols[0]

    s = pd.to_numeric(sub[preferred], errors="coerce").dropna().rename(ticker)
    s.index = pd.to_datetime(sub.index)
    return to_naive_date_index(s)

def yf_download_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start = as_naive_date(start)
    end = as_naive_date(end)

    for attempt in range(YF_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
                group_by="column",
                timeout=YF_TIMEOUT,
            )
            s = _extract_preferred_price(df, ticker)
            if not s.empty:
                return s
        except Exception:
            if attempt < YF_RETRIES - 1:
                time.sleep(1.0 + attempt)

    return pd.Series(dtype=float)

@st.cache_data(ttl=1800, show_spinner=False)
def yf_history_raw(tickers, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
    series_list = []

    for ticker in tickers_list:
        s = yf_download_one(ticker, start, end)
        if not s.empty:
            series_list.append(s.rename(ticker))

    if not series_list:
        return pd.DataFrame()

    out = pd.concat(series_list, axis=1).sort_index().ffill()
    return to_naive_date_index(out)

def rolling_percentile_trading(values: pd.Series, idx: pd.DatetimeIndex, window_trading_days: int) -> pd.Series:
    s = to_naive_date_index(values).reindex(idx)

    def _last_percentile(x):
        arr = pd.Series(x).dropna()
        if arr.empty:
            return np.nan
        return arr.rank(pct=True).iloc[-1]

    out = s.rolling(
        window_trading_days,
        min_periods=max(40, window_trading_days // 6),
    ).apply(_last_percentile, raw=False)

    return to_naive_date_index(out)

def compute_stale_info(original: pd.Series, idx: pd.DatetimeIndex, max_stale: int) -> Tuple[pd.Series, pd.Series]:
    original = to_naive_date_index(original).reindex(idx)
    pos = np.arange(len(idx))
    has = original.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=idx).ffill().values
    age = pos - last_pos
    age_s = pd.Series(age, index=idx).astype(float)
    mask = pd.Series((~np.isnan(age)) & (age <= max_stale), index=idx)
    return mask, age_s

def regime_label(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= REGIME_HI:
        return "High stress"
    if x <= REGIME_LO:
        return "Low stress"
    return "Neutral"

def regime_bucket(x: float) -> str:
    if pd.isna(x):
        return "na"
    if x >= REGIME_HI:
        return "high"
    if x <= REGIME_LO:
        return "low"
    return "neutral"

def dd_bucket(dd: float) -> str:
    if pd.isna(dd):
        return "na"
    if dd < 5:
        return "shallow"
    if dd < 15:
        return "medium"
    return "deep"

def mean_last(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.tail(n).mean())

def generate_commentary(
    as_of_date: pd.Timestamp,
    daily_val: float,
    weekly_val: float,
    monthly_val: float,
    spx_level: float,
    dd_val: float,
    contrib_table: pd.DataFrame,
    active_weight: float,
    active_factors: int,
) -> str:
    d_reg = regime_bucket(daily_val)
    w_reg = regime_bucket(weekly_val)
    m_reg = regime_bucket(monthly_val)
    dd_b = dd_bucket(dd_val)

    date_str = as_of_date.date().isoformat()
    dd_pct = f"{dd_val:.1f}" if not pd.isna(dd_val) else "N/A"
    spx_str = f"{spx_level:,.1f}" if not pd.isna(spx_level) else "N/A"

    top = contrib_table.sort_values("weighted_contribution", ascending=False).head(3)
    drivers = ", ".join(
        [
            f"{row['factor']} ({row['weighted_contribution']:.1f} pts)"
            for _, row in top.iterrows()
            if pd.notna(row["weighted_contribution"]) and row["weighted_contribution"] > 0
        ]
    )
    if not drivers:
        drivers = "no single factor is dominating the tape"

    daily_map = {
        "low": "The composite is signaling a low-stress daily tape, which lines up with contained vol and benign cross-asset spillover.",
        "neutral": "The daily composite sits in neutral territory, pointing to a market that is absorbing local shocks without broad stress transmission.",
        "high": "The daily composite is in high-stress territory, telling you that volatility, credit, funding, and equity internals are leaning defensive.",
        "na": "The daily read is incomplete because too many inputs are missing, so the tape should be interpreted with caution.",
    }
    weekly_map = {
        "low": "On a weekly basis the regime still looks calm, which suggests the market has not yet rolled into a sustained de-risking phase.",
        "neutral": "The weekly profile is neutral, which fits a market oscillating between fear and relief without a stable stress trend.",
        "high": "The weekly composite is elevated, which means the stress is persisting rather than flashing for a single session.",
        "na": "The weekly regime is not fully clean, so the higher-frequency signal deserves more weight than usual.",
    }
    monthly_map = {
        "low": "At the monthly horizon the broader backdrop still looks benign.",
        "neutral": "At the monthly horizon the signal is neutral, which is usually what you see in a mid-cycle market with unstable narratives but no systemic break.",
        "high": "At the monthly horizon the backdrop is elevated, which is where deleveraging feedback loops can start to matter more.",
        "na": "The monthly state is blurred by missing data, so longer-horizon judgment should stay conditional.",
    }
    dd_map = {
        "shallow": f"SPX is near {spx_str} with a shallow drawdown of about {dd_pct}%, so price damage itself has not yet delivered a full reset.",
        "medium": f"SPX is near {spx_str} with a drawdown of about {dd_pct}%, which is usually where the market starts debating whether weakness is cyclical or systemic.",
        "deep": f"SPX is near {spx_str} with a deep drawdown of about {dd_pct}%, the kind of damage that can create reflexive turning points if policy or data improve.",
        "na": "The drawdown context is unclear, so the composite should be read primarily as a cross-asset stress barometer.",
    }

    return (
        f"As of {date_str}, {daily_map[d_reg]} {weekly_map[w_reg]} {monthly_map[m_reg]} "
        f"{dd_map[dd_b]} The main drivers right now are {drivers}. "
        f"Composite confidence is moderate to high with {active_factors} active factors and {active_weight:.0%} active weight."
    )

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return to_naive_date_index(out.sort_index().ffill())

def build_trading_calendar(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DatetimeIndex, str]:
    candidates = ["SPY", "^GSPC", "QQQ", "IWM"]
    for ticker in candidates:
        px = yf_history_raw(ticker, start, end)
        if not px.empty and ticker in px.columns and px[ticker].dropna().shape[0] > 100:
            return pd.DatetimeIndex(px.index.unique()).sort_values(), ticker

    bidx = pd.bdate_range(start, end)
    return pd.DatetimeIndex(bidx), "Business-day fallback"

def to_trade_idx(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(index=idx, dtype=float)
    s = to_naive_date_index(s)
    return s.reindex(idx).ffill()

# ---------------- Load history ----------------
status = st.status("Loading market and macro inputs...", expanded=False)

today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

status.write("Building trading calendar...")
trade_idx_all, calendar_source = build_trading_calendar(start_all, today)

if len(trade_idx_all) == 0:
    status.update(label="Failed to build trading calendar", state="error")
    st.error("Unable to construct any usable calendar.")
    st.stop()

if calendar_source == "Business-day fallback":
    st.warning("Market session calendar could not be loaded from Yahoo. Using business-day fallback.")
else:
    st.caption(f"Calendar source: {calendar_source}")

status.write("Loading FRED series...")
vix_f = fred_series(FRED["vix"], start_all, today)
hy_f = fred_series(FRED["hy_oas"], start_all, today)
yc_f = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f = fred_series(FRED["tbill_3m"], start_all, today)
spx_f = fred_series(FRED["spx"], start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f) if not cp3m_f.empty and not tb3m_f.empty else pd.Series(dtype=float)

status.write("Loading Yahoo market proxies...")
px_eq_credit = yf_history_raw(["HYG", "LQD", "SPY", "RSP"], start_all, today)

hy_lqd_s = pd.Series(dtype=float)
spy_rsp_s = pd.Series(dtype=float)
spy_price_s = pd.Series(dtype=float)

if not px_eq_credit.empty:
    cols = set(px_eq_credit.columns)

    if "SPY" in cols:
        spy_price_s = px_eq_credit["SPY"].dropna()

    if "HYG" in cols and "LQD" in cols:
        ratio = (px_eq_credit["HYG"] / px_eq_credit["LQD"]).replace([np.inf, -np.inf], np.nan)
        hy_lqd_s = -np.log(ratio.replace(0, np.nan)).dropna()

    if "SPY" in cols and "RSP" in cols:
        spy_rsp_s = (px_eq_credit["SPY"] / px_eq_credit["RSP"]).replace([np.inf, -np.inf], np.nan).dropna()

if all(s.dropna().empty for s in [vix_f, hy_f, yc_f, fund_f, spx_f]) and spy_price_s.empty:
    status.update(label="No usable market or macro data could be loaded", state="error")
    st.error("No usable market or macro data could be loaded.")
    st.stop()

panel = pd.DataFrame(index=trade_idx_all)

panel["SPY"] = to_trade_idx(spy_price_s, trade_idx_all) if not spy_price_s.empty else np.nan

if not spx_f.empty:
    panel["SPX"] = to_trade_idx(spx_f, trade_idx_all)
elif not spy_price_s.empty:
    base_spy = safe_float(spy_price_s.iloc[0])
    panel["SPX"] = to_trade_idx((spy_price_s / base_spy) * 100.0, trade_idx_all) if pd.notna(base_spy) and base_spy != 0 else np.nan
else:
    panel["SPX"] = np.nan

panel["VIX"] = to_trade_idx(vix_f, trade_idx_all) if not vix_f.empty else np.nan
panel["HY_OAS"] = to_trade_idx(hy_f, trade_idx_all) if not hy_f.empty else np.nan
panel["T10Y3M"] = to_trade_idx(yc_f, trade_idx_all) if not yc_f.empty else np.nan
panel["FUND"] = to_trade_idx(fund_f, trade_idx_all) if not fund_f.empty else np.nan
panel["HY_LQD"] = to_trade_idx(hy_lqd_s, trade_idx_all) if not hy_lqd_s.empty else np.nan
panel["SPY_RSP"] = to_trade_idx(spy_rsp_s, trade_idx_all) if not spy_rsp_s.empty else np.nan

panel = normalize_frame(panel)

# ---------------- Derived series ----------------
panel["CurveInv"] = -panel["T10Y3M"]
spx_ret = panel["SPX"].pct_change()
panel["RV21"] = np.sqrt(252.0) * spx_ret.rolling(21, min_periods=10).std()

spx_roll_max = panel["SPX"].cummax()
panel["DD_stress"] = -(100.0 * (panel["SPX"] / spx_roll_max - 1.0).clip(upper=0))

# ---------------- Freshness masks ----------------
mask_map = {}
age_map = {}

series_lookup = {
    "VIX": vix_f,
    "HY_OAS": hy_f,
    "HY_LQD": hy_lqd_s,
    "T10Y3M": yc_f,
    "FUND": fund_f,
    "SPX": spx_f if not spx_f.empty else panel["SPX"].dropna(),
    "SPY_RSP": spy_rsp_s,
}

for name, raw_s in series_lookup.items():
    max_stale = STALE_LIMITS.get(name, 2)
    m, age = compute_stale_info(raw_s, panel.index, max_stale)
    mask_map[name] = m
    age_map[name] = age

mask_map["DD"] = mask_map["SPX"].copy()
age_map["DD"] = age_map["SPX"].copy()
mask_map["RV21"] = panel["RV21"].notna()
age_map["RV21"] = pd.Series(np.where(panel["RV21"].notna(), 0, np.nan), index=panel.index)

# ---------------- Percentiles ----------------
window_td = int(PCTL_WINDOW_YEARS * 252)

scores_all = pd.DataFrame(index=panel.index)
scores_all["VIX_p"] = rolling_percentile_trading(panel["VIX"], panel.index, window_td)
scores_all["HY_OAS_p"] = rolling_percentile_trading(panel["HY_OAS"], panel.index, window_td)
scores_all["HY_LQD_p"] = rolling_percentile_trading(panel["HY_LQD"], panel.index, window_td)
scores_all["CurveInv_p"] = rolling_percentile_trading(panel["CurveInv"], panel.index, window_td)
scores_all["Fund_p"] = rolling_percentile_trading(panel["FUND"], panel.index, window_td)
scores_all["DD_p"] = rolling_percentile_trading(panel["DD_stress"], panel.index, window_td)
scores_all["RV21_p"] = rolling_percentile_trading(panel["RV21"], panel.index, window_td)
scores_all["Breadth_p"] = rolling_percentile_trading(panel["SPY_RSP"], panel.index, window_td)
scores_all = normalize_frame(scores_all)

# ---------------- Lookback slice ----------------
start_lb = as_naive_date(today - pd.DateOffset(years=years))
panel_lb = panel.loc[panel.index >= start_lb].copy()
scores = scores_all.loc[scores_all.index >= start_lb].copy()

if smooth_days > 1:
    scores = scores.rolling(smooth_days, min_periods=1).mean()

# ---------------- Masks aligned ----------------
masks = pd.DataFrame(index=scores.index)
masks["VIX_m"] = mask_map["VIX"].reindex(scores.index).fillna(False).astype(float)
masks["HY_OAS_m"] = mask_map["HY_OAS"].reindex(scores.index).fillna(False).astype(float)
masks["HY_LQD_m"] = mask_map["HY_LQD"].reindex(scores.index).fillna(False).astype(float)
masks["Curve_m"] = mask_map["T10Y3M"].reindex(scores.index).fillna(False).astype(float)
masks["Fund_m"] = mask_map["FUND"].reindex(scores.index).fillna(False).astype(float)
masks["DD_m"] = mask_map["DD"].reindex(scores.index).fillna(False).astype(float)
masks["RV21_m"] = mask_map["RV21"].reindex(scores.index).fillna(False).astype(float)
masks["Breadth_m"] = mask_map["SPY_RSP"].reindex(scores.index).fillna(False).astype(float)

score_cols = ["VIX_p", "HY_OAS_p", "HY_LQD_p", "CurveInv_p", "Fund_p", "DD_p", "RV21_p", "Breadth_p"]
mask_cols = ["VIX_m", "HY_OAS_m", "HY_LQD_m", "Curve_m", "Fund_m", "DD_m", "RV21_m", "Breadth_m"]

X = scores[score_cols].values
M = masks[mask_cols].values
W = WEIGHTS_VEC.reshape(1, -1)

X_masked = np.nan_to_num(X, nan=0.0) * M
weight_mask = W * M
active_w = weight_mask.sum(axis=1)
weighted_contrib = np.where(active_w[:, None] > 0, 100.0 * (X_masked * W) / active_w[:, None], np.nan)
comp = np.where(active_w > 0, weighted_contrib.sum(axis=1), np.nan)

comp_s = pd.Series(comp, index=scores.index, name="Composite").dropna()
active_weight_s = pd.Series(active_w, index=scores.index, name="active_weight").reindex(comp_s.index)
active_factors_s = pd.Series(M.sum(axis=1), index=scores.index, name="active_factors").reindex(comp_s.index)

if comp_s.empty:
    status.update(label="Composite has no valid points for the selected lookback", state="error")
    st.error("Composite has no valid points for the selected lookback.")
    st.stop()

status.update(label="Data loaded", state="complete")

# ---------------- Latest stats ----------------
latest_idx = comp_s.index[-1]
latest_val = float(comp_s.iloc[-1])
weekly_val = mean_last(comp_s, 5)
monthly_val = mean_last(comp_s, 21)

plot_idx = comp_s.index.copy()
panel_plot = panel_lb.reindex(plot_idx).ffill()

spx = panel_plot["SPX"].dropna()
base = safe_float(spx.iloc[0]) if len(spx) else np.nan
spx_rebased = (panel_plot["SPX"] / base) * 100.0 if pd.notna(base) and base != 0 else pd.Series(index=plot_idx, dtype=float)
dd_series = -100.0 * (panel_plot["SPX"] / panel_plot["SPX"].cummax() - 1.0)

latest_spx_level = safe_float(panel_plot["SPX"].iloc[-1]) if len(panel_plot) else np.nan
latest_dd_val = safe_float(dd_series.iloc[-1]) if len(dd_series) else np.nan
latest_active_weight = safe_float(active_weight_s.iloc[-1]) if not active_weight_s.empty else np.nan
latest_active_factors = int(active_factors_s.iloc[-1]) if not active_factors_s.empty and pd.notna(active_factors_s.iloc[-1]) else 0

latest_contrib = pd.DataFrame(
    {
        "factor": ["VIX", "HY_OAS", "HY_LQD", "Curve", "Funding", "Drawdown", "RV21", "Breadth"],
        "percentile": scores.loc[latest_idx, score_cols].values * 100.0,
        "mask": masks.loc[latest_idx, mask_cols].values.astype(bool),
        "base_weight": WEIGHTS_VEC,
        "effective_weight": np.where(
            latest_active_weight > 0,
            (WEIGHTS_VEC * masks.loc[latest_idx, mask_cols].values) / latest_active_weight,
            np.nan,
        ),
        "weighted_contribution": weighted_contrib[scores.index.get_loc(latest_idx), :],
    }
).sort_values("weighted_contribution", ascending=False)

# ---------------- Commentary ----------------
commentary_text = generate_commentary(
    as_of_date=latest_idx,
    daily_val=latest_val,
    weekly_val=weekly_val,
    monthly_val=monthly_val,
    spx_level=latest_spx_level,
    dd_val=latest_dd_val,
    contrib_table=latest_contrib,
    active_weight=latest_active_weight,
    active_factors=latest_active_factors,
)
st.info(commentary_text)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Daily regime", regime_label(latest_val), f"{latest_val:.0f}")
c2.metric("Weekly regime", regime_label(weekly_val), f"{weekly_val:.0f}" if not pd.isna(weekly_val) else "N/A")
c3.metric("Monthly regime", regime_label(monthly_val), f"{monthly_val:.0f}" if not pd.isna(monthly_val) else "N/A")
c4.metric("Active factors", f"{latest_active_factors}/8")
c5.metric("Active weight", f"{latest_active_weight:.0%}" if pd.notna(latest_active_weight) else "N/A")

# ---------------- Chart ----------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.62, 0.38],
    specs=[[{"secondary_y": True}], [{}]],
    subplot_titles=("SPX and Drawdown", "Market Stress Composite"),
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=spx_rebased.reindex(plot_idx),
        name="SPX (rebased=100)",
        line=dict(width=2, color="#1f77b4"),
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=dd_series.reindex(plot_idx),
        name="Drawdown (%)",
        line=dict(width=1.5, color="#d62728"),
    ),
    row=1,
    col=1,
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=comp_s.reindex(plot_idx),
        name="Composite",
        line=dict(width=2.3, color="#111111"),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=comp_s.reindex(plot_idx).rolling(21, min_periods=1).mean(),
        name="Composite 21D MA",
        line=dict(width=1.4, color="#7f7f7f", dash="dot"),
    ),
    row=2,
    col=1,
)

fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=2, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=2, col=1)

finite_dd = dd_series.replace([np.inf, -np.inf], np.nan).dropna()
dd_min = float(finite_dd.min()) if not finite_dd.empty else -5.0
dd_min = min(-5.0, dd_min * 1.1)

fig.update_yaxes(title_text="Rebased", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Drawdown %", row=1, col=1, secondary_y=True, range=[dd_min, 0])
fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 100])
fig.update_xaxes(title_text="Date", row=2, col=1, tickformat="%b-%d-%y")

fig.update_layout(
    template="plotly_white",
    height=780,
    margin=dict(l=50, r=30, t=60, b=50),
    legend=dict(orientation="h", x=0, y=1.07, xanchor="left"),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Contribution table ----------------
st.subheader("Current factor contributions")
show_contrib = latest_contrib.copy()
show_contrib["percentile"] = show_contrib["percentile"].round(1)
show_contrib["base_weight"] = (show_contrib["base_weight"] * 100).round(1)
show_contrib["effective_weight"] = (show_contrib["effective_weight"] * 100).round(1)
show_contrib["weighted_contribution"] = show_contrib["weighted_contribution"].round(1)
show_contrib.columns = ["Factor", "Percentile", "Active", "Base Weight %", "Effective Weight %", "Contribution (pts)"]
st.dataframe(show_contrib, use_container_width=True, hide_index=True)

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    latest = plot_idx[-1]

    diag = pd.DataFrame(
        {
            "factor": ["VIX", "HY_OAS", "HY_LQD", "Curve", "Funding", "Drawdown", "RV21", "Breadth"],
            "raw_value": [
                panel_plot["VIX"].loc[latest] if "VIX" in panel_plot.columns else np.nan,
                panel_plot["HY_OAS"].loc[latest] if "HY_OAS" in panel_plot.columns else np.nan,
                panel_plot["HY_LQD"].loc[latest] if "HY_LQD" in panel_plot.columns else np.nan,
                panel_plot["CurveInv"].loc[latest] if "CurveInv" in panel_plot.columns else np.nan,
                panel_plot["FUND"].loc[latest] if "FUND" in panel_plot.columns else np.nan,
                panel_plot["DD_stress"].loc[latest] if "DD_stress" in panel_plot.columns else np.nan,
                panel_plot["RV21"].loc[latest] if "RV21" in panel_plot.columns else np.nan,
                panel_plot["SPY_RSP"].loc[latest] if "SPY_RSP" in panel_plot.columns else np.nan,
            ],
            "percentile": [
                safe_float(scores.loc[latest, "VIX_p"] * 100.0) if "VIX_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "HY_OAS_p"] * 100.0) if "HY_OAS_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "HY_LQD_p"] * 100.0) if "HY_LQD_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "CurveInv_p"] * 100.0) if "CurveInv_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Fund_p"] * 100.0) if "Fund_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "DD_p"] * 100.0) if "DD_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "RV21_p"] * 100.0) if "RV21_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Breadth_p"] * 100.0) if "Breadth_p" in scores.columns else np.nan,
            ],
            "active": [
                bool(masks.loc[latest, "VIX_m"]) if "VIX_m" in masks.columns else False,
                bool(masks.loc[latest, "HY_OAS_m"]) if "HY_OAS_m" in masks.columns else False,
                bool(masks.loc[latest, "HY_LQD_m"]) if "HY_LQD_m" in masks.columns else False,
                bool(masks.loc[latest, "Curve_m"]) if "Curve_m" in masks.columns else False,
                bool(masks.loc[latest, "Fund_m"]) if "Fund_m" in masks.columns else False,
                bool(masks.loc[latest, "DD_m"]) if "DD_m" in masks.columns else False,
                bool(masks.loc[latest, "RV21_m"]) if "RV21_m" in masks.columns else False,
                bool(masks.loc[latest, "Breadth_m"]) if "Breadth_m" in masks.columns else False,
            ],
            "days_since_update": [
                age_map["VIX"].reindex(plot_idx).loc[latest] if "VIX" in age_map else np.nan,
                age_map["HY_OAS"].reindex(plot_idx).loc[latest] if "HY_OAS" in age_map else np.nan,
                age_map["HY_LQD"].reindex(plot_idx).loc[latest] if "HY_LQD" in age_map else np.nan,
                age_map["T10Y3M"].reindex(plot_idx).loc[latest] if "T10Y3M" in age_map else np.nan,
                age_map["FUND"].reindex(plot_idx).loc[latest] if "FUND" in age_map else np.nan,
                age_map["DD"].reindex(plot_idx).loc[latest] if "DD" in age_map else np.nan,
                age_map["RV21"].reindex(plot_idx).loc[latest] if "RV21" in age_map else np.nan,
                age_map["SPY_RSP"].reindex(plot_idx).loc[latest] if "SPY_RSP" in age_map else np.nan,
            ],
            "stale_limit": [
                STALE_LIMITS["VIX"],
                STALE_LIMITS["HY_OAS"],
                STALE_LIMITS["HY_LQD"],
                STALE_LIMITS["T10Y3M"],
                STALE_LIMITS["FUND"],
                STALE_LIMITS["DD"],
                STALE_LIMITS["RV21"],
                STALE_LIMITS["SPY_RSP"],
            ],
        }
    )
    diag["percentile"] = pd.to_numeric(diag["percentile"], errors="coerce").round(1)
    diag["days_since_update"] = pd.to_numeric(diag["days_since_update"], errors="coerce").round(0)
    st.dataframe(diag, use_container_width=True, hide_index=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    export_idx = plot_idx

    export_panel_cols = [
        c for c in ["SPX", "VIX", "HY_OAS", "HY_LQD", "T10Y3M", "CurveInv", "FUND", "DD_stress", "RV21", "SPY_RSP"]
        if c in panel.columns
    ]
    export_panel = panel.reindex(export_idx)[export_panel_cols].copy()

    export_scores = (scores.reindex(export_idx) * 100.0).rename(
        columns={
            "VIX_p": "VIX_pct",
            "HY_OAS_p": "HY_OAS_pct",
            "HY_LQD_p": "HY_LQD_pct",
            "CurveInv_p": "CurveInv_pct",
            "Fund_p": "Fund_pct",
            "DD_p": "DD_pct",
            "RV21_p": "RV21_pct",
            "Breadth_p": "Breadth_pct",
        }
    )

    export_masks = masks.reindex(export_idx).copy()
    export_contrib = pd.DataFrame(
        weighted_contrib,
        index=scores.index,
        columns=[
            "VIX_contrib",
            "HY_OAS_contrib",
            "HY_LQD_contrib",
            "Curve_contrib",
            "Fund_contrib",
            "DD_contrib",
            "RV21_contrib",
            "Breadth_contrib",
        ],
    ).reindex(export_idx)

    export_meta = pd.concat(
        [
            active_weight_s.reindex(export_idx),
            active_factors_s.reindex(export_idx),
            comp_s.reindex(export_idx),
        ],
        axis=1,
    )

    out = pd.concat([export_panel, export_scores, export_masks, export_contrib, export_meta], axis=1)
    out.index.name = "Date"

    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="market_stress_composite.csv",
        mime="text/csv",
    )

st.caption("© 2026 AD Fund Management LP")
