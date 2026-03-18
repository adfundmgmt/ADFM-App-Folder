############################################################
# Cross-Asset RVOL Stress Composite
# AD Fund Management LP
#
# Pure market-data version
# - Removes all FRED dependencies
# - Uses Yahoo Finance only
# - Builds class-level RVOL stress across equities, credit,
#   commodities, FX, and rates
# - Adds cross-asset RVOL breadth and dispersion
# - Keeps diagnostics, contribution table, and CSV export
############################################################

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Constants ----------------
NY_TZ = "America/New_York"
CACHE_DIR = Path(".cache_cross_asset_rvol")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOOKBACK = "3y"
PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1
RV_WINDOW = 21
REGIME_HI = 70
REGIME_LO = 30

REQUEST_TIMEOUT = 18
REQUEST_RETRIES = 5
REQUEST_BACKOFF = 1.6
YF_TIMEOUT = 15
YF_RETRIES = 3

# ---------------- Universe ----------------
ASSET_CLASS_MAP: Dict[str, List[str]] = {
    "Equities": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Credit": ["HYG", "JNK", "LQD", "EMB"],
    "Commodities": ["GLD", "SLV", "USO", "DBA", "CPER"],
    "FX": ["UUP", "FXE", "FXY", "FXB", "CEW"],
    "Rates": ["TLT", "IEF", "SHY", "TIP"],
}

ALL_TICKERS = [t for group in ASSET_CLASS_MAP.values() for t in group]

# ---------------- Weights ----------------
W_EQ = 0.25
W_CREDIT = 0.20
W_CMDTY = 0.15
W_FX = 0.15
W_RATES = 0.15
W_BREADTH = 0.07
W_DISPERSION = 0.03

WEIGHTS = {
    "Equities_p": W_EQ,
    "Credit_p": W_CREDIT,
    "Commodities_p": W_CMDTY,
    "FX_p": W_FX,
    "Rates_p": W_RATES,
    "Breadth_p": W_BREADTH,
    "Dispersion_p": W_DISPERSION,
}
WEIGHTS_VEC = np.array(list(WEIGHTS.values()), dtype=float)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

WEIGHTS_TEXT = (
    f"Equities {W_EQ:.2f}, Credit {W_CREDIT:.2f}, Commodities {W_CMDTY:.2f}, "
    f"FX {W_FX:.2f}, Rates {W_RATES:.2f}, Breadth {W_BREADTH:.2f}, Dispersion {W_DISPERSION:.2f}"
)

# stale limit in trading days
STALE_LIMITS = {
    "Equities": 2,
    "Credit": 2,
    "Commodities": 2,
    "FX": 2,
    "Rates": 2,
    "Breadth": 2,
    "Dispersion": 2,
    "SPX": 2,
    "DD": 2,
}

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
    rv_window = st.slider("RVOL window", 10, 63, RV_WINDOW, 1)
    stress_cutoff = st.slider("Breadth stress threshold", 60, 90, 70, 1)

    st.subheader("Weights (fixed)")
    st.caption(WEIGHTS_TEXT)

    if st.button("Refresh market data now", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared cached data. The next run will force fresh market requests.")

    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: A pure market-data cross-asset stress gauge built from realized volatility across equities, credit, commodities, FX, and rates.

        What to look at
        • Daily, weekly, and monthly regime reads  
        • Which asset class is contributing the most to the stress score  
        • Whether RVOL is concentrated or broadening across markets

        How to use it
        • High readings suggest realized volatility is becoming systemic across assets  
        • Low readings suggest a calmer tape with tighter realized ranges  
        • Breadth tells you whether the stress is localized or spreading

        Construction notes
        • Each instrument is converted into 21D annualized realized volatility  
        • Each RVOL series is converted into a trailing percentile over a rolling trading-day window  
        • Asset-class factors are averages of their underlying RVOL percentiles  
        • Breadth is the share of instruments with RVOL percentile above the threshold  
        • Dispersion is the cross-sectional standard deviation of asset-class percentiles

        Data sources
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

@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRIES,
        connect=REQUEST_RETRIES,
        read=REQUEST_RETRIES,
        backoff_factor=REQUEST_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )
    return session

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
        return to_naive_date_index(s.sort_index())

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
    return to_naive_date_index(s.sort_index())

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
                s = s[~s.index.duplicated(keep="last")]
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
    out = to_naive_date_index(out)
    out = out.loc[~out.index.duplicated(keep="last")]
    return out

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

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out = to_naive_date_index(out.sort_index().ffill())
    out = out.loc[~out.index.duplicated(keep="last")]
    return out

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
        drivers = "no single sleeve is dominating the tape"

    daily_map = {
        "low": "The composite is signaling a low-stress daily tape with realized volatility contained across the major sleeves.",
        "neutral": "The daily composite sits in neutral territory, which fits a market digesting moves without broad cross-asset destabilization.",
        "high": "The daily composite is in high-stress territory, meaning realized volatility is elevated across the tape and the move is becoming more systemic.",
        "na": "The daily read is incomplete because too many market inputs are missing.",
    }
    weekly_map = {
        "low": "On a weekly basis the regime still looks calm.",
        "neutral": "The weekly profile is neutral and unstable rather than cleanly risk-on or risk-off.",
        "high": "The weekly composite is elevated, which means the stress is persisting rather than flashing for one session.",
        "na": "The weekly regime is not fully clean, so the higher-frequency signal deserves more weight.",
    }
    monthly_map = {
        "low": "At the monthly horizon the broader realized-vol backdrop still looks contained.",
        "neutral": "At the monthly horizon the signal is neutral, which usually lines up with a choppier but still tradable market.",
        "high": "At the monthly horizon the backdrop is elevated, which is where volatility regimes can start feeding on themselves.",
        "na": "The monthly state is blurred by missing data.",
    }
    dd_map = {
        "shallow": f"SPX is near {spx_str} with a shallow drawdown of about {dd_pct}%, so the price tape itself has not yet delivered a deep reset.",
        "medium": f"SPX is near {spx_str} with a drawdown of about {dd_pct}%, which is where investors start asking whether the stress is cyclical or something larger.",
        "deep": f"SPX is near {spx_str} with a deep drawdown of about {dd_pct}%, which is where reflexive turns can emerge if realized vol begins to compress.",
        "na": "The drawdown context is unclear, so the composite should be read mainly as a realized-vol barometer.",
    }

    return (
        f"As of {date_str}, {daily_map[d_reg]} {weekly_map[w_reg]} {monthly_map[m_reg]} "
        f"{dd_map[dd_b]} The main drivers right now are {drivers}. "
        f"Composite confidence is moderate to high with {active_factors} active factors and {active_weight:.0%} active weight."
    )

# ---------------- Load history ----------------
status = st.status("Loading cross-asset market inputs...", expanded=False)

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

status.write("Loading Yahoo market proxies...")
px = yf_history_raw(sorted(set(ALL_TICKERS + ["SPY", "^GSPC"])), start_all, today)

if px.empty:
    status.update(label="No usable market data could be loaded", state="error")
    st.error("No usable market data could be loaded.")
    st.stop()

panel = pd.DataFrame(index=trade_idx_all)

for ticker in sorted(set(ALL_TICKERS)):
    panel[ticker] = to_trade_idx(px[ticker].dropna(), trade_idx_all) if ticker in px.columns else np.nan

if "^GSPC" in px.columns and not px["^GSPC"].dropna().empty:
    panel["SPX"] = to_trade_idx(px["^GSPC"].dropna(), trade_idx_all)
elif "SPY" in px.columns and not px["SPY"].dropna().empty:
    base_spy = safe_float(px["SPY"].dropna().iloc[0])
    panel["SPX"] = to_trade_idx((px["SPY"].dropna() / base_spy) * 100.0, trade_idx_all) if pd.notna(base_spy) and base_spy != 0 else np.nan
else:
    panel["SPX"] = np.nan

panel = normalize_frame(panel)

# ---------------- RVOL construction ----------------
ret = panel[sorted(set(ALL_TICKERS))].pct_change()
rvol = np.sqrt(252.0) * ret.rolling(rv_window, min_periods=max(5, rv_window // 2)).std()
rvol = normalize_frame(rvol)

window_td = int(PCTL_WINDOW_YEARS * 252)
rvol_pct = pd.DataFrame(index=panel.index)

status.write("Computing rolling RVOL percentiles...")
for ticker in rvol.columns:
    rvol_pct[f"{ticker}_p"] = rolling_percentile_trading(rvol[ticker], panel.index, window_td)

rvol_pct = normalize_frame(rvol_pct)

# ---------------- Asset-class factors ----------------
factor_panel = pd.DataFrame(index=panel.index)

for asset_class, tickers in ASSET_CLASS_MAP.items():
    cols = [f"{t}_p" for t in tickers if f"{t}_p" in rvol_pct.columns]
    if cols:
        factor_panel[f"{asset_class}_p"] = rvol_pct[cols].mean(axis=1)
    else:
        factor_panel[f"{asset_class}_p"] = np.nan

# breadth: share of universe above stress threshold
all_pct_cols = [c for c in rvol_pct.columns if c.endswith("_p")]
if all_pct_cols:
    breadth_raw = (rvol_pct[all_pct_cols] * 100.0 >= stress_cutoff).mean(axis=1)
    factor_panel["Breadth_p"] = rolling_percentile_trading(breadth_raw, panel.index, window_td)
else:
    factor_panel["Breadth_p"] = np.nan

# dispersion: cross-asset-class dispersion of class percentiles
class_cols = [f"{k}_p" for k in ASSET_CLASS_MAP.keys()]
disp_raw = factor_panel[class_cols].std(axis=1)
factor_panel["Dispersion_p"] = rolling_percentile_trading(disp_raw, panel.index, window_td)

factor_panel = normalize_frame(factor_panel)

# ---------------- Derived series ----------------
spx_ret = panel["SPX"].pct_change()
panel["DD_stress"] = -(100.0 * (panel["SPX"] / panel["SPX"].cummax() - 1.0).clip(upper=0))

# ---------------- Lookback slice ----------------
start_lb = as_naive_date(today - pd.DateOffset(years=years))
panel_lb = panel.loc[panel.index >= start_lb].copy()
scores = factor_panel.loc[factor_panel.index >= start_lb].copy()

if smooth_days > 1:
    scores = scores.rolling(smooth_days, min_periods=1).mean()

# ---------------- Freshness masks ----------------
mask_map = {}
age_map = {}

for asset_class, tickers in ASSET_CLASS_MAP.items():
    raw_series = rvol[[t for t in tickers if t in rvol.columns]].mean(axis=1) if any(t in rvol.columns for t in tickers) else pd.Series(dtype=float)
    m, age = compute_stale_info(raw_series, panel.index, STALE_LIMITS.get(asset_class, 2))
    mask_map[asset_class] = m
    age_map[asset_class] = age

breadth_series = breadth_raw if "breadth_raw" in locals() else pd.Series(dtype=float)
disp_series = disp_raw if "disp_raw" in locals() else pd.Series(dtype=float)

mask_map["Breadth"], age_map["Breadth"] = compute_stale_info(breadth_series, panel.index, STALE_LIMITS["Breadth"])
mask_map["Dispersion"], age_map["Dispersion"] = compute_stale_info(disp_series, panel.index, STALE_LIMITS["Dispersion"])
mask_map["SPX"], age_map["SPX"] = compute_stale_info(panel["SPX"], panel.index, STALE_LIMITS["SPX"])
mask_map["DD"] = mask_map["SPX"].copy()
age_map["DD"] = age_map["SPX"].copy()

# ---------------- Masks aligned ----------------
masks = pd.DataFrame(index=scores.index)
masks["Equities_m"] = mask_map["Equities"].reindex(scores.index).fillna(False).astype(float)
masks["Credit_m"] = mask_map["Credit"].reindex(scores.index).fillna(False).astype(float)
masks["Commodities_m"] = mask_map["Commodities"].reindex(scores.index).fillna(False).astype(float)
masks["FX_m"] = mask_map["FX"].reindex(scores.index).fillna(False).astype(float)
masks["Rates_m"] = mask_map["Rates"].reindex(scores.index).fillna(False).astype(float)
masks["Breadth_m"] = mask_map["Breadth"].reindex(scores.index).fillna(False).astype(float)
masks["Dispersion_m"] = mask_map["Dispersion"].reindex(scores.index).fillna(False).astype(float)

score_cols = ["Equities_p", "Credit_p", "Commodities_p", "FX_p", "Rates_p", "Breadth_p", "Dispersion_p"]
mask_cols = ["Equities_m", "Credit_m", "Commodities_m", "FX_m", "Rates_m", "Breadth_m", "Dispersion_m"]

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

dd_plot = (100.0 * (panel_plot["SPX"] / panel_plot["SPX"].cummax() - 1.0)).clip(upper=0)

latest_spx_level = safe_float(panel_plot["SPX"].iloc[-1]) if len(panel_plot) else np.nan
latest_dd_val = safe_float(panel_plot["DD_stress"].iloc[-1]) if len(panel_plot) else np.nan
latest_active_weight = safe_float(active_weight_s.iloc[-1]) if not active_weight_s.empty else np.nan
latest_active_factors = int(active_factors_s.iloc[-1]) if not active_factors_s.empty and pd.notna(active_factors_s.iloc[-1]) else 0

latest_contrib = pd.DataFrame(
    {
        "factor": ["Equities", "Credit", "Commodities", "FX", "Rates", "Breadth", "Dispersion"],
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
c4.metric("Active factors", f"{latest_active_factors}/7")
c5.metric("Active weight", f"{latest_active_weight:.0%}" if pd.notna(latest_active_weight) else "N/A")

# ---------------- Chart ----------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.62, 0.38],
    specs=[[{"secondary_y": True}], [{}]],
    subplot_titles=("SPX and Drawdown", "Cross-Asset RVOL Stress Composite"),
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=spx_rebased.reindex(plot_idx),
        name="SPX (rebased=100)",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>SPX rebased: %{y:.1f}<extra></extra>",
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=dd_plot.reindex(plot_idx),
        name="Drawdown (%)",
        line=dict(width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
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
        line=dict(width=2.3),
        hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=comp_s.reindex(plot_idx).rolling(21, min_periods=1).mean(),
        name="Composite 21D MA",
        line=dict(width=1.4, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>21D MA: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=2, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=2, col=1)

finite_dd = dd_plot.replace([np.inf, -np.inf], np.nan).dropna()
dd_floor = -5.0 if finite_dd.empty else min(-5.0, float(finite_dd.min()) * 1.1)

fig.update_yaxes(title_text="Rebased", row=1, col=1, secondary_y=False)
fig.update_yaxes(
    title_text="Drawdown %",
    row=1,
    col=1,
    secondary_y=True,
    range=[dd_floor, 0],
    tickformat=".1f",
    zeroline=True,
    zerolinewidth=1,
)
fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 100])
fig.update_xaxes(title_text="Date", row=2, col=1, tickformat="%b-%d-%y")

fig.update_layout(
    template="plotly_white",
    height=780,
    margin=dict(l=50, r=30, t=80, b=50),
    legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
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
            "factor": ["Equities", "Credit", "Commodities", "FX", "Rates", "Breadth", "Dispersion"],
            "raw_value": [
                safe_float(scores.loc[latest, "Equities_p"] * 100.0) if "Equities_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Credit_p"] * 100.0) if "Credit_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Commodities_p"] * 100.0) if "Commodities_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "FX_p"] * 100.0) if "FX_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Rates_p"] * 100.0) if "Rates_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Breadth_p"] * 100.0) if "Breadth_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Dispersion_p"] * 100.0) if "Dispersion_p" in scores.columns else np.nan,
            ],
            "active": [
                bool(masks.loc[latest, "Equities_m"]) if "Equities_m" in masks.columns else False,
                bool(masks.loc[latest, "Credit_m"]) if "Credit_m" in masks.columns else False,
                bool(masks.loc[latest, "Commodities_m"]) if "Commodities_m" in masks.columns else False,
                bool(masks.loc[latest, "FX_m"]) if "FX_m" in masks.columns else False,
                bool(masks.loc[latest, "Rates_m"]) if "Rates_m" in masks.columns else False,
                bool(masks.loc[latest, "Breadth_m"]) if "Breadth_m" in masks.columns else False,
                bool(masks.loc[latest, "Dispersion_m"]) if "Dispersion_m" in masks.columns else False,
            ],
            "days_since_update": [
                age_map["Equities"].reindex(plot_idx).loc[latest] if "Equities" in age_map else np.nan,
                age_map["Credit"].reindex(plot_idx).loc[latest] if "Credit" in age_map else np.nan,
                age_map["Commodities"].reindex(plot_idx).loc[latest] if "Commodities" in age_map else np.nan,
                age_map["FX"].reindex(plot_idx).loc[latest] if "FX" in age_map else np.nan,
                age_map["Rates"].reindex(plot_idx).loc[latest] if "Rates" in age_map else np.nan,
                age_map["Breadth"].reindex(plot_idx).loc[latest] if "Breadth" in age_map else np.nan,
                age_map["Dispersion"].reindex(plot_idx).loc[latest] if "Dispersion" in age_map else np.nan,
            ],
        }
    )
    diag["raw_value"] = pd.to_numeric(diag["raw_value"], errors="coerce").round(1)
    diag["days_since_update"] = pd.to_numeric(diag["days_since_update"], errors="coerce").round(0)
    st.dataframe(diag, use_container_width=True, hide_index=True)

    st.markdown("**Underlying instrument RVOL percentile snapshot**")
    snap = []
    for ticker in sorted(set(ALL_TICKERS)):
        col = f"{ticker}_p"
        if col in rvol_pct.columns:
            snap.append(
                {
                    "Ticker": ticker,
                    "Asset Class": next(k for k, v in ASSET_CLASS_MAP.items() if ticker in v),
                    "RVOL Percentile": safe_float(rvol_pct.loc[latest, col] * 100.0),
                    "RVOL": safe_float(rvol.loc[latest, ticker]) if ticker in rvol.columns else np.nan,
                }
            )
    snap_df = pd.DataFrame(snap).sort_values(["Asset Class", "RVOL Percentile"], ascending=[True, False])
    snap_df["RVOL Percentile"] = pd.to_numeric(snap_df["RVOL Percentile"], errors="coerce").round(1)
    snap_df["RVOL"] = pd.to_numeric(snap_df["RVOL"], errors="coerce").round(3)
    st.dataframe(snap_df, use_container_width=True, hide_index=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    export_idx = plot_idx

    export_panel_cols = [c for c in ["SPX", "DD_stress"] + sorted(set(ALL_TICKERS)) if c in panel.columns]
    export_panel = panel.reindex(export_idx)[export_panel_cols].copy()

    export_rvol = rvol.reindex(export_idx).rename(columns={c: f"{c}_rvol" for c in rvol.columns})
    export_pct = rvol_pct.reindex(export_idx).rename(columns={c: f"{c}_pct" for c in rvol_pct.columns})
    export_scores = (scores.reindex(export_idx) * 100.0).rename(
        columns={
            "Equities_p": "Equities_pct",
            "Credit_p": "Credit_pct",
            "Commodities_p": "Commodities_pct",
            "FX_p": "FX_pct",
            "Rates_p": "Rates_pct",
            "Breadth_p": "Breadth_pct",
            "Dispersion_p": "Dispersion_pct",
        }
    )
    export_masks = masks.reindex(export_idx).copy()
    export_contrib = pd.DataFrame(
        weighted_contrib,
        index=scores.index,
        columns=[
            "Equities_contrib",
            "Credit_contrib",
            "Commodities_contrib",
            "FX_contrib",
            "Rates_contrib",
            "Breadth_contrib",
            "Dispersion_contrib",
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

    out = pd.concat([export_panel, export_rvol, export_pct, export_scores, export_masks, export_contrib, export_meta], axis=1)
    out.index.name = "Date"

    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="cross_asset_rvol_stress_composite.csv",
        mime="text/csv",
    )

st.caption("© 2026 AD Fund Management LP")
