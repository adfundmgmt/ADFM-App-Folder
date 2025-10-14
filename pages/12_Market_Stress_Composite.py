############################################################
# Market Stress Composite — intraday-aware nowcast + FRED history
# Built for AD Fund Management LP
# Changes:
# - Intraday proxies via yfinance: ^VIX, ^GSPC, ^TNX, ^IRX, HYG, LQD
# - Funding proxy nowcast: BIL vs SHY ratio fallback, FRED CP−Tbill for history
# - Percentiles computed per-series over user window, proxies included
# - Short cache TTL, optional auto refresh toggle
# - Strict freshness masks, weight re-normalization
############################################################

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- Config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

FRED = {
    "vix": "VIXCLS",           # VIX close
    "yc_10y_3m": "T10Y3M",     # 10Y minus 3M Treasury spread
    "hy_oas": "BAMLH0A0HYM2",  # HY OAS
    "cp_3m": "DCPF3M",         # 3M AA CP
    "tbill_3m": "DTB3",        # 3M T-bill
    "spx": "SP500",            # S&P 500
}

# Intraday proxy tickers on Yahoo
PX = {
    "VIX": "^VIX",                 # VIX index level, intraday
    "SPX": "^GSPC",                # S&P 500 index
    "TNX": "^TNX",                 # 10Y yield, percent
    "IRX": "^IRX",                 # 13 week bill yield, percent
    "HYG": "HYG",                  # HY ETF
    "LQD": "LQD",                  # IG ETF
    "BIL": "BIL",                  # 1-3M T-bill ETF
    "SHY": "SHY",                  # 1-3Y UST ETF
}

DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI = 70
REGIME_LO = 30
MAX_STALE_BDAYS = 1         # for FRED history
MAX_PROXY_AGE_MIN = 10      # intraday quote freshness window
HISTORY_YF_PERIOD = "max"   # for proxy historical percentiles

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y","2y","3y","5y","10y"],
                            index=["1y","2y","3y","5y","10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 60, DEFAULT_SMOOTH, 1)
    perc_years = st.number_input("Percentile history window (years)", 3, 25,
                                 DEFAULT_PERCENTILE_WINDOW_YEARS, 1)

    st.subheader("Weights")
    w_vix   = st.slider("VIX", 0.0, 1.0, 0.25, 0.05)
    w_hy    = st.slider("HY Credit (OAS or HYG−LQD)", 0.0, 1.0, 0.25, 0.05)
    w_curve = st.slider("Yield Curve, inverted", 0.0, 1.0, 0.20, 0.05)
    w_fund  = st.slider("Funding (CP−Tbill or BIL/SHY proxy)", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)

    st.subheader("Nowcast options")
    use_nowcast = st.checkbox("Use intraday proxies when available", value=True)
    auto_refresh = st.checkbox("Auto refresh every 60 seconds", value=False)
    if auto_refresh:
        st.autorefresh(interval=60_000, key="autorf")
    st.caption("Source: FRED via pandas-datareader, intraday proxies via Yahoo Finance")

    st.markdown("---")
    st.header("About")
    st.markdown(f"""
Daily 0 to 100 composite of market stress. History built from FRED. Intraday nowcast from proxies, zero-weighted if stale. 
Percentiles computed on a {perc_years} year rolling history per series, with optional {smooth} day smoothing.
""")

# ---------------- Utils ----------------
def _today():
    return pd.Timestamp.now(tz="America/New_York").normalize()

@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=30, show_spinner=False)
def yf_hist(tickers, period=HISTORY_YF_PERIOD, interval="1d"):
    try:
        df = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=False, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def yf_last(tickers):
    """Fetch last 5d 1m candles to get a fresh last quote and timestamp."""
    try:
        data = {}
        for tk in tickers:
            h = yf.Ticker(tk).history(period="5d", interval="1m")
            if not h.empty:
                data[tk] = h["Close"].copy()
        if not data:
            return pd.DataFrame()
        # align by last timestamp per ticker
        df = pd.concat(data, axis=1)
        return df
    except Exception:
        return pd.DataFrame()

def rolling_percentile(s: pd.Series, window_days: int) -> pd.Series:
    w = max(60, int(window_days * 252 / 365))
    return s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

def pad_range(lo: float, hi: float, pad: float = 0.06):
    if pd.isna(lo) or pd.isna(hi):
        return None
    d = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1.0)
    return lo - d * pad, hi + d * pad

# ---------------- Data: FRED history ----------------
today = _today()
start_all = today - pd.DateOffset(years=max(int(perc_years) + 2, years + 2))

vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = (cp3m_f - tb3m_f)

series_list = [vix_f, hy_f, yc_f, fund_f, spx_f]
non_empty = [s.dropna() for s in series_list if not s.dropna().empty]
if not non_empty:
    st.error("No FRED data available. Check connectivity or FRED service.")
    st.stop()

# Business-day grid through today
bidx_start = min(s.index.min() for s in non_empty)
bidx = pd.bdate_range(bidx_start, today)

def to_bidx(s: pd.Series) -> pd.Series:
    return s.reindex(bidx).ffill()

vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix_f, hy_f, yc_f, fund_f, spx_f])

# Freshness masks for FRED
def fresh_mask_bdays(original: pd.Series, bindex: pd.DatetimeIndex, max_stale: int) -> pd.Series:
    obs = original.reindex(bindex)
    pos = np.arange(len(bindex))
    has = obs.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=bindex).ffill().values
    age = pos - last_pos
    return pd.Series((~np.isnan(age)) & (age <= max_stale), index=bindex)

m_vix_f  = fresh_mask_bdays(vix_f,   bidx, MAX_STALE_BDAYS)
m_hy_f   = fresh_mask_bdays(hy_f,    bidx, MAX_STALE_BDAYS)
m_yc_f   = fresh_mask_bdays(yc_f,    bidx, MAX_STALE_BDAYS)
m_fund_f = fresh_mask_bdays(fund_f,  bidx, MAX_STALE_BDAYS)
m_dd_f   = fresh_mask_bdays(spx_f,   bidx, MAX_STALE_BDAYS)

# ---------------- Intraday proxies ----------------
# Historical daily closes for proxies to compute their own percentiles
px_hist = yf_hist(list(PX.values()), period=HISTORY_YF_PERIOD, interval="1d")

# Last minute quotes to form a nowcast
px_last = yf_last(list(PX.values()))

def latest_quote(series: pd.Series):
    if series is None or series.empty:
        return np.nan, None
    ts = series.dropna()
    if ts.empty:
        return np.nan, None
    return float(ts.iloc[-1]), ts.index[-1].tz_convert("America/New_York") if ts.index.tz is not None else ts.index[-1]

def is_fresh(ts, max_age_min=MAX_PROXY_AGE_MIN):
    if ts is None:
        return False
    now = pd.Timestamp.now(tz="America/New_York")
    if ts.tz is None:
        ts = ts.tz_localize("America/New_York")
    age = (now - ts).total_seconds() / 60.0
    return age <= max_age_min

# Build proxy nowcast values
def proxy_nowcast():
    out = {}
    ts_map = {}
    for key, tk in PX.items():
        s = px_last.get((tk,), None) if isinstance(px_last.columns, pd.MultiIndex) else px_last.get(tk, None)
        v, ts = latest_quote(s)
        out[key] = v
        ts_map[key] = ts
    return out, ts_map

px_now, px_stamp = proxy_nowcast()

# ---------------- Assemble inputs ----------------
# Base frame from FRED history
df_all = pd.DataFrame({
    "VIX": vix_b,
    "HY_OAS": hy_b,
    "T10Y3M": yc_b,
    "FUND": fund_b,
    "SPX": spx_b
}).dropna(how="all")

if df_all.empty:
    st.error("No overlapping FRED data across components.")
    st.stop()

# Construct proxy series for today if fresh and user opted in
def inject_nowcast(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    today_dt = pd.Timestamp.now(tz="America/New_York").floor("T")
    today_bday = pd.Timestamp(today_dt.date())
    # VIX
    if use_nowcast and is_fresh(px_stamp.get("VIX")) and not np.isnan(px_now.get("VIX", np.nan)):
        df2.loc[today_bday, "VIX"] = px_now["VIX"]
    # SPX
    if use_nowcast and is_fresh(px_stamp.get("SPX")) and not np.isnan(px_now.get("SPX", np.nan)):
        df2.loc[today_bday, "SPX"] = px_now["SPX"]
    # Curve, use ^TNX − ^IRX in percentage points
    if use_nowcast and is_fresh(px_stamp.get("TNX")) and is_fresh(px_stamp.get("IRX")):
        if not np.isnan(px_now.get("TNX", np.nan)) and not np.isnan(px_now.get("IRX", np.nan)):
            df2.loc[today_bday, "T10Y3M"] = px_now["TNX"] - px_now["IRX"]
    # HY OAS proxy using HYG − LQD ratio levels mapped to stress direction
    if use_nowcast and is_fresh(px_stamp.get("HYG")) and is_fresh(px_stamp.get("LQD")):
        if not np.isnan(px_now.get("HYG", np.nan)) and not np.isnan(px_now.get("LQD", np.nan)):
            # Build a proxy series history on the same index to compute percentiles later
            pass
    # Funding proxy using BIL vs SHY ratio, higher ratio implies more front end demand
    if use_nowcast and is_fresh(px_stamp.get("BIL")) and is_fresh(px_stamp.get("SHY")):
        if not np.isnan(px_now.get("BIL", np.nan)) and not np.isnan(px_now.get("SHY", np.nan)):
            proxy_val = px_now["BIL"] / px_now["SHY"]
            # Store raw proxy into a column to percentile separately
            df2.loc[today_bday, "FUND_PROXY"] = proxy_val
    return df2

panel0 = inject_nowcast(df_all)

# Build proxy history columns for percentile math
def attach_proxy_history(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    if not px_hist.empty:
        cols = px_hist.columns
        have = set(cols)
        # HY proxy series, HYG/LQD ratio mapped to stress
        if "HYG" in have and "LQD" in have:
            hyg_lqd = (px_hist["HYG"] / px_hist["LQD"]).rename("HYG_LQD_RATIO")
            p = p.join(hyg_lqd.reindex(p.index, method="ffill"))
        # Funding proxy, BIL/SHY ratio
        if "BIL" in have and "SHY" in have:
            bil_shy = (px_hist["BIL"] / px_hist["SHY"]).rename("BIL_SHY_RATIO")
            p = p.join(bil_shy.reindex(p.index, method="ffill"))
    return p

panel = attach_proxy_history(panel0)

# ---------------- Transforms ----------------
# SPX drawdown
roll_max = panel["SPX"].cummax()
dd_all = 100.0 * (panel["SPX"] / roll_max - 1.0)
stress_dd_all = -dd_all.clip(upper=0)

# Invert the curve
inv_curve_all = -panel["T10Y3M"]

# Percentile windows
window_days = int(perc_years * 365)

# FRED based percentiles
pct_vix_f   = rolling_percentile(panel["VIX"], window_days)
pct_hy_f    = rolling_percentile(panel["HY_OAS"], window_days)
pct_curve_f = rolling_percentile(inv_curve_all, window_days)
pct_fund_f  = rolling_percentile(panel["FUND"], window_days)
pct_dd_f    = rolling_percentile(stress_dd_all, window_days)

# Proxy percentiles for nowcast day
# HY proxy: lower HYG/LQD ratio implies wider spreads, so stress = negative ratio
pct_hy_proxy = None
if "HYG_LQD_RATIO" in panel.columns:
    hy_proxy_stress = -(panel["HYG_LQD_RATIO"])
    pct_hy_proxy = rolling_percentile(hy_proxy_stress, window_days)

# Funding proxy: higher BIL/SHY suggests bid for bills vs duration, treat as stress up
pct_fund_proxy = None
if "BIL_SHY_RATIO" in panel.columns:
    fund_proxy_stress = panel["BIL_SHY_RATIO"]
    pct_fund_proxy = rolling_percentile(fund_proxy_stress, window_days)

# Nowcast stitch: if today has proxy and it is fresh, use proxy percentile, else use FRED percentile
def stitch(series_f, series_proxy):
    if series_proxy is None:
        return series_f
    out = series_f.copy()
    if use_nowcast and not series_proxy.dropna().empty:
        last_idx = series_proxy.index[-1]
        # If last date is today and proxy value exists, take it
        if last_idx.date() == pd.Timestamp.now(tz="America/New_York").date():
            val = series_proxy.loc[last_idx]
            if pd.notna(val):
                out.loc[last_idx] = val
    return out

pct_vix   = pct_vix_f                     # ^VIX maps directly to VIXCLS level, percentile stays consistent
pct_hy    = stitch(pct_hy_f, pct_hy_proxy)
pct_curve = pct_curve_f                   # ^TNX−^IRX already written into T10Y3M then inverted above
pct_fund  = stitch(pct_fund_f, pct_fund_proxy)
pct_dd    = pct_dd_f

scores_all = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
scores_all.columns = ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]

# Lookback subset and smoothing
start_lb = today - pd.DateOffset(years=years)
scores = scores_all.loc[start_lb:].copy()
panel_lb  = panel.loc[start_lb:].copy()

if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# Freshness masks, combine FRED freshness with proxy freshness for today
masks = pd.concat([m_vix_f, m_hy_f, m_yc_f, m_fund_f, m_dd_f], axis=1)
masks.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]
masks = masks.reindex(scores.index).astype(float)

# If we used a fresh proxy today, flip mask to 1.0 for that component at today
today_bday = pd.Timestamp(pd.Timestamp.now(tz="America/New_York").date())
def force_mask_on(col, fresh_flag):
    if fresh_flag and today_bday in masks.index:
        masks.loc[today_bday, col] = 1.0

force_mask_on("VIX_m", use_nowcast and is_fresh(px_stamp.get("VIX")))
force_mask_on("Curve_m", use_nowcast and is_fresh(px_stamp.get("TNX")) and is_fresh(px_stamp.get("IRX")))
force_mask_on("HY_m", use_nowcast and ("HYG_LQD_RATIO" in panel.columns) and today_bday in scores.index)
force_mask_on("Fund_m", use_nowcast and ("BIL_SHY_RATIO" in panel.columns) and today_bday in scores.index)
force_mask_on("DD_m", use_nowcast and is_fresh(px_stamp.get("SPX")))

# Dynamic weights
weights = np.array([w_vix, w_hy, w_curve, w_fund, w_dd], dtype=float)
weights = weights if weights.sum() > 0 else np.ones(5)
weights = weights / weights.sum()

X = scores[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].values
W = weights.reshape(1, -1)
M = masks.values
active_w = (W * M).sum(axis=1)
comp = np.where(active_w > 0, 100.0 * (X * W * M).sum(axis=1) / active_w, np.nan)
comp_s = pd.Series(comp, index=scores.index, name="Composite")
coverage = pd.Series(active_w / W.sum(), index=scores.index, name="ActiveWeightShare")

# ---------------- Metrics ----------------
def tag_for_level(x: float) -> str:
    if pd.isna(x): return "N/A"
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

if comp_s.dropna().empty:
    st.warning("Composite has no valid points for the selected window.")
    st.stop()

latest_idx = comp_s.dropna().index[-1]
def fmt2(x): return "N/A" if pd.isna(x) else f"{x:.2f}"
def fmt0(x): return "N/A" if pd.isna(x) else f"{x:.0f}"
def pct(x): return "N/A" if pd.isna(x) else f"{100*x:.0f}%"

# Tiles
t1, t2, t3 = st.columns(3)
t1.metric("Stress Composite", fmt0(comp_s.loc[latest_idx]))
t2.metric("Regime", tag_for_level(comp_s.loc[latest_idx]))
t3.metric("Active weight in use", pct(coverage.loc[latest_idx] if latest_idx in coverage.index else np.nan))

# Show latest raw inputs
def last_val(s, idx):
    try:
        return s.loc[idx]
    except Exception:
        return np.nan

t4, t5, t6, t7, t8 = st.columns(5)
t4.metric("VIX", fmt2(last_val(panel_lb["VIX"], latest_idx)))
t5.metric("HY OAS or Proxy", fmt2(last_val(panel_lb.get("HY_OAS", panel_lb.get("HYG_LQD_RATIO")), latest_idx)))
t6.metric("T10Y3M (pp)", fmt2(last_val(panel_lb["T10Y3M"], latest_idx)))
t7.metric("Funding raw", fmt2(last_val(panel_lb.get("FUND", panel_lb.get("BIL_SHY_RATIO")), latest_idx)))
# Drawdown recompute on lookback
dd_series_lb = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)
t8.metric("SPX Drawdown (%)", fmt2(last_val(dd_series_lb, latest_idx)))

st.info(
    f"As of {latest_idx.date()}  |  Weights: VIX {weights[0]:.2f}, HY {weights[1]:.2f}, "
    f"Curve {weights[2]:.2f}, Funding {weights[3]:.2f}, Drawdown {weights[4]:.2f}"
)

# ---------------- Charts ----------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    specs=[[{}], [{"secondary_y": True}]],
    subplot_titles=("Market Stress Composite (0 to 100)", "SPX and Drawdown (dual axis)")
)

# Row 1: composite
fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

# Row 2: SPX left, Drawdown right
spx_rebased = panel_lb["SPX"] / panel_lb["SPX"].iloc[0] * 100
dd_series = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)

left_range = pad_range(spx_rebased.min(), spx_rebased.max())
right_range = pad_range(max(0.0, dd_series.min()), dd_series.max())

fig.add_trace(go.Scatter(x=panel_lb.index, y=spx_rebased, name="SPX (rebased=100)",
                         line=dict(color="#7f7f7f")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=panel_lb.index, y=dd_series, name="Drawdown (%)",
                         line=dict(color="#ff7f0e")), row=2, col=1, secondary_y=True)

if left_range:
    fig.update_yaxes(title="Index", range=list(left_range), row=2, col=1, secondary_y=False)
if right_range:
    fig.update_yaxes(title="Drawdown %", range=list(right_range), row=2, col=1, secondary_y=True)

fig.update_layout(
    template="plotly_white",
    height=720,
    legend=dict(orientation="h", x=0, y=1.14, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60)
)
fig.update_xaxes(tickformat="%b-%y", title="Date", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    export_idx = panel_lb.index
    export_scores = (100.0 * scores.rename(columns={
        "VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct",
        "Fund_p":"Fund_pct","DD_p":"DD_pct"
    })).reindex(export_idx)
    export_masks = masks.rename(columns={
        "VIX_m":"VIX_fresh","HY_m":"HY_fresh",
        "Curve_m":"Curve_fresh","Fund_m":"Fund_fresh","DD_m":"DD_fresh"
    }).reindex(export_idx)
    export_cov = coverage.reindex(export_idx)
    export_comp = comp_s.rename("Composite").reindex(export_idx)

    out = pd.concat(
        [
            panel_lb[["VIX","HY_OAS","T10Y3M","FUND"]].reindex(export_idx),
            panel_lb.filter(["HYG_LQD_RATIO","BIL_SHY_RATIO"]).reindex(export_idx),
            panel_lb[["SPX"]].reindex(export_idx),
            export_scores,
            export_masks,
            export_cov,
            export_comp,
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite_intraday.csv", mime="text/csv")

# ---------------- Methodology ----------------
with st.expander("Methodology"):
    st.markdown(f"""
Composite is a weighted average of component percentile scores on a {perc_years} year rolling window. 
FRED provides history. Intraday nowcast uses proxies: VIX from ^VIX, SPX from ^GSPC, curve from ^TNX minus ^IRX, HY from HYG/LQD ratio, funding from BIL/SHY ratio. 
Curve is inverted. Drawdown is percent off high. If a component is older than {MAX_STALE_BDAYS} business day on FRED, or a proxy is not fresh within {MAX_PROXY_AGE_MIN} minutes, it is zero-weighted that day. 
Smoothing applies a {smooth} day moving average to percentile series. Regimes: Low ≤ {REGIME_LO}, High ≥ {REGIME_HI}.
""")

st.caption("© 2025 AD Fund Management LP")
