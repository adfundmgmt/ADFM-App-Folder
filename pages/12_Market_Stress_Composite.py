############################################################
# Market Stress Composite — resilient FRED + proxy-only fallback
# Built for AD Fund Management LP
############################################################

import os
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
    "vix": "VIXCLS",
    "yc_10y_3m": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",
    "cp_3m": "DCPF3M",
    "tbill_3m": "DTB3",
    "spx": "SP500",
}

PX = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "TNX": "^TNX",
    "IRX": "^IRX",
    "HYG": "HYG",
    "LQD": "LQD",
    "BIL": "BIL",
    "SHY": "SHY",
}

DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI, REGIME_LO = 70, 30
MAX_STALE_BDAYS = 1
MAX_PROXY_AGE_MIN = 10

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
    w_hy    = st.slider("HY OAS", 0.0, 1.0, 0.25, 0.05)
    w_curve = st.slider("Yield Curve (inverted)", 0.0, 1.0, 0.20, 0.05)
    w_fund  = st.slider("Funding (CP − T-bill)", 0.0, 1.0, 0.15, 0.05)
    w_dd    = st.slider("SPX Drawdown", 0.0, 1.0, 0.15, 0.05)

    st.subheader("Nowcast options")
    use_nowcast = st.checkbox("Use intraday proxies when available", value=True)
    auto_refresh = st.checkbox("Auto refresh every 60 seconds", value=False)
    if auto_refresh:
        st.autorefresh(interval=60_000, key="autorf")

    st.caption("Sources: FRED, Yahoo Finance")

# ---------------- Utilities ----------------
def now_ny():
    return pd.Timestamp.now(tz="America/New_York")

def ny_bday(ts=None):
    ts = ts or now_ny()
    return pd.Timestamp(ts.date())

@st.cache_data(ttl=3600, show_spinner=False)
def fred_series_api(series: str, start: pd.Timestamp, end: pd.Timestamp):
    """Primary path: fredapi with API key or anonymous if allowed."""
    try:
        from fredapi import Fred
        key = None
        try:
            key = st.secrets.get("FRED_API_KEY", None)
        except Exception:
            key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=key) if key else Fred()
        # fredapi returns a Series indexed by date
        s = fred.get_series(series, observation_start=start.date(), observation_end=end.date())
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s.name = series
        return s
    except Exception as e:
        st.debug(f"[fred_series_api:{series}] {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def fred_series_pdr(series: str, start: pd.Timestamp, end: pd.Timestamp):
    """Fallback path: pandas-datareader FRED endpoint."""
    try:
        df = pdr.DataReader(series, "fred", start, end)
        return df[series]
    except Exception as e:
        st.debug(f"[fred_series_pdr:{series}] {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Resilient fetcher, api first then pdr."""
    s = fred_series_api(series, start, end)
    if s.dropna().empty:
        s = fred_series_pdr(series, start, end)
    return s

@st.cache_data(ttl=30, show_spinner=False)
def yf_hist(tickers, period="max", interval="1d"):
    try:
        df = yf.download(tickers=tickers, period=period, interval=interval,
                         auto_adjust=False, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df
    except Exception as e:
        st.debug(f"[yf_hist] {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def yf_last_minute(tickers):
    """Fetch recent 1m bars to get fresh last quotes and timestamps."""
    try:
        data = {}
        for tk in tickers:
            h = yf.Ticker(tk).history(period="5d", interval="1m")
            if not h.empty:
                data[tk] = h["Close"].copy()
        if not data:
            return pd.DataFrame()
        return pd.concat(data, axis=1)
    except Exception as e:
        st.debug(f"[yf_last_minute] {e}")
        return pd.DataFrame()

def is_fresh(ts, max_age_min=MAX_PROXY_AGE_MIN):
    if ts is None:
        return False
    t = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("America/New_York")
    return (now_ny() - t).total_seconds() / 60.0 <= max_age_min

def rolling_percentile(s: pd.Series, window_days: int) -> pd.Series:
    w = max(60, int(window_days * 252 / 365))
    return s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def pad_range(lo: float, hi: float, pad: float = 0.06):
    if pd.isna(lo) or pd.isna(hi):
        return None
    d = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1.0)
    return lo - d * pad, hi + d * pad

# ---------------- Load data ----------------
today = ny_bday()
start_all = today - pd.DateOffset(years=12)  # a bit larger than perc_years window

# Try FRED
vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = (cp3m_f - tb3m_f)

# Determine availability
series_list = [vix_f, hy_f, yc_f, fund_f, spx_f]
have_any_fred = any((not s.dropna().empty) for s in series_list)

# Always load proxies for fallback and nowcast
px_daily = yf_hist(list(PX.values()), period="max", interval="1d")
px_last  = yf_last_minute(list(PX.values()))

def last_quote(symbol):
    col = symbol
    if isinstance(px_last.columns, pd.MultiIndex):
        # yfinance sometimes returns MultiIndex columns
        try:
            col = [c for c in px_last.columns if (isinstance(c, tuple) and c[0] == symbol)][0]
        except Exception:
            col = symbol
    s = px_last.get(col, pd.Series(dtype=float))
    if s is None or s.empty:
        return np.nan, None
    return float(s.dropna().iloc[-1]), s.dropna().index[-1]

# ---------------- Build panel ----------------
if have_any_fred:
    # Grid and ffill
    bidx_start = min([s.index.min() for s in series_list if not s.dropna().empty])
    bidx = pd.bdate_range(bidx_start, today)
    def to_bidx(s): return s.reindex(bidx).ffill()
    vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix_f, hy_f, yc_f, fund_f, spx_f])

    panel = pd.DataFrame({
        "VIX": vix_b, "HY_OAS": hy_b, "T10Y3M": yc_b, "FUND": fund_b, "SPX": spx_b
    }).dropna(how="all")

    # Freshness masks based on business days
    def fresh_mask_bdays(original, bindex, max_stale):
        obs = original.reindex(bindex)
        pos = np.arange(len(bindex))
        has = obs.notna().values
        last_pos = pd.Series(np.where(has, pos, np.nan), index=bindex).ffill().values
        age = pos - last_pos
        return pd.Series((~np.isnan(age)) & (age <= max_stale), index=bindex)

    m_vix  = fresh_mask_bdays(vix_f,  panel.index, MAX_STALE_BDAYS)
    m_hy   = fresh_mask_bdays(hy_f,   panel.index, MAX_STALE_BDAYS)
    m_yc   = fresh_mask_bdays(yc_f,   panel.index, MAX_STALE_BDAYS)
    m_fund = fresh_mask_bdays(fund_f, panel.index, MAX_STALE_BDAYS)
    m_dd   = fresh_mask_bdays(spx_f,  panel.index, MAX_STALE_BDAYS)

    mode = "FRED+Proxies"
else:
    # Proxy-only mode: synthesize history from proxies
    st.warning(
        "FRED unavailable, running in proxy-only mode. "
        "HY uses HYG/LQD ratio, curve uses ^TNX−^IRX, funding uses BIL/SHY ratio. "
        "Interpret levels as directional proxies."
    )
    if px_daily.empty:
        st.error("Cannot load proxies from Yahoo Finance either. Check network.")
        st.stop()

    # Build synthetic historical series
    # VIX, SPX direct
    spx = px_daily["^GSPC"].rename("SPX").dropna()
    vix = px_daily["^VIX"].rename("VIX").dropna()
    # Curve proxy
    if {"^TNX","^IRX"}.issubset(px_daily.columns):
        curve = (px_daily["^TNX"] - px_daily["^IRX"]).rename("T10Y3M").dropna()
    else:
        curve = pd.Series(dtype=float, name="T10Y3M")
    # HY proxy
    if {"HYG","LQD"}.issubset(px_daily.columns):
        hy_proxy = (px_daily["HYG"] / px_daily["LQD"]).rename("HYG_LQD_RATIO").dropna()
    else:
        hy_proxy = pd.Series(dtype=float, name="HYG_LQD_RATIO")
    # Funding proxy
    if {"BIL","SHY"}.issubset(px_daily.columns):
        fund_proxy = (px_daily["BIL"] / px_daily["SHY"]).rename("BIL_SHY_RATIO").dropna()
    else:
        fund_proxy = pd.Series(dtype=float, name="BIL_SHY_RATIO")

    panel = pd.concat([vix, spx, curve, hy_proxy, fund_proxy], axis=1).dropna(how="all")
    panel.index = pd.DatetimeIndex(panel.index.date)  # normalize to date
    panel = panel.sort_index()

    # Masks: proxies update daily, so treat today as fresh if we have a fresh 1m tick
    idx = panel.index
    m_vix  = pd.Series(True, index=idx)
    m_yc   = pd.Series(True, index=idx)
    m_hy   = pd.Series(True, index=idx)
    m_fund = pd.Series(True, index=idx)
    m_dd   = pd.Series(True, index=idx)

    mode = "Proxy-only"

# Inject nowcast values for today if fresh and requested
if use_nowcast:
    vix_q, vix_ts = last_quote("^VIX")
    spx_q, spx_ts = last_quote("^GSPC")
    tnx_q, tnx_ts = last_quote("^TNX")
    irx_q, irx_ts = last_quote("^IRX")
    hyg_q, hyg_ts = last_quote("HYG")
    lqd_q, lqd_ts = last_quote("LQD")
    bil_q, bil_ts = last_quote("BIL")
    shy_q, shy_ts = last_quote("SHY")
    today_b = ny_bday()

    if is_fresh(vix_ts) and not np.isnan(vix_q):
        panel.loc[today_b, "VIX"] = vix_q
        m_vix.loc[today_b] = True
    if is_fresh(spx_ts) and not np.isnan(spx_q):
        panel.loc[today_b, "SPX"] = spx_q
        m_dd.loc[today_b] = True
    if is_fresh(tnx_ts) and is_fresh(irx_ts) and not np.isnan(tnx_q) and not np.isnan(irx_q):
        panel.loc[today_b, "T10Y3M"] = tnx_q - irx_q
        m_yc.loc[today_b] = True
    if is_fresh(hyg_ts) and is_fresh(lqd_ts) and not np.isnan(hyg_q) and not np.isnan(lqd_q):
        panel.loc[today_b, "HYG_LQD_RATIO"] = hyg_q / lqd_q
        m_hy.loc[today_b] = True
    if is_fresh(bil_ts) and is_fresh(shy_ts) and not np.isnan(bil_q) and not np.isnan(shy_q):
        panel.loc[today_b, "BIL_SHY_RATIO"] = bil_q / shy_q
        m_fund.loc[today_b] = True

# ---------------- Transforms ----------------
# Compute funding and HY series depending on mode/availability
if "FUND" not in panel.columns:
    panel["FUND"] = panel.get("BIL_SHY_RATIO")  # proxy if FRED absent
if "HY_OAS" not in panel.columns:
    # Map HYG/LQD ratio to stress by inversion (lower ratio => wider spreads => higher stress)
    if "HYG_LQD_RATIO" in panel.columns:
        panel["HY_OAS_PROXY"] = -(panel["HYG_LQD_RATIO"])

roll_max = panel["SPX"].dropna().cummax()
dd_all = 100.0 * (panel["SPX"] / roll_max - 1.0)
stress_dd_all = -dd_all.clip(upper=0)
inv_curve_all = -panel["T10Y3M"]

window_days = int(perc_years * 365)

pct_vix   = rolling_percentile(panel["VIX"], window_days)
pct_curve = rolling_percentile(inv_curve_all, window_days)
pct_fund  = rolling_percentile(panel["FUND"], window_days)
pct_dd    = rolling_percentile(stress_dd_all, window_days)

if "HY_OAS" in panel.columns and not panel["HY_OAS"].dropna().empty:
    pct_hy = rolling_percentile(panel["HY_OAS"], window_days)
else:
    pct_hy = rolling_percentile(panel["HY_OAS_PROXY"], window_days) if "HY_OAS_PROXY" in panel else pd.Series(index=panel.index, dtype=float)

scores_all = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
scores_all.columns = ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]

# Lookback and smoothing
start_lb = today - pd.DateOffset(years=years)
scores = scores_all.loc[start_lb:].copy()
panel_lb = panel.loc[start_lb:].copy()

if smooth > 1:
    scores = scores.rolling(smooth, min_periods=1).mean()

# Masks on lookback index
masks = pd.concat([m_vix, m_hy, m_yc, m_fund, m_dd], axis=1).rename(
    columns={0:"VIX_m",1:"HY_m",2:"Curve_m",3:"Fund_m",4:"DD_m"}
)
masks = masks.reindex(scores.index).astype(float).fillna(0.0)
masks.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]

# Dynamic weights and composite
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

# ---------------- UI: tiles ----------------
def tag_for_level(x):
    if pd.isna(x): return "N/A"
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

if comp_s.dropna().empty:
    st.error("Composite has no valid points for the selected window.")
    st.stop()

latest_idx = comp_s.dropna().index[-1]
fmt2 = lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
fmt0 = lambda x: "N/A" if pd.isna(x) else f"{x:.0f}"
pct = lambda x: "N/A" if pd.isna(x) else f"{100*x:.0f}%"

st.info(f"Mode: {mode}. As of {latest_idx.date()} | Weights: VIX {weights[0]:.2f}, HY {weights[1]:.2f}, Curve {weights[2]:.2f}, Funding {weights[3]:.2f}, Drawdown {weights[4]:.2f}")

t1, t2, t3 = st.columns(3)
t1.metric("Stress Composite", fmt0(comp_s.loc[latest_idx]))
t2.metric("Regime", tag_for_level(comp_s.loc[latest_idx]))
t3.metric("Active weight in use", pct(coverage.loc[latest_idx]))

t4, t5, t6, t7, t8 = st.columns(5)
t4.metric("VIX", fmt2(panel_lb["VIX"].reindex([latest_idx]).iloc[0] if "VIX" in panel_lb else np.nan))
t5.metric("HY (OAS or proxy)", fmt2(panel_lb.get("HY_OAS", panel_lb.get("HY_OAS_PROXY")).reindex([latest_idx]).iloc[0] if "HY_OAS" in panel_lb or "HY_OAS_PROXY" in panel_lb else np.nan))
t6.metric("T10Y3M (pp)", fmt2(panel_lb["T10Y3M"].reindex([latest_idx]).iloc[0] if "T10Y3M" in panel_lb else np.nan))
t7.metric("Funding (raw)", fmt2(panel_lb.get("FUND").reindex([latest_idx]).iloc[0] if "FUND" in panel_lb else np.nan))
dd_series_lb = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)
t8.metric("SPX Drawdown (%)", fmt2(dd_series_lb.reindex([latest_idx]).iloc[0] if not dd_series_lb.dropna().empty else np.nan))

# ---------------- Charts ----------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    specs=[[{}], [{"secondary_y": True}]],
    subplot_titles=("Market Stress Composite (0 to 100)", "SPX and Drawdown (dual axis)")
)

fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

spx_rebased = panel_lb["SPX"] / panel_lb["SPX"].iloc[0] * 100
dd_series = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)

left_range = pad_range(spx_rebased.min(), spx_rebased.max())
right_range = pad_range(max(0.0, dd_series.min()), dd_series.max())

fig.add_trace(go.Scatter(x=panel_lb.index, y=spx_rebased, name="SPX (rebased=100)",
                         line=dict(color="#7f7f7f")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=panel_lb.index, y=dd_series, name="Drawdown (%)",
                         line=dict(color="#ff7f0e")), row=2, col=1, secondary_y=True)

if left_range:  fig.update_yaxes(title="Index", range=list(left_range), row=2, col=1, secondary_y=False)
if right_range: fig.update_yaxes(title="Drawdown %", range=list(right_range), row=2, col=1, secondary_y=True)

fig.update_layout(template="plotly_white", height=720,
                  legend=dict(orientation="h", x=0, y=1.14, xanchor="left"),
                  margin=dict(l=60, r=40, t=60, b=60))
fig.update_xaxes(tickformat="%b-%y", title="Date", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    export_idx = panel_lb.index
    export_scores = (100.0 * scores.rename(columns={
        "VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"
    })).reindex(export_idx)
    export_masks = masks.rename(columns={
        "VIX_m":"VIX_fresh","HY_m":"HY_fresh","Curve_m":"Curve_fresh","Fund_m":"Fund_fresh","DD_m":"DD_fresh"
    }).reindex(export_idx)
    export_cov = coverage.reindex(export_idx)
    export_comp = comp_s.rename("Composite").reindex(export_idx)

    out = pd.concat(
        [
            panel_lb.filter(["VIX","HY_OAS","HY_OAS_PROXY","T10Y3M","FUND","HYG_LQD_RATIO","BIL_SHY_RATIO","SPX"]),
            export_scores, export_masks, export_cov, export_comp
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite_resilient.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
