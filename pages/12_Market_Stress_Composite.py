############################################################
# Market Stress Composite — pdr-only FRED + proxy fallback
# Built for AD Fund Management LP
############################################################

# ---- Py3.12 compat shim for pandas_datareader ----
import sys, types
import packaging.version
try:
    import distutils.version  # some pdr installs still import this
except Exception:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

# ---- Imports ----
import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---- Config ----
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

FRED = {
    "vix": "VIXCLS",          # VIX close
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M
    "hy_oas": "BAMLH0A0HYM2", # HY OAS
    "cp_3m": "DCPF3M",        # 3M AA CP
    "tbill_3m": "DTB3",       # 3M T-bill
    "spx": "SP500",           # S&P 500 index
}

# Intraday proxies (no keys)
PX = {"VIX":"^VIX","SPX":"^GSPC","TNX":"^TNX","IRX":"^IRX","HYG":"HYG","LQD":"LQD","BIL":"BIL","SHY":"SHY"}

DEFAULT_LOOKBACK = "5y"
DEFAULT_SMOOTH = 5
DEFAULT_PERCENTILE_WINDOW_YEARS = 10
REGIME_HI, REGIME_LO = 70, 30
MAX_STALE_BDAYS = 1
MAX_PROXY_AGE_MIN = 10

# ---- Sidebar ----
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

    st.subheader("Nowcast")
    use_nowcast = st.checkbox("Use intraday proxies when available", value=True)
    auto_refresh = st.checkbox("Auto refresh every 60 seconds", value=False)
    if auto_refresh:
        st.autorefresh(interval=60_000, key="autorf")
    st.caption("Sources: FRED via pandas-datareader, intraday proxies via Yahoo Finance")

    st.markdown("---")
    st.header("About This Tool")
    st.markdown("""
Daily 0–100 composite of market stress. FRED history, optional intraday nowcast. 
Zero-weights stale inputs and re-normalizes weights.
""")

# ---- Utilities ----
def _today():
    return pd.Timestamp.now(tz="America/New_York").normalize()

@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Keyless FRED via pandas_datareader only."""
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill().dropna()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=30, show_spinner=False)
def yf_hist(tickers, period="max", interval="1d"):
    try:
        df = yf.download(tickers=tickers, period=period, interval=interval,
                         auto_adjust=False, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def yf_last_minute(tickers):
    try:
        data = {}
        for tk in tickers:
            h = yf.Ticker(tk).history(period="5d", interval="1m")
            if not h.empty:
                data[tk] = h["Close"].copy()
        return pd.concat(data, axis=1) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def latest_quote(df_1m: pd.DataFrame, symbol: str):
    s = df_1m.get(symbol, pd.Series(dtype=float))
    if s is None or s.empty:
        return np.nan, None
    ts = s.dropna()
    if ts.empty:
        return np.nan, None
    return float(ts.iloc[-1]), ts.index[-1]

def is_fresh(ts, max_age_min=MAX_PROXY_AGE_MIN):
    if ts is None:
        return False
    now = pd.Timestamp.now(tz="America/New_York")
    t = ts if getattr(ts, "tzinfo", None) else pd.Timestamp(ts).tz_localize("America/New_York")
    return (now - t).total_seconds() / 60.0 <= max_age_min

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

# ---- Load FRED data (no keys) ----
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
have_any_fred = any((not s.dropna().empty) for s in series_list)

# ---- Always load proxies for nowcast/fallback ----
px_daily = yf_hist(list(PX.values()), period="max", interval="1d")
px_last  = yf_last_minute(list(PX.values()))

# ---- Build panel ----
if have_any_fred:
    bidx_start = min(s.index.min() for s in series_list if not s.dropna().empty)
    bidx = pd.bdate_range(bidx_start, today)
    to_bidx = lambda s: s.reindex(bidx).ffill()
    vix_b, hy_b, yc_b, fund_b, spx_b = map(to_bidx, [vix_f, hy_f, yc_f, fund_f, spx_f])
    panel = pd.DataFrame({"VIX": vix_b, "HY_OAS": hy_b, "T10Y3M": yc_b, "FUND": fund_b, "SPX": spx_b}).dropna(how="all")

    # freshness masks by business day age
    def fresh_mask_bdays(original: pd.Series) -> pd.Series:
        obs = original.reindex(panel.index)
        pos = np.arange(len(panel.index))
        has = obs.notna().values
        last_pos = pd.Series(np.where(has, pos, np.nan), index=panel.index).ffill().values
        age = pos - last_pos
        return pd.Series((~np.isnan(age)) & (age <= MAX_STALE_BDAYS), index=panel.index)

    m_vix  = fresh_mask_bdays(vix_f)
    m_hy   = fresh_mask_bdays(hy_f)
    m_yc   = fresh_mask_bdays(yc_f)
    m_fund = fresh_mask_bdays(fund_f)
    m_dd   = fresh_mask_bdays(spx_f)

    mode = "FRED+Proxies"
else:
    # Proxy-only fallback, no CSV, no keys
    if px_daily.empty:
        st.error("Neither FRED nor Yahoo proxies are available. Check network.")
        st.stop()

    spx = px_daily["^GSPC"].rename("SPX").dropna()
    vix = px_daily["^VIX"].rename("VIX").dropna()
    curve = (px_daily["^TNX"] - px_daily["^IRX"]).rename("T10Y3M") if {"^TNX","^IRX"}.issubset(px_daily.columns) else pd.Series(dtype=float, name="T10Y3M")
    hy_ratio = (px_daily["HYG"] / px_daily["LQD"]).rename("HYG_LQD_RATIO") if {"HYG","LQD"}.issubset(px_daily.columns) else pd.Series(dtype=float, name="HYG_LQD_RATIO")
    fund_ratio = (px_daily["BIL"] / px_daily["SHY"]).rename("BIL_SHY_RATIO") if {"BIL","SHY"}.issubset(px_daily.columns) else pd.Series(dtype=float, name="BIL_SHY_RATIO")

    panel = pd.concat([vix, spx, curve, hy_ratio, fund_ratio], axis=1).dropna(how="all")
    panel.index = pd.DatetimeIndex(panel.index.date)
    panel = panel.sort_index()

    m_vix  = pd.Series(True, index=panel.index)
    m_yc   = pd.Series(True, index=panel.index)
    m_hy   = pd.Series(True, index=panel.index)
    m_fund = pd.Series(True, index=panel.index)
    m_dd   = pd.Series(True, index=panel.index)
    mode = "Proxy-only"

# ---- Inject intraday nowcast (optional) ----
if use_nowcast and not px_last.empty:
    tb = pd.Timestamp(pd.Timestamp.now(tz="America/New_York").date())
    vix_q, vix_ts = latest_quote(px_last, "^VIX")
    spx_q, spx_ts = latest_quote(px_last, "^GSPC")
    tnx_q, tnx_ts = latest_quote(px_last, "^TNX")
    irx_q, irx_ts = latest_quote(px_last, "^IRX")
    hyg_q, hyg_ts = latest_quote(px_last, "HYG")
    lqd_q, lqd_ts = latest_quote(px_last, "LQD")
    bil_q, bil_ts = latest_quote(px_last, "BIL")
    shy_q, shy_ts = latest_quote(px_last, "SHY")

    if is_fresh(vix_ts) and not np.isnan(vix_q): panel.loc[tb, "VIX"] = vix_q;       m_vix.loc[tb]  = True
    if is_fresh(spx_ts) and not np.isnan(spx_q): panel.loc[tb, "SPX"] = spx_q;       m_dd.loc[tb]   = True
    if is_fresh(tnx_ts) and is_fresh(irx_ts) and not np.isnan(tnx_q) and not np.isnan(irx_q):
        panel.loc[tb, "T10Y3M"] = tnx_q - irx_q; m_yc.loc[tb] = True
    if is_fresh(hyg_ts) and is_fresh(lqd_ts) and not np.isnan(hyg_q) and not np.isnan(lqd_q):
        panel.loc[tb, "HYG_LQD_RATIO"] = hyg_q / lqd_q; m_hy.loc[tb] = True
    if is_fresh(bil_ts) and is_fresh(shy_ts) and not np.isnan(bil_q) and not np.isnan(shy_q):
        panel.loc[tb, "BIL_SHY_RATIO"] = bil_q / shy_q; m_fund.loc[tb] = True

# Ensure FUND and HY stress proxies exist if FRED fields missing
if "FUND" not in panel.columns and "BIL_SHY_RATIO" in panel.columns:
    panel["FUND"] = panel["BIL_SHY_RATIO"]
if "HY_OAS" not in panel.columns and "HYG_LQD_RATIO" in panel.columns:
    panel["HY_OAS_PROXY"] = -(panel["HYG_LQD_RATIO"])  # lower ratio -> wider spreads -> higher stress

# ---- Transforms ----
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
masks = pd.concat([m_vix, m_hy, m_yc, m_fund, m_dd], axis=1)
masks.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]
masks = masks.reindex(scores.index).astype(float).fillna(0.0)

# Composite
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

# ---- Metrics ----
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

c1, c2, c3 = st.columns(3)
c1.metric("Stress Composite", fmt0(comp_s.loc[latest_idx]))
c2.metric("Regime", tag_for_level(comp_s.loc[latest_idx]))
c3.metric("Active weight in use", pct(coverage.loc[latest_idx]))

c4, c5, c6, c7, c8 = st.columns(5)
c4.metric("VIX", fmt2(panel_lb.get("VIX").reindex([latest_idx]).iloc[0] if "VIX" in panel_lb else np.nan))
c5.metric("HY (OAS or proxy)", fmt2(panel_lb.get("HY_OAS", panel_lb.get("HY_OAS_PROXY")).reindex([latest_idx]).iloc[0] if ("HY_OAS" in panel_lb or "HY_OAS_PROXY" in panel_lb) else np.nan))
c6.metric("T10Y3M (pp)", fmt2(panel_lb.get("T10Y3M").reindex([latest_idx]).iloc[0] if "T10Y3M" in panel_lb else np.nan))
c7.metric("Funding (raw)", fmt2(panel_lb.get("FUND").reindex([latest_idx]).iloc[0] if "FUND" in panel_lb else np.nan))
dd_series_lb = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)
c8.metric("SPX Drawdown (%)", fmt2(dd_series_lb.reindex([latest_idx]).iloc[0] if not dd_series_lb.dropna().empty else np.nan))

# ---- Charts ----
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    specs=[[{}], [{"secondary_y": True}]],
                    subplot_titles=("Market Stress Composite (0 to 100)", "SPX and Drawdown (dual axis)"))

fig.add_trace(go.Scatter(x=comp_s.index, y=comp_s, name="Composite",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)", row=1, col=1)
fig.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)", row=1, col=1)
fig.update_yaxes(title="Score", range=[0,100], row=1, col=1)

spx_rebased = panel_lb["SPX"] / panel_lb["SPX"].iloc[0] * 100
dd_series = -100.0 * (panel_lb["SPX"] / panel_lb["SPX"].cummax() - 1.0)

def _rr(lo, hi):
    pr = pad_range(lo, hi)
    return list(pr) if pr else None

lr = _rr(spx_rebased.min(), spx_rebased.max())
rr = _rr(max(0.0, dd_series.min()), dd_series.max())

fig.add_trace(go.Scatter(x=panel_lb.index, y=spx_rebased, name="SPX (rebased=100)",
                         line=dict(color="#7f7f7f")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=panel_lb.index, y=dd_series, name="Drawdown (%)",
                         line=dict(color="#ff7f0e")), row=2, col=1, secondary_y=True)

if lr: fig.update_yaxes(title="Index", range=lr, row=2, col=1, secondary_y=False)
if rr: fig.update_yaxes(title="Drawdown %", range=rr, row=2, col=1, secondary_y=True)

fig.update_layout(template="plotly_white", height=720,
                  legend=dict(orientation="h", x=0, y=1.14, xanchor="left"),
                  margin=dict(l=60, r=40, t=60, b=60))
fig.update_xaxes(tickformat="%b-%y", title="Date", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---- Download ----
with st.expander("Download Data"):
    export_idx = panel_lb.index
    export_scores = (100.0 * scores.rename(columns={"VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"})).reindex(export_idx)
    export_masks = masks.rename(columns={"VIX_m":"VIX_fresh","HY_m":"HY_fresh","Curve_m":"Curve_fresh","Fund_m":"Fund_fresh","DD_m":"DD_fresh"}).reindex(export_idx)
    export_cov = coverage.reindex(export_idx)
    export_comp = comp_s.rename("Composite").reindex(export_idx)

    out = pd.concat(
        [
            panel_lb.filter(["VIX","HY_OAS","HY_OAS_PROXY","T10Y3M","FUND","HYG_LQD_RATIO","BIL_SHY_RATIO","SPX"]),
            export_scores, export_masks, export_cov, export_comp,
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

# sanity: all major frames must be naive date indexes
for name, obj in [("panel", panel), ("scores_all", scores_all), ("comp_s", comp_s)]:
    assert isinstance(obj.index, pd.DatetimeIndex), f"{name} not datetime index"
    assert obj.index.tz is None, f"{name} index has tz set"
# sanity: start_lb is naive
assert isinstance(start_lb, pd.Timestamp) and start_lb.tz is None


st.caption("© 2025 AD Fund Management LP")
