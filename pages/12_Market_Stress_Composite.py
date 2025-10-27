############################################################
# Market Stress Composite — clean UI, daily panel with live nowcast
# Built for AD Fund Management LP
# Design: single composite, regime bands, Daily/Weekly/Monthly regimes,
#         fixed weights, SPX + Drawdown panel aligned to composite dates.
############################################################

# ---- Py3.12 compat shim for pandas_datareader ----
import sys, types
import packaging.version
try:
    import distutils.version  # some pdr installs import this import
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

# ---- App config ----
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---- Series map ----
FRED = {
    "vix": "VIXCLS",          # VIX close
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M
    "hy_oas": "BAMLH0A0HYM2", # ICE HY OAS
    "cp_3m": "DCPF3M",        # 3M AA CP
    "tbill_3m": "DTB3",       # 3M T-bill
    "spx": "SP500",           # S&P 500 index
}

# Intraday proxies used only to refresh today's row
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

# ---- Fixed parameters tuned for responsiveness without clutter ----
DEFAULT_LOOKBACK = "3y"
PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1
REGIME_HI, REGIME_LO = 70, 30
MAX_STALE_BDAYS = 0        # do not accept yesterday as fresh for the composite
MAX_PROXY_AGE_MIN = 2      # minutes

# ---- LOCKED WEIGHTS (no UI to change) ----
W_VIX   = 0.25
W_HY    = 0.25
W_CURVE = 0.20
W_FUND  = 0.15
W_DD    = 0.15
WEIGHTS_VEC = np.array([W_VIX, W_HY, W_CURVE, W_FUND, W_DD], dtype=float)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y","2y","3y","5y","10y"],
                            index=["1y","2y","3y","5y","10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])

    st.subheader("Nowcast")
    use_nowcast = st.checkbox("Use intraday proxies to update today", value=True)

    st.subheader("Weights (fixed)")
    st.caption(f"VIX {W_VIX:.2f}, HY {W_HY:.2f}, Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}")

    st.markdown("---")
    st.subheader("About this tool")
    st.markdown(
        "- Purpose: a single **risk dial** blending volatility, credit, curve inversion, funding, and SPX drawdown into a 0–100 score.\n"
        "- Data: daily history from **FRED**, intraday quotes only refresh today’s row.\n"
        "- Method: percentile rank each factor over 3 years, then combine with fixed weights.\n"
        "- Interpretation: above 70 = **High stress**, below 30 = **Low stress**, between = **Neutral**.\n"
        "- UI: top shows Daily, Weekly, Monthly regimes. Bottom adds SPX and drawdown for context."
    )

# ---- Time helpers ----
NY_TZ = "America/New_York"

def today_naive() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz=NY_TZ).date())

def to_naive_date_index(obj):
    if isinstance(obj, pd.Series):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
        return obj
    if isinstance(obj, pd.DataFrame):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
        return obj
    return obj

def as_naive_date(ts_like) -> pd.Timestamp:
    t = pd.to_datetime(ts_like)
    return pd.Timestamp(t.date())

def is_fresh(ts, max_age_min=MAX_PROXY_AGE_MIN):
    if ts is None or pd.isna(ts):
        return False
    now = pd.Timestamp.now(tz=NY_TZ)
    t = ts if getattr(ts, "tzinfo", None) else pd.Timestamp(ts).tz_localize(NY_TZ)
    return (now - t).total_seconds() / 60.0 <= max_age_min

# ---- Data loaders ----
@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        s = s.ffill().dropna()
        return to_naive_date_index(s)
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=30, show_spinner=False)
def yf_last_minute(tickers):
    try:
        data = {}
        for tk in tickers:
            h = yf.Ticker(tk).history(period="5d", interval="1m")
            if not h.empty:
                if h.index.tz is None:
                    h.index = h.index.tz_localize(NY_TZ)
                else:
                    h.index = h.index.tz_convert(NY_TZ)
                data[tk] = h["Close"].copy()
        return pd.concat(data, axis=1) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def latest_quote(df_1m: pd.DataFrame, symbol: str):
    if df_1m.empty or symbol not in df_1m.columns:
        return np.nan, None
    s = df_1m[symbol].dropna()
    if s.empty:
        return np.nan, None
    return float(s.iloc[-1]), s.index[-1]

def rolling_percentile_daily(s: pd.Series, window_days: int) -> pd.Series:
    s = to_naive_date_index(s)
    w = max(60, int(window_days * 252 / 365))
    out = s.rolling(w, min_periods=max(20, w // 5)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    return to_naive_date_index(out)

# ---- Load FRED daily history ----
today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f)
series_list = [vix_f, hy_f, yc_f, fund_f, spx_f]
if not any((not s.dropna().empty) for s in series_list):
    st.error("FRED series unavailable. Check network.")
    st.stop()

# ---- Business-day panel ----
bidx_start = min(s.index.min() for s in series_list if not s.dropna().empty)
bidx = pd.bdate_range(as_naive_date(bidx_start), today)

def to_bidx(s):
    s = to_naive_date_index(s)
    return s.reindex(bidx).ffill()

panel = pd.DataFrame({
    "VIX": to_bidx(vix_f),
    "HY_OAS": to_bidx(hy_f),
    "T10Y3M": to_bidx(yc_f),
    "FUND": to_bidx(fund_f),
    "SPX": to_bidx(spx_f),
}).dropna(how="all")
panel = to_naive_date_index(panel)

# ---- Freshness masks for daily data ----
def fresh_mask_bdays(original: pd.Series) -> pd.Series:
    original = to_naive_date_index(original)
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

# ---- Intraday nowcast: refresh today's row only, quietly ----
if use_nowcast:
    px_last = yf_last_minute(list(PX.values()))
    if not px_last.empty:
        tb = today_naive()
        def q(sym):
            v, ts = latest_quote(px_last, sym)
            return v, ts
        vix_q, vix_ts = q(PX["VIX"])
        spx_q, spx_ts = q(PX["SPX"])
        tnx_q, tnx_ts = q(PX["TNX"])
        irx_q, irx_ts = q(PX["IRX"])
        hyg_q, hyg_ts = q(PX["HYG"])
        lqd_q, lqd_ts = q(PX["LQD"])
        bil_q, bil_ts = q(PX["BIL"])
        shy_q, shy_ts = q(PX["SHY"])

        if is_fresh(vix_ts) and not np.isnan(vix_q):
            panel.loc[tb, "VIX"] = vix_q; m_vix.loc[tb] = True
        if is_fresh(spx_ts) and not np.isnan(spx_q):
            panel.loc[tb, "SPX"] = spx_q; m_dd.loc[tb] = True
        if is_fresh(tnx_ts) and is_fresh(irx_ts) and not np.isnan(tnx_q) and not np.isnan(irx_q):
            panel.loc[tb, "T10Y3M"] = tnx_q - irx_q; m_yc.loc[tb] = True
        if is_fresh(hyg_ts) and is_fresh(lqd_ts) and not np.isnan(hyg_q) and not np.isnan(lqd_q):
            panel["HY_OAS_PROXY"] = panel.get("HY_OAS_PROXY", pd.Series(index=panel.index))
            panel.loc[tb, "HY_OAS_PROXY"] = -(hyg_q / lqd_q)  # wider credit = higher stress
        if is_fresh(bil_ts) and is_fresh(shy_ts) and not np.isnan(bil_q) and not np.isnan(shy_q):
            panel.loc[tb, "FUND"] = bil_q / shy_q; m_fund.loc[tb] = True

# Use HY_OAS, but fall back to proxy where HY_OAS is missing
hy_used = panel["HY_OAS"].combine_first(panel.get("HY_OAS_PROXY", pd.Series(index=panel.index)))

# ---- Scores (percentiles over shorter window) ----
window_days = int(PCTL_WINDOW_YEARS * 365)

# SPX drawdown on the full business-day panel
roll_max = panel["SPX"].dropna().cummax()
dd_all = 100.0 * (panel["SPX"] / roll_max - 1.0)
stress_dd = -dd_all.clip(upper=0)

inv_curve = -panel["T10Y3M"]

pct_vix   = rolling_percentile_daily(panel["VIX"], window_days)
pct_curve = rolling_percentile_daily(inv_curve, window_days)
pct_fund  = rolling_percentile_daily(panel["FUND"], window_days)
pct_dd    = rolling_percentile_daily(stress_dd, window_days)
pct_hy    = rolling_percentile_daily(hy_used, window_days)

scores_all = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
scores_all.columns = ["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]
scores_all = to_naive_date_index(scores_all)

# ---- Composite ----
start_lb = as_naive_date(today - pd.DateOffset(years=years))
scores = scores_all.loc[start_lb:].copy()
if DEFAULT_SMOOTH > 1:
    scores = scores.rolling(DEFAULT_SMOOTH, min_periods=1).mean()
    scores = to_naive_date_index(scores)

def _align(m):
    m = to_naive_date_index(m)
    return m.reindex(scores.index).astype(float).fillna(0.0)

masks = pd.concat([_align(m_vix), _align(m_hy), _align(m_yc), _align(m_fund), _align(m_dd)], axis=1)
masks.columns = ["VIX_m","HY_m","Curve_m","Fund_m","DD_m"]

W = WEIGHTS_VEC.reshape(1, -1)
X = scores[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].values
M = masks.values
active_w = (W * M).sum(axis=1)
comp = np.where(active_w > 0, 100.0 * (X * W * M).sum(axis=1) / active_w, np.nan)
comp_s = pd.Series(comp, index=scores.index, name="Composite")

if comp_s.dropna().empty:
    st.error("Composite has no valid points for the selected window.")
    st.stop()

# ---- Validity filter and canonical index ----
valid_idx = comp_s.dropna().index
comp_s = comp_s.reindex(valid_idx)

# ---- Regime helpers (fix SyntaxError source) ----
def regime_label(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= REGIME_HI:
        return "High stress"
    if x <= REGIME_LO:
        return "Low stress"
    return "Neutral"

def bday_mean(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.tail(n).mean())

latest_idx = comp_s.index[-1]
latest_val = float(comp_s.iloc[-1])
weekly_val = bday_mean(comp_s, 5)
monthly_val = bday_mean(comp_s, 21)

# ---- Clean top summary ----
st.info(f"As of {latest_idx.date()} | Weights: VIX {W_VIX:.2f}, HY {W_HY:.2f}, Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("Daily regime", regime_label(latest_val), f"{latest_val:.0f}")
c2.metric("Weekly regime", regime_label(weekly_val), f"{weekly_val:.0f}" if not pd.isna(weekly_val) else "N/A")
c3.metric("Monthly regime", regime_label(monthly_val), f"{monthly_val:.0f}" if not pd.isna(monthly_val) else "N/A")

# ---- Charts ----
# Canonical index for both panes to ensure alignment
canon_idx = valid_idx  # strictly the composite's valid business-day dates
panel_view = panel.reindex(canon_idx).ffill()  # align SPX to composite dates only

# Figure 1: Composite with regime bands
fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                     subplot_titles=("Market Stress Composite",))

fig1.add_trace(go.Scatter(x=canon_idx, y=comp_s.values, name="Composite",
                          line=dict(color="#111111", width=2)))
fig1.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)")
fig1.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)")
fig1.add_shape(type="line", x0=latest_idx, x1=latest_idx, y0=0, y1=latest_val,
               line=dict(color="#888888", width=1, dash="dot"))
fig1.update_yaxes(title="Score", range=[0,100])
fig1.update_xaxes(tickformat="%b-%d-%y", title="Date")
fig1.update_layout(template="plotly_white", height=520,
                   legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
                   margin=dict(l=60, r=40, t=60, b=60))
st.plotly_chart(fig1, use_container_width=True)

# Figure 2: SPX and Drawdown (daily) aligned to canon_idx
if not panel_view.empty and "SPX" in panel_view.columns:
    spx_series = panel_view["SPX"].dropna()
    if not spx_series.empty:
        base_val = spx_series.iloc[0]
        spx_rebased = (panel_view["SPX"] / base_val) * 100
        # Drawdown computed over the aligned window to ensure perfect alignment
        dd_series = -100.0 * (panel_view["SPX"] / panel_view["SPX"].cummax() - 1.0)

        fig2 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                             subplot_titles=("SPX and Drawdown (daily)",),
                             specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=canon_idx, y=spx_rebased.values, name="SPX (rebased=100)",
                                  line=dict(color="#7f7f7f")), row=1, col=1, secondary_y=False)
        fig2.add_trace(go.Scatter(x=canon_idx, y=dd_series.values, name="Drawdown (%)",
                                  line=dict(color="#ff7f0e")), row=1, col=1, secondary_y=True)

        # pad ranges a bit
        def pad_range(lo: float, hi: float, pad: float = 0.06):
            if pd.isna(lo) or pd.isna(hi):
                return None
            d = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1.0)
            return lo - d * pad, hi + d * pad

        lr = pad_range(float(np.nanmin(spx_rebased.values)), float(np.nanmax(spx_rebased.values)))
        rr = pad_range(float(max(0.0, np.nanmin(dd_series.values))), float(np.nanmax(dd_series.values)))
        if lr:
            fig2.update_yaxes(title="Index", range=lr, row=1, col=1, secondary_y=False)
        if rr:
            fig2.update_yaxes(title="Drawdown %", range=rr, row=1, col=1, secondary_y=True)

        fig2.update_xaxes(tickformat="%b-%d-%y", title="Date")
        fig2.update_layout(template="plotly_white", height=520,
                           legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
                           margin=dict(l=60, r=40, t=60, b=60))
        st.plotly_chart(fig2, use_container_width=True)

# ---- Download ----
with st.expander("Download Data"):
    export_idx = canon_idx
    export_scores = (100.0 * scores.reindex(export_idx).rename(columns={
        "VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"
    }))
    cols = ["VIX","HY_OAS","HY_OAS_PROXY","T10Y3M","FUND","SPX"] if "HY_OAS_PROXY" in panel.columns else ["VIX","HY_OAS","T10Y3M","FUND","SPX"]
    out = pd.concat(
        [
            panel.reindex(export_idx)[cols],
            export_scores,
            comp_s.reindex(export_idx).rename("Composite")
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
