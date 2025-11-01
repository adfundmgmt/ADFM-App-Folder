Here’s the full revised Streamlit app code you can paste in directly:

```python
############################################################
# Market Stress Composite — trading-day aligned, hardened
# AD Fund Management LP
# Fixes:
# - True trading-day alignment using SPX dates only
# - Robust daily regime with intraday nowcast for VIX, Curve, SPX, HY proxy
# - No bad FUND proxy; funding omitted intraday if stale
# - Percentiles computed on trading-day windows with NaN-safe math
# - Clear masks and active-weight renormalization
# - Diagnostics pane expanded for verification
############################################################

# Py3.12 compat shim for pandas_datareader
import sys, types
import packaging.version
try:
    import distutils.version
except Exception:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = packaging.version.Version
    _d.version = _dv
    sys.modules.update({"distutils": _d, "distutils.version": _dv})

import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Data maps -----------------
FRED = {
    "vix": "VIXCLS",           # VIX close (daily)
    "yc_10y_3m": "T10Y3M",     # 10Y minus 3M (daily)
    "hy_oas": "BAMLH0A0HYM2",  # ICE HY OAS (daily, 1-3 day lag)
    "cp_3m": "DCPF3M",         # 3M AA CP
    "tbill_3m": "DTB3",        # 3M T-bill
    "spx": "SP500",            # S&P 500 Index
}

# Intraday quotes for nowcast
PX = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "TNX": "^TNX",   # 10y yield *10
    "IRX": "^IRX",   # 13w T-bill *10
    "HYG": "HYG",
    "LQD": "LQD",
}

# ---------------- Parameters ----------------
DEFAULT_LOOKBACK = "3y"
PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1
REGIME_HI, REGIME_LO = 70, 30
MAX_STALE_BDAYS = 0
MAX_PROXY_AGE_MIN = 3

# Locked weights
W_VIX, W_HY, W_CURVE, W_FUND, W_DD = 0.25, 0.25, 0.20, 0.15, 0.15
WEIGHTS_VEC = np.array([W_VIX, W_HY, W_CURVE, W_FUND, W_DD], dtype=float)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

# ---------------- Sidebar -------------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"],
                            index=["1y", "2y", "3y", "5y", "10y"].index(DEFAULT_LOOKBACK))
    years = int(lookback[:-1])

    st.subheader("Nowcast")
    use_nowcast = st.checkbox("Use intraday proxies to update today", value=True)

    st.subheader("Weights (fixed)")
    st.caption(f"VIX {W_VIX:.2f}, HY {W_HY:.2f}, Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}")

    st.markdown("---")
    st.subheader("About this tool")
    st.markdown(
        "- Purpose: a single risk dial blending volatility, credit, curve inversion, funding, and SPX drawdown into a 0-100 score.\n"
        "- Data: FRED for history, intraday quotes only adjust today's row.\n"
        "- Method: 3-year rolling percentiles by factor, fixed weights.\n"
        "- Alignment: all factors evaluated on SPX trading days only.\n"
        "- Note: when a factor is stale today, it is dropped and weights re-normalize across active factors."
    )

# ---------------- Time helpers --------------
NY_TZ = "America/New_York"

def today_naive() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz=NY_TZ).date())

def to_naive_date_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
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

# ---------------- Loaders -------------------
@st.cache_data(ttl=900, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return to_naive_date_index(s.ffill().dropna())
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
        if data:
            return pd.concat(data, axis=1)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def latest_quote(df_1m: pd.DataFrame, symbol: str):
    if df_1m.empty or symbol not in df_1m.columns:
        return np.nan, None
    s = df_1m[symbol].dropna()
    if s.empty:
        return np.nan, None
    return float(s.iloc[-1]), s.index[-1]

# Rolling percentile on trading days, NaN-safe
def rolling_percentile_trading(values: pd.Series, idx: pd.DatetimeIndex, window_trading_days: int) -> pd.Series:
    s = to_naive_date_index(values).reindex(idx)
    def _last_percentile(x):
        arr = pd.Series(x).dropna()
        if arr.empty:
            return np.nan
        return (arr.rank(pct=True).iloc[-1])
    out = s.rolling(window_trading_days, min_periods=max(40, window_trading_days // 6)).apply(_last_percentile, raw=False)
    return to_naive_date_index(out)

# ---------------- Load history --------------
today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f)  # higher = tighter

if all(s.dropna().empty for s in [vix_f, hy_f, yc_f, fund_f, spx_f]):
    st.error("FRED series unavailable. Check network.")
    st.stop()

# Build BD index and align
bidx_start = min(s.index.min() for s in [vix_f, hy_f, yc_f, fund_f, spx_f] if not s.dropna().empty)
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

# ---------------- Freshness masks -----------
def fresh_mask(original: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    original = to_naive_date_index(original).reindex(idx)
    pos = np.arange(len(idx))
    has = original.notna().values
    last_pos = pd.Series(np.where(has, pos, np.nan), index=idx).ffill().values
    age = pos - last_pos
    return pd.Series((~np.isnan(age)) & (age <= MAX_STALE_BDAYS), index=idx)

m_vix  = fresh_mask(vix_f, panel.index)
m_hy   = fresh_mask(hy_f,  panel.index)
m_yc   = fresh_mask(yc_f,  panel.index)
m_fund = fresh_mask(fund_f, panel.index)
m_dd   = fresh_mask(spx_f, panel.index)

# ---------------- Intraday nowcast ----------
hy_proxy_today = np.nan
hy_proxy_ts = None

if use_nowcast:
    px_last = yf_last_minute(list(PX.values()))
    if not px_last.empty:
        tb = today_naive()

        vix_q, vix_ts = latest_quote(px_last, PX["VIX"])
        spx_q, spx_ts = latest_quote(px_last, PX["SPX"])
        tnx_q, tnx_ts = latest_quote(px_last, PX["TNX"])
        irx_q, irx_ts = latest_quote(px_last, PX["IRX"])
        hyg_q, hyg_ts = latest_quote(px_last, PX["HYG"])
        lqd_q, lqd_ts = latest_quote(px_last, PX["LQD"])

        # Update direct series if fresh
        if is_fresh(vix_ts) and not np.isnan(vix_q):
            panel.loc[tb, "VIX"] = vix_q
            m_vix.loc[tb] = True
        if is_fresh(spx_ts) and not np.isnan(spx_q):
            panel.loc[tb, "SPX"] = spx_q
            m_dd.loc[tb] = True
        if is_fresh(tnx_ts) and is_fresh(irx_ts) and not np.isnan(tnx_q) and not np.isnan(irx_q):
            panel.loc[tb, "T10Y3M"] = tnx_q - irx_q
            m_yc.loc[tb] = True

        # HY proxy: use -log(HYG/LQD) so wider credit -> higher number (matches OAS direction)
        if is_fresh(hyg_ts) and is_fresh(lqd_ts) and not np.isnan(hyg_q) and not np.isnan(lqd_q):
            hy_proxy_today = -np.log(hyg_q / lqd_q)
            hy_proxy_ts = hyg_ts if hyg_ts > lqd_ts else lqd_ts
            if "HY_OAS_PROXY" not in panel.columns:
                panel["HY_OAS_PROXY"] = np.nan
            panel.loc[tb, "HY_OAS_PROXY"] = hy_proxy_today  # store for diagnostics

# ---------------- Trading-day index ---------
trade_idx = panel.index[panel["SPX"].notna()].unique()
panel_tr = panel.reindex(trade_idx).ffill()

# Choose HY series: use official OAS where available, else proxy percentile for today only
hy_series_full = panel_tr["HY_OAS"].copy()

# Masks on trading days
m_vix_tr  = m_vix.reindex(trade_idx).fillna(False)
m_yc_tr   = m_yc.reindex(trade_idx).fillna(False)
m_fund_tr = m_fund.reindex(trade_idx).fillna(False)
m_dd_tr   = m_dd.reindex(trade_idx).fillna(False)

# ---------------- Percentiles ----------------
window_td = int(PCTL_WINDOW_YEARS * 252)
inv_curve = -panel_tr["T10Y3M"]                             # more inversion -> higher stress
roll_max = panel_tr["SPX"].dropna().cummax()
dd_all = 100.0 * (panel_tr["SPX"] / roll_max - 1.0)
stress_dd = -dd_all.clip(upper=0)                           # 0 when at highs, positive when drawn down

pct_vix   = rolling_percentile_trading(panel_tr["VIX"],   trade_idx, window_td)
pct_hy    = rolling_percentile_trading(hy_series_full,    trade_idx, window_td)
pct_curve = rolling_percentile_trading(inv_curve,         trade_idx, window_td)
pct_fund  = rolling_percentile_trading(panel_tr["FUND"],  trade_idx, window_td)
pct_dd    = rolling_percentile_trading(stress_dd,         trade_idx, window_td)

# If HY FRED is stale today but we have a proxy, override today's HY percentile with proxy percentile calculated on its own history
if use_nowcast and pd.notna(hy_proxy_today):
    try:
        px_hist = yf.download(["HYG", "LQD"], period=f"{PCTL_WINDOW_YEARS+1}y", interval="1d", auto_adjust=False, progress=False)["Close"]
    except Exception:
        px_hist = pd.DataFrame()
    if isinstance(px_hist, pd.Series):
        px_hist = px_hist.to_frame()
    if not px_hist.empty and "HYG" in px_hist.columns and "LQD" in px_hist.columns:
        px_hist = to_naive_date_index(px_hist.reindex(trade_idx)).ffill()
        proxy_hist = -np.log(px_hist["HYG"] / px_hist["LQD"])
        pct_proxy = rolling_percentile_trading(proxy_hist, trade_idx, window_td)
        today_idx = trade_idx[-1]
        if not m_hy.reindex(trade_idx).iloc[-1] and pd.notna(pct_proxy.loc[today_idx]):
            pct_hy.loc[today_idx] = pct_proxy.loc[today_idx]
            m_hy_tr = m_hy.reindex(trade_idx).fillna(False)
            m_hy_tr.iloc[-1] = True
        else:
            m_hy_tr = m_hy.reindex(trade_idx).fillna(False)
    else:
        m_hy_tr = m_hy.reindex(trade_idx).fillna(False)
else:
    m_hy_tr = m_hy.reindex(trade_idx).fillna(False)

scores_all = pd.concat([pct_vix, pct_hy, pct_curve, pct_fund, pct_dd], axis=1)
scores_all.columns = ["VIX_p", "HY_p", "CurveInv_p", "Fund_p", "DD_p"]
scores_all = to_naive_date_index(scores_all)

# ---------------- Composite -----------------
start_lb = as_naive_date(today - pd.DateOffset(years=years))
scores = scores_all.loc[start_lb:].copy()
if DEFAULT_SMOOTH > 1:
    scores = to_naive_date_index(scores.rolling(DEFAULT_SMOOTH, min_periods=1).mean())

def _align(m):
    return to_naive_date_index(m).reindex(scores.index).astype(float).fillna(0.0)

masks = pd.concat([_align(m_vix_tr), _align(m_hy_tr), _align(m_yc_tr), _align(m_fund_tr), _align(m_dd_tr)], axis=1)
masks.columns = ["VIX_m", "HY_m", "Curve_m", "Fund_m", "DD_m"]

W = WEIGHTS_VEC.reshape(1, -1)
X = scores[["VIX_p", "HY_p", "CurveInv_p", "Fund_p", "DD_p"]].values  # 0..1
M = masks.values  # 0/1

# NaN-safe weighted average with active-weight renormalization
X_masked = np.nan_to_num(X) * M
weight_mask = (W * M)
active_w = weight_mask.sum(axis=1)
num = (X_masked * W).sum(axis=1)
comp = np.where(active_w > 0, 100.0 * num / active_w, np.nan)
comp_s = pd.Series(comp, index=scores.index, name="Composite").dropna()

if comp_s.empty:
    st.error("Composite has no valid points for the selected window.")
    st.stop()

# ---------------- Helpers -------------------
def regime_label(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= REGIME_HI:
        return "High stress"
    if x <= REGIME_LO:
        return "Low stress"
    return "Neutral"

def mean_last(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.tail(n).mean())

latest_idx = comp_s.index[-1]
latest_val = float(comp_s.iloc[-1])
weekly_val = mean_last(comp_s, 5)
monthly_val = mean_last(comp_s, 21)

# ---------------- Header --------------------
st.info(
    f"As of {latest_idx.date()} | Weights: VIX {W_VIX:.2f}, HY {W_HY:.2f}, "
    f"Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}"
)

c1, c2, c3 = st.columns(3)
c1.metric("Daily regime", regime_label(latest_val), f"{latest_val:.0f}")
c2.metric("Weekly regime", regime_label(weekly_val), f"{weekly_val:.0f}" if not pd.isna(weekly_val) else "N/A")
c3.metric("Monthly regime", regime_label(monthly_val), f"{monthly_val:.0f}" if not pd.isna(monthly_val) else "N/A")

# ---------------- Charts --------------------
canon_idx = comp_s.index
panel_view = panel_tr.reindex(canon_idx).ffill()

fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                     subplot_titles=("Market Stress Composite",))
fig1.add_trace(go.Scatter(x=canon_idx, y=comp_s.values, name="Composite",
                          line=dict(color="#111111", width=2)))
fig1.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)")
fig1.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)")
fig1.add_shape(type="line", x0=latest_idx, x1=latest_idx, y0=0, y1=latest_val,
               line=dict(color="#888888", width=1, dash="dot"))
fig1.update_yaxes(title="Score", range=[0, 100])
fig1.update_xaxes(tickformat="%b-%d-%y", title="Date")
fig1.update_layout(template="plotly_white", height=520,
                   legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
                   margin=dict(l=60, r=40, t=60, b=60))
st.plotly_chart(fig1, use_container_width=True)

# SPX and drawdown panel
if not panel_view.empty and "SPX" in panel_view.columns:
    spx_series = panel_view["SPX"].dropna()
    if not spx_series.empty:
        base_val = spx_series.iloc[0]
        spx_rebased = (panel_view["SPX"] / base_val) * 100.0
        dd_series = -100.0 * (panel_view["SPX"] / panel_view["SPX"].cummax() - 1.0)

        fig2 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                             subplot_titles=("SPX and Drawdown (daily)",),
                             specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=canon_idx, y=spx_rebased.values, name="SPX (rebased=100)",
                                  line=dict(color="#7f7f7f")), row=1, col=1, secondary_y=False)
        fig2.add_trace(go.Scatter(x=canon_idx, y=dd_series.values, name="Drawdown (%)",
                                  line=dict(color="#ff7f0e")), row=1, col=1, secondary_y=True)
        fig2.update_xaxes(tickformat="%b-%d-%y", title="Date")
        fig2.update_layout(template="plotly_white", height=520,
                           legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
                           margin=dict(l=60, r=40, t=60, b=60))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics"):
    latest = canon_idx[-1]
    diag = pd.DataFrame({
        "raw_VIX": [panel_view.loc[latest, "VIX"]],
        "raw_HY_OAS": [panel_view.get("HY_OAS", pd.Series(index=canon_idx)).reindex(canon_idx).loc[latest]],
        "raw_HY_proxy(-log(HYG/LQD))": [panel.get("HY_OAS_PROXY", pd.Series(index=panel.index)).reindex(canon_idx).loc[latest] if "HY_OAS_PROXY" in panel.columns else np.nan],
        "raw_curve_inv(-T10Y3M)": [(-panel_view["T10Y3M"]).loc[latest]],
        "raw_fund(CP-TBill)": [panel_view.loc[latest, "FUND"]],
        "raw_dd_stress": [(-100.0 * (panel_view["SPX"]/panel_view["SPX"].cummax() - 1.0)).loc[latest]],
        "pct_VIX": [scores_all.reindex(canon_idx).loc[latest, "VIX_p"]],
        "pct_HY": [scores_all.reindex(canon_idx).loc[latest, "HY_p"]],
        "pct_CurveInv": [scores_all.reindex(canon_idx).loc[latest, "CurveInv_p"]],
        "pct_Fund": [scores_all.reindex(canon_idx).loc[latest, "Fund_p"]],
        "pct_DD": [scores_all.reindex(canon_idx).loc[latest, "DD_p"]],
        "mask_VIX": [bool(masks.iloc[-1,0])],
        "mask_HY": [bool(masks.iloc[-1,1])],
        "mask_Curve": [bool(masks.iloc[-1,2])],
        "mask_Fund": [bool(masks.iloc[-1,3])],
        "mask_DD": [bool(masks.iloc[-1,4])],
        "active_weight_sum": [float((WEIGHTS_VEC * masks.iloc[-1].values).sum())],
        "Composite": [latest_val],
    })
    st.dataframe(diag.T, use_container_width=True)

# ---------------- Download ------------------
with st.expander("Download Data"):
    export_idx = canon_idx
    export_scores = (100.0 * scores.reindex(export_idx).rename(columns={
        "VIX_p": "VIX_pct", "HY_p": "HY_pct", "CurveInv_p": "CurveInv_pct",
        "Fund_p": "Fund_pct", "DD_p": "DD_pct"
    }))
    cols = ["VIX", "HY_OAS", "HY_OAS_PROXY", "T10Y3M", "FUND", "SPX"] if "HY_OAS_PROXY" in panel.columns else ["VIX", "HY_OAS", "T10Y3M", "FUND", "SPX"]
    out = pd.concat(
        [panel_tr.reindex(export_idx)[cols], export_scores, comp_s.reindex(export_idx).rename("Composite")],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
```
