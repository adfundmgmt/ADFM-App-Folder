# market_stress_composite.py
# AD Fund Management LP
# Clean UI, fixed weights, no auto-refresh, dates aligned to SPX business days.

import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Parameters ----------------
FRED = {
    "vix": "VIXCLS",          # VIX close
    "yc_10y_3m": "T10Y3M",    # 10Y minus 3M
    "hy_oas": "BAMLH0A0HYM2", # ICE HY OAS (daily, sometimes lags)
    "cp_3m": "DCPF3M",        # 3M AA CP
    "tbill_3m": "DTB3",       # 3M T-bill
    "spx": "SP500",           # S&P 500 Index
}

LOOKBACK_YEARS = 3
PCTL_WINDOW_YEARS = 3
REGIME_HI, REGIME_LO = 70, 30

# Locked weights
W_VIX, W_HY, W_CURVE, W_FUND, W_DD = 0.25, 0.25, 0.20, 0.15, 0.15
W_VEC = np.array([W_VIX, W_HY, W_CURVE, W_FUND, W_DD], dtype=float)
W_VEC = W_VEC / W_VEC.sum()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback window", ["1y","2y","3y","5y","10y"], index=2)
    lb_years = int(lookback[:-1])
    st.subheader("Weights")
    st.caption(f"VIX {W_VIX:.2f}, HY {W_HY:.2f}, Curve {W_CURVE:.2f}, Funding {W_FUND:.2f}, Drawdown {W_DD:.2f}")
    st.markdown("---")
    st.subheader("About this tool")
    st.markdown(
        "- Purpose: a single risk dial that blends volatility, credit, curve inversion, funding, and SPX drawdown into a 0–100 score.\n"
        "- Data: daily FRED series. No intraday proxies to avoid lag and mismatched dates.\n"
        "- Method: percentile rank each factor over a 3 year rolling window, then combine with fixed weights.\n"
        "- Interpretation: above 70 High stress, below 30 Low stress."
    )

# ---------------- Helpers ----------------
def to_naive_date_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        idx = pd.to_datetime(obj.index)
        obj.index = pd.DatetimeIndex(idx.date)
    return obj

@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = pdr.DataReader(series, "fred", start, end)[series].ffill().dropna()
    return to_naive_date_index(s)

def pct_rank_last_window(arr: np.ndarray) -> float:
    # rolling apply helper, returns percentile rank of the last element within the window
    if np.isnan(arr[-1]):
        return np.nan
    x = arr[~np.isnan(arr)]
    if x.size == 0:
        return np.nan
    v = x[-1]
    # include equals to give stable plateaus
    return 100.0 * (np.count_nonzero(x <= v)) / x.size

def rolling_percentile(series: pd.Series, window_days: int) -> pd.Series:
    # window in trading days
    w = max(60, int(window_days * 252 / 365))
    return series.rolling(w, min_periods=max(20, w // 5)).apply(pct_rank_last_window, raw=True)

# ---------------- Load FRED data ----------------
today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=12)

vix_f   = fred_series(FRED["vix"], start_all, today)
hy_f    = fred_series(FRED["hy_oas"], start_all, today)
yc_f    = fred_series(FRED["yc_10y_3m"], start_all, today)
cp3m_f  = fred_series(FRED["cp_3m"], start_all, today)
tb3m_f  = fred_series(FRED["tbill_3m"], start_all, today)
spx_f   = fred_series(FRED["spx"], start_all, today)

fund_f = to_naive_date_index(cp3m_f - tb3m_f)

# ---------------- Canonical business day index keyed to SPX ----------------
# This guarantees both panes share exactly the same dates.
canon_idx = pd.bdate_range(spx_f.index.min(), spx_f.index.max())
def align(s): return to_naive_date_index(s).reindex(canon_idx).ffill()

panel = pd.DataFrame({
    "SPX": align(spx_f),
    "VIX": align(vix_f),
    "HY_OAS": align(hy_f),
    "T10Y3M": align(yc_f),
    "FUND": align(fund_f),
}).dropna(subset=["SPX"])  # SPX defines the canonical dates

# ---------------- Compute components and scores ----------------
window_days = int(PCTL_WINDOW_YEARS * 365)

roll_max = panel["SPX"].cummax()
drawdown = -100.0 * (panel["SPX"] / roll_max - 1.0)  # positive numbers for stress
inv_curve = -panel["T10Y3M"]                         # more negative curve means more stress

pct_vix   = rolling_percentile(panel["VIX"], window_days)
pct_hy    = rolling_percentile(panel["HY_OAS"], window_days)
pct_curve = rolling_percentile(inv_curve, window_days)
pct_fund  = rolling_percentile(panel["FUND"], window_days)
pct_dd    = rolling_percentile(drawdown, window_days)

scores = pd.concat(
    [pct_vix, pct_hy, pct_curve, pct_fund, pct_dd],
    axis=1
).rename(columns={0:"VIX_p",1:"HY_p",2:"CurveInv_p",3:"Fund_p",4:"DD_p"})

# Limit to lookback and align to SPX dates only
start_lb = today - pd.DateOffset(years=lb_years)
scores = scores.loc[scores.index >= start_lb]
panel_view = panel.reindex(scores.index)

# Composite, mask out rows where a component is missing
mask = scores.notna().astype(float).values
w = W_VEC.reshape(1, -1)
active_w = (w * mask).sum(axis=1)
X = (scores[["VIX_p","HY_p","CurveInv_p","Fund_p","DD_p"]].fillna(0.0).values)
comp_vals = np.where(active_w > 0, (X * w * mask).sum(axis=1) / active_w, np.nan)
comp = pd.Series(comp_vals, index=scores.index, name="Composite")

# ---------------- Regimes summary ----------------
def tag(x):
    if pd.isna(x): return "N/A"
    if x >= REGIME_HI: return "High stress"
    if x <= REGIME_LO: return "Low stress"
    return "Neutral"

latest_idx = comp.dropna().index[-1]
latest_val = float(comp.loc[latest_idx])
weekly_val = float(comp.tail(5).mean())
monthly_val = float(comp.tail(21).mean())

st.info(f"As of {latest_idx.date()}  |  Daily {tag(latest_val)} {latest_val:.0f}  |  Weekly {tag(weekly_val)} {weekly_val:.0f}  |  Monthly {tag(monthly_val)} {monthly_val:.0f}")

# ---------------- Charts ----------------
# 1) Composite with regime bands
fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=("Market Stress Composite",))
fig1.add_trace(go.Scatter(x=comp.index, y=comp, name="Composite", line=dict(color="#111111", width=2)))
fig1.add_hrect(y0=REGIME_HI, y1=100, line_width=0, fillcolor="rgba(214,39,40,0.10)")
fig1.add_hrect(y0=0, y1=REGIME_LO, line_width=0, fillcolor="rgba(44,160,44,0.10)")
fig1.update_yaxes(title="Score", range=[0,100])
fig1.update_xaxes(title="Date", tickformat="%b-%d-%y")
fig1.update_layout(template="plotly_white", height=520, margin=dict(l=60, r=40, t=60, b=60),
                   legend=dict(orientation="h", x=0, y=1.08, xanchor="left"))
st.plotly_chart(fig1, use_container_width=True)

# 2) SPX rebased + drawdown on the same canonical dates
spx_rebased = (panel_view["SPX"] / panel_view["SPX"].iloc[0]) * 100.0
dd_series = -100.0 * (panel_view["SPX"] / panel_view["SPX"].cummax() - 1.0)

fig2 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                     subplot_titles=("SPX and Drawdown (daily)",))
fig2.add_trace(go.Scatter(x=spx_rebased.index, y=spx_rebased, name="SPX (rebased=100)", line=dict(color="#7f7f7f")), secondary_y=False)
fig2.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name="Drawdown (%)", line=dict(color="#ff7f0e")), secondary_y=True)
fig2.update_yaxes(title="Index", secondary_y=False)
fig2.update_yaxes(title="Drawdown %", secondary_y=True)
fig2.update_xaxes(title="Date", tickformat="%b-%d-%y")
fig2.update_layout(template="plotly_white", height=520, margin=dict(l=60, r=40, t=60, b=60),
                   legend=dict(orientation="h", x=0, y=1.08, xanchor="left"))
st.plotly_chart(fig2, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download data"):
    out = pd.concat(
        [
            panel_view[["VIX","HY_OAS","T10Y3M","FUND","SPX"]],
            scores.rename(columns={
                "VIX_p":"VIX_pct","HY_p":"HY_pct","CurveInv_p":"CurveInv_pct","Fund_p":"Fund_pct","DD_p":"DD_pct"
            }),
            comp.rename("Composite"),
        ],
        axis=1
    )
    out.index.name = "Date"
    st.download_button("Download CSV", out.to_csv(), file_name="market_stress_composite.csv", mime="text/csv")

st.caption("© 2025 AD Fund Management LP")
