############################################################
# Liquidity & Fed Policy Tracker - clean metrics, no deltas
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"
FRED = {
    "fed_bs": "WALCL",   # Fed total assets (millions on FRED; weekly)
    "rrp": "RRPONTSYD",  # ON RRP (billions on FRED; daily)
    "tga": "WDTGAL",     # Treasury General Account (millions on FRED; daily)
    "effr": "EFFR",      # Effective Fed Funds Rate (percent; daily)
}
DEFAULT_SMOOTH_DAYS = 5

# Robust rebase parameters
REBASE_BASE_WINDOW = 10   # use median of first 10 points
RRP_BASE_FLOOR_B   = 5.0  # billions; prevents divide-by-near-zero

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Goal: track Net Liquidity and relate it to policy.

        Definition: Net Liquidity = WALCL − RRP − TGA  
        Units are billions. Sustained rises often coincide with risk-on conditions.

        Series  
        • WALCL: Federal Reserve total assets (millions; weekly H.4.1).  
        • RRPONTSYD (RRP): ON RRP usage (billions; daily).  
        • WDTGAL (TGA): Treasury cash at the Fed (millions; daily).  
        • EFFR: effective fed funds rate (percent; daily).

        Panels  
        1) Net Liquidity (billions)  
        2) Components rebased to 100 at the lookback start  
        3) EFFR
        """
    )
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "10y"], index=2)
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 30, DEFAULT_SMOOTH_DAYS, 1)
    st.caption("Data source: FRED via pandas-datareader")

# ---------------- Data ----------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series_raw(series, start, end):
    """Raw series from FRED without forward-fill (for integrity)."""
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s  # <-- NO ffill here
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series_ffill(series, start, end):
    """Forward-filled copy for chart continuity only."""
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill()
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb  = today - pd.DateOffset(years=years)

# Raw for metrics (no ffill)
walcl_raw = fred_series_raw(FRED["fed_bs"], start_all, today)   # millions; weekly
rrp_raw   = fred_series_raw(FRED["rrp"],    start_all, today)   # billions; daily
tga_raw   = fred_series_raw(FRED["tga"],    start_all, today)   # millions; daily
effr_raw  = fred_series_raw(FRED["effr"],   start_all, today)   # percent; daily

# Guard required series
if walcl_raw.empty or rrp_raw.empty or tga_raw.empty:
    st.error("Missing required series from FRED. (WALCL/RRPONTSYD/WDTGAL)")
    st.stop()

# Integrity: compute only on exact same-date intersection (no ffill)
df_exact = pd.concat([walcl_raw, rrp_raw, tga_raw], axis=1, join="inner")
df_exact.columns = ["WALCL_m", "RRP_b", "TGA_m"]
# Apply lookback
df_exact = df_exact[df_exact.index >= start_lb]
if df_exact.empty:
    st.error("No overlapping observations within the selected lookback.")
    st.stop()

# Convert to billions and compute Net Liquidity
df_exact["WALCL_b"] = df_exact["WALCL_m"] / 1000.0
df_exact["TGA_b"]   = df_exact["TGA_m"] / 1000.0
df_exact["NetLiq"]  = df_exact["WALCL_b"] - df_exact["RRP_b"] - df_exact["TGA_b"]

# Latest (integrity)
latest_date = df_exact.index[-1]
latest = {
    "date": latest_date.date(),
    "netliq": df_exact["NetLiq"].iloc[-1],
    "walcl": df_exact["WALCL_b"].iloc[-1],
    "rrp":   df_exact["RRP_b"].iloc[-1],
    "tga":   df_exact["TGA_b"].iloc[-1],
    # EFFR at or before latest_date (no future-peeking)
    "effr":  effr_raw.loc[:latest_date].iloc[-1] if not effr_raw.loc[:latest_date].empty else np.nan,
}

# Chart data: forward-filled for continuity (does not affect metrics)
walcl_ff = fred_series_ffill(FRED["fed_bs"], start_all, today)
rrp_ff   = fred_series_ffill(FRED["rrp"],    start_all, today)
tga_ff   = fred_series_ffill(FRED["tga"],    start_all, today)
effr_ff  = fred_series_ffill(FRED["effr"],   start_all, today)

df_chart = pd.concat([walcl_ff, rrp_ff, tga_ff, effr_ff], axis=1).ffill()
df_chart.columns = ["WALCL_m", "RRP_b", "TGA_m", "EFFR"]
df_chart = df_chart[df_chart.index >= start_lb]

# Scale + Net Liquidity for charts
df_chart["WALCL_b"] = df_chart["WALCL_m"] / 1000.0
df_chart["TGA_b"]   = df_chart["TGA_m"] / 1000.0
df_chart["NetLiq"]  = df_chart["WALCL_b"] - df_chart["RRP_b"] - df_chart["TGA_b"]

# Smoothing (charts only)
cols_to_smooth = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]
if smooth > 1:
    for col in cols_to_smooth:
        df_chart[f"{col}_s"] = df_chart[col].rolling(smooth, min_periods=1).mean()
else:
    for col in cols_to_smooth:
        df_chart[f"{col}_s"] = df_chart[col]

# Rebased panel (robust against near-zero bases) — charts only
def rebase(series, base_window=REBASE_BASE_WINDOW, min_base=None):
    s = series.dropna()
    if s.empty:
        return series * 0 + 100
    head = s.iloc[:max(1, base_window)]
    base = head.median()
    if min_base is not None:
        base = max(base, float(min_base))
    if pd.isna(base) or base == 0:
        return series * 0 + 100
    return (series / base) * 100

reb = pd.DataFrame(index=df_chart.index)
reb["WALCL_idx"] = rebase(df_chart["WALCL_b"])
reb["RRP_idx"]   = rebase(df_chart["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df_chart["TGA_b"])

# ---------------- Metrics (integrity-driven) ----------------
def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]), help="WALCL - RRP - TGA (same-date only)")
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))

# ---------------- Charts ----------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100 at Lookback Start",
        "Effective Fed Funds Rate"
    )
)

# Row 1: Net Liquidity (charts use smoothed/ffill version)
fig.add_trace(
    go.Scatter(x=df_chart.index, y=df_chart["NetLiq_s"], name="Net Liquidity (B)", line=dict(color="#000000", width=2)),
    row=1, col=1
)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Row 2: Components rebased
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx", line=dict(color="#1f77b4")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"],   name="RRP idx",   line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"],   name="TGA idx",   line=dict(color="#d62728")), row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1, rangemode="tozero")

# Row 3: EFFR
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["EFFR_s"], name="EFFR", line=dict(color="#ff7f0e")), row=3, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=3, col=1)

fig.update_layout(
    template="plotly_white",
    height=820,
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60),
)
fig.update_xaxes(tickformat="%b-%y", row=3, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    # Headline integrity series (same-date only)
    out = pd.DataFrame(index=df_exact.index)
    out["WALCL_B"]  = df_exact["WALCL_b"]
    out["RRP_B"]    = df_exact["RRP_b"]
    out["TGA_B"]    = df_exact["TGA_b"]
    out["NetLiq_B"] = df_exact["NetLiq"]
    out.index.name = "Date"

    st.download_button(
        "Download CSV (headline integrity series)",
        out.to_csv(),
        file_name="liquidity_tracker.csv",
        mime="text/csv",
    )

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
**Accuracy:** Headline metrics use only exact same-date observations across WALCL (weekly), RRP (daily), and TGA (daily).  
No forward-fill is used in the calculation of Net Liquidity.

Charts may use forward-filled series solely for visual continuity; they do not affect the headline values.

Net Liquidity = WALCL − RRP − TGA.  
Units: WALCL and TGA are reported by FRED in **millions** (divided by 1,000); RRP is **billions**.

Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base.  
For RRP, a {RRP_BASE_FLOOR_B:.0f}B floor avoids near-zero base artifacts.  
Smoothing applies a simple moving average over {smooth} points for charts only.
        """
    )

st.caption("© 2025 AD Fund Management LP")
