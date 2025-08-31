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
    "fed_bs": "WALCL",   # Fed total assets (millions; weekly, H.4.1 Wednesday)
    "rrp": "RRPONTSYD",  # ON RRP (billions; daily)
    "tga": "WDTGAL",     # Treasury General Account (millions; daily)
    "effr": "EFFR",      # Effective Fed Funds Rate (percent; daily)
}
DEFAULT_SMOOTH_DAYS = 5

# Robust rebase parameters
REBASE_BASE_WINDOW = 10   # median of first 10 points
RRP_BASE_FLOOR_B   = 5.0  # billions; avoids near-zero base

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
        • WALCL: Fed total assets, H.4.1 weekly (millions).  
        • RRPONTSYD (RRP): ON RRP usage (billions).  
        • WDTGAL (TGA): Treasury cash at the Fed (millions).  
        • EFFR: effective fed funds rate (percent).

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
    align_to_walcl = st.checkbox(
        "Align chart to WALCL H.4.1 dates (avoid forward-fill artifacts)",
        value=True
    )
    st.caption("Data source: FRED via pandas-datareader")

# ---------------- Data ----------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series, start, end):
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill()
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

walcl = fred_series(FRED["fed_bs"], start_all, today)  # millions; weekly Wed
rrp    = fred_series(FRED["rrp"],    start_all, today) # billions; daily
tga    = fred_series(FRED["tga"],    start_all, today) # millions; daily
effr   = fred_series(FRED["effr"],   start_all, today) # percent; daily

# Base concat on the union of indices then ffill
df = pd.concat(
    [walcl, rrp, tga, effr],
    axis=1,
    keys=["WALCL", "RRP", "TGA", "EFFR"]
).ffill()

# Optional: align everything strictly to WALCL observation dates
if align_to_walcl and not walcl.empty:
    df = df.loc[walcl.index]  # restrict to weekly H.4.1 dates
    # Guard: drop anything beyond last WALCL date
    df = df[df.index <= walcl.index.max()]

# Ensure components exist and apply lookback
df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]
if df.empty:
    st.error("No data for selected lookback. Try a shorter window.")
    st.stop()

# Net Liquidity in billions
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]
df["TGA_b"]   = df["TGA"] / 1000.0
df["NetLiq"]  = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Smoothing
cols_to_smooth = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]
if smooth > 1:
    for col in cols_to_smooth:
        df[f"{col}_s"] = df[col].rolling(smooth, min_periods=1).mean()
else:
    for col in cols_to_smooth:
        df[f"{col}_s"] = df[col]

# Rebased panel
def rebase(series, base_window=REBASE_BASE_WINDOW, min_base=None):
    s = series.dropna()
    if s.empty:
        return series * 0 + 100
    head = s.iloc[:max(1, base_window)]
    base = head.median()
    if min_base is not None:
        base = max(base, float(min_base))
    if base == 0 or pd.isna(base):
        return series * 0 + 100
    return (series / base) * 100

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"]   = rebase(df["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df["TGA_b"])

# ---------------- Lookback change ----------------
def change_since_lookback(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    start_val = s.iloc[0]
    end_val = s.iloc[-1]
    abs_chg = end_val - start_val
    pct_chg = np.nan if start_val == 0 else (abs_chg / start_val) * 100.0
    return start_val, end_val, pct_chg

nl_start, nl_end, nl_pct = change_since_lookback(df["NetLiq"])

# ---------------- Metrics ----------------
def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.3f}%"

latest = {
    "date": df.index[-1].date(),
    "netliq": df["NetLiq"].iloc[-1],
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
    "nl_abs_chg": nl_end - nl_start if pd.notna(nl_start) else np.nan,
    "nl_pct_chg": nl_pct,
    "walcl_last_date": df["WALCL_b"].dropna().index[-1].date() if not df["WALCL_b"].dropna().empty else None,
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]), help="WALCL - RRP - TGA")
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))
m6.metric(
    f"Net Liquidity since {lookback}",
    fmt_b(latest["nl_abs_chg"]),
    delta=fmt_pct(latest["nl_pct_chg"])
)

st.caption(
    f"WALCL last observation date: {latest['walcl_last_date']}  |  Frequency: weekly (H.4.1, Wednesday)."
)

# ---------------- Charts ----------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100 at Lookback Start",
        "Effective Fed Funds Rate"
    )
)

# Row 1: Net Liquidity
fig.add_trace(
    go.Scatter(x=df.index, y=df["NetLiq_s"], name="Net Liquidity (B)", line=dict(color="#000000", width=2)),
    row=1, col=1
)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Row 2: Components rebased
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx", line=dict(color="#1f77b4")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"],   name="RRP idx",   line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"],   name="TGA idx",   line=dict(color="#d62728")), row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1, rangemode="tozero")

# Row 3: EFFR
fig.add_trace(go.Scatter(x=df.index, y=df["EFFR_s"], name="EFFR", line=dict(color="#ff7f0e")), row=3, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=3, col=1)

fig.update_layout(
    template="plotly_white",
    height=860,
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60),
)
fig.update_xaxes(tickformat="%b-%y", row=3, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Lookback Change Summary (below charts) ----------------
with st.expander("Lookback Change Summary"):
    st.markdown(
        f"""
**Lookback start date:** {df.index[0].date()}

• **Start Net Liquidity:** {fmt_b(nl_start)}  
• **End Net Liquidity:** {fmt_b(nl_end)}  
• **Absolute Change:** {fmt_b(latest["nl_abs_chg"])}  
• **Percent Change:** {fmt_pct(latest["nl_pct_chg"])}
        """
    )

# ---------------- Download ----------------
with st.expander("Download Data"):
    out = pd.DataFrame(index=df.index)
    # Export in billions for consistency
    out["WALCL_B"]    = df["WALCL_b"]
    out["RRP_B"]      = df["RRP_b"]
    out["TGA_B"]      = df["TGA_b"]
    out["EFFR_%"]     = df["EFFR"]
    out["NetLiq_B"]   = df["NetLiq"]
    out["NetLiq_s_B"] = df["NetLiq_s"]
    out.index.name = "Date"

    summary = pd.DataFrame(
        {
            "Lookback": [lookback],
            "StartDate": [df.index[0].date()],
            "Start_NetLiq_B": [nl_start],
            "End_NetLiq_B": [nl_end],
            "AbsChange_B": [latest["nl_abs_chg"]],
            "PctChange": [latest["nl_pct_chg"]],
            "AlignedToWALCL": [align_to_walcl],
        }
    )

    st.download_button(
        "Download Main CSV",
        out.to_csv(),
        file_name="liquidity_tracker.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Lookback Summary (CSV)",
        summary.to_csv(index=False),
        file_name="liquidity_tracker_lookback_summary.csv",
        mime="text/csv",
    )

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
Net Liquidity = WALCL − RRP − TGA.  
Units: WALCL and TGA are reported by FRED in **millions**; RRP is in **billions**.  
We scale WALCL and TGA by 1,000 to billions before calculating Net Liquidity.

**WALCL integrity:** WALCL is a weekly H.4.1 series with Wednesday observation dates.  
When **Align to WALCL dates** is enabled, all series are restricted to WALCL's index to avoid forward-filled distortion.  
This is usually the most accurate view if you are sanity-checking against the FRED WALCL chart.

Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base.  
For RRP, a {RRP_BASE_FLOOR_B:.0f}B floor prevents divide-by-near-zero artifacts.

Smoothing is a simple moving average over {smooth} points on the selected index.  
Percent change since lookback is computed on the **raw** Net Liquidity series:
((last − first) ÷ first) × 100. If the first value is zero, percent is N/A.
        """
    )

st.caption("© 2025 AD Fund Management LP")
