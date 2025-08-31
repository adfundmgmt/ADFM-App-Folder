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
    "fed_bs": "WALCL",   # Fed total assets (millions on FRED)
    "rrp": "RRPONTSYD",  # ON RRP (billions on FRED)
    "tga": "WDTGAL",     # Treasury General Account (millions on FRED)
    "effr": "EFFR",      # Effective Fed Funds Rate (percent)
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
        • WALCL: Federal Reserve total assets on the consolidated balance sheet (millions on FRED).  
        • RRPONTSYD (RRP): overnight reverse repo facility usage by counterparties (billions on FRED).  
        • WDTGAL (TGA): U.S. Treasury cash balance at the Fed (millions on FRED).  
        • EFFR: effective federal funds rate (volume-weighted overnight rate).

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
def fred_series(series, start, end):
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill()
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

fed_bs = fred_series(FRED["fed_bs"], start_all, today)  # millions
rrp    = fred_series(FRED["rrp"],    start_all, today)  # billions
tga    = fred_series(FRED["tga"],    start_all, today)  # millions
effr   = fred_series(FRED["effr"],   start_all, today)  # percent

# Align and subset
df = pd.concat(
    [fed_bs, rrp, tga, effr],
    axis=1,
    keys=["WALCL", "RRP", "TGA", "EFFR"]
).ffill()

# Ensure all components exist
df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]
if df.empty:
    st.error("No data for selected lookback. Try a shorter window.")
    st.stop()

# Net Liquidity in billions
# WALCL and TGA are millions -> /1000; RRP already billions -> no scaling
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

# Rebased panel (robust against near-zero bases)
def rebase(series, base_window=REBASE_BASE_WINDOW, min_base=None):
    s = series.copy()
    if s.isna().all():
        return s * 0 + 100
    head = s.dropna().iloc[:max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]
    if min_base is not None:
        base = max(base, float(min_base))
    if pd.isna(base) or base == 0:
        return s * 0 + 100
    return (s / base) * 100

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"]   = rebase(df["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df["TGA_b"])

# ---------------- Helper: lookback change ----------------
def change_since_lookback(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    start_val = s.iloc[0]
    end_val = s.iloc[-1]
    abs_chg = end_val - start_val
    pct_chg = np.nan
    if pd.notna(start_val) and start_val != 0:
        pct_chg = (abs_chg / start_val) * 100.0
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

# Optional audit table for the change calculation
with st.expander("Change audit for selected lookback"):
    audit = pd.DataFrame(
        {
            "Start": [nl_start],
            "End": [nl_end],
            "Abs Change (B)": [latest["nl_abs_chg"]],
            "Percent Change": [latest["nl_pct_chg"]],
        },
        index=[df.index[0].date()]
    )
    audit.index.name = "Lookback start date"
    st.dataframe(audit.style.format({"Start": "{:,.0f}", "End": "{:,.0f}", "Abs Change (B)": "{:,.0f}", "Percent Change": "{:.3f}%"}))

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

# ---------------- Download ----------------
with st.expander("Download Data"):
    out = pd.DataFrame(index=df.index)
    # Export in billions for consistency
    out["WALCL_B"]      = df["WALCL_b"]
    out["RRP_B"]        = df["RRP_b"]
    out["TGA_B"]        = df["TGA_b"]
    out["EFFR_%"]       = df["EFFR"]
    out["NetLiq_B"]     = df["NetLiq"]
    out["NetLiq_s_B"]   = df["NetLiq_s"]
    # Append single-row lookback summary at the end for auditing
    summary = pd.DataFrame(
        {
            "Lookback": [lookback],
            "StartDate": [df.index[0].date()],
            "Start_NetLiq_B": [nl_start],
            "End_NetLiq_B": [nl_end],
            "AbsChange_B": [latest["nl_abs_chg"]],
            "PctChange": [latest["nl_pct_chg"]],
        }
    )
    st.download_button(
        "Download CSV",
        pd.concat([out,], axis=1).to_csv(),
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
        Units: WALCL and TGA are reported by FRED in millions; RRP is reported in billions.  
        All panels display values in billions after scaling.

        Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base.  
        For RRP, a {RRP_BASE_FLOOR_B:.0f}B floor is applied to avoid divide-by-near-zero artifacts at regimes where usage was ~0.

        Smoothing applies a simple moving average over {smooth} days.  
        Percent change since lookback is computed on the raw Net Liquidity series:
        ((last − first) ÷ first) × 100. If the first value is zero, percent is reported as N/A.

        Note: RRP daily series begins in 2013. Long lookbacks truncate to available history.
        """
    )

st.caption("© 2025 AD Fund Management LP")
