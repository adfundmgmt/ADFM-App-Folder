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
    "fed_bs": "WALCL",   # Fed total assets (millions on FRED, weekly H.4.1 Wednesday)
    "rrp": "RRPONTSYD",  # ON RRP (billions on FRED, daily)
    "tga": "WDTGAL",     # Treasury General Account (millions on FRED, daily)
    "effr": "EFFR",      # Effective Fed Funds Rate (percent, daily)
}
DEFAULT_SMOOTH_DAYS = 5
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B   = 5.0

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Goal: track Net Liquidity and relate it to policy.

        Definition: Net Liquidity = WALCL − RRP − TGA  
        Units are billions.

        Series  
        • WALCL: Fed total assets, weekly H.4.1 (millions).  
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
    st.caption("Data source: FRED via pandas-datareader")

# ---------------- Data ----------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series, start, end, ffill=False):
    """
    Fetch a single FRED series. Optionally forward-fill for chart continuity,
    but keep raw series without ffill for integrity calculations.
    """
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill() if ffill else s
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

# Raw series (no forward fill) for integrity and headline metrics
walcl_raw = fred_series(FRED["fed_bs"], start_all, today, ffill=False)  # millions; weekly
rrp_raw   = fred_series(FRED["rrp"],    start_all, today, ffill=False)  # billions; daily
tga_raw   = fred_series(FRED["tga"],    start_all, today, ffill=False)  # millions; daily
effr_raw  = fred_series(FRED["effr"],   start_all, today, ffill=False)  # percent; daily

# Guard rails
if walcl_raw.empty or rrp_raw.empty or tga_raw.empty:
    st.error("Missing required series from FRED. Try again later.")
    st.stop()

# Build integrity-aligned Net Liquidity series:
# Use only exact dates where all three have observations. No forward fill.
df_exact = pd.concat(
    [walcl_raw, rrp_raw, tga_raw],
    axis=1, join="inner"
)
df_exact.columns = ["WALCL_m", "RRP_b", "TGA_m"]

# Convert to billions and compute Net Liquidity
df_exact["WALCL_b"] = df_exact["WALCL_m"] / 1000.0
df_exact["TGA_b"]   = df_exact["TGA_m"] / 1000.0
df_exact["NetLiq"]  = df_exact["WALCL_b"] - df_exact["RRP_b"] - df_exact["TGA_b"]

# Apply lookback to exact-date series
df_exact_lb = df_exact[df_exact.index >= start_lb]
if df_exact_lb.empty:
    st.error("No overlapping WALCL, RRP, and TGA observations within the selected lookback.")
    st.stop()

# Latest exact-date point used for headline
latest_common_date = df_exact.index.max()
latest_exact = df_exact.loc[latest_common_date]

# Separate chart-friendly dataframe with optional ffill for smoother visuals.
# This does not affect headline metrics.
walcl_ff = walcl_raw.ffill()
rrp_ff   = rrp_raw.ffill()
tga_ff   = tga_raw.ffill()
effr_ff  = effr_raw.ffill()

df_chart = pd.concat([walcl_ff, rrp_ff, tga_ff, effr_ff], axis=1, join="outer")
df_chart.columns = ["WALCL_m", "RRP_b", "TGA_m", "EFFR"]
df_chart = df_chart.sort_index()
df_chart_lb = df_chart[df_chart.index >= start_lb].copy()

# Scale for charts and compute Net Liquidity for charts only
df_chart_lb["WALCL_b"] = df_chart_lb["WALCL_m"] / 1000.0
df_chart_lb["TGA_b"]   = df_chart_lb["TGA_m"] / 1000.0
df_chart_lb["NetLiq"]  = df_chart_lb["WALCL_b"] - df_chart_lb["RRP_b"] - df_chart_lb["TGA_b"]

# Smoothing for charts
cols_to_smooth = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]
if smooth > 1:
    for col in cols_to_smooth:
        if col in df_chart_lb:
            df_chart_lb[f"{col}_s"] = df_chart_lb[col].rolling(smooth, min_periods=1).mean()
else:
    for col in cols_to_smooth:
        if col in df_chart_lb:
            df_chart_lb[f"{col}_s"] = df_chart_lb[col]

# Rebased panel for charts
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

reb = pd.DataFrame(index=df_chart_lb.index)
reb["WALCL_idx"] = rebase(df_chart_lb["WALCL_b"])
reb["RRP_idx"]   = rebase(df_chart_lb["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df_chart_lb["TGA_b"])

# ---------------- Lookback change on integrity series ----------------
def change_since_lookback(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    start_val = s.iloc[0]
    end_val = s.iloc[-1]
    abs_chg = end_val - start_val
    pct_chg = np.nan if start_val == 0 else (abs_chg / start_val) * 100.0
    return start_val, end_val, pct_chg

nl_start, nl_end, nl_pct = change_since_lookback(df_exact_lb["NetLiq"])

# ---------------- Metrics (integrity-driven) ----------------
def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.3f}%"

latest_metrics = {
    "date": latest_common_date.date(),
    "netliq": latest_exact["NetLiq"],
    "walcl": latest_exact["WALCL_b"],
    "rrp": latest_exact["RRP_b"],
    "tga": latest_exact["TGA_b"],
    "effr": effr_ff.loc[:latest_common_date].iloc[-1] if not effr_ff.loc[:latest_common_date].empty else np.nan,
    "nl_abs_chg": nl_end - nl_start if pd.notna(nl_start) else np.nan,
    "nl_pct_chg": nl_pct,
    "walcl_last_obs": walcl_raw.dropna().index.max().date() if not walcl_raw.dropna().empty else None,
    "rrp_last_obs": rrp_raw.dropna().index.max().date() if not rrp_raw.dropna().empty else None,
    "tga_last_obs": tga_raw.dropna().index.max().date() if not tga_raw.dropna().empty else None,
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest_metrics["netliq"]), help="Exact-date WALCL - RRP - TGA")
m2.metric("WALCL", fmt_b(latest_metrics["walcl"]))
m3.metric("RRP", fmt_b(latest_metrics["rrp"]))
m4.metric("TGA", fmt_b(latest_metrics["tga"]))
m5.metric("EFFR", fmt_pct(latest_metrics["effr"]))
m6.metric(
    f"Net Liquidity since {lookback}",
    fmt_b(latest_metrics["nl_abs_chg"]),
    delta=fmt_pct(latest_metrics["nl_pct_chg"])
)
st.caption(
    f"Headline date: {latest_metrics['date']}  |  WALCL last obs: {latest_metrics['walcl_last_obs']}  |  RRP last obs: {latest_metrics['rrp_last_obs']}  |  TGA last obs: {latest_metrics['tga_last_obs']}"
)

# ---------------- Charts (visuals can use smoothed/ffill) ----------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)  [chart uses ffill for continuity, headline uses exact-date intersection]",
        "Components Rebased to 100 at Lookback Start",
        "Effective Fed Funds Rate"
    )
)

# Row 1: Net Liquidity
fig.add_trace(
    go.Scatter(x=df_chart_lb.index, y=df_chart_lb["NetLiq_s"], name="Net Liquidity (B)", line=dict(color="#000000", width=2)),
    row=1, col=1
)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Row 2: Components rebased
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx", line=dict()), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"],   name="RRP idx",   line=dict()), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"],   name="TGA idx",   line=dict()), row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1, rangemode="tozero")

# Row 3: EFFR
fig.add_trace(go.Scatter(x=df_chart_lb.index, y=df_chart_lb["EFFR_s"], name="EFFR", line=dict()), row=3, col=1)
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
**Net Liquidity is computed only on exact dates where WALCL, RRP, and TGA all report.**

**Lookback window:** {lookback}  
**Start date:** {df_exact_lb.index[0].date()}  
**End date:** {df_exact_lb.index[-1].date()}  

• **Start Net Liquidity:** {fmt_b(nl_start)}  
• **End Net Liquidity:** {fmt_b(nl_end)}  
• **Absolute Change:** {fmt_b(latest_metrics["nl_abs_chg"])}  
• **Percent Change:** {fmt_pct(latest_metrics["nl_pct_chg"])}
        """
    )

# ---------------- Download ----------------
with st.expander("Download Data"):
    # Raw exact-date series used for headline metrics
    exact_export = df_exact.copy()
    exact_export = exact_export.assign(
        WALCL_B=exact_export["WALCL_b"],
        TGA_B=exact_export["TGA_b"],
        NetLiq_B=exact_export["NetLiq"]
    )[["WALCL_B", "RRP_b", "TGA_B", "NetLiq_B"]]
    exact_export.index.name = "Date"

    # Chart data export (continuous with ffill)
    chart_export = df_chart_lb.copy()
    chart_export = chart_export.assign(
        WALCL_B=chart_export["WALCL_b"],
        TGA_B=chart_export["TGA_b"],
        NetLiq_B=chart_export["NetLiq"],
        NetLiq_s_B=chart_export["NetLiq_s"]
    )[["WALCL_B", "RRP_b", "TGA_B", "EFFR", "NetLiq_B", "NetLiq_s_B"]]
    chart_export.index.name = "Date"

    # One-row lookback summary
    summary = pd.DataFrame(
        {
            "Lookback": [lookback],
            "StartDate": [df_exact_lb.index[0].date()],
            "EndDate": [df_exact_lb.index[-1].date()],
            "Start_NetLiq_B": [nl_start],
            "End_NetLiq_B": [nl_end],
            "AbsChange_B": [latest_metrics["nl_abs_chg"]],
            "PctChange": [latest_metrics["nl_pct_chg"]],
        }
    )

    st.download_button(
        "Download Headline Integrity Series (CSV)",
        exact_export.to_csv(),
        file_name="liquidity_tracker_exact_integrity.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Chart Series (CSV)",
        chart_export.to_csv(),
        file_name="liquidity_tracker_chart_series.csv",
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
WALCL and TGA are reported in **millions**; we divide by 1,000 to billions. RRP is already **billions**.

**Integrity rule:** headline Net Liquidity and metrics are computed using only dates where all three series have a same-day observation (inner join, no forward fill).  
Charts may use forward fill for visual continuity, but do not affect the headline numbers.

Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base for each component index.  
A {RRP_BASE_FLOOR_B:.0f}B floor is applied for RRP rebase.

Percent change since lookback uses the integrity Net Liquidity series: ((last − first) ÷ first) × 100.
        """
    )

st.caption("© 2025 AD Fund Management LP")
