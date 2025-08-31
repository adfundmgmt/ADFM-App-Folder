############################################################
# Liquidity & Fed Policy Tracker - clean metrics, no deltas
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple, Optional

# Primary loader
from pandas_datareader import data as pdr

# Optional fallback loader if available in your environment
try:
    from fredapi import Fred  # pip install fredapi
    HAS_FREDAPI = True
except Exception:
    HAS_FREDAPI = False

from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

    st.markdown("---")
    st.subheader("FRED access")
    st.caption("If FRED rate limits or blocks anonymous requests, add an API key.")
    fred_key = st.text_input("FRED API key (optional, kept in session only)", value=st.secrets.get("FRED_API_KEY", ""))

    st.caption("Source: FRED via pandas-datareader or fredapi (fallback)")

# ---------------- Dates ----------------
today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

# ---------------- Data loaders ----------------
def fetch_with_pdr(series: str, start, end) -> Tuple[pd.Series, Optional[str]]:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s, None
    except Exception as e:
        return pd.Series(dtype=float), f"pandas_datareader error: {repr(e)}"

def fetch_with_fredapi(series: str, start, end, api_key: str) -> Tuple[pd.Series, Optional[str]]:
    if not HAS_FREDAPI:
        return pd.Series(dtype=float), "fredapi not installed"
    try:
        fred = Fred(api_key=api_key) if api_key else Fred()
        # fred.get_series returns pandas Series indexed by datetime
        s = fred.get_series(series, observation_start=start, observation_end=end)
        if isinstance(s, pd.Series):
            s.index = pd.to_datetime(s.index)
        return s, None
    except Exception as e:
        return pd.Series(dtype=float), f"fredapi error: {repr(e)}"

@st.cache_data(ttl=24*60*60, show_spinner=False)
def robust_fred(series: str, start, end, retries: int = 3, sleep_sec: float = 0.8, api_key: str = ""):
    """
    Try pandas_datareader with retries. If it fails and api_key is provided,
    try fredapi as a fallback. Return raw series (no ffill) and a status dict.
    """
    attempts = []
    # Primary: pandas_datareader with backoff
    for i in range(retries):
        s, err = fetch_with_pdr(series, start, end)
        if err is None and not s.empty:
            return s.sort_index(), {"loader": "pandas_datareader", "attempts": attempts}
        attempts.append(err or "unknown error")
        time.sleep(sleep_sec * (i + 1))

    # Fallback: fredapi if key provided or available
    s2, err2 = fetch_with_fredapi(series, start, end, api_key)
    if err2 is None and not s2.empty:
        return s2.sort_index(), {"loader": "fredapi", "attempts": attempts}
    attempts.append(err2 or "unknown fredapi error")

    # Failed
    return pd.Series(dtype=float), {"loader": "failed", "attempts": attempts}

# ---------------- Fetch raw series (no forward fill) ----------------
walcl_raw, walcl_status = robust_fred(FRED["fed_bs"], start_all, today, api_key=fred_key)
rrp_raw,   rrp_status   = robust_fred(FRED["rrp"],    start_all, today, api_key=fred_key)
tga_raw,   tga_status   = robust_fred(FRED["tga"],    start_all, today, api_key=fred_key)
effr_raw,  effr_status  = robust_fred(FRED["effr"],   start_all, today, api_key=fred_key)

# ---------------- Integrity checks ----------------
missing = []
if walcl_raw.empty: missing.append("WALCL")
if rrp_raw.empty:   missing.append("RRPONTSYD")
if tga_raw.empty:   missing.append("WDTGAL")

if missing:
    st.error(
        "Missing required series from FRED: " + ", ".join(missing) +
        ". Add a FRED API key in the sidebar and try again. If this persists, FRED may be rate limiting."
    )
    with st.expander("Fetch diagnostics"):
        st.write("WALCL loader:", walcl_status)
        st.write("RRP loader:", rrp_status)
        st.write("TGA loader:", tga_status)
        st.write("EFFR loader:", effr_status)
    st.stop()

# Exact same-date intersection for headline integrity
df_exact = pd.concat([walcl_raw, rrp_raw, tga_raw], axis=1, join="inner")
df_exact.columns = ["WALCL_m", "RRP_b", "TGA_m"]

df_exact["WALCL_b"] = df_exact["WALCL_m"] / 1000.0
df_exact["TGA_b"]   = df_exact["TGA_m"] / 1000.0
df_exact["NetLiq"]  = df_exact["WALCL_b"] - df_exact["RRP_b"] - df_exact["TGA_b"]

df_exact_lb = df_exact[df_exact.index >= start_lb]
if df_exact_lb.empty:
    st.error("No overlapping WALCL, RRP, and TGA observations within the selected lookback.")
    with st.expander("Fetch diagnostics"):
        st.write("WALCL last obs:", None if walcl_raw.empty else walcl_raw.index.max().date())
        st.write("RRP last obs:", None if rrp_raw.empty else rrp_raw.index.max().date())
        st.write("TGA last obs:", None if tga_raw.empty else tga_raw.index.max().date())
    st.stop()

latest_common_date = df_exact.index.max()
latest_exact = df_exact.loc[latest_common_date]

# Build chart series with ffill for continuity - does not affect headline metrics
walcl_ff = walcl_raw.ffill()
rrp_ff   = rrp_raw.ffill()
tga_ff   = tga_raw.ffill()
effr_ff  = effr_raw.ffill()

df_chart = pd.concat([walcl_ff, rrp_ff, tga_ff, effr_ff], axis=1, join="outer").sort_index()
df_chart.columns = ["WALCL_m", "RRP_b", "TGA_m", "EFFR"]
df_chart_lb = df_chart[df_chart.index >= start_lb].copy()

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

# Lookback change on integrity series
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

# ---------------- Metrics ----------------
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
    "walcl_last_obs": walcl_raw.index.max().date(),
    "rrp_last_obs": rrp_raw.index.max().date(),
    "tga_last_obs": tga_raw.index.max().date(),
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest_metrics["netliq"]), help="Exact same-date WALCL - RRP - TGA")
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
    f"Headline date: {latest_metrics['date']}  |  WALCL last obs: {latest_metrics['walcl_last_obs']}  |  "
    f"RRP last obs: {latest_metrics['rrp_last_obs']}  |  TGA last obs: {latest_metrics['tga_last_obs']}"
)

# ---------------- Charts ----------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)  [chart uses ffill for continuity, headline uses exact intersection]",
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

# ---------------- Data Integrity panel ----------------
with st.expander("Data integrity and diagnostics"):
    st.markdown(
        f"""
**Latest raw observations**
- WALCL last obs: **{latest_metrics['walcl_last_obs']}**  
- RRP last obs: **{latest_metrics['rrp_last_obs']}**  
- TGA last obs: **{latest_metrics['tga_last_obs']}**

**Headline Net Liquidity date:** **{latest_metrics['date']}**  
Computed only where WALCL, RRP, and TGA all report on the same date. No forward fill.

**Loaders**
- WALCL: {walcl_status.get('loader')}
- RRP: {rrp_status.get('loader')}
- TGA: {tga_status.get('loader')}
- EFFR: {effr_status.get('loader')}

If a series ever fails, expand Fetch diagnostics below to see the last error and try adding a FRED API key.
        """
    )
with st.expander("Fetch diagnostics"):
    st.write("WALCL attempts:", walcl_status.get("attempts"))
    st.write("RRP attempts:", rrp_status.get("attempts"))
    st.write("TGA attempts:", tga_status.get("attempts"))
    st.write("EFFR attempts:", effr_status.get("attempts"))

# ---------------- Lookback Change Summary ----------------
with st.expander("Lookback Change Summary"):
    st.markdown(
        f"""
**Lookback window:** {lookback}  
**Start date:** {df_exact_lb.index[0].date()}  
**End date:** {df_exact_lb.index[-1].date()}  

• **Start Net Liquidity:** {fmt_b(nl_start)}  
• **End Net Liquidity:** {fmt_b(nl_end)}  
• **Absolute Change:** {fmt_b(latest_metrics['nl_abs_chg'])}  
• **Percent Change:** {fmt_pct(latest_metrics['nl_pct_chg'])}
        """
    )

# ---------------- Downloads ----------------
with st.expander("Download Data"):
    exact_export = df_exact.copy()
    exact_export = exact_export.assign(
        WALCL_B=exact_export["WALCL_b"],
        TGA_B=exact_export["TGA_b"],
        NetLiq_B=exact_export["NetLiq"]
    )[["WALCL_B", "RRP_b", "TGA_B", "NetLiq_B"]]
    exact_export.index.name = "Date"

    chart_export = df_chart_lb.copy()
    chart_export = chart_export.assign(
        WALCL_B=chart_export["WALCL_b"],
        TGA_B=chart_export["TGA_b"],
        NetLiq_B=chart_export["NetLiq"],
        NetLiq_s_B=chart_export["NetLiq_s"]
    )[["WALCL_B", "RRP_b", "TGA_B", "EFFR", "NetLiq_B", "NetLiq_s_B"]]
    chart_export.index.name = "Date"

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
WALCL and TGA are reported in millions and are divided by 1,000 to billions. RRP is already in billions.

Integrity rule: headline values use only exact same-date observations across WALCL, RRP, and TGA.  
Charts may use forward fill for continuity and do not affect headline numbers.

Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base for each component index.  
A {RRP_BASE_FLOOR_B:.0f}B floor is applied for RRP rebase.

Percent change since lookback is computed on the integrity Net Liquidity series.
        """
    )

st.caption("© 2025 AD Fund Management LP")
