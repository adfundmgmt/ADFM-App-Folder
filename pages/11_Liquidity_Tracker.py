############################################################
# Liquidity & Fed Policy Tracker - integrity-first
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from typing import Tuple, Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas_datareader import data as pdr

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"
FRED = {
    "fed_bs": "WALCL",   # Fed total assets (millions, weekly)
    "rrp": "RRPONTSYD",  # ON RRP (billions, daily)
    "tga": "WDTGAL",     # Treasury General Account (millions, daily)
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

Definition: Net Liquidity = WALCL − RRP − TGA. Units are billions.

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
    smooth = st.number_input("Smoothing window (points)", 1, 30, DEFAULT_SMOOTH_DAYS, 1)
    st.markdown("---")
    st.subheader("FRED access")
    st.caption("If FRED blocks pandas_datareader, the app will use the official FRED JSON API.")
    fred_key = st.text_input("FRED API key (required for fallback)", value=st.secrets.get("FRED_API_KEY", ""))
    st.caption("Source: FRED via pandas-datareader or FRED JSON API")

# ---------------- Dates ----------------
today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

# ---------------- Loaders ----------------
def fetch_with_pdr(series: str, start, end) -> Tuple[pd.Series, Optional[str]]:
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s, None
    except Exception as e:
        return pd.Series(dtype=float), f"pandas_datareader error: {repr(e)}"

def fetch_with_fred_http(series: str, start, end, api_key: str) -> Tuple[pd.Series, Optional[str]]:
    """Use FRED official JSON API. Requires API key."""
    if not api_key:
        return pd.Series(dtype=float), "fred_http requires API key"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "observation_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "sort_order": "asc",
        "units": "lin",
        "frequency": "",  # native frequency
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float), "fred_http empty"
        idx = []
        vals = []
        for o in obs:
            v = o.get("value")
            # FRED uses "." for missing
            if v is None or v == ".":
                continue
            vals.append(float(v))
            idx.append(pd.to_datetime(o["date"]))
        s = pd.Series(vals, index=pd.to_datetime(idx), name=series).sort_index()
        return s, None
    except Exception as e:
        return pd.Series(dtype=float), f"fred_http error: {repr(e)}"

@st.cache_data(ttl=24*60*60, show_spinner=False)
def robust_fred(series: str, start, end, retries: int = 2, sleep_sec: float = 0.8, api_key: str = ""):
    """
    Try pandas_datareader with retries. If blocked, use FRED JSON API with key.
    Returns raw series (no ffill) and diagnostics.
    """
    attempts = []
    # pdr first
    for i in range(retries):
        s, err = fetch_with_pdr(series, start, end)
        if err is None and not s.empty:
            return s.sort_index(), {"loader": "pandas_datareader", "attempts": attempts}
        attempts.append(err or "unknown pdr error")
        time.sleep(sleep_sec * (i + 1))
    # FRED JSON API
    s2, err2 = fetch_with_fred_http(series, start, end, api_key)
    if err2 is None and not s2.empty:
        return s2.sort_index(), {"loader": "fred_http", "attempts": attempts}
    attempts.append(err2 or "unknown fred_http error")
    return pd.Series(dtype=float), {"loader": "failed", "attempts": attempts}

# ---------------- Fetch raw series (no ffill for integrity) ----------------
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
    st.error("Missing required FRED series: " + ", ".join(missing) + ". Add a FRED API key and rerun.")
    with st.expander("Fetch diagnostics"):
        st.write("WALCL:", walcl_status)
        st.write("RRPONTSYD:", rrp_status)
        st.write("WDTGAL:", tga_status)
        st.write("EFFR:", effr_status)
    st.stop()

# Exact same-date intersection for headline integrity
df_exact = pd.concat([walcl_raw, rrp_raw, tga_raw], axis=1, join="inner")
df_exact.columns = ["WALCL_m", "RRP_b", "TGA_m"]
df_exact["WALCL_b"] = df_exact["WALCL_m"] / 1000.0
df_exact["TGA_b"]   = df_exact["TGA_m"] / 1000.0
df_exact["NetLiq"]  = df_exact["WALCL_b"] - df_exact["RRP_b"] - df_exact["TGA_b"]

df_exact_lb = df_exact[df_exact.index >= start_lb]
if df_exact_lb.empty:
    st.error("No overlapping WALCL, RRP, and TGA observations in the selected lookback.")
    st.stop()

latest_common_date = df_exact.index.max()
latest_exact = df_exact.loc[latest_common_date]

# Build chart series with ffill for continuity (visuals only)
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
    "walcl_last_obs": walcl_raw.index.max().date(),
    "rrp_last_obs": rrp_raw.index.max().date(),
    "tga_last_obs": tga_raw.index.max().date(),
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest_metrics["netliq"]), help="Exact same-date WALCL − RRP − TGA")
m2.metric("WALCL", fmt_b(latest_metrics["walcl"]))
m3.metric("RRP", fmt_b(latest_metrics["rrp"]))
m4.metric("TGA", fmt_b(latest_metrics["tga"]))
m5.metric("EFFR", fmt_pct(latest_metrics["effr"]))
m6.metric(f"Net Liquidity since {lookback}", fmt_b(latest_metrics["nl_abs_chg"]), delta=fmt_pct(latest_metrics["nl_pct_chg"]))

st.caption(
    f"Headline date: {latest_metrics['date']}  |  WALCL last obs: {latest_metrics['walcl_last_obs']}  |  "
    f"RRP last obs: {latest_metrics['rrp_last_obs']}  |  TGA last obs: {latest_metrics['tga_last_obs']}"
)

# ---------------- Charts ----------------
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)  [chart uses ffill for continuity; headline uses exact intersection]",
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
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx"), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"],   name="RRP idx"),   row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"],   name="TGA idx"),   row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1, rangemode="tozero")

# Row 3: EFFR
fig.add_trace(go.Scatter(x=df_chart_lb.index, y=df_chart_lb["EFFR_s"], name="EFFR"), row=3, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=3, col=1)

fig.update_layout(
    template="plotly_white",
    height=860,
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60),
)
fig.update_xaxes(tickformat="%b-%y", row=3, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Data integrity and change summary ----------------
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
        """
    )

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
    )

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
Net Liquidity = WALCL − RRP − TGA.
WALCL and TGA reported in millions; divided by 1,000 to billions. RRP already in billions.

Integrity rule: headline values use only exact same-date observations across WALCL, RRP, and TGA.
Charts may use forward fill for continuity and do not affect headline numbers.

Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base.
A {RRP_BASE_FLOOR_B:.0f}B floor is applied for RRP rebase.

Percent change since lookback is computed on the integrity Net Liquidity series.
        """
    )

st.caption("© 2025 AD Fund Management LP")
