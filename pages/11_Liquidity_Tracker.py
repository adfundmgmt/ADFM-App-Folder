############################################################
# Liquidity & Fed Policy Tracker  —  Scale-optimized layout
# Built by AD Fund Management LP
############################################################

import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"
FRED = {
    "fed_bs": "WALCL",        # Fed total assets (millions)
    "rrp": "RRPONTSYD",       # ON RRP (millions)
    "tga": "WDTGAL",          # Treasury General Account (millions)
    "effr": "EFFR",           # Effective Fed Funds Rate (percent)
    "spx": "SP500",           # S&P 500 index level
    "nasdaq": "NASDAQCOM",    # Nasdaq Composite index level
}
DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_SMOOTH_DAYS = 5
DEFAULT_CORR_WINDOW = 60  # trading days

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Track Net Liquidity and relate it to policy and equities.

        **Definition:** Net Liquidity = WALCL - RRP - TGA  
        Units are billions. Higher is generally risk-on if sustained.

        **Panels**  
        1) **Net Liquidity (B):** primary signal. Look for trend turns and 5d or 20d inflections.  
        2) **Components Rebased:** WALCL, RRP, TGA each set to 100 at the lookback start to show direction without scale distortion.  
        3) **Policy Rate:** EFFR level.  
        4) **Equities Rebased:** SPX and optional Nasdaq, base 100 at lookback start.
        """
    )
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 30, DEFAULT_SMOOTH_DAYS, 1)
    corr_win = st.number_input("Correlation window (trading days)", 20, 120, DEFAULT_CORR_WINDOW, 5)
    show_nasdaq = st.checkbox("Show Nasdaq in Equities panel", value=False)
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
start_all = today - pd.DateOffset(years=10)  # fetch wide, subset later
start_lb = today - pd.DateOffset(years=years)

# Load series
fed_bs = fred_series(FRED["fed_bs"], start_all, today)       # millions
rrp    = fred_series(FRED["rrp"],    start_all, today)       # millions
tga    = fred_series(FRED["tga"],    start_all, today)       # millions
effr   = fred_series(FRED["effr"],   start_all, today)       # percent
spx    = fred_series(FRED["spx"],    start_all, today)       # index
nasdaq = fred_series(FRED["nasdaq"], start_all, today)       # index

# Align and subset
df = pd.concat(
    [fed_bs, rrp, tga, effr, spx, nasdaq],
    axis=1,
    keys=["WALCL", "RRP", "TGA", "EFFR", "SPX", "NASDAQ"]
).ffill()
df = df[df.index >= start_lb]
if df.empty:
    st.error("No data for selected lookback.")
    st.stop()

# Net Liquidity in billions
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]   / 1000.0
df["TGA_b"]   = df["TGA"]   / 1000.0
df["NetLiq"]  = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Smoothing
if smooth > 1:
    for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]:
        df[f"{col}_s"] = df[col].rolling(smooth, min_periods=1).mean()
else:
    for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]:
        df[f"{col}_s"] = df[col]

# Rebased panels
def rebase(series):
    base = series.iloc[0]
    return (series / base) * 100 if base and base != 0 else series * 0 + 100

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"]   = rebase(df["RRP_b"])
reb["TGA_idx"]   = rebase(df["TGA_b"])
reb["SPX_idx"]   = rebase(df["SPX"])
if show_nasdaq:
    reb["NASDAQ_idx"] = rebase(df["NASDAQ"])

# Correlations on changes
returns = pd.DataFrame(index=df.index)
returns["NetLiq_chg"] = df["NetLiq"].diff()
returns["SPX_ret"]    = df["SPX"].pct_change()
returns["NASDAQ_ret"] = df["NASDAQ"].pct_change()
corr_spx = returns["NetLiq_chg"].rolling(DEFAULT_CORR_WINDOW).corr(returns["SPX_ret"])
corr_nasdaq = returns["NetLiq_chg"].rolling(DEFAULT_CORR_WINDOW).corr(returns["NASDAQ_ret"])

# Quick deltas for takeaways
def delta(series, days):
    try:
        return series.iloc[-1] - series.shift(days).iloc[-1]
    except Exception:
        return pd.NA

def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_bp(x):  return "N/A" if pd.isna(x) else f"{x:+,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"

latest = {
    "date": df.index[-1].date(),
    "netliq": df["NetLiq"].iloc[-1],
    "netliq_5d": delta(df["NetLiq"], 5),
    "netliq_20d": delta(df["NetLiq"], 20),
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
    "corr_spx": corr_spx.iloc[-1],
    "corr_nasdaq": corr_nasdaq.iloc[-1] if show_nasdaq else pd.NA,
}

# ---------------- Metrics ----------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]),
          f"5d {fmt_bp(latest['netliq_5d'])} | 20d {fmt_bp(latest['netliq_20d'])}",
          help="WALCL - RRP - TGA")
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))

m6, m7 = st.columns(2)
m6.metric(f"Corr(NetLiq Δ, SPX ret) {DEFAULT_CORR_WINDOW}d",
          "N/A" if pd.isna(latest["corr_spx"]) else f"{latest['corr_spx']:.2f}")
if show_nasdaq:
    m7.metric(f"Corr(NetLiq Δ, Nasdaq ret) {DEFAULT_CORR_WINDOW}d",
              "N/A" if pd.isna(latest["corr_nasdaq"]) else f"{latest['corr_nasdaq']:.2f}")

# ---------------- Charts ----------------
# Layout: 4 rows to avoid scale compression
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100 at Lookback Start",
        "Effective Fed Funds Rate",
        "Equities Rebased to 100"
    )
)

# Row 1: Net Liquidity only — clean, big signal
fig.add_trace(go.Scatter(x=df.index, y=df["NetLiq_s"], name="Net Liquidity (B)",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Row 2: Components rebased — direction without scale distortion
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx",
                         line=dict(color="#1f77b4")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"], name="RRP idx",
                         line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"], name="TGA idx",
                         line=dict(color="#d62728")), row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1)

# Row 3: Policy rate
fig.add_trace(go.Scatter(x=df.index, y=df["EFFR_s"], name="EFFR",
                         line=dict(color="#ff7f0e")), row=3, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=3, col=1)

# Row 4: Equities rebased
fig.add_trace(go.Scatter(x=reb.index, y=reb["SPX_idx"], name="SPX idx",
                         line=dict(color="#9467bd")), row=4, col=1)
if show_nasdaq:
    fig.add_trace(go.Scatter(x=reb.index, y=reb["NASDAQ_idx"], name="NASDAQ idx",
                             line=dict(color="#8c564b")), row=4, col=1)
fig.update_yaxes(title="Index = 100", row=4, col=1)

fig.update_layout(
    template="plotly_white",
    height=980,
    legend=dict(orientation="h", x=0, y=1.13, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60),
)
fig.update_xaxes(tickformat="%b-%y", row=4, col=1, title="Date")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
with st.expander("Download Data"):
    out = df[["WALCL", "RRP", "TGA", "EFFR", "SPX", "NASDAQ", "NetLiq"]].copy()
    out.index.name = "Date"
    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="liquidity_tracker.csv",
        mime="text/csv",
    )

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
        **Net Liquidity** = WALCL - RRP - TGA.  
        Units: WALCL, RRP, TGA are millions on FRED. Displayed in billions.

        **Smoothing:** simple moving average over {smooth} days on Net Liquidity and EFFR.

        **Rebased panels:** set the first observation in the lookback window to 100.  
        This avoids scale compression and makes direction of change clear.

        **Correlations:** Pearson correlation of daily Net Liquidity changes and equity daily returns over {DEFAULT_CORR_WINDOW} trading days.
        """
    )

st.caption("© 2025 AD Fund Management LP")
