############################################################
# Liquidity & Fed Policy Tracker
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
    "fed_bs": "WALCL",        # Fed total assets
    "rrp": "RRPONTSYD",       # ON RRP
    "tga": "WDTGAL",          # Treasury General Account
    "effr": "EFFR",           # Effective Fed Funds Rate
    "spx": "SP500",           # S&P 500 index
    "nasdaq": "NASDAQCOM",    # Nasdaq Composite
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
        Tracks Fed balance sheet, ON RRP, and TGA to derive a **Net Liquidity** proxy, and relates it to equity markets.
        - Net Liquidity = WALCL − RRP − TGA
        - Panels: Liquidity stack with net line, policy rate, and equities.
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
df["NetLiq"] = (df["WALCL"] - df["RRP"] - df["TGA"]) / 1000.0
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]   / 1000.0
df["TGA_b"]   = df["TGA"]   / 1000.0

# Smoothing
if smooth > 1:
    for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]:
        df[f"{col}_s"] = df[col].rolling(smooth, min_periods=1).mean()
else:
    for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]:
        df[f"{col}_s"] = df[col]

# Returns for correlation
returns = pd.DataFrame(index=df.index)
returns["NetLiq_chg"] = df["NetLiq"].diff()  # daily change in billions
returns["SPX_ret"] = df["SPX"].pct_change()
returns["NASDAQ_ret"] = df["NASDAQ"].pct_change()

def rolling_corr(a, b, w):
    return a.rolling(w).corr(b)

corr_spx = rolling_corr(returns["NetLiq_chg"], returns["SPX_ret"], corr_win)
corr_nasdaq = rolling_corr(returns["NetLiq_chg"], returns["NASDAQ_ret"], corr_win)

# Metrics
def last_delta(series, lag_days=7):
    try:
        return series.iloc[-1] - series.shift(lag_days).iloc[-1]
    except Exception:
        return pd.NA

latest = {
    "date": df.index[-1].date(),
    "netliq": df["NetLiq"].iloc[-1],
    "netliq_d1": df["NetLiq"].iloc[-1] - df["NetLiq"].iloc[-2] if len(df) > 1 else pd.NA,
    "walcl": df["WALCL_b"].iloc[-1],
    "walcl_wow": last_delta(df["WALCL_b"]),
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
    "corr_spx": corr_spx.iloc[-1],
    "corr_nasdaq": corr_nasdaq.iloc[-1] if show_nasdaq else pd.NA,
}

def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_bp(x):  return "N/A" if pd.isna(x) else f"{x:+,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net Liquidity", fmt_b(latest["netliq"]), fmt_bp(latest["netliq_d1"]), help="WALCL - RRP - TGA")
c2.metric("Fed Balance Sheet", fmt_b(latest["walcl"]), fmt_bp(latest["walcl_wow"]), help="Level and 7d change")
c3.metric("ON RRP", fmt_b(latest["rrp"]))
c4.metric("TGA", fmt_b(latest["tga"]))
c5.metric(f"EFFR", fmt_pct(latest["effr"]))

c6, c7 = st.columns(2)
c6.metric(f"Corr(NetLiq Δ, SPX ret) {corr_win}d", "N/A" if pd.isna(latest["corr_spx"]) else f"{latest['corr_spx']:.2f}")
if show_nasdaq:
    c7.metric(f"Corr(NetLiq Δ, Nasdaq ret) {corr_win}d", "N/A" if pd.isna(latest["corr_nasdaq"]) else f"{latest['corr_nasdaq']:.2f}")

# ---------------- Charts ----------------
nrows = 3
fig = make_subplots(
    rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=("Liquidity Stack and Net Liquidity",
                    "Policy Rate",
                    "Equities (rebased)"))

# Panel 1: Liquidity stack and Net Liquidity
fig.add_trace(go.Scatter(x=df.index, y=df["WALCL_b_s"], name="WALCL (B)", line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=-df["RRP_b_s"], name="RRP as negative (B)", line=dict(color="#2ca02c")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=-df["TGA_b_s"], name="TGA as negative (B)", line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["NetLiq_s"], name="Net Liquidity (B)", line=dict(color="#000000", width=2)), row=1, col=1)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Panel 2: EFFR
fig.add_trace(go.Scatter(x=df.index, y=df["EFFR_s"], name="EFFR", line=dict(color="#ff7f0e")), row=2, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=2, col=1)

# Panel 3: Equities rebased
rebased = pd.DataFrame(index=df.index)
rebased["SPX"] = df["SPX"] / df["SPX"].iloc[0] * 100
fig.add_trace(go.Scatter(x=rebased.index, y=rebased["SPX"], name="SPX (rebased=100)", line=dict(color="#9467bd")), row=3, col=1)
if show_nasdaq:
    rebased["NASDAQ"] = df["NASDAQ"] / df["NASDAQ"].iloc[0] * 100
    fig.add_trace(go.Scatter(x=rebased.index, y=rebased["NASDAQ"], name="NASDAQ (rebased=100)", line=dict(color="#8c564b")), row=3, col=1)
fig.update_yaxes(title="Index (rebased)", row=3, col=1)

fig.update_layout(
    template="plotly_white",
    height=900,
    legend=dict(orientation="h", x=0, y=1.12, xanchor="left"),
    margin=dict(l=60, r=40, t=60, b=60),
)
fig.update_xaxes(tickformat="%b-%y", row=3, col=1, title="Date")

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
        """
        **Net Liquidity** = WALCL − RRP − TGA.  
        - WALCL, RRP, TGA are in millions on FRED. This app converts to billions for readability.  
        - Smoothing applies a simple moving average over the chosen window.  
        - Correlation uses Pearson correlation between daily Net Liquidity changes and equity daily returns over the selected window.
        """
    )

st.caption("© 2025 AD Fund Management LP")
