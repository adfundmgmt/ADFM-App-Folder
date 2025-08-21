############################################################
# Liquidity & Fed Policy Tracker - clean metrics, no deltas
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
    "fed_bs": "WALCL",        # Fed total assets (millions on FRED)
    "rrp": "RRPONTSYD",       # ON RRP (billions on FRED)
    "tga": "WDTGAL",          # Treasury General Account (millions on FRED)
    "effr": "EFFR",           # Effective Fed Funds Rate (percent)
    "spx": "SP500",           # S&P 500 index level
    "nasdaq": "NASDAQCOM",    # Nasdaq Composite index level
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
        Goal: track Net Liquidity and relate it to policy and equities.

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
        4) Equities rebased to 100
        """
    )
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "10y", "25y"], index=2)
    years = int(lookback[:-1])
    smooth = st.number_input("Smoothing window (days)", 1, 30, DEFAULT_SMOOTH_DAYS, 1)
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
start_all = today - pd.DateOffset(years=15)
start_lb = today - pd.DateOffset(years=years)

fed_bs = fred_series(FRED["fed_bs"], start_all, today)       # millions
rrp    = fred_series(FRED["rrp"],    start_all, today)       # billions
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

# Rebased panels (robust against near-zero bases)
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
reb["SPX_idx"]   = rebase(df["SPX"])
if show_nasdaq:
    reb["NASDAQ_idx"] = rebase(df["NASDAQ"])

# ---------------- Metrics ----------------
def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"

latest = {
    "date": df.index[-1].date(),
    "netliq": df["NetLiq"].iloc[-1],
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
}

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]), help="WALCL - RRP - TGA")
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))

# ---------------- Charts ----------------
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100 at Lookback Start",
        "Effective Fed Funds Rate",
        "Equities Rebased to 100"
    )
)

# Row 1: Net Liquidity
fig.add_trace(go.Scatter(x=df.index, y=df["NetLiq_s"], name="Net Liquidity (B)",
                         line=dict(color="#000000", width=2)), row=1, col=1)
fig.update_yaxes(title="Billions USD", row=1, col=1)

# Row 2: Components rebased
fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx",
                         line=dict(color="#1f77b4")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"], name="RRP idx",
                         line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"], name="TGA idx",
                         line=dict(color="#d62728")), row=2, col=1)
fig.update_yaxes(title="Index = 100", row=2, col=1, rangemode="tozero")

# Row 3: EFFR
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
    out = pd.DataFrame(index=df.index)
    # Export in billions for consistency
    out["WALCL_B"]  = df["WALCL_b"]
    out["RRP_B"]    = df["RRP_b"]
    out["TGA_B"]    = df["TGA_b"]
    out["EFFR_%"]   = df["EFFR"]
    out["SPX"]      = df["SPX"]
    out["NASDAQ"]   = df["NASDAQ"]
    out["NetLiq_B"] = df["NetLiq"]
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
        Net Liquidity = WALCL − RRP − TGA.  
        Units: WALCL and TGA are reported by FRED in millions; RRP is reported in billions.  
        All panels display values in billions after scaling.

        Rebase uses the median of the first {REBASE_BASE_WINDOW} observations as the base.  
        For RRP, a {RRP_BASE_FLOOR_B:.0f}B floor is applied to avoid divide-by-near-zero artifacts at regimes where usage was ~0.

        Smoothing applies a simple moving average over {smooth} days.  
        Components are rebased to 100 at the lookback start using the robust base above.

        Note: RRP daily series begins in 2013. Long lookbacks truncate to available history.
        """
    )

st.caption("© 2025 AD Fund Management LP")
