############################################################
# Liquidity & Fed Policy Tracker — scale-optimized and robust deltas
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
DEFAULT_SMOOTH_DAYS = 5
DEFAULT_CORR_WINDOW = 60  # trading days

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Goal**: Track Net Liquidity and relate it to policy and equities.

        **Definition**: Net Liquidity = WALCL − RRP − TGA  
        Units are billions. Sustained rises often coincide with risk-on conditions.

        **Series**  
        • **WALCL**: Federal Reserve total assets on the consolidated balance sheet.  
        • **RRPONTSYD (RRP)**: Overnight reverse repo facility usage by counterparties.  
        • **WDTGAL (TGA)**: U.S. Treasury’s cash balance at the Federal Reserve.  
        • **EFFR**: Effective federal funds rate (volume-weighted overnight rate).

        **Panels**  
        1) Net Liquidity (billions).  
        2) Components rebased to 100 at the lookback start.  
        3) EFFR.  
        4) Equities rebased to 100.
        """
    )
    st.markdown("---")
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y", index=2)
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
# Fetch wide enough so 25y is available
start_all = today - pd.DateOffset(years=40)
start_lb = today - pd.DateOffset(years=years)

fed_bs = fred_series(FRED["fed_bs"], start_all, today)       # millions
rrp    = fred_series(FRED["rrp"],    start_all, today)       # millions (starts 2013)
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

# If early years lack RRP (pre-2013), drop until all three exist
df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]
if df.empty:
    st.error("No data for selected lookback. Try a shorter window.")
    st.stop()

# Net Liquidity in billions
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]   / 1000.0
df["TGA_b"]   = df["TGA"]   / 1000.0
df["NetLiq"]  = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Smoothing
cols_to_smooth = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR"]
if smooth > 1:
    for col in cols_to_smooth:
        df[f"{col}_s"] = df[col].rolling(smooth, min_periods=1).mean()
else:
    for col in cols_to_smooth:
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
corr_spx = returns["NetLiq_chg"].rolling(corr_win).corr(returns["SPX_ret"])
corr_nasdaq = returns["NetLiq_chg"].rolling(corr_win).corr(returns["NASDAQ_ret"])

# Robust deltas that never disappear
def safe_delta_num(series: pd.Series, periods: int):
    n = series.size
    if n < 2:
        return pd.NA
    k = min(periods, n - 1)
    return series.iloc[-1] - series.iloc[-1 - k]

def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"
def signed_html(x):
    if pd.isna(x): return "N/A"
    color = "#2ca02c" if x >= 0 else "#d62728"
    sign = "+" if x >= 0 else ""
    return f'<span style="color:{color}">{sign}{x:,.0f} B</span>'

latest = {
    "date": df.index[-1].date(),
    "netliq": df["NetLiq"].iloc[-1],
    "netliq_5d": safe_delta_num(df["NetLiq"], 5),
    "netliq_20d": safe_delta_num(df["NetLiq"], 20),
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp": df["RRP_b"].iloc[-1],
    "tga": df["TGA_b"].iloc[-1],
    "effr": df["EFFR"].iloc[-1],
    "corr_spx": corr_spx.iloc[-1],
    "corr_nasdaq": corr_nasdaq.iloc[-1] if show_nasdaq else pd.NA,
}

# ---------------- Metrics ----------------
m1, m2, m3, m4, m5 = st.columns(5)
# Use numeric delta for correct Streamlit coloring; show 20d below with sign-aware HTML
m1.metric("Net Liquidity", fmt_b(latest["netliq"]), latest["netliq_5d"], help="5d change shown; 20d below")
m1.markdown(f"**20d** {signed_html(latest['netliq_20d'])}", unsafe_allow_html=True)
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))

m6, m7 = st.columns(2)
m6.metric(f"Corr(NetLiq Δ, SPX ret) {corr_win}d",
          "N/A" if pd.isna(latest["corr_spx"]) else f"{latest['corr_spx']:.2f}")
if show_nasdaq:
    m7.metric(f"Corr(NetLiq Δ, Nasdaq ret) {corr_win}d",
              "N/A" if pd.isna(latest["corr_nasdaq"]) else f"{latest['corr_nasdaq']:.2f}")

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
fig.update_yaxes(title="Index = 100", row=2, col=1)

# Row 3: EFFR
fig.add_trace(go.Scatter(x=df.index, y=df["EFFR_s"], name="EFFR",
                         line=dict(color="#ff7f0e")), row=3, col=1)
fig.update_yaxes(title="Percent", tickformat=".2f", row=3, col=1)

# Row 4: Equities rebased
reb_equities = pd.DataFrame(index=df.index)
reb_equities["SPX_idx"] = (df["SPX"] / df["SPX"].iloc[0]) * 100
fig.add_trace(go.Scatter(x=reb_equities.index, y=reb_equities["SPX_idx"], name="SPX idx",
                         line=dict(color="#9467bd")), row=4, col=1)
if show_nasdaq:
    reb_equities["NASDAQ_idx"] = (df["NASDAQ"] / df["NASDAQ"].iloc[0]) * 100
    fig.add_trace(go.Scatter(x=reb_equities.index, y=reb_equities["NASDAQ_idx"], name="NASDAQ idx",
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
        Net Liquidity = WALCL − RRP − TGA.  
        WALCL, RRP, TGA are millions on FRED and are displayed in billions.

        Smoothing applies a simple moving average over {smooth} days.  
        Components are rebased to 100 at the lookback start to remove scale distortion.

        Correlations use Pearson correlation between daily Net Liquidity changes and equity returns over {corr_win} trading days.

        Note: RRP daily series begins in 2013. Selecting longer lookbacks will truncate to available data.
        """
    )

st.caption("© 2025 AD Fund Management LP")
