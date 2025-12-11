############################################################
# Liquidity, Fed Policy & Financial Conditions Tracker
# AD Fund Management LP
# Includes: Net Liquidity + NFCI
# Clean version with no recession shading or SPX overlays
############################################################

import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"

FRED = {
    "fed_bs": "WALCL",
    "rrp": "RRPONTSYD",
    "tga": "WDTGAL",
    "effr": "EFFR",
    "nfci": "NFCI"
}

LOOKBACK_OPTIONS = {
    "1 year": 1,
    "3 years": 3,
    "5 years": 5,
    "10 years": 10,
    "Max (25 years)": 25
}

DEFAULT_SMOOTH_DAYS = 5
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0
MAX_YEARS = 25

# ----------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard tracks how the Federal Reserve and Treasury  
        influence system liquidity and financial conditions.

        **Net Liquidity = WALCL − RRP − TGA**  
        Higher liquidity often reflects easier conditions for risk-taking.

        **Inputs:**  
        • **WALCL** – Federal Reserve balance sheet  
        • **RRP** – Reverse Repo usage  
        • **TGA** – Treasury's cash balance  
        • **EFFR** – Effective Fed Funds Rate  
        • **NFCI** – Chicago Fed Financial Conditions Index  

        **Interpretation:**  
        • Rising Net Liquidity often signals a supportive policy environment.  
        • NFCI above zero reflects tighter-than-average conditions.  
        """
    )

    st.markdown("---")
    st.header("Settings")

    lookback_label = st.selectbox(
        "Lookback window",
        list(LOOKBACK_OPTIONS.keys()),
        index=4  # Default: Max (25 years)
    )
    lookback_years = LOOKBACK_OPTIONS[lookback_label]

    smooth = st.number_input(
        "Smoothing window (days)",
        1, 30, DEFAULT_SMOOTH_DAYS, 1
    )

    st.caption("Data source: FRED via pandas-datareader")

# ---------------- Data Fetch ----------------
@st.cache_data(ttl=24*60*60, show_spinner=False)
def fred_series(series, start, end):
    try:
        s = pdr.DataReader(series, "fred", start, end)[series]
        return s.ffill()
    except Exception:
        return pd.Series(dtype=float)

today = pd.Timestamp.today().normalize()
start_all = today - pd.DateOffset(years=MAX_YEARS)

# Lookback logic
if lookback_label == "Max (25 years)":
    start_lb = start_all
else:
    start_lb = today - pd.DateOffset(years=int(lookback_years))

# Fetch FRED series
fed_bs = fred_series(FRED["fed_bs"], start_all, today)
rrp    = fred_series(FRED["rrp"],    start_all, today)
tga    = fred_series(FRED["tga"],    start_all, today)
effr   = fred_series(FRED["effr"],   start_all, today)
nfci   = fred_series(FRED["nfci"],   start_all, today)

df = pd.concat(
    [fed_bs, rrp, tga, effr, nfci],
    axis=1,
    keys=["WALCL", "RRP", "TGA", "EFFR", "NFCI"]
).ffill()

df = df.dropna(subset=["WALCL", "RRP", "TGA"])
df = df[df.index >= start_lb]

if df.empty:
    st.error("No data for selected lookback.")
    st.stop()

# ---------------- Transformations ----------------
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]
df["TGA_b"]   = df["TGA"] / 1000.0
df["NetLiq"]  = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Smoothing
for col in ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR", "NFCI"]:
    df[f"{col}_s"] = (
        df[col].rolling(smooth, min_periods=1).mean() if smooth > 1 else df[col]
    )

# Rebased indices
def rebase(series, base_window=REBASE_BASE_WINDOW, min_base=None):
    s = series.copy()
    head = s.dropna().iloc[:max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]
    if min_base is not None:
        base = max(base, float(min_base))
    if base == 0 or pd.isna(base):
        return s * 0 + 100
    return (s / base) * 100

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"]   = rebase(df["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df["TGA_b"])

# ---------------- Metrics ----------------
def fmt_b(x):   return f"{x:,.0f} B" if pd.notna(x) else "N/A"
def fmt_pct(x): return f"{x:.2f}%" if pd.notna(x) else "N/A"
def fmt_nfci(x): return f"{x:.3f}" if pd.notna(x) else "N/A"

latest = df.iloc[-1]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest["NetLiq"]))
m2.metric("WALCL", fmt_b(latest["WALCL_b"]))
m3.metric("RRP", fmt_b(latest["RRP_b"]))
m4.metric("TGA", fmt_b(latest["TGA_b"]))
m5.metric("EFFR", fmt_pct(latest["EFFR"]))
m6.metric("NFCI", fmt_nfci(latest["NFCI"]), help=">0 = tighter conditions")

# ---------------- Main Liquidity Chart ----------------
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions (NFCI)"
    )
)

fig.add_trace(go.Scatter(
    x=df.index, y=df["NetLiq_s"],
    name="Net Liquidity", line=dict(color="#000000", width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx"), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["RRP_idx"], name="RRP idx"), row=2, col=1)
fig.add_trace(go.Scatter(x=reb.index, y=reb["TGA_idx"], name="TGA idx"), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["EFFR_s"],
    name="EFFR", line=dict(color="#ff7f0e")
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["NFCI_s"],
    name="NFCI", line=dict(color="#1a1a1a", width=2)
), row=4, col=1)

fig.update_layout(
    template="plotly_white",
    height=1100,
    legend=dict(orientation="h", x=0, y=1.14),
    margin=dict(l=60, r=40, t=60, b=60)
)

fig.update_xaxes(tickformat="%b-%y", title="Date", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download Data ----------------
with st.expander("Download Data"):
    out = pd.DataFrame(index=df.index)
    out["WALCL_B"]  = df["WALCL_b"]
    out["RRP_B"]    = df["RRP_b"]
    out["TGA_B"]    = df["TGA_b"]
    out["EFFR_%"]   = df["EFFR"]
    out["NFCI"]     = df["NFCI"]
    out["NetLiq_B"] = df["NetLiq"]
    out.index.name = "Date"

    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="liquidity_tracker.csv",
        mime="text/csv"
    )

st.caption("© 2025 AD Fund Management LP")
