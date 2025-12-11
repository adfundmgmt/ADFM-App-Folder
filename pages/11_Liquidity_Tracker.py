import streamlit as st
import pandas as pd
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
    "nfci": "NFCI",
}
DEFAULT_SMOOTH_DAYS = 5
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B   = 5.0

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Sidebar ----------------
with st.sidebar():
    st.header("About This Tool")
    st.markdown(
        """
        Tracks systemic liquidity and policy stance using five core series.

        **Net Liquidity = WALCL − RRP − TGA**

        **Series**
        • WALCL Fed balance sheet  
        • RRPONTSYD reverse repo  
        • TGA Treasury General Account  
        • EFFR effective fed funds rate  
        • NFCI Chicago Fed financial conditions index  

        **Panels**
        • Net Liquidity  
        • Rebased components  
        • Fed Funds Rate  
        • Financial conditions (NFCI)
        """
    )

    st.markdown("---")
    st.header("Settings")

    LOOKBACK_MAP = {
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "10 years": 10,
        "25 years": 25,
    }

    lookback = st.selectbox(
        "Lookback",
        list(LOOKBACK_MAP.keys()),
        index=4
    )

    years = LOOKBACK_MAP[lookback]

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
start_all = today - pd.DateOffset(years=25)
start_lb  = today - pd.DateOffset(years=years)

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

# Net Liquidity (billions)
df["WALCL_b"] = df["WALCL"] / 1000.0
df["RRP_b"]   = df["RRP"]
df["TGA_b"]   = df["TGA"] / 1000.0
df["NetLiq"]  = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Apply smoothing
cols = ["WALCL_b", "RRP_b", "TGA_b", "NetLiq", "EFFR", "NFCI"]
for col in cols:
    df[f"{col}_s"] = (
        df[col].rolling(smooth, min_periods=1).mean()
        if smooth > 1 else df[col]
    )

# Rebase
def rebase(series, base_window=REBASE_BASE_WINDOW, min_base=None):
    s = series.copy()
    if s.isna().all():
        return s * 0 + 100
    head = s.dropna().iloc[:max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]
    if min_base is not None:
        base = max(base, float(min_base))
    return (s / base) * 100 if base else s * 0 + 100

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"]   = rebase(df["RRP_b"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"]   = rebase(df["TGA_b"])

# ---------------- Metrics ----------------
def fmt_b(x):    return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x):  return "N/A" if pd.isna(x) else f"{x:.2f}%"
def fmt_nfci(x): return "N/A" if pd.isna(x) else f"{x:.3f}"

latest = {
    "netliq": df["NetLiq"].iloc[-1],
    "walcl": df["WALCL_b"].iloc[-1],
    "rrp":   df["RRP_b"].iloc[-1],
    "tga":   df["TGA_b"].iloc[-1],
    "effr":  df["EFFR"].iloc[-1],
    "nfci":  df["NFCI"].iloc[-1],
}

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest["netliq"]))
m2.metric("WALCL", fmt_b(latest["walcl"]))
m3.metric("RRP", fmt_b(latest["rrp"]))
m4.metric("TGA", fmt_b(latest["tga"]))
m5.metric("EFFR", fmt_pct(latest["effr"]))
m6.metric("NFCI", fmt_nfci(latest["nfci"]), help=">0 = tighter conditions")

# ---------------- Charts ----------------
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions (NFCI)"
    )
)

# Row 1
fig.add_trace(go.Scatter(
    x=df.index, y=df["NetLiq_s"],
    name="Net Liquidity",
    line=dict(color="#000000", width=2)
), row=1, col=1)

# Row 2
fig.add_trace(go.Scatter(
    x=reb.index, y=reb["WALCL_idx"], name="WALCL idx"
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=reb.index, y=reb["RRP_idx"], name="RRP idx"
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=reb.index, y=reb["TGA_idx"], name="TGA idx"
), row=2, col=1)

# Row 3
fig.add_trace(go.Scatter(
    x=df.index, y=df["EFFR_s"],
    name="EFFR",
    line=dict(color="#ff7f0e")
), row=3, col=1)

# Row 4 (NFCI)
fig.add_trace(go.Scatter(
    x=df.index, y=df["NFCI_s"],
    name="NFCI",
    line=dict(color="#1f1f1f", width=2)
), row=4, col=1)

# Layout improvements
fig.update_layout(
    template="plotly_white",
    height=1080,
    hovermode="x unified",
    legend=dict(orientation="h", x=0, y=1.16),
    margin=dict(l=60, r=40, t=70, b=60),
)
fig.update_xaxes(tickformat="%b-%y")

st.plotly_chart(fig, use_container_width=True)

# ---------------- Download ----------------
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

# ---------------- Notes ----------------
with st.expander("Methodology"):
    st.markdown(
        f"""
        **Net Liquidity = WALCL − RRP − TGA**

        **NFCI interpretation**
        • Combined measure of credit, leverage, and funding markets  
        • Values above zero imply tighter financial conditions relative to history  
        • Useful complement to net liquidity and EFFR  

        Smoothing uses {smooth}-day averages.  
        Rebase uses median of first {REBASE_BASE_WINDOW} observations.
        """
    )

st.caption("© 2025 AD Fund Management LP")
