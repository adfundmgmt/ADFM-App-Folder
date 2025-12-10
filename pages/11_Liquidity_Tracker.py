############################################################
# Liquidity, Financial Conditions & Market Overlay Dashboard
# AD Fund Management LP
# Includes: Net Liquidity, NFCI, Recessions, SPX Overlay,
#           Rolling Beta (NetLiq → SPX) [NumPy OLS]
############################################################

import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import yfinance as yf

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
    "3 months": 0.25,
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

# NBER recession list
US_RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30")
]

# ----------------------------------------------------------
# Rolling Beta Function (NumPy-based OLS)
# ----------------------------------------------------------
def rolling_beta(x, y, window):
    """
    Compute rolling beta of y ~ x using simple OLS:
        y = alpha + beta * x
    """
    betas = []
    idx = []

    for i in range(window, len(x)):
        xw = x.iloc[i-window:i]
        yw = y.iloc[i-window:i]

        mask = xw.notna() & yw.notna()
        xw = xw[mask]
        yw = yw[mask]

        if len(xw) < 5:
            betas.append(np.nan)
        else:
            x_arr = xw.values
            y_arr = yw.values

            if np.var(x_arr) == 0:
                betas.append(np.nan)
            else:
                b = np.cov(x_arr, y_arr)[0, 1] / np.var(x_arr)
                betas.append(b)

        idx.append(x.index[i])

    return pd.Series(betas, index=idx)

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
        Net Liquidity = WALCL − RRP − TGA  
        NFCI = Chicago Fed National Financial Conditions Index  

        New Panels Added:  
        • SPX overlay with Net Liquidity  
        • Rolling 90-day beta (Net Liquidity → SPX)  
        • NBER recession shading  
        """
    )
    st.markdown("---")
    st.header("Settings")

    lookback_label = st.selectbox(
        "Lookback",
        list(LOOKBACK_OPTIONS.keys()),
        index=5
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
if lookback_label == "3 months":
    start_lb = today - pd.DateOffset(months=3)
elif lookback_label == "Max (25 years)":
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

# Rebased components
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

# ---------------- SPX & Rolling Beta ----------------
spx = yf.download("^GSPC", start=start_lb, end=today)["Adj Close"].ffill()
spx.name = "SPX"

combo = pd.concat([df["NetLiq"], spx], axis=1).dropna()

combo["spx_ret"] = combo["SPX"].pct_change()
combo["nl_ret"]  = combo["NetLiq"].pct_change()

# Compute rolling beta
window = 90
beta_series = rolling_beta(combo["nl_ret"], combo["spx_ret"], window)

combo = combo.iloc[window:].copy()
combo["beta"] = beta_series

# ---------------- Metrics ----------------
def fmt_b(x):   return "N/A" if pd.isna(x) else f"{x:,.0f} B"
def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x:.2f}%"
def fmt_nfci(x): return "N/A" if pd.isna(x) else f"{x:.3f}"

latest = df.iloc[-1]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Net Liquidity", fmt_b(latest["NetLiq"]))
m2.metric("WALCL", fmt_b(latest["WALCL_b"]))
m3.metric("RRP", fmt_b(latest["RRP_b"]))
m4.metric("TGA", fmt_b(latest["TGA_b"]))
m5.metric("EFFR", fmt_pct(latest["EFFR"]))
m6.metric("NFCI", fmt_nfci(latest["NFCI"]), help=">0 = tighter financial conditions")

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

# ---------------- SPX Overlay Panel ----------------
fig2 = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        "SPX vs Net Liquidity",
        "Rolling 90-Day Beta (NetLiq → SPX)",
        "Recession Periods (NBER)"
    )
)

fig2.add_trace(go.Scatter(
    x=combo.index, y=combo["SPX"],
    name="SPX", line=dict(color="#1f77b4")
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=combo.index, y=combo["NetLiq"],
    name="Net Liquidity", line=dict(color="#000000")
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=combo.index, y=combo["beta"],
    name="Beta (NetLiq → SPX)", line=dict(color="#ff7f0e")
), row=2, col=1)

# Recession shading
fig2.add_trace(go.Scatter(
    x=combo.index, y=[0]*len(combo),
    showlegend=False
), row=3, col=1)

for start_rec, end_rec in US_RECESSIONS:
    fig2.add_vrect(
        x0=start_rec, x1=end_rec,
        fillcolor="gray", opacity=0.25,
        line_width=0, row=3, col=1
    )

fig2.update_layout(
    template="plotly_white",
    height=950,
    legend=dict(orientation="h", x=0, y=1.15),
    margin=dict(l=60, r=40, t=60, b=60)
)

fig2.update_xaxes(tickformat="%b-%y", title="Date", row=3, col=1)
st.plotly_chart(fig2, use_container_width=True)

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

st.caption("© 2025 AD Fund Management LP")
