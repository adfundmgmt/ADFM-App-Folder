# theme_monitor.py

"""
AI + Datacenter Theme Monitor
No logins. No API keys. Uses Yahoo Finance via yfinance.

Upgrades in this version
- Cleaner layout with tabs and KPI cards
- Consistent percent formatting in tables
- Better legends and margins on charts
- Controls grouped in sidebar with presets + custom tickers

Run
pip install streamlit yfinance pandas numpy plotly
streamlit run theme_monitor.py
"""

import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Datacenter Theme Monitor", page_icon="📊", layout="wide")

# ---------------------
# Preset baskets
# ---------------------
HYPERS = ["AMZN","MSFT","GOOGL","META","ORCL"]
VENDORS = ["NVDA","AMD","AVGO","MRVL","SMCI","TSM","ASML","MU"]
NETWORK_POWER = ["ANET","CSCO","ACLS","AMAT","LRCX","KLAC","HPE","NTAP"]
REITS_POWER = ["EQIX","DLR","IRM","VST","NEE","DUK"]
DEFAULT = HYPERS + VENDORS + NETWORK_POWER + REITS_POWER

# ---------------------
# Helpers
# ---------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker")
    frames = []
    for t in tickers:
        try:
            frames.append(data[(t, "Close")].rename(t))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).dropna(how="all")
    df.index.name = "Date"
    return df

def ew_index(df: pd.DataFrame) -> pd.Series:
    return (df.pct_change().fillna(0).add(1).cumprod()).mean(axis=1)

def returns_table(df: pd.DataFrame) -> pd.DataFrame:
    rets = pd.DataFrame(index=df.columns)
    for label, win in {"1d":1, "5d":5, "20d":20}.items():
        rets[label] = df.pct_change(win).iloc[-1]*100.0
    ytd_start = df.index[df.index.year==df.index[-1].year][0]
    rets["YTD"] = (df.iloc[-1]/df.loc[ytd_start]-1)*100.0
    return rets.sort_values("20d", ascending=False).round(2)

def breadth(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for w in [20,50,200]:
        ma = df.rolling(w).mean()
        out[f">{w}DMA"] = (df.iloc[-1] > ma.iloc[-1]).mean()*100.0
    win = 63
    out["3m highs"] = int((df.iloc[-1] >= df.rolling(win).max().iloc[-1]).sum())
    out["3m lows"] = int((df.iloc[-1] <= df.rolling(win).min().iloc[-1]).sum())
    return pd.DataFrame(out, index=["Breadth"]).T

def rolling_beta(series: pd.Series, market: pd.Series, window: int = 20) -> pd.Series:
    x = market.pct_change()
    y = series.pct_change()
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov/var

# ---------------------
# Sidebar controls
# ---------------------
st.sidebar.title("AI Datacenter Theme Monitor")
with st.sidebar.expander("About This Tool", expanded=True):
    st.write(
        """
        Price-based trackers for the AI compute and datacenter buildout.
        - Equal-weight basket vs SPY
        - Leaders and laggards over 1d, 5d, 20d, YTD
        - Breadth: % above 20/50/200 DMA, 3m highs and lows
        - Risk: rolling beta to SPY
        """
    )

preset = st.sidebar.selectbox("Preset", ["Core US","Vendors only","Hypers only","With REITs/Power","Everything"], index=0)
period = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=2)

if preset == "Core US":
    tickers = HYPERS + VENDORS
elif preset == "Vendors only":
    tickers = VENDORS
elif preset == "Hypers only":
    tickers = HYPERS
elif preset == "With REITs/Power":
    tickers = HYPERS + VENDORS + REITS_POWER
else:
    tickers = DEFAULT

custom = st.sidebar.text_input("Add tickers (comma separated)", value="")
if custom.strip():
    add = [t.strip().upper() for t in custom.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers + add))

# ---------------------
# Data
# ---------------------
spy = fetch_prices(("SPY",), period=period).squeeze()
px = fetch_prices(tuple(tickers), period=period)

if px.empty or spy.empty:
    st.warning("No price data returned. Try fewer tickers or a shorter history.")
    st.stop()

px = px.dropna(how="all").dropna(axis=1, how="all")
px = px.loc[px.index.intersection(spy.index)]
spy = spy.loc[px.index]

# ---------------------
# Top KPIs
# ---------------------
idx = ew_index(px)
rel_vs_spy = idx/(spy/spy.iloc[0])

k1,k2,k3,k4 = st.columns(4)
k1.metric("Basket total return", f"{(idx.iloc[-1]-1)*100:,.1f}%")
k2.metric("Basket vs SPY", f"{(rel_vs_spy.iloc[-1]-1)*100:,.1f}%")
k3.metric("1m avg return", f"{px.pct_change(21).mean(axis=1).iloc[-1]*100:,.1f}%")
k4.metric("20d dispersion σ", f"{px.pct_change(20).iloc[-1].std()*100:,.1f}%")

# ---------------------
# Tabs
# ---------------------
overview, leaders_tab, risk_tab = st.tabs(["Overview","Leaders & Breadth","Risk & Beta"])

with overview:
    st.subheader("Equal-weight basket vs SPY")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx.index, y=idx, mode="lines", name="Basket EW"))
    fig.add_trace(go.Scatter(x=spy.index, y=spy/spy.iloc[0], mode="lines", name="SPY"))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

with leaders_tab:
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Leaders & laggards")
        tbl = returns_table(px)
        st.dataframe(
            tbl,
            use_container_width=True,
            column_config={
                "1d": st.column_config.NumberColumn(format="%.2f%%"),
                "5d": st.column_config.NumberColumn(format="%.2f%%"),
                "20d": st.column_config.NumberColumn(format="%.2f%%"),
                "YTD": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    with c2:
        st.subheader("Breadth")
        br = breadth(px)
        st.dataframe(
            br,
            use_container_width=True,
            column_config={
                ">20DMA": st.column_config.NumberColumn(format="%.1f%%"),
                ">50DMA": st.column_config.NumberColumn(format="%.1f%%"),
                ">200DMA": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

with risk_tab:
    st.subheader("Rolling 20d beta to SPY (key names)")
    key_show = [t for t in ["NVDA","AMD","SMCI","AMZN","MSFT","GOOGL","EQIX","DLR"] if t in px.columns][:6]
    fig2 = go.Figure()
    for t in key_show:
        fig2.add_trace(go.Scatter(x=px.index, y=rolling_beta(px[t], spy, 20), mode="lines", name=t))
    fig2.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig2, use_container_width=True)

st.caption("All series equal-weight. For decisions, combine with positioning and flows.")
