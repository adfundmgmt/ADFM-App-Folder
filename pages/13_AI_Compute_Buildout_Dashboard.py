# theme_monitor.py

"""
AI + Datacenter Theme Monitor
No logins. No API keys. Uses Yahoo Finance via yfinance.

Panels
- Basket overview: equal-weight index, vs SPY, KPIs
- Leaders & laggards: 1d, 5d, 20d returns
- Breadth: % above 20/50/200 DMA, 3m highs/lows
- Risk: rolling beta to SPY, drawdowns
- Ratio charts: NVDA vs basket, Vendors vs Hyperscalers

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

# =====================
# Preset baskets (tickers are liquid US listings / ADRs)
# =====================
HYPERS = ["AMZN","MSFT","GOOGL","META","ORCL"]
VENDORS = ["NVDA","AMD","AVGO","MRVL","SMCI","TSM","ASML","MU"]
NETWORK_POWER = ["ANET","CSCO","ACLS","AMAT","LRCX","KLAC","HPE","NTAP"]
REITS_POWER = ["EQIX","DLR","IRM","VST","NEE","DUK"]
ASIA_ADR = ["BIDU","BABA","JD","NTES","TCEHY"]
EU_NAMES = ["SAP","ADBE","SNOW","ORCL","ASML"]

DEFAULT = HYPERS + VENDORS + NETWORK_POWER + REITS_POWER

# =====================
# Helpers
# =====================

def fetch_prices(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="ticker")
    # Normalize to a flat column MultiIndex -> close only
    frames = []
    for t in tickers:
        try:
            close = data[(t, "Close")].rename(t)
            frames.append(close)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).dropna(how="all")
    df.index.name = "Date"
    return df

@st.cache_data(show_spinner=False)
def get_prices_cached(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_prices(list(tickers), period=period)

def ew_index(df: pd.DataFrame) -> pd.Series:
    return (df.pct_change().fillna(0).add(1).cumprod()).mean(axis=1)

def returns_table(df: pd.DataFrame) -> pd.DataFrame:
    rets = pd.DataFrame(index=df.columns)
    for label, win in {"1d":1, "5d":5, "20d":20}.items():
        rets[label] = df.pct_change(win).iloc[-1]*100.0
    rets["YTD"] = (df.iloc[-1]/df[df.index.year==df.index[-1].year].iloc[0]-1)*100.0
    rets = rets.sort_values("20d", ascending=False)
    return rets.round(2)

def breadth(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for w in [20,50,200]:
        ma = df.rolling(w).mean()
        out[f">{w}DMA"] = (df.iloc[-1] > ma.iloc[-1]).mean()*100.0
    # 3m highs/lows
    win = 63
    highs = (df.iloc[-1] >= df.rolling(win).max().iloc[-1]).sum()
    lows = (df.iloc[-1] <= df.rolling(win).min().iloc[-1]).sum()
    out["3m highs"] = int(highs)
    out["3m lows"] = int(lows)
    return pd.DataFrame(out, index=["Breadth"]).T

def rolling_beta(series: pd.Series, market: pd.Series, window: int = 20) -> pd.Series:
    x = market.pct_change()
    y = series.pct_change()
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov/var
    return beta

# =====================
# Sidebar
# =====================
st.sidebar.title("AI Datacenter Theme Monitor")
with st.sidebar.expander("About This Tool", expanded=True):
    st.write(
        """
        Track price-based proxies for the AI compute and datacenter buildout.
        No logins. No keys. Data from Yahoo Finance via yfinance.
        - Equal-weight basket index vs SPY
        - Leaders and laggards over 1d, 5d, 20d, YTD
        - Breadth: % above 20/50/200 DMA, 3m highs and lows
        - Risk: rolling beta to SPY and drawdowns
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

# =====================
# Data
# =====================
spy = get_prices_cached(("SPY",), period=period)
px = get_prices_cached(tuple(tickers), period=period)

if px.empty or spy.empty:
    st.warning("No price data returned. Try fewer tickers or a shorter history.")
    st.stop()

# Align
px = px.dropna(how="all").dropna(axis=1, how="all")
spy = spy.squeeze()
px = px.loc[px.index.intersection(spy.index)]
spy = spy.loc[px.index]

# =====================
# KPIs
# =====================
idx = ew_index(px)
rel = idx / (spy/spy.iloc[0])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Basket total return", f"{(idx.iloc[-1]-1)*100:,.1f}%")
c2.metric("Basket vs SPY", f"{(rel.iloc[-1]-1)*100:,.1f}%")
c3.metric("1m return", f"{px.pct_change(21).mean(axis=1).iloc[-1]*100:,.1f}%")
c4.metric("Dispersion 20d σ", f"{px.pct_change(20).iloc[-1].std()*100:,.1f}%")

# =====================
# Charts
# =====================
st.subheader("Equal-weight index vs SPY")
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx.index, y=idx, mode="lines", name="Basket EW"))
fig.add_trace(go.Scatter(x=spy.index, y=spy/spy.iloc[0], mode="lines", name="SPY"))
fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Rolling 20d beta to SPY (key names)")
key_show = [t for t in ["NVDA","AMD","SMCI","AMZN","MSFT","GOOGL","EQIX","DLR"] if t in px.columns][:6]
fig2 = go.Figure()
for t in key_show:
    fig2.add_trace(go.Scatter(x=px.index, y=rolling_beta(px[t], spy, 20), mode="lines", name=t))
fig2.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig2, use_container_width=True)

# =====================
# Tables
# =====================
left, right = st.columns([1,1])
with left:
    st.subheader("Leaders & laggards")
    st.dataframe(returns_table(px), use_container_width=True)
with right:
    st.subheader("Breadth")
    st.dataframe(breadth(px), use_container_width=True)

st.caption("All series equal-weight. For investment decisions, combine with positioning and flows.")
