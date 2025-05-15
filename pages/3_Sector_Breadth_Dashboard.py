import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime

SECTOR_ETFS = {
    "S&P 500": "SPY",
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

st.set_page_config("Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

# Sidebar with About
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
**Sector Rotation Quadrant (Interactive):**  
Compares short-term (1W) and medium-term (3M) returns for each S&P 500 sector, visualizing leadership and laggards. The selected sector in the relative strength chart is automatically highlighted for easy comparison.

**Sector Relative Strength vs. S&P 500 (Interactive):**  
Plots the ratio of any sector ETF to the S&P 500 (SPY) for trend and leadership signals.

**Performance Heatmap & Breadth Table:**  
See which sectors are outperforming, and what percentage of each is above their 20, 50, and 200-day moving averages.
---
"""
    )
    lookback = st.slider("Lookback window (days)", 90, 365, 180, 15)
    st.caption("All charts use end-of-day ETF closes. Charts are interactive.")

# Data
@st.cache_data(ttl=21600)
def fetch_prices(tickers, period="max"):
    px = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    px = px.ffill().bfill()
    return px

tickers = list(SECTOR_ETFS.values())
prices = fetch_prices(tickers, period="max")
prices = prices.iloc[-lookback:]

# Performance Calculations
def perf_calc(s, d):
    if len(s) < d: return np.nan
    return (s.iloc[-1] / s.iloc[-d] - 1) * 100

def ytd_perf(s):
    ytd_idx = s.index.get_loc(s[s.index.year == datetime.now().year].index[0])
    return (s.iloc[-1] / s.iloc[ytd_idx] - 1) * 100

perf_windows = [("1W", 5), ("1M", 21), ("3M", 63)]
perf = {}
for name, ticker in SECTOR_ETFS.items():
    s = prices[ticker]
    perf[name] = {label: perf_calc(s, days) for label, days in perf_windows}
    try:
        perf[name]["YTD"] = ytd_perf(s)
    except Exception:
        perf[name]["YTD"] = np.nan
perf_df = pd.DataFrame(perf).T[["1W", "1M", "3M", "YTD"]].round(2)

# ====================
# Top Section: Interactive Charts
# ====================
col1, col2 = st.columns([1.25, 1.0])

# 1. Sector Relative Strength (Single select, Plotly)
with col1:
    st.markdown("Sector Relative Strength vs. S&P 500 (Interactive)")
    sector = st.selectbox("Select sector to compare to S&P 500:", [k for k in SECTOR_ETFS if k != "S&P 500"], index=0)
    ratio = prices[SECTOR_ETFS[sector]] / prices["SPY"]
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(
        x=ratio.index, y=ratio, name=f"{sector}/SPY", mode="lines",
        line=dict(width=2.5, color="#7d3c98"),
        hovertemplate=f"{sector}/SPY<br>Date: %{x|%Y-%m-%d}<br>Ratio: %{y:.3f}<extra></extra>"
    ))
    fig_rs.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=370, margin=dict(l=25, r=15, t=35, b=25),
        xaxis_title="", yaxis_title="Ratio",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig_rs, use_container_width=True)

# 2. Sector Rotation Quadrant (Plotly, only highlight selected sector)
with col2:
    st.markdown("Sector Rotation Quadrant (Interactive)")
    x = perf_df["3M"]
    y = perf_df["1W"]
    figq = go.Figure()
    # Quadrant backgrounds
    figq.add_shape(type="rect", x0=0, y0=0, x1=max(x.max(),1), y1=max(y.max(),1),
                   fillcolor="rgba(55,200,55,0.12)", line=dict(width=0)) # Leading
    figq.add_shape(type="rect", x0=min(x.min(),0), y0=0, x1=0, y1=max(y.max(),1),
                   fillcolor="rgba(60,160,255,0.10)", line=dict(width=0)) # Improving
    figq.add_shape(type="rect", x0=0, y0=min(y.min(),0), x1=max(x.max(),1), y1=0,
                   fillcolor="rgba(255,180,70,0.10)", line=dict(width=0)) # Weakening
    figq.add_shape(type="rect", x0=min(x.min(),0), y0=min(y.min(),0), x1=0, y1=0,
                   fillcolor="rgba(240,60,60,0.11)", line=dict(width=0)) # Lagging

    for sector_name in perf_df.index:
        if sector_name == sector:
            marker = dict(
                size=21, symbol="circle", color="#7d3c98",
                line=dict(width=2.5, color="black"),
            )
        else:
            marker = dict(
                size=16, symbol="circle", color="#cccccc",
                line=dict(width=1, color="#bbb"),
            )
        figq.add_trace(go.Scatter(
            x=[x[sector_name]], y=[y[sector_name]], mode="markers+text",
            marker=marker, text=[sector_name], textposition="bottom center",
            hovertemplate=f"{sector_name}<br>3M: {x[sector_name]:+.2f}%<br>1W: {y[sector_name]:+.2f}%",
            name=sector_name, showlegend=False
        ))
    figq.add_hline(0, line_width=1, line_dash="dot", line_color="gray")
    figq.add_vline(0, line_width=1, line_dash="dot", line_color="gray")
    figq.update_layout(
        xaxis_title="3M Return (%)", yaxis_title="1W Return (%)",
        height=370, margin=dict(l=35, r=20, t=35, b=30),
        template="plotly_white",
    )
    st.plotly_chart(figq, use_container_width=True)

# ======================
# Bottom Section: Heatmaps & Breadth
# ======================
st.markdown("Sector Performance Heatmap (%)")
st.dataframe(
    perf_df.style.background_gradient(cmap="RdYlGn", axis=0).format("{:+.2f}%"),
    use_container_width=True,
)

# Breadth Table
def pct_above_ma(series, window):
    ma = series.rolling(window).mean()
    return (series > ma).mean() * 100
ma_windows = [20, 50, 200]
breadth = {
    name: {f"%>{w}D": pct_above_ma(prices[ticker], w) for w in ma_windows}
    for name, ticker in SECTOR_ETFS.items()
}
breadth_df = pd.DataFrame(breadth).T[[f"%>{w}D" for w in ma_windows]].round(1)
st.markdown("Breadth Table: % Above Moving Averages")
st.dataframe(
    breadth_df.style.format("{:.1f}%").background_gradient(cmap="YlGn", axis=None),
    use_container_width=True,
)

# Mini Panel for Best/Worst YTD
best = perf_df["YTD"].idxmax()
worst = perf_df["YTD"].idxmin()
st.markdown(
    f"<div style='margin-top:1em;font-size:1.13em'>"
    f"<b>Best YTD Sector:</b> <span style='color:green'>{best} ({perf_df.loc[best, 'YTD']:+.2f}%)</span><br>"
    f"<b>Worst YTD Sector:</b> <span style='color:red'>{worst} ({perf_df.loc[worst, 'YTD']:+.2f}%)</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.caption("Data: Yahoo Finance | Calculations: AD Fund Management LP | All rights reserved.")
