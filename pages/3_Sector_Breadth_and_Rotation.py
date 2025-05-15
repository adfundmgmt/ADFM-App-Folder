import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---- Sectors dictionary in alphabetical order ----
SECTORS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

st.set_page_config(page_title="Sector Breadth & Rotation Monitor", layout="wide")

st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.markdown("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.\n\n")

# Sidebar with About This Tool description
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This tool visualizes key market dynamics for S&P 500 sectors using data from Yahoo Finance.

        **Features:**
        - Relative Strength: Selected sector price ratio vs S&P 500 (SPY)
        - Sector Rotation: 1-month vs 3-month returns quadrant
        - Breadth Table: % of sectors above their 20, 50, 100, and 200-day moving averages

        Data refreshes on load. Use for market breadth insights and sector rotation signals.
        """
    )

# Append SPY to tickers for benchmark
tickers = list(SECTORS.values()) + ["SPY"]

@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    df = yf.download(tickers, period=period, interval=interval, progress=False)
    # Extract adjusted close prices, handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    return df.dropna(how="all")

prices = fetch_prices(tickers)

if prices.empty:
    st.error("Failed to fetch data from Yahoo Finance. Check your internet connection or ticker symbols.")
    st.stop()

# Sector selector (alphabetically by sector name)
selected_sector = st.selectbox("Select sector to compare with S&P 500:", list(SECTORS.keys()))
selected_ticker = SECTORS[selected_sector]

# --- Sector Relative Strength Chart ---
sector_ratio = prices[selected_ticker] / prices["SPY"]

fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=sector_ratio.index,
    y=sector_ratio,
    mode="lines",
    name=f"{selected_sector} / SPY",
    line=dict(color="orange", width=3)
))
fig_rs.update_layout(
    title=f"{selected_sector} Relative Strength vs. S&P 500",
    xaxis_title="Date",
    yaxis_title="Ratio",
    template="plotly_white",
    height=400
)
st.plotly_chart(fig_rs, use_container_width=True)

# --- Sector Rotation Quadrant (1M vs 3M returns) ---
end_date = prices.index[-1]
start_1m = end_date - timedelta(days=30)
start_3m = end_date - timedelta(days=90)

# Find nearest available dates in the index (backfill)
date_1m = prices.index[prices.index.get_loc(start_1m, method='bfill')]
date_3m = prices.index[prices.index.get_loc(start_3m, method='bfill')]

returns_1m = prices.loc[end_date] / prices.loc[date_1m] - 1
returns_3m = prices.loc[end_date] / prices.loc[date_3m] - 1

rotation_df = pd.DataFrame({
    "Ticker": list(SECTORS.values()),
    "Sector": list(SECTORS.keys()),
    "1M Return": returns_1m[list(SECTORS.values())].values,
    "3M Return": returns_3m[list(SECTORS.values())].values
})

fig_rot = go.Figure()
fig_rot.add_trace(go.Scatter(
    x=rotation_df["1M Return"],
    y=rotation_df["3M Return"],
    mode="markers+text",
    text=rotation_df["Sector"],
    textposition="top center",
    marker=dict(size=12, color="blue"),
    hovertemplate="%{text}<br>1M Return: %{x:.2%}<br>3M Return: %{y:.2%}<extra></extra>"
))
fig_rot.update_layout(
    title="Sector Rotation Quadrant (1M vs 3M Returns)",
    xaxis_title="1 Month Return",
    yaxis_title="3 Month Return",
    template="plotly_white",
    height=500,
    xaxis=dict(tickformat=".0%", zeroline=True),
    yaxis=dict(tickformat=".0%", zeroline=True),
)
st.plotly_chart(fig_rot, use_container_width=True)

# --- Breadth Table: % of sectors above key moving averages ---
ma_periods = [20, 50, 100, 200]
breadth = {}
for period in ma_periods:
    ma = prices[list(SECTORS.values())].rolling(period).mean()
    above_ma = (prices[list(SECTORS.values())].iloc[-1] > ma.iloc[-1])
    breadth[f"Above {period}D"] = above_ma.mean() * 100

breadth_df = pd.DataFrame({
    "MA Window": [f"{p}D" for p in ma_periods],
    "% Sectors Above MA": [round(breadth[f"Above {p}D"], 1) for p in ma_periods]
})

st.subheader("% of Sectors Above Key Moving Averages")
st.dataframe(breadth_df, use_container_width=True)
