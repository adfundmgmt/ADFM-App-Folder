import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")

st.title("S&P 500 Sector Breadth & Rotation Monitor")

# Sector tickers and names in alphabetical order
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

# Timeframe for data: last 12 months
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)

@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    raw_data = yf.download(tickers, period=period, interval=interval, progress=False)
    # Handle multi-index columns (multiple tickers)
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            adj_close = raw_data.loc[:, "Adj Close"]
        else:
            adj_close = raw_data.loc[:, "Close"]
    else:
        # Single ticker case
        if "Adj Close" in raw_data.columns:
            adj_close = raw_data["Adj Close"]
        else:
            adj_close = raw_data["Close"]

    # Flatten multiindex columns if still present
    if isinstance(adj_close.columns, pd.MultiIndex):
        adj_close.columns = adj_close.columns.get_level_values(1)
    return adj_close.dropna(how="all")

tickers = list(SECTORS.keys()) + ["SPY"]

# Fetch price data
prices = fetch_prices(tickers, period="1y", interval="1d")

if prices.empty:
    st.error("No price data found for sectors or SPY. Check Yahoo Finance or your internet connection.")
    st.stop()

# Calculate sector relative strength vs SPY (ratio)
relative_strength = prices[SECTORS.keys()].div(prices["SPY"], axis=0)

# Sidebar with About This Tool description
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard monitors the relative performance and rotation of S&P 500 sectors to help identify leadership trends and market breadth dynamics.

        **Features:**

        - **Sector Relative Strength vs. S&P 500**  
          Visualizes the price ratio of the selected sector ETF against SPY over the past 12 months, helping to track sector leadership or weakness.

        - **Sector Rotation Quadrant**  
          A scatter plot comparing 1-month vs 3-month total returns for all sectors. Quickly identifies which sectors are leading or lagging in the current market cycle.

        - **Data Source**  
          All price data is pulled from Yahoo Finance via yfinance, refreshed hourly.

        *Built by AD Fund Management LP. For informational use only.*
        """
    )

# Select sector to display (alphabetical order)
selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=sorted(SECTORS.keys()),
    format_func=lambda x: SECTORS[x]
)

# Plot relative strength ratio chart with Plotly for interactivity
fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs. S&P 500 (SPY)",
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=40))
st.plotly_chart(fig_rs, use_container_width=True)

# Sector Rotation Quadrant (1M vs 3M total return)
returns_1m = prices.pct_change(21).iloc[-1]  # Approx 21 trading days = 1 month
returns_3m = prices.pct_change(63).iloc[-1]  # Approx 63 trading days = 3 months

rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m,
    "Sector": [SECTORS.get(t, t) for t in returns_1m.index]
}).reset_index(drop=True)

fig_rot = px.scatter(
    rotation_df,
    x="3M Return",
    y="1M Return",
    text="Sector",
    title="Sector Rotation Quadrant (1M vs 3M Total Return)",
    labels={"3M Return": "3-Month Return", "1M Return": "1-Month Return"},
    color="Sector",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    hover_data={"3M Return": ':.2%', "1M Return": ':.2%'}
)
fig_rot.update_traces(textposition="top center")
fig_rot.update_layout(height=500, margin=dict(l=40, r=40, t=60, b=40), showlegend=False)
st.plotly_chart(fig_rot, use_container_width=True)
