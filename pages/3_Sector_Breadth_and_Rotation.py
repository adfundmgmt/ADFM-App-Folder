import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---- Sector ETFs and names in alphabetical order ----
SECTOR_ETFS = {
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
    "S&P 500": "SPY",
}

SECTORS = sorted([s for s in SECTOR_ETFS.keys() if s != "S&P 500"])

st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.markdown("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

# ---- Fetch prices with caching ----
@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    raw_data = yf.download(tickers, period=period, interval=interval, progress=False)
    if isinstance(raw_data.columns, pd.MultiIndex):
        adj_close = raw_data["Adj Close"]
    else:
        adj_close = raw_data
    if isinstance(adj_close.columns, pd.MultiIndex):
        adj_close.columns = adj_close.columns.get_level_values(1)
    return adj_close

# Fetch prices for all sectors + SPY
tickers = [SECTOR_ETFS[s] for s in SECTORS] + [SECTOR_ETFS["S&P 500"]]
prices = fetch_prices(tickers)

if prices.empty or SECTOR_ETFS["S&P 500"] not in prices.columns:
    st.error("No sector or SPY data found. Check Yahoo Finance or your internet connection.")
    st.stop()

# ---- User selection ----
selected_sector = st.selectbox("Select sector to compare with S&P 500:", SECTORS)

# Use last 12 months for all analysis
end_date = prices.index[-1]
start_date = end_date - pd.DateOffset(months=12)
prices_12m = prices.loc[start_date:end_date]

# ---- Sector Relative Strength vs SPY ----
st.subheader(f"Sector Relative Strength vs. S&P 500: {selected_sector}")

# Calculate ratio of sector / SPY
ratio = prices_12m[selected_sector] / prices_12m[SECTOR_ETFS["S&P 500"]]

fig_rs = px.line(
    x=ratio.index,
    y=ratio.values,
    labels={"x": "Date", "y": f"{selected_sector} / SPY Ratio"},
    title=f"{selected_sector} Relative Strength vs SPY (Last 12 Months)",
)
st.plotly_chart(fig_rs, use_container_width=True)

# ---- Sector Rotation Quadrant (1M vs 3M returns) ----
st.subheader("Sector Rotation Quadrant (1M vs 3M Returns)")

returns_1m = prices_12m.pct_change(21).iloc[-1]  # approx 21 trading days in 1 month
returns_3m = prices_12m.pct_change(63).iloc[-1]  # approx 63 trading days in 3 months

rotation_df = pd.DataFrame({
    "Sector": SECTORS,
    "1M Return": returns_1m[[SECTOR_ETFS[s] for s in SECTORS]].values,
    "3M Return": returns_3m[[SECTOR_ETFS[s] for s in SECTORS]].values,
})

fig_rot = px.scatter(
    rotation_df,
    x="1M Return",
    y="3M Return",
    text="Sector",
    labels={"1M Return": "1-Month Return", "3M Return": "3-Month Return"},
    title="Sector Rotation Quadrant (1M vs 3M Returns)",
    size_max=15,
)
fig_rot.update_traces(textposition="top center")
fig_rot.add_shape(
    type="line", x0=0, y0=rotation_df["3M Return"].min(), x1=0, y1=rotation_df["3M Return"].max(),
    line=dict(dash="dash", color="gray")
)
fig_rot.add_shape(
    type="line", x0=rotation_df["1M Return"].min(), y0=0, x1=rotation_df["1M Return"].max(), y1=0,
    line=dict(dash="dash", color="gray")
)
st.plotly_chart(fig_rot, use_container_width=True)

# ---- Breadth Table: % Above Moving Averages ----
st.subheader("Percentage of Sector Constituents Above Moving Averages")

# Fetch individual sector constituent data and calculate % above moving averages
# For simplicity here: calculate % above 20, 50, 100, 200D for sector ETFs themselves (proxy)

def pct_above_ma(series, window):
    ma = series.rolling(window).mean()
    return (series > ma).mean() * 100

breadth_data = []
for sector in SECTORS:
    series = prices_12m[SECTOR_ETFS[sector]]
    breadth_data.append({
        "Sector": sector,
        "% Above 20D": pct_above_ma(series, 20),
        "% Above 50D": pct_above_ma(series, 50),
        "% Above 100D": pct_above_ma(series, 100),
        "% Above 200D": pct_above_ma(series, 200),
    })

breadth_df = pd.DataFrame(breadth_data).set_index("Sector").sort_index()

st.dataframe(breadth_df.style.background_gradient(cmap="Greens"))

