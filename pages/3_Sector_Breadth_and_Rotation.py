import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# ----- Sector ETFs -----
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

# Alphabetical order of sectors
SECTORS = sorted([k for k in SECTOR_ETFS.keys() if k != "S&P 500"])

st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

# Sidebar
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Sector Relative Strength vs. S&P 500: Plot the ratio of your chosen sector to the S&P 500.

    Sector Rotation Quadrant: 1M vs 3M total returns for all sectors to spot leaders/laggards.

    Breadth Table: Percentage of sectors above their 20, 50, 100, and 200-day moving averages.
    """
)

# Select sector
selected_sector = st.selectbox("Select sector to compare with S&P 500:", SECTORS)

# ----- Fetch data -----
@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    # Download price data
    data = yf.download(tickers, period=period, interval=interval, progress=False)["Adj Close"]
    return data

tickers = [SECTOR_ETFS[s] for s in SECTORS] + [SECTOR_ETFS["S&P 500"]]
prices = fetch_prices(tickers)

if prices.empty or SECTOR_ETFS["S&P 500"] not in prices.columns:
    st.error("No sector or SPY data found. Check Yahoo Finance or your internet connection.")
    st.stop()

# ----- Sector Relative Strength vs SPY -----
def plot_relative_strength(prices, sector_ticker, spy_ticker="SPY"):
    ratio = prices[sector_ticker] / prices[spy_ticker]
    df = ratio.reset_index().rename(columns={sector_ticker: "Relative Strength"})
    fig = px.line(df, x="Date", y="Relative Strength",
                  title=f"{selected_sector} Relative Strength vs. S&P 500",
                  labels={"Relative Strength": f"{selected_sector} / SPY", "Date": "Date"})
    fig.update_layout(height=400)
    return fig

fig_rs = plot_relative_strength(prices, SECTOR_ETFS[selected_sector])
st.plotly_chart(fig_rs, use_container_width=True)

# ----- Sector Rotation Quadrant (1M vs 3M Total Returns) -----
def get_total_return(prices, latest_date, months_ago):
    date_ago = latest_date - timedelta(days=months_ago*30)
    # Find closest date on or before date_ago
    valid_dates = prices.index[prices.index <= date_ago]
    if len(valid_dates) == 0:
        return None
    closest_date = valid_dates.max()
    return prices.loc[latest_date] / prices.loc[closest_date] - 1

latest_date = prices.index.max()

returns_1m = {}
returns_3m = {}
for ticker in tickers:
    ret_1m = get_total_return(prices[ticker], latest_date, 1)
    ret_3m = get_total_return(prices[ticker], latest_date, 3)
    if ret_1m is not None and ret_3m is not None:
        returns_1m[ticker] = ret_1m
        returns_3m[ticker] = ret_3m

rot_df = pd.DataFrame({
    "1M Return": pd.Series(returns_1m),
    "3M Return": pd.Series(returns_3m),
})

# Map tickers back to sector names (exclude SPY)
inv_sector_etfs = {v: k for k, v in SECTOR_ETFS.items()}
rot_df["Sector"] = rot_df.index.map(inv_sector_etfs)
rot_df = rot_df[rot_df["Sector"] != "S&P 500"].reset_index(drop=True)

fig_rot = px.scatter(rot_df, x="1M Return", y="3M Return", text="Sector",
                     title="Sector Rotation Quadrant (1M vs 3M Returns)",
                     labels={"1M Return": "1 Month Return", "3M Return": "3 Month Return"},
                     height=450)
fig_rot.update_traces(marker=dict(size=15, color="#1f77b4"))
fig_rot.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
st.plotly_chart(fig_rot, use_container_width=True)

# ----- Breadth Table: % Above Moving Averages -----
def calculate_breadth(prices, windows=[20, 50, 100, 200]):
    breadth_data = {}
    for window in windows:
        ma = prices.rolling(window=window).mean()
        breadth = (prices > ma).tail(1).T.astype(float) * 100
        breadth_data[f"Above {window}D"] = breadth.iloc[:, 0]
    breadth_df = pd.DataFrame(breadth_data)
    breadth_df["Sector"] = breadth_df.index.map(inv_sector_etfs)
    breadth_df = breadth_df[breadth_df["Sector"] != "S&P 500"]
    return breadth_df.set_index("Sector").sort_index()

breadth_df = calculate_breadth(prices)

st.markdown("### % of Sectors Above Moving Averages")
st.dataframe(breadth_df.style.format("{:.2f}").background_gradient(cmap="Greens"), height=350)

