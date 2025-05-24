import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")

SECTORS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    raw_data = yf.download(tickers, period=period, interval=interval, progress=False)
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            adj_close = raw_data.loc[:, "Adj Close"]
        else:
            adj_close = raw_data.loc[:, "Close"]
    else:
        if "Adj Close" in raw_data.columns:
            adj_close = raw_data["Adj Close"]
        else:
            adj_close = raw_data["Close"]
    if isinstance(adj_close.columns, pd.MultiIndex):
        adj_close.columns = adj_close.columns.get_level_values(1)
    return adj_close.dropna(how="all")

tickers = list(SECTORS.keys()) + ["SPY"]
prices = fetch_prices(tickers, period="1y", interval="1d")

available_sector_tickers = [t for t in SECTORS.keys() if t in prices.columns]
if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data.")
    st.stop()

# --- Returns Data ---
returns_1m = prices.pct_change(21).iloc[-1]
returns_3m = prices.pct_change(63).iloc[-1]

df = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "1M Return": [returns_1m[t] for t in available_sector_tickers],
    "3M Return": [returns_3m[t] for t in available_sector_tickers]
}).set_index("Sector")

# --- Single, sortable percent table ---
st.subheader("Sector Total Returns Table")
st.dataframe(df.style.format("{:.2%}"), height=400)

# --- CSV Download ---
csv = df.to_csv().encode()
st.download_button(
    label="Download returns table as CSV",
    data=csv,
    file_name="sector_returns.csv",
    mime="text/csv"
)

st.caption("© 2025 AD Fund Management LP")
