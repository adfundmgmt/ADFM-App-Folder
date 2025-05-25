import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
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
missing_sectors = [SECTORS[t] for t in SECTORS if t not in prices.columns]

# Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard monitors the relative performance and rotation of S&P 500 sectors to help identify leadership trends and market breadth dynamics.

        **Features:**
        - Sector Relative Strength vs. S&P 500  
        - Sector Rotation Quadrant (1M vs 3M returns)  
        - Downloadable & sortable sector performance table  
        - Robust to missing sector tickers

        **Data Source:**  
        Yahoo Finance (via yfinance), refreshed hourly.

        *Built by AD Fund Management LP.*
        """
    )
    st.markdown("---")
    st.write(f"**Sectors available:** {', '.join([SECTORS[t] for t in available_sector_tickers])}")
    if missing_sectors:
        st.warning(f"Missing data for: {', '.join(missing_sectors)}")

if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data.")
    st.stop()

# --- Relative Strength vs SPY Chart ---
relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)
selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=available_sector_tickers,
    format_func=lambda x: SECTORS[x]
)

fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs. S&P 500 (SPY)",
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=40))
st.plotly_chart(fig_rs, use_container_width=True)

# --- Sector Rotation Quadrant (Scatter) ---
returns_1m = prices.pct_change(21).iloc[-1]
returns_3m = prices.pct_change(63).iloc[-1]

rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m,
    "Ticker": returns_1m.index
})
rotation_df = rotation_df[rotation_df["Ticker"].isin(available_sector_tickers)]
rotation_df["Sector"] = rotation_df["Ticker"].map(SECTORS)
rotation_df = rotation_df.sort_values("Sector")

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

# --- One Table: Sortable, Percentage Format ---
df = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "1M Return": [returns_1m[t] for t in available_sector_tickers],
    "3M Return": [returns_3m[t] for t in available_sector_tickers]
}).set_index("Sector")

st.subheader("Sector Total Returns Table")
st.dataframe(df.style.format("{:.2%}"), height=400)

csv = df.to_csv().encode()
st.download_button(
    label="Download returns table as CSV",
    data=csv,
    file_name="sector_returns.csv",
    mime="text/csv"
)

st.caption("© 2025 AD Fund Management LP")
