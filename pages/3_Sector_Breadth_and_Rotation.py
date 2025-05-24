import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")

# S&P 500 sector ETF tickers and names
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

# Data window: last 12 months
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)

@st.cache_data(ttl=3600)
def fetch_prices(tickers, period="1y", interval="1d"):
    raw_data = yf.download(tickers, period=period, interval=interval, progress=False)
    # Multi-index (multi-ticker) handling
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            adj_close = raw_data.loc[:, "Adj Close"]
        else:
            adj_close = raw_data.loc[:, "Close"]
    else:
        # Single ticker fallback
        if "Adj Close" in raw_data.columns:
            adj_close = raw_data["Adj Close"]
        else:
            adj_close = raw_data["Close"]
    # Flatten multiindex columns if needed
    if isinstance(adj_close.columns, pd.MultiIndex):
        adj_close.columns = adj_close.columns.get_level_values(1)
    return adj_close.dropna(how="all")

tickers = list(SECTORS.keys()) + ["SPY"]
prices = fetch_prices(tickers, period="1y", interval="1d")

if prices.empty:
    st.error("No price data found for sectors or SPY. Check Yahoo Finance or your internet connection.")
    st.stop()

# Identify which sector ETFs are available in the data
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

# Only proceed if at least one sector is available
if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data. Please try again later.")
    st.stop()

# Calculate sector relative strength vs SPY
relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)

# User selects which sector to display
selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=available_sector_tickers,
    format_func=lambda x: SECTORS[x]
)

# Plot relative strength line chart
fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs. S&P 500 (SPY)",
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=40))
st.plotly_chart(fig_rs, use_container_width=True)

# Sector Rotation Quadrant (1M vs 3M total return)
returns_1m = prices.pct_change(21).iloc[-1]  # 21 trading days ≈ 1 month
returns_3m = prices.pct_change(63).iloc[-1]  # 63 trading days ≈ 3 months

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

# --- Sector Performance Table (sortable and formatted) ---

st.subheader("Sector Total Returns Table (sortable)")
# Show numeric values for sorting
st.dataframe(
    rotation_df[["Sector", "1M Return", "3M Return"]].set_index("Sector"),
    height=350
)

# Show pretty-formatted, non-sortable summary table for reporting
st.write("Formatted for screenshots/reporting (not sortable):")
rotation_df_show = rotation_df[["Sector", "1M Return", "3M Return"]].copy()
rotation_df_show["1M Return"] = rotation_df_show["1M Return"].apply(lambda x: f"{x:.2%}")
rotation_df_show["3M Return"] = rotation_df_show["3M Return"].apply(lambda x: f"{x:.2%}")
st.table(rotation_df_show.set_index("Sector"))

# Option to download returns table as CSV (numeric for sorting in Excel)
csv = rotation_df[["Sector", "1M Return", "3M Return"]].to_csv(index=False).encode()
st.download_button(
    label="Download returns table as CSV",
    data=csv,
    file_name="sector_returns.csv",
    mime="text/csv"
)

st.caption("© 2025 AD Fund Management LP")
