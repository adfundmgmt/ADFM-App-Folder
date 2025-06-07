import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

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
def fetch_adj_close(tickers, period="1y", interval="1d"):
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame(), []
    if data is None or data.empty:
        return pd.DataFrame(), []

    # Handle single ticker (simple columns) or multi ticker (MultiIndex)
    if isinstance(data.columns, pd.Index) and "Adj Close" in data.columns:
        # Only one ticker - ensure always DataFrame shape [date, ticker]
        t = tickers[0] if isinstance(tickers, list) else tickers
        df = data[["Adj Close"]].rename(columns={"Adj Close": t})
        available = [t]
        return df, available

    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: get 'Adj Close' for all tickers
        if "Adj Close" in data.columns.get_level_values(0):
            df = data["Adj Close"].copy()
            df = df.dropna(how="all")
            available = list(df.columns)
            return df, available

    # Fallback
    return pd.DataFrame(), []

# --- FETCH DATA ---
tickers = list(SECTORS.keys()) + ["SPY"]
prices, available_columns = fetch_adj_close(tickers, period="1y", interval="1d")

if prices.empty:
    st.error("No valid data retrieved for sector ETFs or SPY. Check Yahoo Finance or try again later.")
    st.stop()

available_sector_tickers = [t for t in SECTORS if t in available_columns]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in available_columns]

if "SPY" not in available_columns:
    st.error("SPY data is missing. No relative analytics can be displayed.")
    st.stop()

with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard monitors the relative performance and rotation of S&P 500 sectors to identify leadership trends and market breadth.

    **Features:**
    - Sector Relative Strength vs. S&P 500
    - Sector Rotation Quadrant (1M vs 3M returns)
    - Downloadable & sortable sector performance table

    **Data Source:** Yahoo Finance (via yfinance)
    
    *Built by AD Fund Management LP.*
    """)
    if available_sector_tickers:
        st.success(f"Sectors available: {', '.join([SECTORS[t] for t in available_sector_tickers])}")
    if missing_sectors:
        st.warning(f"Missing data for: {', '.join(missing_sectors)}")
    if "SPY" not in available_columns:
        st.error("'SPY' is missing from price columns.")

if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data. Wait a few minutes or check your network/API access.")
    st.stop()

# --- RELATIVE STRENGTH ---
try:
    relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)
except Exception as e:
    st.error(f"Failed to compute relative strength: {e}")
    st.stop()

selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=available_sector_tickers,
    format_func=lambda x: SECTORS[x]
)

try:
    fig_rs = px.line(
        relative_strength[selected_sector].dropna(),
        title=f"Relative Strength: {SECTORS[selected_sector]} vs. S&P 500 (SPY)",
        labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
    )
    st.plotly_chart(fig_rs, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to display relative strength chart for {selected_sector}: {e}")

# --- ROTATION QUADRANT ---
try:
    returns_1m = prices.pct_change(21).iloc[-1]
    returns_3m = prices.pct_change(63).iloc[-1]
except Exception as e:
    st.warning(f"Failed to calculate 1M/3M returns: {e}")
    returns_1m = pd.Series(dtype=float)
    returns_3m = pd.Series(dtype=float)

rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m,
    "Ticker": returns_1m.index if not returns_1m.empty else []
})
if not rotation_df.empty:
    rotation_df = rotation_df[rotation_df["Ticker"].isin(available_sector_tickers)]
    rotation_df["Sector"] = rotation_df["Ticker"].map(SECTORS)
    try:
        fig_rot = px.scatter(
            rotation_df,
            x="3M Return",
            y="1M Return",
            text="Sector",
            title="Sector Rotation Quadrant (1M vs 3M Total Return)",
            labels={"3M Return": "3-Month Return", "1M Return": "1-Month Return"},
            color="Sector",
        )
        fig_rot.update_traces(textposition="top center")
        st.plotly_chart(fig_rot, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to render sector rotation chart: {e}")
else:
    st.info("Not enough data to display rotation quadrant.")

# --- SECTOR TABLE ---
try:
    if not rotation_df.empty:
        df = rotation_df.set_index("Sector")[["1M Return", "3M Return"]]
        st.subheader("Sector Total Returns Table")
        st.dataframe(df.style.format("{:.2%}"), height=400)
        csv = df.to_csv().encode()
        st.download_button("Download returns table as CSV", csv, "sector_returns.csv", "text/csv")
    else:
        st.info("No sector returns to display or download.")
except Exception as e:
    st.warning(f"Unable to display or download returns table: {e}")

st.caption("© 2025 AD Fund Management LP")
