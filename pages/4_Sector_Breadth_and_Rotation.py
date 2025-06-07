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
def robust_fetch(tickers, period="1y", interval="1d"):
    close_series = {}
    full_index = None
    for t in tickers:
        try:
            data = yf.download(t, period=period, interval=interval, progress=False)
            close = data.get("Adj Close", data.get("Close"))
            if close is not None and not close.empty:
                close = close.dropna()
                close_series[t] = close
                full_index = close.index if full_index is None else full_index.union(close.index)
        except Exception:
            continue

    if not close_series or full_index is None:
        return pd.DataFrame()

    for t in close_series:
        close_series[t] = close_series[t].reindex(full_index)

    df = pd.DataFrame(close_series).dropna(how="all")
    return df

tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch(tickers, period="1y", interval="1d")

if prices.empty:
    st.error("No valid data retrieved for the selected tickers. Please retry later.")
    st.stop()

if "SPY" not in prices.columns:
    st.error("'SPY' data not available. Cannot calculate relative strength without SPY.")
    st.stop()

available_sector_tickers = [t for t in SECTORS if t in prices.columns]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in prices.columns]

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

if missing_sectors:
    st.warning(f"Missing data for: {', '.join(missing_sectors)}")

if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data.")
    st.stop()

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
st.plotly_chart(fig_rs, use_container_width=True)

returns_1m = prices.pct_change(21).iloc[-1]
returns_3m = prices.pct_change(63).iloc[-1]

rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m,
    "Ticker": returns_1m.index
})
rotation_df = rotation_df[rotation_df["Ticker"].isin(available_sector_tickers)]
rotation_df["Sector"] = rotation_df["Ticker"].map(SECTORS)

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

df = rotation_df.set_index("Sector")[["1M Return", "3M Return"]]
st.subheader("Sector Total Returns Table")
st.dataframe(df.style.format("{:.2%}"), height=400)

csv = df.to_csv().encode()
st.download_button("Download returns table as CSV", csv, "sector_returns.csv", "text/csv")

st.caption("© 2025 AD Fund Management LP")
