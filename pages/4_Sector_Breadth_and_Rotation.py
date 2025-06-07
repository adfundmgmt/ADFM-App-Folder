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
            # For multi-index, flatten
            if isinstance(data.columns, pd.MultiIndex):
                if ('Adj Close', t) in data.columns:
                    close = data[('Adj Close', t)]
                elif ('Close', t) in data.columns:
                    close = data[('Close', t)]
                else:
                    continue
            else:
                if "Adj Close" in data.columns:
                    close = data["Adj Close"]
                elif "Close" in data.columns:
                    close = data["Close"]
                else:
                    continue
            close = close.dropna()
            close_series[t] = close
            if full_index is None:
                full_index = close.index
            else:
                full_index = full_index.union(close.index)
        except Exception:
            continue

    if not close_series:
        return pd.DataFrame()
    for t in close_series:
        close_series[t] = close_series[t].reindex(full_index)
    df = pd.DataFrame(close_series)
    df = df.dropna(how="all")
    return df

tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch(tickers, period="1y", interval="1d")

if prices.empty:
    st.error("Downloaded data is empty. Yahoo Finance API might be rate-limited or unavailable.")
    st.stop()

price_cols = [str(c).strip() for c in prices.columns]
available_sector_tickers = [t for t in SECTORS.keys() if t in price_cols]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in price_cols]

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
    if available_sector_tickers:
        st.write(f"**Sectors available:** {', '.join([SECTORS[t] for t in available_sector_tickers])}")
    if missing_sectors:
        st.warning(f"Missing data for: {', '.join(missing_sectors)}")

if not available_sector_tickers:
    st.error("No sector tickers found in the downloaded data. Try refreshing in a few minutes.")
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
