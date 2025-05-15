import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---- S&P 500 Sector ETFs ----
SECTOR_ETFS = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
SPY_TICKER = "SPY"

st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")
st.caption("Built by AD Fund Management LP. Data: Yahoo Finance. For informational use only.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
- **Sector Relative Strength vs. S&P 500:** Plot the ratio of your chosen sector to the S&P 500.
- **Sector Rotation Quadrant:** 1M vs 3M returns for all sectors — instantly see leaders/laggards.
- **Breadth Table:** % of sectors above their 20, 50, 100, and 200-day moving averages.
    """)

sector_list = list(SECTOR_ETFS.keys())
selected_sector = st.selectbox("Select sector to compare with S&P 500:", sector_list, index=0)

# ---- Fetch price data for all sectors + SPY (last 12 months) ----
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data(ttl=3600)
def fetch_all_sector_prices():
    tickers = list(SECTOR_ETFS.values()) + [SPY_TICKER]
    price_data = {}
    failed = []
    for t in tickers:
        df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
        # Valid data: must be a pd.Series with at least 10 dates
        if not df.empty and "Close" in df.columns:
            series = df["Close"].dropna()
            if isinstance(series, pd.Series) and len(series) > 10 and isinstance(series.index, pd.DatetimeIndex):
                price_data[t] = series
            else:
                failed.append(t)
        else:
            failed.append(t)
    # Only build the DataFrame from valid series
    if not price_data:
        st.error("No sector or SPY data found. Check Yahoo Finance or your internet connection.")
        st.stop()
    if failed:
        st.sidebar.warning(f"Could not fetch data for: {', '.join(failed)}")
    # Outer join so all series align by date
    return pd.DataFrame(price_data)

prices = fetch_all_sector_prices()

if SECTOR_ETFS[selected_sector] not in prices or SPY_TICKER not in prices:
    st.error("Could not fetch data for selected sector or SPY.")
    st.stop()

# ---- 1. Sector Relative Strength (Selected) ----
rel_strength = prices[SECTOR_ETFS[selected_sector]] / prices[SPY_TICKER]
fig_rel = px.line(
    rel_strength,
    title=f"{selected_sector} / S&P 500 (SPY) – 12M Relative Strength",
    labels={"value": "Sector / SPY (Ratio)", "index": "Date"},
    height=400,
)
fig_rel.update_traces(line_color="orange", name=f"{selected_sector}/SPY")
fig_rel.update_layout(title_x=0.01, showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
st.subheader(f"Sector Relative Strength vs S&P 500 ({selected_sector})")
st.plotly_chart(fig_rel, use_container_width=True)

# ---- 2. Sector Rotation Quadrant (1M vs 3M return) ----
returns_1m = prices.pct_change(21).iloc[-1]  # ~21 trading days = 1M
returns_3m = prices.pct_change(63).iloc[-1]  # ~63 trading days = 3M
rotation_df = pd.DataFrame({
    "Sector": [s for s, t in SECTOR_ETFS.items() if t in prices],
    "1M Return": [returns_1m[SECTOR_ETFS[s]] for s in SECTOR_ETFS if SECTOR_ETFS[s] in prices],
    "3M Return": [returns_3m[SECTOR_ETFS[s]] for s in SECTOR_ETFS if SECTOR_ETFS[s] in prices],
})
highlight_color = "#ffa600"
rotation_fig = px.scatter(
    rotation_df,
    x="1M Return",
    y="3M Return",
    text="Sector",
    title="Sector Rotation Quadrant: 1M vs 3M Returns",
    labels={"1M Return":"1 Month Return", "3M Return":"3 Month Return"},
    height=400,
)
rotation_fig.update_traces(
    marker=dict(size=13, color=[highlight_color if s == selected_sector else "#636efa" for s in rotation_df["Sector"]]),
    selector=dict(mode='markers+text')
)
rotation_fig.update_layout(title_x=0.01, margin=dict(l=40, r=40, t=60, b=40))
st.subheader("Sector Rotation Quadrant (1M vs 3M Returns)")
st.plotly_chart(rotation_fig, use_container_width=True)

# ---- 3. Sector Breadth Table ----
breadth = {}
for s, t in SECTOR_ETFS.items():
    if t in prices:
        close = prices[t]
        breadth[s] = {
            "Above 20D": "✅" if close.iloc[-1] > close.rolling(20).mean().iloc[-1] else "",
            "Above 50D": "✅" if close.iloc[-1] > close.rolling(50).mean().iloc[-1] else "",
            "Above 100D": "✅" if close.iloc[-1] > close.rolling(100).mean().iloc[-1] else "",
            "Above 200D": "✅" if close.iloc[-1] > close.rolling(200).mean().iloc[-1] else "",
        }
breadth_df = pd.DataFrame(breadth).T

st.subheader("% of Sectors Above 20, 50, 100, 200-Day MAs")
st.dataframe(
    breadth_df,
    use_container_width=True,
    hide_index=False,
    height=420,
)

# Footer
st.caption("© 2025 AD Fund Management LP")
