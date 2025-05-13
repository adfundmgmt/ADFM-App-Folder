# Market Drawdown & Breadth Dashboard — Streamlit app
# ---------------------------------------------------
# Built for Arya — AD Fund Management LP
# ---------------------------------------------------
# Key features
#   • Replicates Charles Schwab-style tables: index drawdowns & breadth (% of members beating the index)
#   • Dynamic date-picker ("as-of"), index selector (S&P 500, Nasdaq-100, Russell 2000, Dow 30)
#   • Pulls live prices via yfinance; constituents scraped on-the-fly (cached) from Wikipedia/Official index pages
#   • Bells & whistles:
#       – KPI tiles (st.metric) on top
#       – Interactive Plotly line chart with peak/valley markers
#       – Download buttons for each table (CSV)
#       – Optional dark/light theme toggle (uses Streamlit theme config)
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --------------------------
# Config & helper functions
# --------------------------
st.set_page_config(
    page_title="Market Drawdown & Breadth Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def get_index_members(index: str) -> list:
    """Return constituent tickers for a given index."""
    if index == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tbl = pd.read_html(url, header=0)[0]
        return tbl["Symbol"].tolist()
    if index == "Nasdaq 100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tbl = pd.read_html(url, header=0, match="Ticker")[0]
        return tbl["Ticker"].str.replace("\u200a", "").tolist()
    if index == "Russell 2000":
        # Russell publishes a CSV; easier to use ETF holdings as proxy
        holdings = yf.Ticker("IWM").fund_holdings
        return holdings["symbol"].tolist()
    if index == "Dow 30":
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tbl = pd.read_html(url, header=0, match="Symbol")[1]
        return tbl["Symbol"].tolist()
    return []

@st.cache_data(show_spinner=False)
def get_price_series(tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    """Download Adj Close prices for list of tickers."""
    df = yf.download(tickers=tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)["Adj Close"]
    # If a single ticker, yfinance returns a Series; convert to DF for consistency
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df

def calc_drawdown(prices: pd.Series) -> float:
    cummax = prices.cummax()
    draw = (prices / cummax - 1).min()
    return draw

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Settings")
selected_index = st.sidebar.selectbox(
    "Select index", ("S&P 500", "Nasdaq 100", "Russell 2000", "Dow 30"))

as_of = st.sidebar.date_input("As-of date", value=datetime.today())
if as_of > datetime.today().date():
    st.sidebar.error("As-of date cannot be in the future")
    st.stop()

year_start = datetime(as_of.year, 1, 1)

# --------------------------
# Retrieve index-level data
# --------------------------
index_ticker_map = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow 30": "^DJI",
}
index_tk = index_ticker_map[selected_index]

idx_px = get_price_series([index_tk], start=year_start, end=as_of + timedelta(days=1))[index_tk]

# Return metrics
ytd_ret = idx_px.iloc[-1] / idx_px.iloc[0] - 1
trough_price = idx_px.min()
peak_price = idx_px.max()
ret_from_trough = idx_px.iloc[-1] / trough_price - 1
max_dd = idx_px.iloc[-1] / peak_price - 1  # negative value

# KPI tiles
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(label="YTD Return", value=f"{ytd_ret:+.1%}")
kpi2.metric(label="Return from YTD Low", value=f"{ret_from_trough:+.1%}")
kpi3.metric(label="Max Drawdown", value=f"{max_dd:.1%}")

# --------------------------
# Index drawdown table (Schwab-style)
# --------------------------
# For demonstration, compute same metrics for all indices regardless of sidebar selection
full_tbl = []
for name, tk in index_ticker_map.items():
    series = get_price_series([tk], start=year_start, end=as_of + timedelta(days=1))[tk]
    yret = series.iloc[-1] / series.iloc[0] - 1
    r_from_low = series.iloc[-1] / series.min() - 1
    mdd = series.iloc[-1] / series.max() - 1
    full_tbl.append((name, yret, r_from_low, mdd))

dd_df = pd.DataFrame(full_tbl, columns=["Index", "YTD return", "Return from YTD low", "Max drawdown from YTD high"])

dd_df_style = dd_df.style.format({c: "{:+.1%}" for c in dd_df.columns[1:]}) \
    .apply(lambda s: ["color:red" if v < 0 else "color:green" for v in s] if s.name != "Index" else [""] * len(s))

st.subheader("Major indexes and maximum drawdowns")
st.dataframe(dd_df_style, use_container_width=True, height=225)

# --------------------------
# Member-level stats
# --------------------------
with st.spinner("Fetching constituent data & calculating breadth …"):
    members = get_index_members("S&P 500")  # breadth table is always S&P like Schwab example
    mem_prices = get_price_series(members, start=as_of - timedelta(days=365), end=as_of + timedelta(days=1))

# Helper function for rolling horizon returns
def pct_ret(df: pd.DataFrame, window_days: int):
    return df.pct_change(periods=window_days).iloc[-1]

horizons = {
    "1m": 21,
    "2m": 42,
    "3m": 63,
    "4m": 84,
    "5m": 105,
    "6m": 126,
    "1y": 252,
}

breadth = {}
for label, days in horizons.items():
    member_r = pct_ret(mem_prices, days)
    idx_r = pct_ret(idx_px.to_frame("index"), days)[0]
    breadth[label] = (member_r > idx_r).mean()  # fraction beating index

breadth_df = pd.DataFrame(breadth, index=["% of S&P 500 members outperforming"]).T.reset_index()

breadth_df.columns = ["Horizon", "% members > index"]

breadth_style = breadth_df.style.format({"% members > index": "{:.0%}"}) \
    .apply(lambda s: ["background-color:#004d99;color:white" if v > 0.5 else "" for v in s] if s.name == "% members > index" else [""] * len(s))

st.subheader("Breadth — S&P 500 members beating the index")
st.dataframe(breadth_style, use_container_width=True, height=200)

# --------------------------
# Price chart with peak-trough markers
# --------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx_px.index, y=idx_px, mode="lines", name=selected_index))
fig.add_trace(go.Scatter(x=[idx_px.idxmax()], y=[peak_price], mode="markers", marker_symbol="triangle-up", marker_size=10, name="YTD High"))
fig.add_trace(go.Scatter(x=[idx_px.idxmin()], y=[trough_price], mode="markers", marker_symbol="triangle-down", marker_size=10, name="YTD Low"))
fig.update_layout(title=f"{selected_index} price YTD", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Downloads
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "⬇️ Download index drawdowns (CSV)",
        dd_df.to_csv(index=False).encode(),
        file_name="index_drawdowns.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "⬇️ Download breadth table (CSV)",
        breadth_df.to_csv(index=False).encode(),
        file_name="sp500_breadth.csv",
        mime="text/csv",
    )

# --------------------------
# Footer
# --------------------------
st.markdown("""
<sub>Source: Yahoo Finance, Wikipedia, AD Fund Management LP calculations. Data auto-adjusted for splits/dividends.<br>
Past performance is no guarantee of future results.</sub>
""", unsafe_allow_html=True)
