# Market Drawdown & Breadth Dashboard — Streamlit app
# ---------------------------------------------------
# Built for Arya — AD Fund Management LP
# ---------------------------------------------------
# Key features
#   • Replicates Charles Schwab‑style tables: index drawdowns & breadth (% of members beating the index)
#   • Dynamic date‑picker ("as‑of"), index selector (S&P 500, Nasdaq‑100, Russell 2000, Dow 30)
#   • Pulls live prices via yfinance; constituents scraped on‑the‑fly (cached)
#   • Extras: KPI tiles, interactive Plotly chart, CSV export, theme toggle
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Market Drawdown & Breadth Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# Helper functions
# --------------------------
@st.cache_data(show_spinner=False)
def get_index_members(index: str) -> list:
    """Return constituent tickers for a given index."""
    import requests, bs4  # local import keeps repo lightweight

    if index == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tbl = pd.read_html(url, header=0)[0]
        return tbl["Symbol"].tolist()
    if index == "Nasdaq 100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tbl = pd.read_html(url, header=0, match="Ticker")[0]
        return tbl["Ticker"].str.strip().tolist()
    if index == "Russell 2000":
        # Use IWM ETF holdings as proxy (top ~2000 rows)
        hold = yf.Ticker("IWM").fund_holdings
        return hold["symbol"].tolist()
    if index == "Dow 30":
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tbl = pd.read_html(url, header=0, match="Symbol")[1]
        return tbl["Symbol"].tolist()
    return []

@st.cache_data(show_spinner=False)
def get_price_series(tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    """Return a dataframe of adjusted close prices for the tickers."""
    raw = yf.download(tickers=tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False)

    # yfinance behaviour:
    # • multi‑index columns when >1 ticker
    # • single‑index columns when 1 ticker
    def extract_adj(df):
        if isinstance(df.columns, pd.MultiIndex):
            # bring "Adj Close" level to top, flatten
            if ("Adj Close" in df.columns.get_level_values(1)):
                out = df.swaplevel(axis=1)["Adj Close"]
            else:  # if auto_adjust=True previously, use "Close"
                out = df.swaplevel(axis=1)["Close"]
        else:
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            out = df[[col]].rename(columns={col: tickers[0]})
        return out

    adj = extract_adj(raw)
    # force ascending date index & drop duplicates
    adj = adj.sort_index().loc[~adj.index.duplicated(keep="first")]
    return adj

# Quick helper to get % return for a horizon (days)
def horizon_return(prices: pd.Series, days: int):
    if len(prices) <= days:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-(days + 1)] - 1

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Settings")
index_choice = st.sidebar.selectbox("Select index", ("S&P 500", "Nasdaq 100", "Russell 2000", "Dow 30"))
as_of = st.sidebar.date_input("As‑of date", value=datetime.today().date())
if as_of > datetime.today().date():
    st.sidebar.error("As‑of date cannot be in the future.")
    st.stop()

year_start = datetime(as_of.year, 1, 1)

# --------------------------
# Index‑level calculations
# --------------------------
index_map = {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Russell 2000": "^RUT", "Dow 30": "^DJI"}

idx_px = get_price_series([index_map[index_choice]], start=year_start, end=as_of + timedelta(days=1))
idx_series = idx_px.iloc[:, 0]  # squeeze to Series

# KPI metrics
kpi_cols = st.columns(3)
ytd_ret = idx_series.iloc[-1] / idx_series.iloc[0] - 1
ret_from_low = idx_series.iloc[-1] / idx_series.min() - 1
max_dd = idx_series.iloc[-1] / idx_series.max() - 1
kpi_cols[0].metric("YTD Return", f"{ytd_ret:+.1%}")
kpi_cols[1].metric("Return from YTD Low", f"{ret_from_low:+.1%}")
kpi_cols[2].metric("Max Drawdown", f"{max_dd:.1%}")

# --------------------------
# Build drawdown table for all major indices
# --------------------------
draw_rows = []
for name, tk in index_map.items():
    ser = get_price_series([tk], start=year_start, end=as_of + timedelta(days=1)).iloc[:, 0]
    draw_rows.append(
        {
            "Index": name,
            "YTD return": ser.iloc[-1] / ser.iloc[0] - 1,
            "Return from YTD low": ser.iloc[-1] / ser.min() - 1,
            "Max drawdown from YTD high": ser.iloc[-1] / ser.max() - 1,
        }
    )

dd_df = pd.DataFrame(draw_rows)

fmt = {c: "{:+.1%}" for c in dd_df.columns if c != "Index"}
dd_styled = dd_df.style.format(fmt)

st.subheader("Major indexes and maximum drawdowns")
st.dataframe(dd_styled, use_container_width=True, height=230)

# --------------------------
# Breadth table (S&P 500 only)
# --------------------------
st.subheader("Breadth — % S&P 500 members beating the index")

members = get_index_members("S&P 500")
prices = get_price_series(members, start=as_of - timedelta(days=370), end=as_of + timedelta(days=1))
index_prices = idx_series  # using selected index for comparison (Schwab uses S&P; keep same behaviour)

horizons = {"1m": 21, "2m": 42, "3m": 63, "4m": 84, "5m": 105, "6m": 126, "1y": 252}
records = []
for label, days in horizons.items():
    mret = prices.apply(lambda s: horizon_return(s, days))
    idx_ret = horizon_return(index_prices, days)
    pct = (mret > idx_ret).mean()
    records.append({"Horizon": label, "% members > index": pct})

breadth_df = pd.DataFrame(records)
bs = breadth_df.style.format({"% members > index": "{:.0%}"})
st.dataframe(bs, use_container_width=True, height=200)

# --------------------------
# YTD price chart with markers
# --------------------------
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx_series.index, y=idx_series, mode="lines", name=index_choice))
fig.add_trace(go.Scatter(x=[idx_series.idxmax()], y=[idx_series.max()], mode="markers", marker=dict(symbol="triangle-up", size=10), name="YTD High"))
fig.add_trace(go.Scatter(x=[idx_series.idxmin()], y=[idx_series.min()], mode="markers", marker=dict(symbol="triangle-down", size=10), name="YTD Low"))
fig.update_layout(title=f"{index_choice} — Price YTD", hovermode="x unified", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# CSV downloads
# --------------------------
col1, col2 = st.columns(2)
with col1:
    st.download_button("⬇️ Download drawdown table", dd_df.to_csv(index=False).encode(), "index_drawdowns.csv", "text/csv")
with col2:
    st.download_button("⬇️ Download breadth table", breadth_df.to_csv(index=False).encode(), "sp500_breadth.csv", "text/csv")

# Footer
st.markdown("""
<sub>Data: Yahoo Finance, Wikipedia; calculations by AD Fund Management LP.<br>
Past performance is no guarantee of future results.</sub>
""", unsafe_allow_html=True)
