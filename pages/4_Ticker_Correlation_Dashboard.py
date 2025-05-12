# correlation_dashboard.py
"""
Streamlit app: Correlation Dashboard
-----------------------------------
Compare the historical correlation of **Ticker X** against **Ticker Y** (and an optional **Index Z**) across
multiple look‑back windows:

• Year‑to‑Date (YTD)
• 12 months
• 24 months
• 36 months
• 60 months

The app fetches daily adjusted closes via *yfinance*, computes daily log returns, and reports Pearson
correlations. A rolling‑window chart (default 60‑day) is included for additional colour.
"""

import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

# ── Streamlit page setup ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Correlation Dashboard — AD Fund Management LP",
    layout="wide",
    menu_items={"About": "Internal analytics tool — built in‑house."},
)

st.title("📈 Correlation Dashboard")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Security to analyse against others.")
    ticker_y = st.text_input("Ticker Y", value="MSFT")
    ticker_z = st.text_input("Index Z (optional)", value="^GSPC", help="Benchmark / index — leave blank to skip.")

    years_back = st.slider("Data history (years)", 1, 10, value=6)
    roll_window = st.slider("Rolling window (days)", 20, 120, value=60)
    run = st.button("Run Analysis")

if not run:
    st.stop()

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: dt.date, end: dt.date):
    """Download daily adjusted close prices for the given symbols."""
    data = (
        yf.download(
            symbols,
            start=start,
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=True,
        )["Adj Close"]
        .dropna(how="all")
    )
    # Ensure DataFrame even when one ticker
    if isinstance(data, pd.Series):
        data = data.to_frame(symbols[0])
    return data


def log_returns(prices: pd.DataFrame):
    return np.log(prices / prices.shift(1)).dropna()


def period_slice(returns: pd.DataFrame, start_date: dt.date):
    return returns.loc[start_date:]


# ── Data download ────────────────────────────────────────────────────────────
end_date = dt.date.today()
start_date = end_date - relativedelta(years=years_back)

symbols = [ticker_x.strip().upper(), ticker_y.strip().upper()]
if ticker_z.strip():
    symbols.append(ticker_z.strip().upper())

prices = fetch_prices(symbols, start=start_date, end=end_date)
if prices.empty:
    st.error("No price data returned — check tickers and try again.")
    st.stop()

rets = log_returns(prices)

# ── Define look‑back windows ────────────────────────────────────────────────
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "12 m": end_date - relativedelta(months=12),
    "24 m": end_date - relativedelta(months=24),
    "36 m": end_date - relativedelta(months=36),
    "60 m": end_date - relativedelta(months=60),
}

# ── Correlation computation ─────────────────────────────────────────────────
results = {}
for label, start_dt in windows.items():
    sliced = period_slice(rets, start_dt)
    corr = sliced.corr(method="pearson").round(3)
    results[label] = corr

# ── Display correlation tables ──────────────────────────────────────────────
st.subheader("Correlation Matrix")

tabs = st.tabs(list(windows.keys()))
for tab, period_label in zip(tabs, windows.keys()):
    with tab:
        st.write(f"**{period_label}** correlation (daily returns)")
        st.dataframe(results[period_label])

# ── Rolling correlation chart ───────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Rolling {roll_window}-day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")
    pair_corr = (
        rets[[ticker_x.upper(), ticker_y.upper()]]
        .rolling(roll_window)
        .corr()
        .dropna()
        .unstack()
        .droplevel(0)
        .iloc[:, 1]  # correlation column
        .rename("corr")
    )
    chart = (
        alt.Chart(pair_corr.reset_index())
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("corr:Q", title="Correlation"),
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.markdown(
        f"""
        **Quick Notes**

        • Correlation is Pearson, based on daily log returns.
        • Rolling window = **{roll_window}** trading days (≈ {roll_window/21:.1f} months).
        • Data auto‑adjusted for splits/dividends via *yfinance*.
        • Index Z inclusion is optional; tables adjust dynamically.
        """
    )

st.caption("© 2025 AD Fund Management LP — Internal use only")
