# correlation_dashboard.py
"""
Correlation Dashboard — AD Fund Management LP
--------------------------------------------
Compare **Ticker X** against **Ticker Y** (plus an optional benchmark **Index Z**) across multiple look‑back
windows:

• Year‑to‑Date (YTD)
• 12 months
• 24 months
• 36 months
• 60 months

The app fetches daily adjusted closes via *yfinance*, computes daily log returns, and shows:
  – Pearson correlation matrices for each window
  – A rolling‑window correlation line chart (default = 60 trading days)

Written for quick internal use — no fancy bells, just clean numbers.
"""

import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Correlation Dashboard — AD Fund Management LP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Correlation Dashboard")

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security to analyse.")
    ticker_y = st.text_input("Ticker Y", value="MSFT")
    ticker_z = st.text_input("Index Z (optional)", value="^GSPC", help="Benchmark / index — leave blank to skip.")

    years_back = st.slider("Data history (years)", 1, 10, value=6)
    roll_window = st.slider("Rolling window (days)", 20, 120, value=60)

    run = st.button("Run Analysis")

if not run:
    st.stop()

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily *adjusted close* prices for each symbol."""
    raw = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,  # keep Adj Close so we can select it reliably
    )

    # When multiple tickers, columns are a MultiIndex (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw["Adj Close"].copy()
        adj.columns = adj.columns.get_level_values(0)  # flatten to ticker names
    else:
        # Single ticker → raw is a normal DataFrame
        adj = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    return adj.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def slice_since(returns: pd.DataFrame, since: dt.date) -> pd.DataFrame:
    return returns.loc[since:]

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

returns = log_returns(prices)

# ── Look‑back windows ───────────────────────────────────────────────────────
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "12 m": end_date - relativedelta(months=12),
    "24 m": end_date - relativedelta(months=24),
    "36 m": end_date - relativedelta(months=36),
    "60 m": end_date - relativedelta(months=60),
}

# ── Correlation matrices ────────────────────────────────────────────────────
st.subheader("Correlation Matrix (daily log returns)")

tabs = st.tabs(list(windows.keys()))

for tab, label in zip(tabs, windows.keys()):
    with tab:
        since = windows[label]
        corr = slice_since(returns, since).corr(method="pearson").round(3)
        st.write(f"**{label}** window starting {since}")
        st.dataframe(corr, use_container_width=True)

# ── Rolling correlation chart ───────────────────────────────────────────────
if ticker_y.strip():
    pair = returns[[ticker_x.upper(), ticker_y.upper()]].dropna()
    rolling = (
        pair.rolling(window=roll_window)
        .corr()
        .dropna()
        .unstack()
        .droplevel(0)
        .iloc[:, 1]  # correlation column
        .rename("Correlation")
    )

    st.subheader(f"Rolling {roll_window}-day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")

    line = (
        alt.Chart(rolling.reset_index())
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Correlation:Q", title="Correlation"),
        )
        .properties(height=400)
    )
    st.altair_chart(line, use_container_width=True)

# ── Footnotes ───────────────────────────────────────────────────────────────
with st.expander("Methodology / Notes"):
    st.markdown(
        f"""
        • **Pearson correlation** on *daily log returns*.
        • Rolling window = **{roll_window}** trading days (≈ {roll_window/21:.1f} months).
        • Price data via **yfinance**; *Adj Close* used, so splits/dividends are accounted for.
        • Missing data rows are dropped before calculations.
        """
    )

st.caption("© 2025 AD Fund Management LP — Internal use only")
