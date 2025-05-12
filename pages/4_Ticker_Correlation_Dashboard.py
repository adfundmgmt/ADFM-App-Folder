# correlation_dashboard.py
"""
Correlation Dashboard — AD Fund Management LP
--------------------------------------------
Quantify how **Ticker X** co-moves with **Ticker Y** (plus an optional benchmark **Index Z**) across
five standard windows (YTD, 12 m, 24 m, 36 m, 60 m) and inspect a rolling correlation trace. Designed as
a lightweight, real-time reference for portfolio construction and risk calibration.
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

# ── Sidebar: About + Inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About this tool")
    st.markdown(
        """
        **Purpose**  
        Provide an at-a-glance read on historical and rolling correlations to inform relative-value work
        and risk budgeting.

        **Key features**  
        • Pearson correlations on *daily log returns* for five look-back windows (YTD, 12 / 24 / 36 / 60 m).  
        • Normalised price-path plot for each window with the correlation value highlighted.  
        • Rolling-window correlation chart.  
        • Data sourced on-demand from **yfinance** (*Adj Close*).
        """
    )
    st.markdown("---")

    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security to analyse.")
    ticker_y = st.text_input("Ticker Y", value="MSFT")
    ticker_z = st.text_input("Index Z (optional)", value="^GSPC", help="Benchmark / index — leave blank to skip.")

    years_back = st.slider("Data history (years)", 1, 10, value=6)
    roll_window = st.slider("Rolling window (days)", 20, 120, value=60)

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download adjusted-close prices for the symbols."""
    data = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"].copy()
        adj.columns = adj.columns.get_level_values(0)
    else:
        adj = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    return adj.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def normalise(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.divide(prices.iloc[0]).multiply(100)


def slice_since(df: pd.DataFrame, since: dt.date) -> pd.DataFrame:
    return df.loc[since:]

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

# ── Look-back windows ───────────────────────────────────────────────────────
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "12 m": end_date - relativedelta(months=12),
    "24 m": end_date - relativedelta(months=24),
    "36 m": end_date - relativedelta(months=36),
    "60 m": end_date - relativedelta(months=60),
}

st.subheader("Correlation & Price Paths")

tabs = st.tabs(list(windows.keys()))

for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    sub_prices = slice_since(prices, since)
    sub_returns = slice_since(returns, since)

    if sub_prices.empty or sub_returns.empty:
        with tab:
            st.warning("Insufficient data for this window.")
        continue

    corr_val = sub_returns[ticker_x.upper()].corr(sub_returns[ticker_y.upper()])

    # Normalise prices to 100 at window start
    norm_prices = normalise(sub_prices)
    tidy = norm_prices.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Index")

    with tab:
        st.write(f"**{label}** window starting {since} — correlation ({ticker_x}/{ticker_y}): **{corr_val:.2f}**")

        line_chart = (
            alt.Chart(tidy)
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y("Index:Q", title="Indexed to 100"),
                color="Ticker:N",
                tooltip=["Date:T", "Ticker:N", alt.Tooltip("Index:Q", format=".2f")],
            )
            .properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)

        # Show correlation matrix beneath for completeness
        st.dataframe(sub_returns.corr(method="pearson").round(3), use_container_width=True)

# ── Rolling correlation chart ───────────────────────────────────────────────
if ticker_y.strip():
    rolling_corr = (
        returns[ticker_x.upper()].rolling(roll_window).corr(returns[ticker_y.upper()])
        .dropna()
        .to_frame(name="Correlation")
    )

    st.subheader(f"Rolling {roll_window}-day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")

    roll_line = (
        alt.Chart(rolling_corr.reset_index())
        .mark_line()
        .encode(
            x="Date:T",
            y
