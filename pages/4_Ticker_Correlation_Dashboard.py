# correlation_dashboard.py
"""
Correlation Dashboard — AD Fund Management LP
--------------------------------------------
Quantify how **Ticker X** co‑moves with **Ticker Y** (plus optional **Index Z**) across eight
look‑back windows (YTD, 3 m, 6 m, 9 m, 1 y, 3 y, 5 y, 10 y) and monitor rolling correlation. Designed as a
quick reference for portfolio construction and risk alignment.
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
        Provide an at‑a‑glance view of historical and rolling correlations to support relative‑value
        work, diversification checks, and position sizing.

        **Key features**  
        • Pearson correlations on *daily log returns* for eight windows (YTD, 3 m, 6 m, 9 m, 1 y, 3 y, 5 y, 10 y).  
        • Dynamic overlay charts of the selected tickers for each window.  
        • Rolling‑window correlation line to monitor regime shifts.  
        • Live data sourced via **yfinance** (*Adj Close*).
        """
    )
    st.markdown("---")

    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security to analyse.")
    ticker_y = st.text_input("Ticker Y", value="MSFT")
    ticker_z = st.text_input("Index Z (optional)", value="^GSPC", help="Benchmark / index — leave blank to skip.")

    roll_window = st.slider("Rolling window (days)", 20, 120, value=60)

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download adjusted‑close prices for the symbols."""
    raw = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw["Adj Close"].copy()
        adj.columns = adj.columns.get_level_values(0)
    else:
        adj = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    return adj.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def slice_since(df: pd.DataFrame, since: dt.date) -> pd.DataFrame:
    return df.loc[since:]


def normalize(prices: pd.Series) -> pd.Series:
    return prices / prices.iloc[0] * 100

# ── Data download ────────────────────────────────────────────────────────────
end_date = dt.date.today()

# Define look‑back windows
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "3 m": end_date - relativedelta(months=3),
    "6 m": end_date - relativedelta(months=6),
    "9 m": end_date - relativedelta(months=9),
    "1 y": end_date - relativedelta(years=1),
    "3 y": end_date - relativedelta(years=3),
    "5 y": end_date - relativedelta(years=5),
    "10 y": end_date - relativedelta(years=10),
}

# Fetch data back to earliest window (add 1 month cushion)
earliest_date = min(windows.values()) - relativedelta(months=1)

symbols = [ticker_x.strip().upper(), ticker_y.strip().upper()]
if ticker_z.strip():
    symbols.append(ticker_z.strip().upper())

prices = fetch_prices(symbols, start=earliest_date, end=end_date)

if prices.empty:
    st.error("No price data returned — check tickers and try again.")
    st.stop()

returns = log_returns(prices)

# ── Correlation + Price Overlay by Window ───────────────────────────────────
st.subheader("Window‑Specific Correlation & Price Paths")

tabs = st.tabs(list(windows.keys()))

for tab, label in zip(tabs, windows.keys()):
    since = windows[label]

    price_slice = slice_since(prices, since)
    ret_slice = slice_since(returns, since)

    # Correlation only between X & Y
    corr_value = ret_slice.corr().loc[ticker_x.upper(), ticker_y.upper()].round(3)

    with tab:
        st.markdown(f"**{label}** window starting {since} — **Correlation (X vs Y): {corr_value}**")

        overlay_tickers = [ticker_x.upper(), ticker_y.upper()]
        if ticker_z.strip():
            overlay_tickers.append(ticker_z.strip().upper())

        norm_df = price_slice[overlay_tickers].apply(normalize)
        norm_df = norm_df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="IndexedPrice")

        line_chart = (
            alt.Chart(norm_df)
            .mark_line()
            .encode(
                x="Date:T",
                y=alt.Y("IndexedPrice:Q", title="Indexed Price (Base = 100)", scale=alt.Scale(nice=True)),
                color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
            )
            .properties(height=320)
        )
        st.altair_chart(line_chart, use_container_width=True)

# ── Rolling correlation chart ───────────────────────────────────────────────
if ticker_y.strip():
    rolling_corr = (
        returns[ticker_x.upper()].rolling(roll_window).corr(returns[ticker_y.upper()])
        .dropna()
        .to_frame(name="Correlation")
    )

    st.subheader(f"Rolling {roll_window}-Day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")

    line_chart_roll = (
        alt.Chart(rolling_corr.reset_index())
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Correlation:Q", title="Correlation", scale=alt.Scale(domain=[-1, 1])),
        )
        .properties(height=400)
    )
    st.altair_chart(line_chart_roll, use_container_width=True)

# ── Footnotes ───────────────────────────────────────────────────────────────
with st.expander("Methodology / Notes"):
    st.markdown(
        f"""
        • Pearson correlation on *daily log returns*.  
        • Rolling window = **{roll_window}** trading days (≈ {roll_window/21:.1f} months).  
        • Prices via **yfinance** (*Adj Close*); splits / dividends included.  
        • Missing rows dropped prior to calculation.
        """
    )

st.caption("© 2025 AD Fund Management LP — Internal use only")
