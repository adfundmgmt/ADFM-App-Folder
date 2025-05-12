# correlation_dashboard.py
"""
Correlation Dashboard — AD Fund Management LP
--------------------------------------------
Quantify how **Ticker X** co‑moves with **Ticker Y** (plus an optional benchmark **Index Z**) across five
standard windows — YTD, 12 m, 24 m, 36 m, and 60 m — and view a rolling correlation trace. Designed as a
lightweight, real‑time reference for portfolio construction and risk alignment.
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
        Provide an at‑a‑glance view of historical and rolling correlations to inform relative‑value
        analysis, diversification checks, and position sizing.

        **Key features**  
        • Pearson correlations on *daily log returns* for five look‑back windows (YTD, 12 / 24 / 36 / 60 m).  
        • Interactive heat‑maps and a rolling‑window line chart.  
        • Data sourced on‑the‑fly from **yfinance** using adjusted closes (splits / dividends reflected).  
        """
    )
    st.markdown("---")

    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security to analyse.")
    ticker_y = st.text_input("Ticker Y", value="MSFT")
    ticker_z = st.text_input("Index Z (optional)", value="^GSPC", help="Benchmark / index — leave blank to skip.")

    years_back = st.slider("Data history (years)", 1, 10, value=6)
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

    # Multi‑ticker download returns a MultiIndex; single ticker returns a Series/DF
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw["Adj Close"].copy()
        adj.columns = adj.columns.get_level_values(0)
    else:
        adj = raw[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    return adj.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


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
    "12 m": end_date - relativedelta(months=12),
    "24 m": end_date - relativedelta(months=24),
    "36 m": end_date - relativedelta(months=36),
    "60 m": end_date - relativedelta(months=60),
}

# ── Correlation matrices & heat‑maps ────────────────────────────────────────
st.subheader("Correlation Matrix (daily log returns)")

tabs = st.tabs(list(windows.keys()))

ticker_order = returns.columns.tolist()

for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    subset = slice_since(returns, since)
    corr = subset.corr(method="pearson").round(3)

    with tab:
        st.write(f"**{label}** window starting {since}")
        st.dataframe(corr, use_container_width=True)

        # Heat‑map visual
        heat_df = corr.copy()
        heat_df["Ticker1"] = heat_df.index
        heat_tidy = heat_df.melt(id_vars="Ticker1", var_name="Ticker2", value_name="Correlation")

        heat_chart = (
            alt.Chart(heat_tidy)
            .mark_rect()
            .encode(
                x=alt.X("Ticker1:O", sort=ticker_order, title=""),
                y=alt.Y("Ticker2:O", sort=ticker_order, title=""),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="blueorange", domain=[-1, 0, 1]),
                    legend=alt.Legend(orient="right"),
                ),
                tooltip=["Ticker1", "Ticker2", "Correlation"],
            )
            .properties(width=200, height=200)
        )
        st.altair_chart(heat_chart, use_container_width=False)

# ── Rolling correlation chart ───────────────────────────────────────────────
if ticker_y.strip():
    rolling_corr = (
        returns[ticker_x.upper()].rolling(roll_window).corr(returns[ticker_y.upper()])
        .dropna()
        .to_frame(name="Correlation")
    )

    st.subheader(f"Rolling {roll_window}-day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")

    line_chart = (
        alt.Chart(rolling_corr.reset_index())
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Correlation:Q", title="Correlation"),
        )
        .properties(height=400)
    )
    st.altair_chart(line_chart, use_container_width=True)

# ── Footnotes ───────────────────────────────────────────────────────────────
with st.expander("Methodology / Notes"):
    st.markdown(
        f"""
        • Pearson correlation on *daily log returns*.  
        • Rolling window = **{roll_window}** trading days (≈ {roll_window/21:.1f} months).  
        • Raw prices via **yfinance** (*Adj Close*).  
        • Splits / dividends automatically reflected.  
        • Missing data rows dropped prior to calculation.
        """
    )

st.caption("© 2025 AD Fund Management LP — Internal use only")
