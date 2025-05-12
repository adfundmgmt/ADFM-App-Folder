# correlation_dashboard.py
"""
Correlation Dashboard — AD Fund Management LP
--------------------------------------------
Quickly measure how **Ticker X** moves with **Ticker Y** (and an optional benchmark **Index Z**) across
five look‑back windows (YTD, 12 m, 24 m, 36 m, 60 m). Built for fast intuition‑building rather than
heavy‑duty quant work.
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

# ── About this tool ──────────────────────────────────────────────────────────
with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown(
        """
        **What it does**  
        • Calculates **Pearson correlations** on *daily log returns* for the selected tickers.  
        • Shows the numbers for five standard windows **(YTD / 12 / 24 / 36 / 60 months)**.  
        • Plots a **rolling‑window correlation** line so you can see how relationships evolve.  

        **Why it matters**  
        We use correlation as a first‑pass gauge of diversification and regime shifts.  
        Having the numbers one click away helps sanity‑check position sizing and basket risk.
        """
    )

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
    """Download daily *adjusted close* prices."""
    raw = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # MultiIndex → pick Adj Close and flatten
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
    "12 m": end_date - relativedelta(months=12),
    "24 m": end_date - relativedelta(months=24),
    "36 m": end_date - relativedelta(months=36),
    "60 m": end_date - relativedelta(months=60),
}

# ── Correlation matrices ────────────────────────────────────────────────────
st.subheader("Correlation Matrix (daily log returns)")

tabs = st.tabs(list(windows.keys()))

for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    subset = slice_since(returns, since)
    corr = subset.corr(method="pearson").round(3)

    with tab:
        st.write(f"**{label}** window starting {since}")
        st.dataframe(corr, use_container_width=True)

        # Heatmap visual
        heat_data = (
            corr.reset_index()
            .melt(id_vars="index", var_name="Ticker2", value_name="Correlation")
            .rename(columns={"index": "Ticker1"})
        )
        heat = (
            alt.Chart(heat_data)
            .mark_rect()
            .encode(
                x=alt.X("Ticker1:O", title=""),
                y=alt.Y("Ticker2:O", title=""),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redyellowblue", domain=[-1, 1])),
                tooltip=["Ticker1", "Ticker2", "Correlation"],
            )
            .properties(height=200)
        )
        st.altair_chart(heat, use_container_width=True)

# ── Rolling correlation chart ───────────────────────────────────────────────
if ticker_y.strip():
    ser_corr = returns[ticker_x.upper()].rolling(roll_window).corr(returns[ticker_y.upper()])
    ser_corr = ser_corr.dropna().to_frame(name="Correlation")

    st.subheader(f"Rolling {roll_window}-day Correlation — {ticker_x.upper()} vs {ticker_y.upper()}")

    line = (
        alt.Chart(ser_corr.reset_index())
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
        • Rolling window = **{roll_window}** trading days (≈ {roll_window/21:.1f} months).  
        • Data via **yfinance** (*Adj Close*).  
        • Splits/dividends are automatically accounted for.  
        • Any missing rows are dropped prior to calculation.
        """
    )

st.caption("© 2025 AD Fund Management LP — Internal use only")
