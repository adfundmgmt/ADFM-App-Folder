# correlation_dashboard.py

import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt
plt.style.use("default")

st.set_page_config(
    page_title="Correlation Dashboard — AD Fund Management LP",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Ticker Correlation Dashboard")

# ── Sidebar: About + Inputs ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose**  
        Provides a quick, at-a-glance view of historical and rolling correlations for up to three tickers, supporting portfolio construction and risk assessment.

        **Key features**  
        • Computes Pearson correlations on *daily log returns* for eight windows: YTD, 3m, 6m, 9m, 1y, 3y, 5y, 10y.  
        • Shows indexed price overlays for each window.  
        • Plots rolling correlation, highlights potential regime shifts.  
        • Pulls split- and dividend-adjusted data from **Yahoo Finance** (Adj Close).
        """
    )
    st.markdown("---")
    st.header("Inputs")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security to analyse.").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT").strip().upper()
    ticker_z = st.text_input(
        "Index Z (optional)", 
        value="", 
        help="Benchmark/index (optional; leave blank to skip)."
    ).strip().upper()

    roll_window = st.slider("Rolling window (days)", 20, 120, value=60)

# ── Helper functions ────────────────────────────────────────────────────────
def fetch_prices(symbols: list, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end + dt.timedelta(days=1),
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()
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

def emoji_corr(val: float) -> str:
    if pd.isnull(val):
        return ""
    if val > 0.7: return "🟢"
    if val > 0.3: return "🟡"
    if val < -0.3: return "🔴"
    return "⚪️"

def dynamic_axis(data, buffer_ratio=0.04, axis='y'):
    min_val = float(np.nanmin(data))
    max_val = float(np.nanmax(data))
    buffer = (max_val - min_val) * buffer_ratio
    if min_val == max_val:
        # flat line, just buffer a bit
        return [min_val - 1, max_val + 1]
    return [min_val - buffer, max_val + buffer]

# ── Data download ───────────────────────────────────────────────────────────
end_date = dt.date.today()
windows = {
    "YTD": dt.date(end_date.year, 1, 1),
    "3 Months": end_date - relativedelta(months=3),
    "6 Months": end_date - relativedelta(months=6),
    "9 Months": end_date - relativedelta(months=9),
    "1 Year": end_date - relativedelta(years=1),
    "3 Years": end_date - relativedelta(years=3),
    "5 Years": end_date - relativedelta(years=5),
    "10 Years": end_date - relativedelta(years=10),
}
earliest_date = min(windows.values()) - relativedelta(months=1)

# Remove blanks/dupes
symbols = sorted(set(filter(bool, [ticker_x, ticker_y, ticker_z])))

if not (ticker_x and ticker_y):
    st.warning("You must enter at least two tickers.")
    st.stop()

prices = fetch_prices(symbols, start=earliest_date, end=end_date)
if prices.empty:
    st.error("No price data returned — check ticker symbols and try again.")
    st.stop()
returns = log_returns(prices)

# Check available data range
if prices.index[0].date() > earliest_date:
    st.info(f"Note: Oldest available price is {prices.index[0].date()}, so not all look-back windows may be shown.")

# ── Window‑Specific Overlay Charts (dynamic Y axis) ────────────────────────
st.subheader("Indexed Price Overlays")
tabs = st.tabs(list(windows.keys()))
for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    price_slice = slice_since(prices, since)
    overlay_tickers = [ticker_x, ticker_y] + ([ticker_z] if ticker_z else [])
    overlay_tickers = [t for t in overlay_tickers if t in price_slice.columns]
    norm_df = price_slice[overlay_tickers].apply(normalize)
    norm_df = norm_df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="IndexedPrice")
    y_axis_range = dynamic_axis(norm_df['IndexedPrice'], buffer_ratio=0.06)
    chart = (
        alt.Chart(norm_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("IndexedPrice:Q", title="Indexed Price (Base=100)", scale=alt.Scale(domain=y_axis_range)),
            color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
            tooltip=["Date:T", "Ticker:N", alt.Tooltip("IndexedPrice:Q", format=".2f")],
        )
        .properties(height=320)
        .interactive()
    )
    with tab:
        st.altair_chart(chart, use_container_width=True)

# ── Rolling correlation chart (dynamic Y axis) ──────────────────────────────
st.subheader("Rolling Correlation Chart")
corr_df = pd.DataFrame(index=returns.index)
corr_df[f"{ticker_x} vs {ticker_y}"] = (
    returns[ticker_x].rolling(roll_window).corr(returns[ticker_y])
)
if ticker_z and ticker_z in returns.columns:
    corr_df[f"{ticker_x} vs {ticker_z}"] = (
        returns[ticker_x].rolling(roll_window).corr(returns[ticker_z])
    )
    corr_df[f"{ticker_y} vs {ticker_z}"] = (
        returns[ticker_y].rolling(roll_window).corr(returns[ticker_z])
    )
corr_df = corr_df.dropna(how="all")

corr_long = corr_df.reset_index().melt(id_vars="Date", var_name="Pair", value_name="Correlation")
y_axis_corr = dynamic_axis(corr_long['Correlation'].dropna(), buffer_ratio=0.06)
roll_chart = (
    alt.Chart(corr_long)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Correlation:Q", title="Correlation", scale=alt.Scale(domain=y_axis_corr)),
        color=alt.Color("Pair:N", legend=alt.Legend(title="Pairs")),
        tooltip=["Date:T", "Pair:N", alt.Tooltip("Correlation:Q", format=".2f")],
    )
    .properties(height=400)
    .interactive()
)
st.altair_chart(roll_chart, use_container_width=True)

# ── Correlation Table for Each Window (moved below charts) ─────────────────
st.subheader("Correlation Table by Look-Back Window")
rows = []
for label, since in windows.items():
    ret_slice = slice_since(returns, since)
    row = {"Window": label}
    # X↔Y
    try:
        row["X vs Y"] = ret_slice.corr().loc[ticker_x, ticker_y].round(3)
    except: row["X vs Y"] = np.nan
    # X↔Z and Y↔Z if Z provided
    if ticker_z:
        try:
            row["X vs Z"] = ret_slice.corr().loc[ticker_x, ticker_z].round(3)
        except: row["X vs Z"] = np.nan
        try:
            row["Y vs Z"] = ret_slice.corr().loc[ticker_y, ticker_z].round(3)
        except: row["Y vs Z"] = np.nan
    rows.append(row)
df_corr = pd.DataFrame(rows)
df_corr_display = df_corr.copy()
for c in df_corr.columns[1:]:
    df_corr_display[c] = df_corr_display[c].apply(lambda v: f"{v:.2f} {emoji_corr(v)}" if pd.notnull(v) else "")
st.dataframe(df_corr_display.set_index("Window"), height=320)

# ── Data Download Option ────────────────────────────────────────────────────
with st.expander("Download Data"):
    st.download_button(
        "Download Indexed Prices (CSV)", 
        prices.to_csv(index=True), 
        file_name="correlation_dashboard_prices.csv", 
        mime="text/csv"
    )
    st.download_button(
        "Download Log Returns (CSV)",
        returns.to_csv(index=True),
        file_name="correlation_dashboard_returns.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download Correlation Table (CSV)",
        df_corr.to_csv(index=False),
        file_name="correlation_dashboard_correlation_table.csv",
        mime="text/csv"
    )

st.caption("© 2025 AD Fund Management LP")
