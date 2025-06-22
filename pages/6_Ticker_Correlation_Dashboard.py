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
st.markdown(
    """
    <div style='display:flex; align-items:center; margin-bottom: -18px;'>
        <h1 style='margin:0 12px 0 0; font-size:2.2rem;'>Correlation Dashboard</h1>
        <span style='font-size:1.2rem; color:#767676; font-weight:500;'>AD Fund Management LP</span>
    </div>
    """, unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("How it Works")
    st.markdown(
        """
        1. Enter two or three tickers (US equities, ETF, or index).
        2. Select log-return frequency and rolling window.
        3. View *live* correlations, ratio charts, and indexed overlays.
        ---
        - **Correlation** uses Pearson on log returns (Adj Close).
        - **Relative Value Chart** shows ratio paths (A/B, A/C, ...).
        - **Charts** are interactive (zoom/hover).
        """
    )
    st.markdown("---")
    ticker_x = st.text_input("Ticker X", value="AAPL", help="Primary security.").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="MSFT").strip().upper()
    ticker_z = st.text_input("Ticker Z (optional)", value="", help="Benchmark/index (optional).").strip().upper()
    freq = st.selectbox("Return Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling window (periods)", 20, 120, value=60)
    st.markdown("---")
    st.caption("© 2025 AD Fund Management LP")

# --- Header Inputs Card ---
with st.container():
    c1, c2, c3, c4 = st.columns([2,2,2,3])
    c1.markdown(f"<b>Primary</b>: <span style='color:#1976D2'>{ticker_x}</span>", unsafe_allow_html=True)
    c2.markdown(f"<b>Secondary</b>: <span style='color:#D2691E'>{ticker_y}</span>", unsafe_allow_html=True)
    if ticker_z:
        c3.markdown(f"<b>Benchmark</b>: <span style='color:#2ca02c'>{ticker_z}</span>", unsafe_allow_html=True)
    c4.markdown(f"<b>Freq</b>: {freq} <br><b>Rolling</b>: {roll_window}", unsafe_allow_html=True)

# --- Data download + log-returns ---
@st.cache_data(show_spinner=False)
def fetch_prices(symbols: list, start: dt.date, end: dt.date, freq: str) -> pd.DataFrame:
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
    adj = adj.dropna(how="all")
    if freq == "Weekly":
        adj = adj.resample("W-FRI").last()
    elif freq == "Monthly":
        adj = adj.resample("M").last()
    return adj.dropna(how="all")

@st.cache_data(show_spinner=False)
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")

def slice_since(df: pd.DataFrame, since: dt.date) -> pd.DataFrame:
    return df.loc[since:]

def normalize(prices: pd.Series) -> pd.Series:
    return prices / prices.iloc[0] * 100

def highlight_corr(val):
    if pd.isnull(val): return ''
    if val > 0.85: return 'background-color: #C8FFD4'
    if val < -0.3: return 'background-color: #FFD6D6'
    return ''

def emoji_corr(val: float) -> str:
    if pd.isnull(val): return ""
    if val > 0.7: return "🟢"
    if val > 0.3: return "🟡"
    if val < -0.3: return "🔴"
    return "⚪️"

# --- Data Download Logic ---
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
symbols = sorted(set(filter(bool, [ticker_x, ticker_y, ticker_z])))

if not (ticker_x and ticker_y):
    st.warning("You must enter at least two tickers.")
    st.stop()

prices = fetch_prices(symbols, start=earliest_date, end=end_date, freq=freq)
if prices.empty:
    st.error("No price data returned — check ticker symbols and try again.")
    st.stop()
returns = log_returns(prices)

missing = prices.isnull().sum()
if missing.any():
    st.warning(f"Data gaps detected: {dict(missing[missing > 0])}")
if prices.index[0].date() > earliest_date:
    st.info(f"Note: Oldest available price is {prices.index[0].date()}, so not all look-back windows may be shown.")

# --- Rolling Correlation Chart (First!) ---
st.markdown("### Rolling Correlation (live regime shifts)")
corr_df = pd.DataFrame(index=returns.index)
corr_df[f"{ticker_x} vs {ticker_y}"] = returns[ticker_x].rolling(roll_window).corr(returns[ticker_y])
if ticker_z and ticker_z in returns.columns:
    corr_df[f"{ticker_x} vs {ticker_z}"] = returns[ticker_x].rolling(roll_window).corr(returns[ticker_z])
    corr_df[f"{ticker_y} vs {ticker_z}"] = returns[ticker_y].rolling(roll_window).corr(returns[ticker_z])
corr_df = corr_df.dropna(how="all")

regime_lines = []
if f"{ticker_x} vs {ticker_y}" in corr_df.columns:
    diff = corr_df[f"{ticker_x} vs {ticker_y}"].diff().abs()
    threshold = 0.4
    regime_points = corr_df.index[diff > threshold]
    for date in regime_points:
        regime_lines.append({"Date": date})

if not corr_df.empty:
    latest_corrs = corr_df.iloc[-1].dropna()
    corr_display = " &nbsp; | &nbsp; ".join([f"<b>{col}:</b> <span style='color:#1976D2'>{val:.2f}</span>" for col, val in latest_corrs.items()])
    st.markdown(f"<div style='margin-bottom:-6px'>{corr_display}</div>", unsafe_allow_html=True)

corr_long = corr_df.reset_index().melt(id_vars="Date", var_name="Pair", value_name="Correlation")
roll_chart = (
    alt.Chart(corr_long)
    .mark_line(strokeWidth=3)
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Correlation:Q", title="Correlation", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("Pair:N", legend=alt.Legend(title="Pairs")),
        tooltip=["Date:T", "Pair:N", alt.Tooltip("Correlation:Q", format=".2f")],
    )
    .properties(height=340)
    .interactive()
)
if regime_lines:
    regime_df = pd.DataFrame(regime_lines)
    rule_chart = (
        alt.Chart(regime_df)
        .mark_rule(color="red", opacity=0.21)
        .encode(x="Date:T")
    )
    roll_chart = roll_chart + rule_chart
st.altair_chart(roll_chart, use_container_width=True)

# --- Relative Value (Ratio) Charts ---
from itertools import permutations

def compute_ratio(prices, t1, t2):
    s1, s2 = prices[t1].dropna(), prices[t2].dropna()
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) < 5:
        return pd.Series(dtype=float)
    ratio = s1.loc[common_idx] / s2.loc[common_idx]
    return ratio

pair_list = [(a, b) for a, b in permutations([t for t in [ticker_x, ticker_y, ticker_z] if t], 2)]

if pair_list:
    st.markdown("### Relative Value (Ratio) Chart(s)")
    for t1, t2 in pair_list:
        ratio = compute_ratio(prices, t1, t2)
        if ratio.empty:
            continue
        ratio_df = ratio.reset_index()
        ratio_df.columns = ["Date", "Ratio"]
        chart = (
            alt.Chart(ratio_df)
            .mark_line(strokeWidth=2.3)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Ratio:Q", title=f"{t1} / {t2} Ratio", scale=alt.Scale(nice=True)),
                tooltip=["Date:T", alt.Tooltip("Ratio:Q", format=".3f")]
            )
            .properties(height=210, title=f"{t1} / {t2}")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

# --- Window overlays ---
st.markdown("### Indexed Price Overlays")
tabs = st.tabs(list(windows.keys()))
for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    price_slice = slice_since(prices, since)
    overlay_tickers = [ticker_x, ticker_y] + ([ticker_z] if ticker_z else [])
    overlay_tickers = [t for t in overlay_tickers if t in price_slice.columns]
    norm_df = price_slice[overlay_tickers].apply(normalize)
    norm_df = norm_df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="IndexedPrice")
    chart = (
        alt.Chart(norm_df)
        .mark_line(strokeWidth=2.3)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("IndexedPrice:Q", title="Indexed Price (Base=100)", scale=alt.Scale(nice=True)),
            color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
            tooltip=["Date:T", "Ticker:N", alt.Tooltip("IndexedPrice:Q", format=".2f")],
        )
        .properties(height=250)
        .interactive()
    )
    with tab:
        st.altair_chart(chart, use_container_width=True)

# --- Correlation Table (after charts) ---
st.markdown("### Correlation Table by Look-Back Window")
rows = []
for label, since in windows.items():
    ret_slice = slice_since(returns, since)
    row = {"Window": label}
    try:
        row["X vs Y"] = ret_slice.corr().loc[ticker_x, ticker_y].round(3)
    except: row["X vs Y"] = np.nan
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
st.dataframe(df_corr_display.set_index("Window"), height=350)

# --- Data Download Option ---
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
    st.download_button(
        "Download Rolling Correlation (CSV)",
        corr_df.to_csv(index=True),
        file_name="correlation_dashboard_rolling_corr.csv",
        mime="text/csv"
    )
