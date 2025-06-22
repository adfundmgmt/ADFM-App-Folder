import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

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
    ticker_x = st.text_input("Ticker X", value="AAPL").strip().upper()
    ticker_y = st.text_input("Ticker Y", value="SPY").strip().upper()
    ticker_z = st.text_input("Ticker Z (optional)", value="", help="Benchmark/index (optional).").strip().upper()
    freq = st.selectbox("Return Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    roll_window = st.slider("Rolling window (periods)", 20, 120, value=60)
    st.markdown("---")
    st.caption("© 2025 AD Fund Management LP")

with st.container():
    c1, c2, c3, c4 = st.columns([2,2,2,3])
    c1.markdown(f"<b>Primary</b>: <span style='color:#1976D2'>{ticker_x}</span>", unsafe_allow_html=True)
    c2.markdown(f"<b>Secondary</b>: <span style='color:#D2691E'>{ticker_y}</span>", unsafe_allow_html=True)
    if ticker_z:
        c3.markdown(f"<b>Benchmark</b>: <span style='color:#2ca02c'>{ticker_z}</span>", unsafe_allow_html=True)
    c4.markdown(f"<b>Freq</b>: {freq} <br><b>Rolling</b>: {roll_window}", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_prices(symbols: list, start: dt.date, end: dt.date, freq: str) -> pd.DataFrame:
    raw = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),
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

# --- Data setup ---
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

# --- Rolling Correlation Chart (tight axes, no legend) ---
st.markdown("### Rolling Correlation (live regime shifts)")
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
    ratio_tabs = st.tabs(list(windows.keys()))
    for tab, label in zip(ratio_tabs, windows.keys()):
        since = windows[label]
        price_slice = slice_since(prices, since)
        with tab:
            for t1, t2 in pair_list:
                if t1 not in price_slice.columns or t2 not in price_slice.columns:
                    continue
                ratio = compute_ratio(price_slice, t1, t2)
                if ratio.empty:
                    continue
                rmin = ratio.min() - 0.07 * abs(ratio.min())
                rmax = ratio.max() + 0.07 * abs(ratio.max())
                if rmin == rmax:
                    rmin, rmax = None, None
                ratio_df = ratio.reset_index()
                ratio_df.columns = ["Date", "Ratio"]
                chart = (
                    alt.Chart(ratio_df)
                    .mark_line(strokeWidth=2.2)
                    .encode(
                        x=alt.X("Date:T", title="Date", axis=alt.Axis(labelAngle=0, format="%b %Y")),
                        y=alt.Y("Ratio:Q", title=f"{t1} / {t2} Ratio", scale=alt.Scale(domain=[rmin, rmax] if rmin is not None and rmax is not None else None)),
                        tooltip=["Date:T", alt.Tooltip("Ratio:Q", format=".3f")]
                    )
                    .properties(height=200, title=f"{t1} / {t2}")
                    .interactive()
                ).configure_legend(disable=True)
                st.altair_chart(chart, use_container_width=True)


# --- Relative Value (Ratio) Charts (tight axes, no legend) ---
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
        rmin = ratio.min() - 0.07 * abs(ratio.min())
        rmax = ratio.max() + 0.07 * abs(ratio.max())
        if rmin == rmax:  # fallback for degenerate
            rmin, rmax = None, None
        ratio_df = ratio.reset_index()
        ratio_df.columns = ["Date", "Ratio"]
        chart = (
            alt.Chart(ratio_df)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("Date:T", title="Date", axis=alt.Axis(labelAngle=0, format="%b %Y")),
                y=alt.Y("Ratio:Q", title=f"{t1} / {t2} Ratio", scale=alt.Scale(domain=[rmin, rmax] if rmin is not None and rmax is not None else None)),
                tooltip=["Date:T", alt.Tooltip("Ratio:Q", format=".3f")]
            )
            .properties(height=180, title=f"{t1} / {t2}")
            .interactive()
        ).configure_legend(disable=True)
        st.altair_chart(chart, use_container_width=True)

# --- Indexed Price Overlays (tight axes, no legend) ---
st.markdown("### Indexed Price Overlays")
tabs = st.tabs(list(windows.keys()))
for tab, label in zip(tabs, windows.keys()):
    since = windows[label]
    price_slice = slice_since(prices, since)
    overlay_tickers = [ticker_x, ticker_y] + ([ticker_z] if ticker_z else [])
    overlay_tickers = [t for t in overlay_tickers if t in price_slice.columns]
    norm_df = price_slice[overlay_tickers].apply(normalize)
    # Y axis range tight
    min_val, max_val = norm_df.min().min(), norm_df.max().max()
    ymin = min_val - 0.07 * abs(min_val)
    ymax = max_val + 0.07 * abs(max_val)
    if ymin == ymax:  # fallback
        ymin, ymax = None, None
    norm_df = norm_df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="IndexedPrice")
    chart = (
        alt.Chart(norm_df)
        .mark_line(strokeWidth=2.1)
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(labelAngle=0, format="%b %Y")),
            y=alt.Y("IndexedPrice:Q", title="Indexed Price (Base=100)", scale=alt.Scale(domain=[ymin, ymax] if ymin is not None and ymax is not None else None)),
            color=alt.Color("Ticker:N", legend=None if len(overlay_tickers) == 1 else alt.Legend(title="Ticker")),
            tooltip=["Date:T", "Ticker:N", alt.Tooltip("IndexedPrice:Q", format=".2f")],
        )
        .properties(height=180)
        .interactive()
    )
    with tab:
        st.altair_chart(chart, use_container_width=True)
