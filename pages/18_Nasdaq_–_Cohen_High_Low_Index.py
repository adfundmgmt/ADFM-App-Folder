# nasdaq_cohen_high_low.py
# Cohen-style High Low Index for Nasdaq (approx via Nasdaq-100 components)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Nasdaq – Cohen High Low Index",
    layout="wide"
)

st.title("Nasdaq – Cohen High Low Index (Streamlit)")

st.write(
    """
This app approximates the **Nasdaq Cohen High Low Index** using Nasdaq-100 constituents:

- For each stock we compute daily 52-week highs and lows (based on adjusted close).
- Each day:  
  `Record High Percent = NewHighs / (NewHighs + NewLows)`  
- The **High Low Index** is the 10-day simple moving average of Record High Percent.
    """
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_nasdaq100_tickers() -> list[str]:
    """
    Fetch current Nasdaq-100 constituents from Wikipedia.
    Returns a list of ticker symbols suitable for Yahoo Finance / yfinance.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)

    tickers = []
    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]
        if any("ticker" in c or "symbol" in c for c in cols_lower):
            # pick the first ticker/symbol column
            col_idx = next(
                i
                for i, c in enumerate(cols_lower)
                if "ticker" in c or "symbol" in c
            )
            raw = tbl.iloc[:, col_idx].dropna().astype(str)
            # clean footnotes etc (e.g. "AAPL[1]" -> "AAPL")
            cleaned = (
                raw.str.strip()
                .str.replace(r"\s*\(.*\)", "", regex=True)
                .str.replace(r"\[.*\]", "", regex=True)
            )
            tickers.extend(cleaned.tolist())

    # De-duplicate and sort
    tickers = sorted(pd.unique(pd.Series(tickers)))
    # Some tickers have periods for classes (e.g. BRK.B) that Yahoo uses as "-B"
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


@st.cache_data(ttl=24 * 3600)
def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers from `start` to today.
    Returns a DataFrame indexed by date, columns = tickers.
    """
    data = yf.download(
        tickers,
        start=start,
        progress=False,
        auto_adjust=True
    )

    # yfinance returns a multi-index columns if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"]
    else:
        adj_close = data.rename(columns={"Adj Close": tickers[0]})[tickers]

    # Ensure columns exactly match tickers we requested where possible
    adj_close = adj_close.sort_index(axis=1)
    return adj_close


@st.cache_data(ttl=24 * 3600)
def compute_high_low_index(adj_close: pd.DataFrame) -> pd.DataFrame:
    """
    Given adjusted close prices (dates x tickers), compute:
    - new highs count
    - new lows count
    - Record High Percent (0..1)
    - High Low Index (10-day SMA of Record High Percent, 0..100)
    Returns a DataFrame with these series.
    """
    # 252 trading days ~ 1 year
    window = 252

    # Rolling 52-week max/min per stock
    rolling_max = adj_close.rolling(window=window, min_periods=40).max()
    rolling_min = adj_close.rolling(window=window, min_periods=40).min()

    # New high if today's close equals rolling max and we have enough history
    new_highs_bool = (adj_close >= rolling_max) & adj_close.notna()
    new_lows_bool = (adj_close <= rolling_min) & adj_close.notna()

    new_highs = new_highs_bool.sum(axis=1)
    new_lows = new_lows_bool.sum(axis=1)

    total = new_highs + new_lows
    record_high_pct = new_highs / total.replace(0, np.nan)  # 0..1

    high_low_index = record_high_pct.rolling(window=10, min_periods=5).mean() * 100.0

    out = pd.DataFrame(
        {
            "new_highs": new_highs,
            "new_lows": new_lows,
            "record_high_pct": record_high_pct * 100.0,  # percent
            "high_low_index": high_low_index,
        }
    )
    return out


@st.cache_data(ttl=24 * 3600)
def download_nasdaq_composite(start: str) -> pd.Series:
    """
    Download Nasdaq Composite (^IXIC) adjusted close from Yahoo.
    """
    ixic = yf.download("^IXIC", start=start, progress=False, auto_adjust=True)
    return ixic["Adj Close"].rename("Nasdaq Composite")


def make_chart(breadth: pd.DataFrame, ixic: pd.Series):
    """
    Build a 2-panel Plotly figure:
    - Top: Cohen High Low Index with 30 / 70 levels
    - Bottom: Nasdaq Composite price
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.03,
        subplot_titles=(
            "Nasdaq Cohen High Low Index (10-day SMA of Record High Percent)",
            "Nasdaq Composite Index (^IXIC)",
        ),
    )

    # Top panel: High Low Index
    fig.add_trace(
        go.Scatter(
            x=breadth.index,
            y=breadth["high_low_index"],
            name="High Low Index",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # 30 / 50 / 70 reference lines
    for level, name in [(30, "30"), (50, "50"), (70, "70")]:
        fig.add_trace(
            go.Scatter(
                x=breadth.index,
                y=[level] * len(breadth),
                name=f"Level {level}",
                mode="lines",
                line=dict(width=1, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(
        title_text="High Low Index (0–100)",
        range=[0, 100],
        row=1,
        col=1,
    )

    # Bottom panel: Nasdaq Composite
    fig.add_trace(
        go.Scatter(
            x=ixic.index,
            y=ixic,
            name="Nasdaq Composite",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Nasdaq Composite",
        row=2,
        col=1,
        tickformat=",",
    )

    fig.update_layout(
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return fig


# -----------------------------
# UI controls
# -----------------------------
default_years = 2
min_date = date.today() - timedelta(days=5 * 365)

st.sidebar.header("Settings")

years_back = st.sidebar.slider(
    "Lookback (years)",
    min_value=1,
    max_value=5,
    value=default_years,
    step=1,
)

start_date = date.today() - timedelta(days=365 * years_back)
st.sidebar.write(f"Start date: {start_date.isoformat()}")

# -----------------------------
# Main computation
# -----------------------------
with st.spinner("Loading Nasdaq-100 constituents..."):
    nasdaq100 = get_nasdaq100_tickers()

st.sidebar.write(f"Nasdaq-100 tickers loaded: {len(nasdaq100)}")

with st.spinner("Downloading price data and computing High Low Index..."):
    prices = download_prices(nasdaq100, start=start_date.isoformat())
    breadth = compute_high_low_index(prices)
    ixic = download_nasdaq_composite(start=start_date.isoformat())

# Align indices
common_index = breadth.index.intersection(ixic.index)
breadth = breadth.loc[common_index]
ixic = ixic.loc[common_index]

fig = make_chart(breadth, ixic)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Breadth data (last 10 observations)")
st.dataframe(breadth.tail(10))
