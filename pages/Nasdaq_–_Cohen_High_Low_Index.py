# nasdaq_cohen_high_low_all.py
# Cohen-style High Low Index using all Nasdaq-listed securities

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
    page_title="Nasdaq – Cohen High Low Index (All Listings)",
    layout="wide"
)

st.title("Nasdaq – Cohen High Low Index")

st.write(
    """
This app approximates the **Nasdaq Cohen High Low Index** using **all Nasdaq-listed
securities** from Nasdaq Trader's symbol directory:

- Universe: all symbols in `nasdaqlisted.txt` with `Test Issue = N`
- Each day we compute how many symbols are making 52-week highs and lows
- `Record High Percent = NewHighs / (NewHighs + NewLows)`
- **High Low Index** = 10-day simple moving average of `Record High Percent` (0 to 100)
    """
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_all_nasdaq_tickers() -> list[str]:
    """
    Fetch all Nasdaq-listed tickers from Nasdaq Trader's official symbol file.
    Filters out test issues and extracts the Symbol column.
    """
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")

    # Drop the last row which is "File Creation Time"
    df = df[df["Symbol"].str.contains("File Creation Time") == False]

    # Exclude test issues
    df = df[df["Test Issue"] == "N"]

    tickers = df["Symbol"].astype(str).str.strip().tolist()
    tickers = sorted(pd.unique(pd.Series(tickers)))
    return tickers


@st.cache_data(ttl=24 * 3600)
def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers from `start` to today.
    Uses chunking so we do not overload yfinance with thousands of tickers at once.
    Returns a DataFrame indexed by date, columns = tickers.
    """
    if not tickers:
        return pd.DataFrame()

    all_frames = []
    chunk_size = 200

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        data = yf.download(
            chunk,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if data.empty:
            continue

        # yfinance returns different shapes depending on chunk size
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index: outer level = ticker, inner = OHLCV
            sub_frames = []
            for t in chunk:
                if t in data.columns.get_level_values(0):
                    try:
                        sub = data[(t, "Adj Close")].rename(t)
                        sub_frames.append(sub)
                    except KeyError:
                        continue
            if sub_frames:
                chunk_close = pd.concat(sub_frames, axis=1)
            else:
                continue
        else:
            # Single ticker case
            if "Adj Close" in data.columns:
                chunk_close = data[["Adj Close"]].rename(columns={"Adj Close": chunk[0]})
            else:
                continue

        all_frames.append(chunk_close)

    if not all_frames:
        return pd.DataFrame()

    adj_close = pd.concat(all_frames, axis=1)
    # Remove duplicate columns if any and sort
    adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
    adj_close = adj_close.sort_index(axis=1)
    # Drop days where everything is NaN
    adj_close = adj_close.dropna(how="all")
    return adj_close


@st.cache_data(ttl=24 * 3600)
def compute_high_low_index(adj_close: pd.DataFrame) -> pd.DataFrame:
    """
    Given adjusted close prices (dates x tickers), compute:
    - new highs count
    - new lows count
    - Record High Percent (0..100)
    - High Low Index (10-day SMA of Record High Percent, 0..100)
    """
    if adj_close.empty:
        return pd.DataFrame()

    window = 252  # about 1 trading year

    rolling_max = adj_close.rolling(window=window, min_periods=40).max()
    rolling_min = adj_close.rolling(window=window, min_periods=40).min()

    new_highs_bool = (adj_close >= rolling_max) & adj_close.notna()
    new_lows_bool = (adj_close <= rolling_min) & adj_close.notna()

    new_highs = new_highs_bool.sum(axis=1)
    new_lows = new_lows_bool.sum(axis=1)

    total = new_highs + new_lows
    record_high_pct = new_highs / total.replace(0, np.nan)  # 0 to 1

    high_low_index = record_high_pct.rolling(window=10, min_periods=5).mean() * 100.0

    out = pd.DataFrame(
        {
            "new_highs": new_highs,
            "new_lows": new_lows,
            "record_high_pct": record_high_pct * 100.0,
            "high_low_index": high_low_index,
        }
    )
    return out


@st.cache_data(ttl=24 * 3600)
def download_nasdaq_composite(start: str) -> pd.Series:
    """
    Download Nasdaq Composite (^IXIC) adjusted close from Yahoo.
    """
    ixic = yf.download("^IXIC", start=start, auto_adjust=True, progress=False)
    if "Adj Close" not in ixic.columns:
        return pd.Series(dtype=float)
    return ixic["Adj Close"].rename("Nasdaq Composite")


def make_chart(breadth: pd.DataFrame, ixic: pd.Series):
    """
    Two-panel Plotly figure:
    - Top: Cohen High Low Index with reference levels
    - Bottom: Nasdaq Composite
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.03,
        subplot_titles=(
            "Nasdaq Cohen High Low Index (All Nasdaq Listings, 10-day SMA)",
            "Nasdaq Composite Index (^IXIC)",
        ),
    )

    # Top panel
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

    for level in [30, 50, 70]:
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
        title_text="High Low Index (0 to 100)",
        range=[0, 100],
        row=1,
        col=1,
    )

    # Bottom panel
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig


# -----------------------------
# UI controls
# -----------------------------
st.sidebar.header("Settings")

years_back = st.sidebar.slider(
    "Lookback (years)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
)

start_date = date.today() - timedelta(days=365 * years_back)
st.sidebar.write(f"Start date: {start_date.isoformat()}")

# -----------------------------
# Main computation
# -----------------------------
with st.spinner("Loading all Nasdaq-listed symbols from Nasdaq Trader..."):
    all_nasdaq = get_all_nasdaq_tickers()

st.sidebar.write(f"Nasdaq-listed symbols loaded: {len(all_nasdaq)}")

with st.spinner("Downloading price data and computing High Low Index..."):
    prices = download_prices(all_nasdaq, start=start_date.isoformat())
    breadth = compute_high_low_index(prices)
    ixic = download_nasdaq_composite(start=start_date.isoformat())

# Align indices
common_index = breadth.index.intersection(ixic.index)
breadth = breadth.loc[common_index]
ixic = ixic.loc[common_index]

if breadth.empty or ixic.empty:
    st.error("No data available. Check internet access and try again.")
else:
    fig = make_chart(breadth, ixic)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Breadth data (last 10 observations)")
    st.dataframe(breadth.tail(10))
