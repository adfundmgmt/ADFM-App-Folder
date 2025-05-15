import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Sector ETFs and tickers
SECTOR_ETFS = {
    'S&P 500': 'SPY',
    'Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}

st.set_page_config(page_title="Sector Breadth & Rotation", layout="wide")
st.title("📊 S&P 500 Sector Breadth & Rotation Monitor")
st.markdown(
    "Tracks sector leadership, momentum, and internal market breadth using free Yahoo Finance data. "
    "Helps you spot turning points, regime shifts, and emerging sector trends—no paywalls or data headaches."
)

# --- Inputs
with st.sidebar:
    st.header("Analysis Window")
    lookback = st.slider("Lookback (days)", 60, 365, 180, 10)
    show_ratio = st.checkbox("Show Sector/SPY Relative Strength", value=True)
    st.caption("Powered by Yahoo Finance. All data is delayed.")

# --- Fetch prices
@st.cache_data(ttl=21600)
def fetch_prices(tickers, period="1y"):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):  # one ticker
        data = data.to_frame()
    return data.ffill().bfill()

tickers = list(SECTOR_ETFS.values())
prices = fetch_prices(tickers, period="max")
prices = prices[-lookback:]

# --- Breadth: % above MA (20D, 50D, 200D)
def pct_above_ma(s, window):
    return (s > s.rolling(window).mean()).mean() * 100

ma_windows = [20, 50, 200]
breadth_stats = {
    name: {
        f"%>{w}D": pct_above_ma(prices[ticker], w)
        for w in ma_windows
    }
    for name, ticker in SECTOR_ETFS.items()
}

breadth_df = pd.DataFrame(breadth_stats).T
breadth_df = breadth_df[[f"%>{w}D" for w in ma_windows]]
st.subheader("Sector Breadth: % Above Moving Average")
st.dataframe(breadth_df.style.format("{:.1f}%"))

# --- Performance Heatmap
perf_windows = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "YTD": (prices.index[0] - pd.offsets.BDay(prices.index[0].weekday())).days if prices.index[0].year == datetime.now().year else None,
}
perf = {}
for name, ticker in SECTOR_ETFS.items():
    perf[name] = {}
    for label, days in perf_windows.items():
        if days is not None and len(prices[ticker]) > days:
            p = (prices[ticker].iloc[-1] / prices[ticker].iloc[-days] - 1) * 100
            perf[name][label] = p
        elif label == "YTD":
            ytd = (prices[ticker].iloc[-1] / prices[ticker][prices[ticker].index.year == datetime.now().year].iloc[0] - 1) * 100
            perf[name][label] = ytd
        else:
            perf[name][label] = np.nan
perf_df = pd.DataFrame(perf).T[["1W", "1M", "3M", "YTD"]]
st.subheader("Sector Performance Heatmap (%)")
st.dataframe(perf_df.style.background_gradient(cmap="RdYlGn", axis=0).format("{:+.2f}%"))

# --- Relative Strength Plot
if show_ratio:
    st.subheader("Sector Relative Strength vs. SPY")
    sector = st.selectbox("Select sector for relative chart:", [k for k in SECTOR_ETFS if k != "S&P 500"])
    ratio = prices[SECTOR_ETFS[sector]] / prices["SPY"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ratio.index, ratio, color="purple", lw=2)
    ax.set_title(f"{sector} / SPY Ratio")
    ax.set_ylabel("Relative Strength")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

# --- Mini-rotation chart
st.subheader("Sector Rotation Plot (3M vs 1W Returns)")
fig, ax = plt.subplots(figsize=(7, 7))
x = perf_df["3M"]
y = perf_df["1W"]
for i, name in enumerate(perf_df.index):
    ax.scatter(x[name], y[name], s=150, label=name)
    ax.annotate(name, (x[name], y[name]), fontsize=9, ha='center', va='bottom')
ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("3M Return (%)")
ax.set_ylabel("1W Return (%)")
ax.set_title("Sector Rotation: Momentum vs Short-term Reversal")
st.pyplot(fig)

st.caption("Built by AD Fund Management LP — All data via Yahoo Finance. Use for informational purposes only.")
