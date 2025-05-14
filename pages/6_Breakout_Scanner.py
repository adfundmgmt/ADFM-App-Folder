import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Description and Input
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📘 About This Tool")
st.sidebar.markdown("""
This breakout scanner helps identify stocks breaking out to:
- 🔺 **20-day or 50-day highs**
- 🔺 High **momentum** via 14-day RSI

Use it to spot potential trend continuation setups.
""")

st.sidebar.subheader("🧩 Input Settings")
user_input = st.sidebar.text_area(
    "Enter tickers (comma-separated):",
    "AAPL, MSFT, NVDA, TSLA, AMD"
)
tickers_raw = [t.strip().upper() for t in user_input.split(",") if t.strip()]
tickers = tickers_raw[0] if len(tickers_raw) == 1 else tickers_raw

# ─────────────────────────────────────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Breakout Scanner")
st.caption("Live scanner for 20D / 50D highs and RSI signals")

lookback_days = 90
start_date = datetime.today() - timedelta(days=lookback_days)

# ─────────────────────────────────────────────────────────────────────────────
# Data Fetcher
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tickers):
    raw = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), progress=False)

    if isinstance(tickers, str) or len(tickers) == 1:
        # Single ticker: wrap Series into DataFrame
        if "Adj Close" in raw:
            return pd.DataFrame({tickers if isinstance(tickers, str) else tickers[0]: raw["Adj Close"]})
        else:
            raise ValueError("No 'Adj Close' data found.")
    else:
        if "Adj Close" in raw:
            return raw["Adj Close"]
        else:
            raise ValueError("No 'Adj Close' data returned from Yahoo Finance.")

try:
    price_data = load_data(tickers)
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RSI Calculator
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────────────────────────────────────
# Signal Engine
# ─────────────────────────────────────────────────────────────────────────────
signals = []

for ticker in (tickers_raw if isinstance(tickers, list) else [tickers]):
    try:
        series = price_data[ticker].dropna()
        if len(series) < 50:
            continue

        price = series.iloc[-1]
        high_20d = series[-20:].max()
        high_50d = series[-50:].max()
        breakout_20d = price >= high_20d
        breakout_50d = price >= high_50d
        rsi = compute_rsi(series).iloc[-1]

        signals.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "20D High": round(high_20d, 2),
            "50D High": round(high_50d, 2),
            "Breakout 20D": breakout_20d,
            "Breakout 50D": breakout_50d,
            "RSI (14D)": round(rsi, 2)
        })
    except Exception as e:
        continue

# ─────────────────────────────────────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────────────────────────────────────
if signals:
    df = pd.DataFrame(signals)
    df = df.sort_values(by=["Breakout 50D", "Breakout 20D"], ascending=False)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No valid tickers returned data or breakouts.")
