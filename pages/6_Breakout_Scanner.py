import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- Sidebar: Tool Description ---
st.sidebar.header("📘 About This Tool")
st.sidebar.markdown("""
This breakout scanner flags stocks that:
- Hit **20-day or 50-day highs**
- Show **RSI signals** of momentum
- Belong to any custom list of tickers

Useful for spotting high-momentum breakout setups favored by CTAs or trend followers.

**Data Source:** Yahoo Finance  
**Indicators Used:**  
- 20D & 50D highs  
- 14-day RSI
""")

# --- User Input for Tickers ---
st.sidebar.subheader("🧩 Input Settings")
user_input = st.sidebar.text_area("Enter tickers (comma-separated):", "AAPL, MSFT, NVDA, TSLA, AMD")
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

# --- Page Title ---
st.title("📈 Breakout Scanner")
st.caption("Live scanner for 20D / 50D highs and RSI signals")

# --- Define Lookback Window ---
lookback_days = 90
start_date = datetime.today() - timedelta(days=lookback_days)

# --- Fetch Price Data ---
@st.cache_data(ttl=3600)
def load_data(tickers):
    data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), progress=False)["Adj Close"]
    return data

try:
    price_data = load_data(tickers)
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()

# --- RSI Calculation ---
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- Signal Generation ---
signals = []

for ticker in tickers:
    try:
        series = price_data[ticker].dropna()
        if len(series) < 50:
            continue

        today_price = series.iloc[-1]
        high_20d = series[-20:].max()
        high_50d = series[-50:].max()
        breakout_20d = today_price >= high_20d
        breakout_50d = today_price >= high_50d
        rsi = compute_rsi(series).iloc[-1]

        signals.append({
            "Ticker": ticker,
            "Price": round(today_price, 2),
            "20D High": round(high_20d, 2),
            "50D High": round(high_50d, 2),
            "Breakout 20D": breakout_20d,
            "Breakout 50D": breakout_50d,
            "RSI (14D)": round(rsi, 2)
        })
    except Exception as e:
        continue

# --- Display Results ---
df = pd.DataFrame(signals)
st.dataframe(df.sort_values(by=["Breakout 50D", "Breakout 20D"], ascending=False), use_container_width=True)
