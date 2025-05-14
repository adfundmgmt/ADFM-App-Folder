import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: About & Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📘 About This Tool")
st.sidebar.markdown("""
This scanner flags stocks breaking out to:
- 🔺 **20D or 50D highs**
- ⚡️ Strong momentum via **14-day RSI**

Useful for CTA-style setups and trend followers.

**Data Source:** Yahoo Finance  
**Indicators Used:** RSI(14), 20D / 50D highs
""")

st.sidebar.subheader("🧩 Ticker Settings")
user_input = st.sidebar.text_area(
    "Enter tickers (comma-separated):",
    "AAPL, MSFT, NVDA, TSLA, AMD, META, AVGO, GOOGL"
)
tickers_raw = [t.strip().upper() for t in user_input.split(",") if t.strip()]
if not tickers_raw:
    st.warning("Please enter at least one valid ticker.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Breakout Scanner")
st.caption("Live scanner for 20D / 50D highs and RSI signals")

lookback_days = 90
start_date = datetime.today() - timedelta(days=lookback_days)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Data Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_valid_ticker_data(tickers):
    valid_data = {}
    bad_tickers = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), progress=False)
            if "Adj Close" in df and not df["Adj Close"].isna().all():
                valid_data[ticker] = df["Adj Close"]
            else:
                bad_tickers.append(ticker)
        except Exception:
            bad_tickers.append(ticker)

    if not valid_data:
        return None, bad_tickers

    combined = pd.DataFrame(valid_data)
    return combined, bad_tickers

price_data, failed_tickers = load_valid_ticker_data(tickers_raw)

if failed_tickers:
    st.sidebar.warning(f"Ignored invalid tickers: {', '.join(failed_tickers)}")

if price_data is None or price_data.empty:
    st.error("❌ No valid data returned from Yahoo Finance.")
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
for ticker in price_data.columns:
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

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
if signals:
    df = pd.DataFrame(signals)
    df = df.sort_values(by=["Breakout 50D", "Breakout 20D"], ascending=False)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("✅ No breakouts yet — all tickers valid, but no signals hit.")
