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
- 🔺 **20‑day or 50‑day highs**
- ⚡️ Strong momentum via **14‑day RSI**

Use it to spot potential trend‑continuation setups.

**Data Source:** Yahoo Finance  
**Indicators:** RSI(14), 20D / 50D highs
""")

st.sidebar.subheader("🧩 Input Settings")
user_input = st.sidebar.text_area(
    "Enter tickers (comma‑separated):",
    "AAPL, MSFT, NVDA, TSLA, AMD, META, AVGO, GOOGL"
)
tickers_raw = [t.strip().upper() for t in user_input.split(",") if t.strip()]
if not tickers_raw:
    st.warning("Please enter at least one valid ticker.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Breakout Scanner")
st.caption("Live scanner for 20D / 50D highs and RSI signals")

lookback_days = 90
start_date = datetime.today() - timedelta(days=lookback_days)

# ─────────────────────────────────────────────────────────────────────────────
# Robust Data Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tickers):
    valid = {}
    bad = []
    for t in tickers:
        try:
            df = yf.download(t, start=start_date.strftime("%Y-%m-%d"), progress=False)
            if "Adj Close" in df and not df["Adj Close"].isna().all():
                valid[t] = df["Adj Close"]
            else:
                bad.append(t)
        except Exception:
            bad.append(t)
    if not valid:
        return None, bad
    return pd.DataFrame(valid), bad

price_data, failed = load_data(tickers_raw)
if failed:
    st.sidebar.warning(f"Ignored invalid tickers: {', '.join(failed)}")
if price_data is None:
    st.error("❌ No valid data returned. Make sure you’re running with internet access and valid tickers.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RSI Calculator
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(s, window=14):
    delta = s.diff()
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
for t in price_data.columns:
    series = price_data[t].dropna()
    if len(series) < 50:
        continue
    p = series.iloc[-1]
    h20 = series[-20:].max()
    h50 = series[-50:].max()
    bo20 = p >= h20
    bo50 = p >= h50
    rsi = compute_rsi(series).iloc[-1]
    signals.append({
        "Ticker": t,
        "Price": round(p, 2),
        "20D High": round(h20, 2),
        "50D High": round(h50, 2),
        "Breakout 20D": bo20,
        "Breakout 50D": bo50,
        "RSI (14D)": round(rsi, 2)
    })

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
if signals:
    df = pd.DataFrame(signals).sort_values(
        by=["Breakout 50D", "Breakout 20D"], ascending=False
    )
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No breakouts detected for the provided tickers.")
