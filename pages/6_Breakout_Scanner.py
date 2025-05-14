import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Breakout Scanner: 20D/50D Highs & RSI")

st.sidebar.header("About")
st.sidebar.info(
    "Enter tickers (comma‑separated). Fetches last 3 months of prices, "
    "flags 20D/50D highs, and shows 14‑day RSI."
)

# User input
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):", "AAPL, MSFT, NVDA, TSLA, AMD"
).upper()
tickers = [s.strip() for s in tickers_input.split(",") if s.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# Download data per ticker
failed = []
price_df = pd.DataFrame()
for t in tickers:
    df = yf.download(t, period="3mo", progress=False)
    if df is None or "Adj Close" not in df.columns or df["Adj Close"].dropna().empty:
        failed.append(t)
    else:
        price_df[t] = df["Adj Close"]

if failed:
    st.sidebar.warning(f"Ignored invalid tickers: {','.join(failed)}")

if price_df.empty:
    st.error("No valid price data. Check tickers or internet access.")
    st.stop()

# RSI calculation
@st.cache_data(ttl=3600)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return float((100 - (100 / (1 + rs))).iloc[-1])

# Build results
records = []
for sym in price_df.columns:
    s = price_df[sym].dropna()
    if len(s) < 50:
        continue
    price = s.iat[-1]
    h20 = s[-20:].max()
    h50 = s[-50:].max()
    rsi = compute_rsi(s)
    records.append({
        "Ticker": sym,
        "Price": round(price,2),
        "20D High": round(h20,2),
        "50D High": round(h50,2),
        "Breakout 20D": price>=h20,
        "Breakout 50D": price>=h50,
        "RSI (14D)": round(rsi,2)
    })

# Display
if records:
    out = pd.DataFrame(records).sort_values(
        by=["Breakout 50D","Breakout 20D"], ascending=False
    )
    st.dataframe(out, use_container_width=True)
else:
    st.info("No current breakouts detected.")
