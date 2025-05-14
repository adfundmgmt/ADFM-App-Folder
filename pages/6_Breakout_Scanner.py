import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Breakout Scanner: 20D/50D Highs & RSI")

st.sidebar.header("About")
st.sidebar.info(
    "Enter comma‑separated tickers below. "
    "This tool pulls the last 3 months of closing prices via yfinance, "
    "then flags stocks at 20‑day or 50‑day highs and shows current 14‑day RSI."
)

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):", 
    "AAPL, MSFT, NVDA, TSLA, AMD"
).upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# ─── Fetch Prices per Ticker ───────────────────────────────────────────────────
period = "3mo"
failed = []
price_dict = {}

for t in tickers:
    hist = yf.Ticker(t).history(period=period)
    if hist.empty or "Close" not in hist.columns:
        failed.append(t)
    else:
        price_dict[t] = hist["Close"].dropna()

if failed:
    st.sidebar.warning(f"Ignored invalid/unreachable tickers: {', '.join(failed)}")
if not price_dict:
    st.error("No valid price data returned. Check tickers or your internet connection.")
    st.stop()

price_df = pd.DataFrame(price_dict)

# ─── RSI Calculator ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def compute_rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return float((100 - (100 / (1 + rs))).iloc[-1])

# ─── Build Signal Table ────────────────────────────────────────────────────────
records = []
for sym, s in price_df.items():
    if len(s) < 50:
        continue
    price = s.iat[-1]
    high20 = s[-20:].max()
    high50 = s[-50:].max()
    rsi14 = compute_rsi(s)
    records.append({
        "Ticker": sym,
        "Price": round(price, 2),
        "20D High": round(high20, 2),
        "50D High": round(high50, 2),
        "Breakout 20D": price >= high20,
        "Breakout 50D": price >= high50,
        "RSI (14D)": round(rsi14, 2)
    })

# ─── Display Results ───────────────────────────────────────────────────────────
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(by=["Breakout 50D", "Breakout 20D"], ascending=False)
    st.dataframe(df, use_container_width=True)
else:
    st.info("✅ No current breakouts detected among the provided tickers.")
