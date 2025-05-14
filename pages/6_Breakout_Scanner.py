import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Breakout Scanner: 20D/50D Highs & RSI")

st.sidebar.header("About")
st.sidebar.info(
    "Enter comma‑separated tickers below. "
    "This tool fetches the last 3 months of adjusted close prices, "
    "then flags stocks at 20‑day or 50‑day highs and shows current 14‑day RSI."
)

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):", 
    value="AAPL, MSFT, NVDA, TSLA, AMD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker in the sidebar.")
    st.stop()

# ─── Fetch Prices ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(tickers):
    df = pd.DataFrame()
    failed = []
    for t in tickers:
        data = yf.download(t, period="3mo", progress=False)
        if "Adj Close" in data and not data["Adj Close"].isna().all():
            df[t] = data["Adj Close"]
        else:
            failed.append(t)
    df.dropna(axis=1, how="all", inplace=True)
    return df, failed

price_df, failed = fetch_prices(tickers)

if failed:
    st.sidebar.warning(f"Ignored invalid tickers: {', '.join(failed)}")
if price_df.empty:
    st.error("No valid price data returned. Check your tickers or internet connection.")
    st.stop()

# ─── Indicator Functions ──────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return float((100 - (100 / (1 + rs))).iloc[-1])

# ─── Generate Signals ───────────────────────────────────────────────────────────
records = []
for sym in price_df.columns:
    s = price_df[sym].dropna()
    if len(s) < 50:
        continue
    price = s.iloc[-1]
    high20 = s[-20:].max()
    high50 = s[-50:].max()
    records.append({
        "Ticker": sym,
        "Price": round(price, 2),
        "20D High": round(high20, 2),
        "50D High": round(high50, 2),
        "Breakout 20D": price >= high20,
        "Breakout 50D": price >= high50,
        "RSI (14D)": round(compute_rsi(s), 2)
    })

# ─── Display Results ─────────────────────────────────────────────────────────────
if records:
    out = pd.DataFrame(records)
    out = out.sort_values(by=["Breakout 50D", "Breakout 20D"], ascending=False)
    st.dataframe(out, use_container_width=True)
else:
    st.info("No current breakouts detected in your ticker list.")
