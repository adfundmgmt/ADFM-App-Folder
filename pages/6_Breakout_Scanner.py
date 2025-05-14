import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Breakout Scanner")
st.sidebar.header("About")
st.sidebar.info(
    "Flags stocks hitting 20‑day or 50‑day highs, "
    "and shows current 14‑day RSI for momentum context."
)

# ── User Input ─────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):",
    value="AAPL, MSFT, NVDA, TSLA, AMD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

# ── Fetch Price Data ────────────────────────────────────────────────────────────
lookback_days = 90
start_date = datetime.now() - timedelta(days=lookback_days)

@st.cache_data(ttl=3600)
def get_adj_close(tickers, start):
    data = yf.download(tickers, start=start, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(axis=1, how="all")

price_df = get_adj_close(tickers, start_date)
if price_df.empty:
    st.error("No price data found. Check tickers or internet connection.")
    st.stop()

# ── Indicator Functions ─────────────────────────────────────────────────────────
def compute_rsi(s: pd.Series, window=14) -> float:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return float(100 - (100 / (1 + rs.iloc[-1])))

# ── Generate Signals ────────────────────────────────────────────────────────────
records = []
for sym in price_df.columns:
    series = price_df[sym].dropna()
    if len(series) < 50:
        continue
    price = series.iloc[-1]
    high20 = series[-20:].max()
    high50 = series[-50:].max()
    records.append({
        "Ticker": sym,
        "Price": round(price, 2),
        "20D High": round(high20, 2),
        "50D High": round(high50, 2),
        "Breakout 20D": price >= high20,
        "Breakout 50D": price >= high50,
        "RSI (14)": round(compute_rsi(series), 2)
    })

# ── Display Results ─────────────────────────────────────────────────────────────
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["Breakout 50D", "Breakout 20D"], ascending=False
    )
    st.dataframe(df, use_container_width=True)
else:
    st.info("No breakouts detected with your ticker list.")
