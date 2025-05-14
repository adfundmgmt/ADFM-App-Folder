import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breakout Scanner", layout="wide")
st.title("📈 Breakout Scanner")

st.sidebar.header("About")
st.sidebar.info(
    "Flags stocks at 20‑day or 50‑day highs and shows 14‑day RSI for momentum.\n\n"
    "Enter comma‑separated tickers below."
)

# ─── Inputs ───────────────────────────────────────────────────────────────────
tickers_input = st.sidebar.text_input(
    "Tickers (comma‑separated):", 
    "AAPL, MSFT, NVDA, TSLA, AMD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

# ─── Fetch Adjusted Close ──────────────────────────────────────────────────────
lookback = 90
start_date = datetime.now() - timedelta(days=lookback)

@st.cache_data(ttl=3600)
def get_adj_close(tickers, start):
    raw = yf.download(tickers, start=start, progress=False)
    # For multi-ticker, raw.columns is a MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        # pick the 'Adj Close' subframe
        if 'Adj Close' in raw.columns.get_level_values(0):
            data = raw['Adj Close']
        else:
            return pd.DataFrame()
    else:
        # single ticker case: wrap into DataFrame
        if 'Adj Close' in raw.columns:
            data = raw[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        else:
            return pd.DataFrame()
    return data.dropna(axis=1, how='all')

price_df = get_adj_close(tickers, start_date)
if price_df.empty:
    st.error("No price data returned. Check tickers or internet access.")
    st.stop()

# ─── Indicator Functions ──────────────────────────────────────────────────────
def compute_rsi(s, window=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ─── Signal Generation ────────────────────────────────────────────────────────
records = []
for sym in price_df.columns:
    series = price_df[sym].dropna()
    if len(series) < 50:
        continue
    price = series.iloc[-1]
    high20 = series[-20:].max()
    high50 = series[-50:].max()
    rsi = compute_rsi(series).iloc[-1]
    records.append({
        "Ticker": sym,
        "Price": round(price, 2),
        "20D High": round(high20, 2),
        "50D High": round(high50, 2),
        "Breakout 20D": price >= high20,
        "Breakout 50D": price >= high50,
        "RSI (14)": round(rsi, 2)
    })

# ─── Display ─────────────────────────────────────────────────────────────────
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["Breakout 50D", "Breakout 20D"], ascending=False
    )
    st.dataframe(df, use_container_width=True)
else:
    st.info("No current breakouts detected.")
