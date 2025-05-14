import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Ticker Technical Chart")
st.title("📊 Technical Chart Explorer")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
period = st.sidebar.selectbox("Period", ["6mo","1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# Fetch data
df = yf.Ticker(ticker).history(period=period, interval=interval)
if df.empty:
    st.error("No data for this ticker. Check symbol or internet.")
    st.stop()

# Compute moving averages
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# Compute RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI14"] = 100 - (100 / (1 + rs))

# Compute MACD(12,26,9)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"] = df["MACD"] - df["Signal"]

# Plotting
fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
    3, 1, figsize=(12, 9), sharex=True,
    gridspec_kw={"height_ratios":[3,1,1], "hspace":0.05}
)

# Price + MAs
ax_price.plot(df.index, df["Close"], label="Close", color="black", lw=1.5)
for w,col in zip((20,50,100,200), ("purple","blue","orange","gray")):
    ax_price.plot(df.index, df[f"MA{w}"], label=f"MA{w}", color=col, lw=1)
ax_price.set_title(f"{ticker} Price & Moving Averages")
ax_price.legend(loc="upper left", fontsize=8)
ax_price.grid(alpha=0.3)

# Volume as bar underneath price axis
ax_vol = ax_price.twinx()
ax_vol.bar(df.index, df["Volume"], color="lightgray", alpha=0.3, width=1)
ax_vol.set_ylabel("Volume", color="gray")
ax_vol.tick_params(axis="y", labelcolor="gray")

# RSI
ax_rsi.plot(df.index, df["RSI14"], color="purple", lw=1.2)
ax_rsi.axhline(70, ls="--", color="gray", lw=0.8)
ax_rsi.axhline(30, ls="--", color="gray", lw=0.8)
ax_rsi.set_title("RSI (14)")
ax_rsi.grid(alpha=0.3)

# MACD
ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue", lw=1.2)
ax_macd.plot(df.index, df["Signal"], label="Signal", color="orange", lw=1)
ax_macd.bar(df.index, df["Hist"], label="Hist", color="gray", alpha=0.5, width=1)
ax_macd.set_title("MACD (12,26,9)")
ax_macd.legend(loc="upper left", fontsize=8)
ax_macd.grid(alpha=0.3)

# Final layout
fig.tight_layout()
st.pyplot(fig)
