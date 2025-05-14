import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
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
    st.error("No data for ticker. Check symbol or internet connection.")
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

# Volume profile (distribution of Close weighted by Volume)
price_bins = np.linspace(df["Low"].min(), df["High"].max(), 30)
vol_profile, _ = np.histogram(df["Close"], bins=price_bins, weights=df["Volume"])
# normalize for plotting
vol_profile = vol_profile / vol_profile.max()

# Build mplfinance style
mc = mpf.make_marketcolors(up="g", down="r", inherit=True)
s  = mpf.make_mpf_style(marketcolors=mc)
add_plots = [
    mpf.make_addplot(df["MA20"], color="purple", width=1),
    mpf.make_addplot(df["MA50"], color="blue",  width=1),
    mpf.make_addplot(df["MA100"], color="orange",width=1),
    mpf.make_addplot(df["MA200"], color="black", width=1),
]

# Create figure
fig = mpf.figure(style=s, figsize=(16,10), dpi=100)
ax_candle = fig.add_subplot(4,4, (1,3))
ax_rsi    = fig.add_subplot(4,4,  5)
ax_macd   = fig.add_subplot(4,4,  9)
ax_volpro = fig.add_subplot(4,4,  (4,8))

# Plot candlesticks + MAs
mpf.plot(df,
         type="candle",
         ax=ax_candle,
         addplot=add_plots,
         volume=False,
         show_nontrading=False)

ax_candle.set_title(f"{ticker} Price ({period}, {interval})")
ax_candle.grid(alpha=0.2)

# Plot RSI
ax_rsi.plot(df.index, df["RSI14"], color="purple")
ax_rsi.axhline(70, ls="--", color="gray")
ax_rsi.axhline(30, ls="--", color="gray")
ax_rsi.set_ylabel("RSI(14)")
ax_rsi.grid(alpha=0.2)

# Plot MACD
ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue")
ax_macd.plot(df.index, df["Signal"], label="Signal", color="orange")
ax_macd.bar(df.index, df["Hist"], label="Hist", color="gray", alpha=0.5)
ax_macd.legend(loc="upper left", fontsize=8)
ax_macd.set_ylabel("MACD")
ax_macd.grid(alpha=0.2)

# Plot Volume Profile as horizontal bars on the right
# align bins to middle
bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
ax_volpro.barh(bin_centers, vol_profile, height=price_bins[1]-price_bins[0], color="gold", alpha=0.6)
ax_volpro.invert_xaxis()
ax_volpro.set_ylabel("Price")
ax_volpro.set_xlabel("Relative Volume")
ax_volpro.grid(alpha=0.2)

# Tight layout
fig.tight_layout()
st.pyplot(fig)

