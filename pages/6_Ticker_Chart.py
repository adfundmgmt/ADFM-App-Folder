import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

st.set_page_config(layout="wide", page_title="Ticker Technical Chart")
st.title("Technical Chart Explorer")

# Sidebar
st.sidebar.header("About This Tool")
st.sidebar.info(
    """
    This tool visualizes technical indicators for any publicly traded stock using Yahoo Finance data.
    It displays:
    - Price as bars, color-coded by up/down days
    - 20/50/100/200-day moving averages
    - Volume
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    You can customize the ticker, historical range, and data interval from the sidebar.
    """
)

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
period = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y", "max"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# Fetch data
df = yf.Ticker(ticker).history(period=period, interval=interval)
if df.empty:
    st.error("No data for this ticker. Check symbol or internet.")
    st.stop()

# Compute MAs
for w in (20, 50, 100, 200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI14"] = 100 - (100 / (1 + rs))

# MACD
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"] = df["MACD"] - df["Signal"]

# Bullish/bearish volume color
vol_colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]
price_colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]

# Plot
fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
    3, 1, figsize=(12, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05}
)

# ── Price as bars + MA
ax_price.set_title(f"{ticker} Price & Moving Averages", fontsize=16, weight="bold", pad=14)
bar_width = 0.8 if interval == "1d" else 1.0
ax_price.bar(df.index, df["Close"], color=price_colors, width=bar_width, label="Close")
for w, col in zip((20, 50, 100, 200), ("purple", "blue", "orange", "gray")):
    ax_price.plot(df.index, df[f"MA{w}"], label=f"MA{w}", color=col, lw=1)

# Volume bars
ax_vol = ax_price.twinx()
ax_vol.bar(df.index, df["Volume"], color=vol_colors, alpha=0.3, width=1)
ax_vol.set_ylabel("Volume", color="gray")
ax_vol.tick_params(axis="y", labelcolor="gray", labelsize=8)

ax_price.legend(loc="upper left", fontsize=8)
ax_price.grid(alpha=0.3)

# ── RSI
ax_rsi.set_title("Relative Strength Index (14-day)", fontsize=14, weight="bold", pad=10)
ax_rsi.plot(df.index, df["RSI14"], color="purple", lw=1.2)
ax_rsi.axhline(70, linestyle="--", color="gray", lw=0.8)
ax_rsi.axhline(30, linestyle="--", color="gray", lw=0.8)
ax_rsi.grid(alpha=0.3)

# ── MACD
ax_macd.set_title("MACD (12,26,9)", fontsize=14, weight="bold", pad=10)
ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue", lw=1.2)
ax_macd.plot(df.index, df["Signal"], label="Signal", color="orange", lw=1)
ax_macd.bar(df.index, df["Hist"], label="Hist", color="gray", alpha=0.5, width=1)
ax_macd.legend(loc="upper left", fontsize=8)
ax_macd.grid(alpha=0.3)

# Format X-axis
date_format = DateFormatter("%Y-%m")
ax_macd.xaxis.set_major_formatter(date_format)

fig.tight_layout()
st.pyplot(fig)
