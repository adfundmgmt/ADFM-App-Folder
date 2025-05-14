import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Ticker Technical Chart")
st.title("Technical Chart Explorer")

# --------------------------------------------------
# SIDEBAR: About & Settings
# --------------------------------------------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Visualize key technical indicators for any publicly traded stock using Yahoo Finance data.
        
        **Features:**
        - Price & 20/50/100/200-day moving averages
        - Volume bars color-coded by up/down days
        - RSI (14-day)
        - MACD (12,26,9)
        
        Customize the ticker symbol, data range, and interval below.
        """
    )
    st.subheader("Settings")
    ticker = st.text_input("Ticker", "NVDA").upper()
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y", "max"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# --------------------------------------------------
# DATA FETCH WITH BUFFER FOR MAs
# --------------------------------------------------
# Map user period to days
period_map = {"1y":365, "2y":365*2, "3y":365*3, "5y":365*5}
# Buffer days to cover max MA window
buffer_days = 250
if period != "max":
    days = period_map.get(period, 365) + buffer_days
    start_dt = datetime.today() - timedelta(days=days)
    df = yf.Ticker(ticker).history(start=start_dt, interval=interval)
else:
    df = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data for this ticker. Check symbol or internet.")
    st.stop()

# Trim to user window for plotting after computing MAs
# Ensure index is datetime
try:
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
except Exception:
    df.index = pd.to_datetime(df.index)

display_days = period_map.get(period, None)
if period != "max" and display_days:
    cutoff = datetime.today() - timedelta(days=display_days)
    cutoff = pd.to_datetime(cutoff)
    df_plot = df.loc[df.index >= cutoff]
else:
    df_plot = df.copy()

# --------------------------------------------------
# INDICATORS
# --------------------------------------------------
# Moving Averages
for w in (20, 50, 100, 200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI14"] = 100 - (100 / (1 + rs))

# MACD(12,26,9)
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"] = df["MACD"] - df["Signal"]

# Volume colors
vol_colors = ["green" if c >= o else "red" for c, o in zip(df_plot["Close"], df_plot["Open"])]

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
# Create 4-row subplot: Price, Volume, RSI, MACD
grid_opts = {"height_ratios": [4, 1, 1, 1], "hspace": 0.2}
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw=grid_opts)
ax_price, ax_vol, ax_rsi, ax_macd = axes

# Price & MAs on trimmed data
ax_price.set_title(f"{ticker} Price & Moving Averages", fontsize=18, weight="bold", pad=20)
ax_price.plot(df_plot.index, df_plot["Close"], label="Close", color="black", linewidth=1.5)
for w, col in zip((20, 50, 100, 200), ("purple", "blue", "orange", "gray")):
    ax_price.plot(df_plot.index, df_plot[f"MA{w}"], label=f"MA{w}", color=col, linewidth=1)
ax_price.legend(loc="upper left", fontsize=10)
ax_price.grid(alpha=0.3)

# Volume subplot
df_vol = df_plot
ax_vol.bar(df_vol.index, df_vol["Volume"], color=vol_colors, alpha=0.3, width=1)
ax_vol.set_ylabel("Volume", color="gray", fontsize=10)
ax_vol.tick_params(axis="y", labelcolor="gray", labelsize=8)
ax_vol.grid(alpha=0.3)

# RSI subplot
df_rsi = df_plot
ax_rsi.set_title("Relative Strength Index (14-day)", fontsize=14, weight="bold", pad=12)
ax_rsi.plot(df_rsi.index, df_rsi["RSI14"], color="purple", linewidth=1.5)
ax_rsi.axhline(70, linestyle="--", color="gray", linewidth=1)
ax_rsi.axhline(30, linestyle="--", color="gray", linewidth=1)
ax_rsi.set_ylabel("RSI", fontsize=10)
ax_rsi.grid(alpha=0.3)

# MACD subplot
df_macd = df_plot
ax_macd.set_title("MACD (12,26,9)", fontsize=14, weight="bold", pad=12)
ax_macd.plot(df_macd.index, df_macd["MACD"], label="MACD", color="blue", linewidth=1.5)
ax_macd.plot(df_macd.index, df_macd["Signal"], label="Signal", color="orange", linewidth=1.5)
ax_macd.bar(df_macd.index, df_macd["Hist"], label="Hist", color="gray", alpha=0.5, width=1)
ax_macd.set_ylabel("MACD", fontsize=10)
ax_macd.legend(loc="upper left", fontsize=10)
ax_macd.grid(alpha=0.3)

# X-axis formatting
date_format = DateFormatter("%Y-%m")
ax_macd.xaxis.set_major_formatter(date_format)

fig.tight_layout()
st.pyplot(fig)
