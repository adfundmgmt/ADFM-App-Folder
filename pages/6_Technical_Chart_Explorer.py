import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, date2num
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime, timedelta

# ── Page config
st.set_page_config(layout="wide", page_title="Ticker Technical Chart")
st.title("Technical Chart Explorer")

# ── Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Visualize key technical indicators for any stock using Yahoo Finance data.

        **Features:**
        - OHLC candlesticks + 20/50/100/200‑day moving averages
        - Volume bars color‑coded by up/down days
        - RSI (14‑day)
        - MACD (12,26,9)

        Customize ticker, date range, and interval below.
        """
    )
    st.subheader("Settings")
    ticker   = st.text_input("Ticker",   "NVDA").upper()
    period   = st.selectbox("Period",   ["1y","2y","3y","5y","max"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk","1mo"],     index=0)

# ── Data fetch (with buffer for MAs)
period_map  = {"1y":365,"2y":730,"3y":1095,"5y":1825}
buffer_days = 250
if period != "max":
    days     = period_map[period] + buffer_days
    start_dt = datetime.today() - timedelta(days=days)
    df       = yf.Ticker(ticker).history(start=start_dt, interval=interval)
else:
    df       = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data returned. Check symbol or connection.")
    st.stop()

# drop any tz info
df.index = pd.to_datetime(df.index).tz_localize(None)

# ── Indicators
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

delta       = df["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain / loss
df["RSI14"] = 100 - (100 / (1 + rs))

df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

# trim to exact window
if period != "max":
    cutoff  = df.index.max() - pd.Timedelta(days=period_map[period])
    df      = df.loc[df.index >= cutoff]

# ── Prepare OHLC for candlestick_ohlc
df_plot = df.copy()
ohlc     = df_plot[["Open","High","Low","Close"]].reset_index()
ohlc["DateNum"] = ohlc["Date"].apply(date2num)
ohlc_vals      = ohlc[["DateNum","Open","High","Low","Close"]].values

# color map for volume
vol_colors = [
    "green" if row.Close >= row.Open else "red"
    for _, row in df_plot.iterrows()
]

# ── Plot layout
grid_opts = {"height_ratios":[5,1,1,1],"hspace":0.25}
fig, (ax_price, ax_vol, ax_rsi, ax_macd) = plt.subplots(
    4,1, figsize=(14,10), sharex=True, gridspec_kw=grid_opts
)

# ── Candlesticks + MAs
ax_price.set_title(f"{ticker} Price & Moving Averages", fontsize=18, weight="bold", pad=20)
candlestick_ohlc(
    ax_price,
    ohlc_vals,
    width=0.6,
    colorup="green",
    colordown="red",
    alpha=0.8
)
for w,col in zip((20,50,100,200),("purple","blue","orange","gray")):
    ax_price.plot(df_plot.index, df_plot[f"MA{w}"], label=f"MA{w}", color=col, linewidth=1)
ax_price.legend(loc="upper left", fontsize=10)
ax_price.grid(alpha=0.3)
ax_price.xaxis_date()

# ── Volume
ax_vol.bar(df_plot.index, df_plot["Volume"], color=vol_colors, width=1, alpha=0.3)
ax_vol.set_ylabel("Volume", color="gray", fontsize=10)
ax_vol.tick_params(axis="y", labelcolor="gray", labelsize=8)
ax_vol.grid(alpha=0.3)

# ── RSI
ax_rsi.set_title("Relative Strength Index (14-day)", fontsize=14, weight="bold", pad=12)
ax_rsi.plot(df_plot.index, df_plot["RSI14"], color="purple", linewidth=1.5)
ax_rsi.axhline(70, linestyle="--", color="gray", linewidth=1)
ax_rsi.axhline(30, linestyle="--", color="gray", linewidth=1)
ax_rsi.set_ylabel("RSI", fontsize=10)
ax_rsi.grid(alpha=0.3)

# ── MACD
ax_macd.set_title("MACD (12,26,9)", fontsize=14, weight="bold", pad=12)
ax_macd.plot(df_plot.index, df_plot["MACD"],   label="MACD",   color="blue",   linewidth=1.5)
ax_macd.plot(df_plot.index, df_plot["Signal"], label="Signal", color="orange", linewidth=1.5)
ax_macd.bar(df_plot.index, df_plot["Hist"],     label="Hist",    color="gray",   alpha=0.5, width=1)
ax_macd.set_ylabel("MACD", fontsize=10)
ax_macd.legend(loc="upper left", fontsize=10)
ax_macd.grid(alpha=0.3)

# ── X‑axis formatting
ax_macd.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

fig.tight_layout()
st.pyplot(fig)
st.caption("© 2025 AD Fund Management LP")
