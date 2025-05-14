import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

# ── Page config
st.set_page_config(layout="wide", page_title="Technical Chart Explorer")
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
buff = 250
if period != "max":
    days = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period] + buff
    start = datetime.today() - timedelta(days=days)
    df    = yf.download(ticker, start=start, interval=interval)
else:
    df    = yf.download(ticker, period="max", interval=interval)

if df.empty:
    st.error("No data returned. Check your ticker/connection.")
    st.stop()

df.index = pd.to_datetime(df.index).tz_localize(None)

# ── Indicators
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = -delta.clip(upper=0).rolling(14).mean()
rs    = gain/loss
df["RSI14"] = 100 - (100/(1+rs))

df["EMA12"]   = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]   = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]    = df["EMA12"] - df["EMA26"]
df["Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]    = df["MACD"] - df["Signal"]

# Trim buffer
if period!="max":
    window = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period]
    df     = df.loc[df.index>=df.index.max()-pd.Timedelta(days=window)]

# ── mplfinance add‑plots
apds = [
    mpf.make_addplot(df["MA20"],  color="purple"),
    mpf.make_addplot(df["MA50"],  color="blue"),
    mpf.make_addplot(df["MA100"], color="orange"),
    mpf.make_addplot(df["MA200"], color="gray"),
    mpf.make_addplot(df["RSI14"], panel=1, color="purple", ylabel="RSI (14)"),
    mpf.make_addplot(df["MACD"],  panel=2, color="blue", ylabel="MACD"),
    mpf.make_addplot(df["Signal"],panel=2, color="orange"),
    mpf.make_addplot(df["Hist"],   panel=2, type="bar", color="dimgray", alpha=0.5)
]

# ── Plot (fixed!)
mpf_style = mpf.make_mpf_style(base_mpf_style="charles", gridstyle=":")
fig, axes = mpf.plot(
    df,
    type="candle",
    mav=(20, 50, 100, 200),
    volume=True,
    addplot=apds,
    panel_ratios=(6, 2, 2),
    figsize=(14, 10),
    style=mpf_style,
    returnfig=True
)

st.pyplot(fig)
st.caption("© 2025 AD Fund Management LP")
