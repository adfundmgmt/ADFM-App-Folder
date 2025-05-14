import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num
from mplfinance.original_flavor import candlestick_ohlc
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
        - OHLC candlesticks + 20/50/100/200-day moving averages
        - Volume bars color-coded by up/down days
        - RSI (14-day)
        - MACD (12,26,9)
        
        Customize the ticker symbol, data range, and interval below.
        """
    )
    st.subheader("Settings")
    ticker = st.text_input("Ticker", "NVDA").upper()
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y", "max"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# --------------------------------------------------
# DATA FETCH WITH BUFFER FOR MAs
# --------------------------------------------------
period_map = {"1y": 365, "2y": 365*2, "3y": 365*3, "5y": 365*5}
buffer_days = 250
if period != "max":
    days = period_map[period] + buffer_days
    start_dt = datetime.today() - timedelta(days=days)
    df = yf.Ticker(ticker).history(start=start_dt, interval=interval)
else:
    df = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data for this ticker. Check symbol or internet.")
    st.stop()

# Normalize index
df.index = pd.to_datetime(df.index).tz_localize(None)

# --------------------------------------------------
# INDICATOR CALCULATIONS
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

# --------------------------------------------------
# TRIM TO DISPLAY WINDOW
