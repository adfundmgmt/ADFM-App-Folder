import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    st.error("No data for this ticker.")
    st.stop()

# Moving averages
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# RSI(14)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI14"] = 100 - (100 / (1 + rs))

# MACD(12,26,9)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"] = df["MACD"] - df["Signal"]

# Volume profile
bins = pd.interval_range(df["Low"].min(), df["High"].max(), periods=30)
profile = pd.cut(df["Close"], bins).groupby(bins).apply(lambda rng: df.loc[rng.index, "Volume"].sum())
profile = profile / profile.max()

# Build subplots: rows 4 (candles+volume), 2 (RSI), 2 (MACD), cols 2 with volume‑profile on right
fig = make_subplots(
    rows=4, cols=2,
    column_widths=[0.85,0.15],
    row_heights=[0.5,0.15,0.15,0.2],
    specs=[
        [{"type":"candlestick","rowspan":2}, {"type":"bar"}],
        [           None               ,               None],
        [{"type":"scatter"},          {"type":"scatter"}],
        [{"type":"bar"},              {"type":"bar"}]
    ],
    shared_xaxes=True,
    vertical_spacing=0.02,
    horizontal_spacing=0.02
)

# Candlestick + MAs
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"), row=1, col=1)
for w, color in zip((20,50,100,200), ("purple","blue","orange","black")):
    fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{w}"], mode="lines", 
                             line=dict(color=color,width=1), name=f"MA{w}"), row=1, col=1)

# Volume bars
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="grey", name="Volume"), row=2, col=1)

# Volume profile
fig.add_trace(go.Barh(y=profile.index.mid, x=profile.values, marker_color="gold", name="Vol Profile"), row=1, col=2)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", line_color="purple", name="RSI(14)"), row=3, col=1)
fig.add_hline(70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(30, line_dash="dash", line_color="gray", row=3, col=1)

# MACD & Signal
fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", line_color="blue", name="MACD"), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode="lines", line_color="orange", name="Signal"), row=4, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Hist"], marker_color="lightgray", name="Hist"), row=4, col=1)

# Layout tweaks
fig.update_layout(
    showlegend=False,
    xaxis_rangeslider_visible=False,
    height=900,
    margin=dict(l=50,r=20,t=50,b=50)
)

st.plotly_chart(fig, use_container_width=True)
