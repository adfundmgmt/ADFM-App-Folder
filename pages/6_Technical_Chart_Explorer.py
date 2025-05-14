import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Sidebar / Data Fetch (same as before) ──────────────────────────────────
ticker   = st.sidebar.text_input("Ticker", "NVDA").upper()
period   = st.sidebar.selectbox("Period", ["1y","2y","3y","5y","max"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

buff = 250
if period != "max":
    days  = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period] + buff
    start = datetime.today() - timedelta(days=days)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)

df.index = pd.to_datetime(df.index).tz_localize(None)
if df.empty:
    st.error("No data—check symbol or connection."); st.stop()

# ── Indicators (MA only for this example) ───────────────────────────────────
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# trim buffer
if period!="max":
    window = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period]
    df     = df.loc[df.index >= df.index.max() - pd.Timedelta(days=window)]

# ── Build Plotly Figure ─────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7,0.3],
    vertical_spacing=0.02,
    specs=[[{"type":"candlestick"}],
           [{"type":"bar"}]]
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red"
    ),
    row=1, col=1
)

# Moving Averages
for w,color in zip((20,50,100,200),("purple","blue","orange","gray")):
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f"MA{w}"],
            mode="lines", line=dict(color=color, width=1),
            name=f"MA{w}"
        ),
        row=1, col=1
    )

# Volume
fig.add_trace(
    go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=[
            "green" if c>=o else "red"
            for c,o in zip(df["Close"], df["Open"])
        ],
        name="Volume"
    ),
    row=2, col=1
)

# Layout tweaks
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(
    height=800, width=1000,
    title=f"{ticker} — Interactive OHLC Chart",
    hovermode="x unified"
)

# ── Render ─────────────────────────────────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)
