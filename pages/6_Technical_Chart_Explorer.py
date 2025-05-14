import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Sidebar / Inputs ────────────────────────────────────────────────────────
st.sidebar.header("Settings")
ticker   = st.sidebar.text_input("Ticker",    "NVDA").upper()
period   = st.sidebar.selectbox("Period",    ["1y","2y","3y","5y","max"], index=1)
interval = st.sidebar.selectbox("Interval",  ["1d","1wk","1mo"],      index=0)

# ── Fetch data (with buffer for MAs) ────────────────────────────────────────
buffer_days = 250
if period != "max":
    days  = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period] + buffer_days
    start = datetime.today() - timedelta(days=days)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data returned. Check ticker/connection.")
    st.stop()

df.index = pd.to_datetime(df.index).tz_localize(None)

# ── Indicators ──────────────────────────────────────────────────────────────
# Moving Averages
for w,color in zip((20,50,100,200),("purple","blue","orange","gray")):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

# RSI(14)
delta       = df["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain / loss
df["RSI14"] = 100 - (100/(1+rs))

# MACD(12,26,9)
df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

# Trim off buffer so main window = exact period
if period != "max":
    window = {"1y":365,"2y":730,"3y":1095,"5y":1825}[period]
    df     = df.loc[df.index >= df.index.max() - pd.Timedelta(days=window)]

# ── Build Plotly figure with 4 rows ─────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.5, 0.1, 0.15, 0.25],
    vertical_spacing=0.02,
    specs=[
        [{"type":"candlestick"}],
        [{"type":"bar"}],
        [{"type":"scatter"}],
        [{"type":"scatter"}],
    ]
)

# 1) Candlestick + MAs
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
for w,color in zip((20,50,100,200),("purple","blue","orange","gray")):
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f"MA{w}"],
            mode="lines",
            line=dict(color=color, width=1),
            name=f"MA{w}"
        ),
        row=1, col=1
    )

# 2) Volume
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=[
            "green" if c>=o else "red"
            for c,o in zip(df["Close"], df["Open"])
        ],
        name="Volume"
    ),
    row=2, col=1
)

# 3) RSI
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["RSI14"],
        mode="lines", line=dict(color="purple", width=1),
        name="RSI (14)"
    ),
    row=3, col=1
)
# RSI overbought/oversold lines
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Signal + Histogram
fig.add_trace(
    go.Bar(
        x=df.index, y=df["Hist"],
        marker_color="gray", name="MACD Hist"
    ),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["MACD"],
        mode="lines", line=dict(color="blue", width=1.5),
        name="MACD"
    ),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Signal"],
        mode="lines", line=dict(color="orange", width=1),
        name="Signal"
    ),
    row=4, col=1
)

# ── Layout tweaks ────────────────────────────────────────────────────────────
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(
    height=900, width=1000,
    title=f"{ticker} — Interactive OHLC with RSI & MACD",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# ── Render ───────────────────────────────────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)
