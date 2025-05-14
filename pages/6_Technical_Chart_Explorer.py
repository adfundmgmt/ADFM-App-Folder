import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sidebar
st.sidebar.header("Settings")
ticker   = st.sidebar.text_input("Ticker", "NVDA").upper()
period   = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","3y","5y","max"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# Fetch
buff = 250
pm   = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"3y":1095,"5y":1825}
if period!="max":
    start = datetime.today() - timedelta(days=pm[period]+buff)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data"); st.stop()

# Prep
df.index = pd.to_datetime(df.index).tz_localize(None)
if period!="max":
    cutoff = df.index.max() - pd.Timedelta(days=pm[period])
    df     = df.loc[df.index>=cutoff]
df = df[df.index.weekday<5]
df["DateStr"] = df.index.strftime("%Y-%m-%d")

# Indicators
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = -delta.clip(upper=0).rolling(14).mean()
rs    = gain/loss
df["RSI14"] = 100 - (100/(1+rs))
df["EMA12"]  = df["Close"].ewm(span=12).mean()
df["EMA26"]  = df["Close"].ewm(span=26).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

mas = [w for w in (20,50,100,200) if len(df)>=w]

# Figure
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.6,0.1,0.18,0.12],
    vertical_spacing=0.03,
    specs=[
        [{"type":"candlestick"}],
        [{"type":"bar"}],       # <-- only volume here
        [{"type":"scatter"}],
        [{"type":"scatter"}]
    ]
)

# Row 1: Price + MAs
fig.add_trace(go.Candlestick(
    x=df["DateStr"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="green", decreasing_line_color="red",
    name="Price"
), row=1, col=1)
for w,c in zip(mas,("purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"],
        line=dict(color=c, width=1), name=f"MA{w}"
    ), row=1, col=1)

# Row 2: Volume only
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Volume"],
    marker_color=[ "green" if c>=o else "red"
                   for c,o in zip(df["Close"], df["Open"])],
    name="Volume"
), row=2, col=1)

# Row 3: RSI
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["RSI14"],
    line=dict(color="purple", width=1), name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# Row 4: MACD + Hist
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Hist"], marker_color="gray",
    name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["MACD"],
    line=dict(color="blue", width=1.5), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["Signal"],
    line=dict(color="orange", width=1), name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Layout
fig.update_layout(
    height=900, width=1000,
    title=f"{ticker} — OHLC + RSI & MACD",
    hovermode="x unified",
    xaxis=dict(type="category"),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
)

st.plotly_chart(fig, use_container_width=True)
