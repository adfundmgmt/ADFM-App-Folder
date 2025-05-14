import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    • OHLC bars (with open/close ticks)  
    • 20/50/100/200‑day moving averages  
    • Volume bars color‑coded by up/down days  
    • RSI (14) & MACD panels  
    • Volume‑profile on right  
    """
)
ticker   = st.sidebar.text_input("Ticker", "NVDA").upper()
period   = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y","3y","5y","max"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Download & prep ─────────────────────────────────────────────────────────
buff       = 250
period_map = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"3y":1095,"5y":1825}
if period!="max":
    start = datetime.today() - timedelta(days=period_map[period]+buff)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)
if df.empty:
    st.error("No data"); st.stop()

df.index = pd.to_datetime(df.index).tz_localize(None)
if period!="max":
    cutoff = df.index.max() - pd.Timedelta(days=period_map[period])
    df     = df.loc[df.index>=cutoff]
df = df[df.index.weekday<5]
df["DateStr"] = df.index.strftime("%Y-%m-%d")

# ── Indicators ─────────────────────────────────────────────────────────────
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

delta       = df["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain/loss
df["RSI14"] = 100 - (100/(1+rs))

df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

# ── Volume‐profile ──────────────────────────────────────────────────────────
price_min, price_max = df["Low"].min(), df["High"].max()
bins                = np.linspace(price_min, price_max, 30)
df["Bin"]           = pd.cut(df["Close"], bins)
profile             = df.groupby("Bin")["Volume"].sum()
centers             = [interval.mid for interval in profile.index]

# ── Build figure ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.5,0.1,0.15,0.25],
    vertical_spacing=0.02,
    specs=[
        [{"type":"ohlc"}],
        [{"type":"bar"}],
        [{"type":"scatter"}],
        [{"type":"scatter"}]
    ],
    column_widths=[0.85]
)

# 1) OHLC bars + MAs
fig.add_trace(
    go.Ohlc(
        x=df["DateStr"],
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        width=0.6,
        increasing=dict(line=dict(color="green", width=1)),
        decreasing=dict(line=dict(color="red",   width=1)),
        name="Price"
    ),
    row=1, col=1
)
for w,col in zip((20,50,100,200),("purple","blue","orange","gray")):
    fig.add_trace(
        go.Scatter(
            x=df["DateStr"], y=df[f"MA{w}"],
            mode="lines", line=dict(color=col, width=1),
            name=f"MA{w}"
        ),
        row=1, col=1
    )

# 2) Volume
fig.add_trace(
    go.Bar(
        x=df["DateStr"], y=df["Volume"],
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
        x=df["DateStr"], y=df["RSI14"],
        mode="lines", line=dict(color="purple", width=1),
        name="RSI"
    ),
    row=3, col=1
)
fig.update_yaxes(title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD
fig.add_trace(
    go.Bar(x=df["DateStr"], y=df["Hist"], marker_color="gray", name="Hist"),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=df["DateStr"], y=df["MACD"],
        mode="lines", line=dict(color="blue", width=1.5),
        name="MACD"
    ),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=df["DateStr"], y=df["Signal"],
        mode="lines", line=dict(color="orange", width=1),
        name="Signal"
    ),
    row=4, col=1
)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# 5) Volume profile on right
fig.add_trace(
    go.Bar(
        x=profile.values,
        y=centers,
        orientation="h",
        marker_color="gold",
        showlegend=False,
        xaxis="x2", yaxis="y1"
    )
)

# ── Layout tweaks ───────────────────────────────────────────────────────────
fig.update_layout(
    title=f"{ticker} — OHLC with Volume Profile",
    hovermode="y",
    xaxis1=dict(domain=[0,0.85], type="category"),
    xaxis2=dict(domain=[0.85,1], showticklabels=False),
    yaxis1=dict(domain=[0.5,1], title="Price"),
    height=900, width=1100,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right")
)
# sync all x‑axes
fig.update_xaxes(matches="x")

st.plotly_chart(fig, use_container_width=True)
