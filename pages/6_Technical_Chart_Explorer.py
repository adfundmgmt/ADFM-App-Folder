import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Sidebar: About + Settings ────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Visualize key technical indicators for any stock using Yahoo Finance data.

    **Features:**
    - Interactive OHLC candlesticks
    - 20/50/100/200‑day moving averages
    - Volume bars color‑coded by up/down days
    - RSI (14‑day) panel
    - MACD (12,26,9) panel

    Hover anywhere to inspect OHLC, MA values, RSI and MACD.
    """
)
st.sidebar.subheader("Settings")
ticker   = st.sidebar.text_input("Ticker",   "NVDA").upper()
period   = st.sidebar.selectbox(
    "Period",
    ["1mo","3mo","6mo","1y","2y","3y","5y","max"],
    index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Fetch with buffer for MAs ────────────────────────────────────────────────
buffer_days = 250
period_map = {
    "1mo": 30, "3mo": 90,  "6mo": 180,
    "1y": 365, "2y": 365*2, "3y": 365*3, "5y": 365*5
}

if period != "max":
    days  = period_map[period] + buffer_days
    start = datetime.today() - timedelta(days=days)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# strip tz
df.index = pd.to_datetime(df.index).tz_localize(None)

# ── Compute Indicators ───────────────────────────────────────────────────────
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

delta       = df["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain / loss
df["RSI14"] = 100 - (100 / (1+rs))

df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

# trim back to the exact window
if period != "max":
    window = period_map[period]
    cutoff = df.index.max() - pd.Timedelta(days=window)
    df     = df.loc[df.index >= cutoff]

# drop weekends if any slipped through
df = df[df.index.weekday < 5]

# ── Make a string‐date column to use as a category x‑axis ────────────────────
df["DateStr"] = df.index.strftime("%Y-%m-%d")
x            = df["DateStr"]

# ── Select which MAs fit in your window ──────────────────────────────────────
n = len(df)
available_mas = [w for w in (20,50,100,200) if n >= w]

# ── Build Plotly Figure (4 rows) ────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    # increased RSI & MACD height, slightly less volume
    row_heights=[0.5, 0.08, 0.22, 0.20],
    vertical_spacing=0.03,
    specs=[
        [{"type":"candlestick"}],
        [{"type":"bar"}],
        [{"type":"scatter"}],
        [{"type":"scatter"}]
    ]
)

# 1) Candlestick + MAs
fig.add_trace(
    go.Candlestick(
        x=x,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=True
    ),
    row=1, col=1
)
for w,color in zip(available_mas,("purple","blue","orange","gray")):
    fig.add_trace(
        go.Scatter(
            x=x, y=df[f"MA{w}"],
            mode="lines",
            line=dict(color=color, width=1),
            name=f"MA{w}"
        ),
        row=1, col=1
    )

# 2) Volume
fig.add_trace(
    go.Bar(
        x=x,
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
        x=x, y=df["RSI14"],
        mode="lines", line=dict(color="purple", width=1),
        name="RSI (14)"
    ),
    row=3, col=1
)
fig.update_yaxes(title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Signal + Hist
fig.add_trace(
    go.Bar(
        x=x, y=df["Hist"],
        marker_color="gray", name="MACD Hist"
    ), row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=x, y=df["MACD"],
        mode="lines", line=dict(color="blue", width=1.5),
        name="MACD"
    ), row=4, col=1
)
fig.add_trace(
    go.Scatter(
        x=x, y=df["Signal"],
        mode="lines", line=dict(color="orange", width=1),
        name="Signal"
    ), row=4, col=1
)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# ── Layout tweaks ────────────────────────────────────────────────────────────
fig.update_layout(
    height=900, width=1000,
    title=f"{ticker} — Interactive OHLC + RSI & MACD",
    hovermode="x unified",
    xaxis=dict(type="category"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
