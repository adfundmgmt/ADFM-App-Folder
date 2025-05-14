import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
Visualize key technical indicators for any stock using Yahoo Finance data.

**Features:**
- Interactive OHLC candlesticks  
- 20/50/100/200‑day moving averages (full length)  
- Volume bars color‑coded by up/down days  
- RSI (14‑day) panel (0–100 scale)  
- MACD (12,26,9) panel  
""")
ticker   = st.sidebar.text_input("Ticker", "NVDA").upper()
period   = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Fetch with buffer for full-length MAs ─────────────────────────────────
base_days   = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"3y":1095,"5y":1825}
buffer_days = max(base_days.get(period,365), 500)

if period != "max":
    start   = datetime.today() - timedelta(days=base_days[period] + buffer_days)
    df_full = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df_full = yf.Ticker(ticker).history(period="max", interval=interval)

if df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# ── PREP & INDICATORS on full history ─────────────────────────────────────
df_full.index = pd.to_datetime(df_full.index).tz_localize(None)
df_full = df_full[df_full.index.weekday < 5]  # drop weekends

# rolling MAs
for w in (20,50,100,200):
    df_full[f"MA{w}"] = df_full["Close"].rolling(w).mean()

# RSI(14)
delta       = df_full["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain / loss
df_full["RSI14"] = 100 - (100/(1+rs))

# MACD(12,26,9)
df_full["EMA12"]  = df_full["Close"].ewm(span=12, adjust=False).mean()
df_full["EMA26"]  = df_full["Close"].ewm(span=26, adjust=False).mean()
df_full["MACD"]   = df_full["EMA12"] - df_full["EMA26"]
df_full["Signal"] = df_full["MACD"].ewm(span=9, adjust=False).mean()
df_full["Hist"]   = df_full["MACD"] - df_full["Signal"]

# ── Trim to display window ─────────────────────────────────────────────────
if period != "max":
    cutoff = df_full.index.max() - pd.Timedelta(days=base_days[period])
    df     = df_full.loc[df_full.index >= cutoff]
else:
    df = df_full.copy()

df["DateStr"] = df.index.strftime("%Y-%m-%d")
available_mas = [w for w in (20,50,100,200) if len(df) >= w]

# ── Build Figure ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.55,0.15,0.15,0.15],
    vertical_spacing=0.04,
    specs=[
        [{"type":"candlestick"}],  # 1 Price
        [{"type":"bar"}],          # 2 Volume
        [{"type":"scatter"}],      # 3 RSI
        [{"type":"scatter"}],      # 4 MACD
    ]
)

# 1) Price + MAs (show legend + label)
fig.add_trace(go.Candlestick(
    x=df["DateStr"],
    open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="green", decreasing_line_color="red",
    name="Price", showlegend=True
), row=1, col=1)
for w,color in zip(available_mas,("purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1),
        name=f"MA{w}", showlegend=True
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# 2) Volume
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Volume"], width=1,
    marker_color=["green" if c>=o else "red" for c,o in zip(df["Close"], df["Open"])],
    name="Volume", showlegend=False
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 3) RSI
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["RSI14"],
    mode="lines", line=dict(color="purple", width=1),
    name="RSI (14)", showlegend=False
), row=3, col=1)
fig.update_yaxes(
    range=[0,100],
    title_text="RSI",
    row=3, col=1,
    title_standoff=15
)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Hist
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Hist"], marker_color="gray",
    name="MACD Hist", showlegend=False
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["MACD"],
    mode="lines", line=dict(color="blue", width=1.5),
    name="MACD", showlegend=False
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["Signal"],
    mode="lines", line=dict(color="orange", width=1),
    name="Signal", showlegend=False
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# ── Monthly ticks ───────────────────────────────────────────────────────────
month_starts = df.index.to_series().groupby(df.index.to_period("M")).first()
tickvals     = month_starts.dt.strftime("%Y-%m-%d").tolist()
ticktext     = month_starts.dt.strftime("%b-%y").tolist()
fig.update_xaxes(
    type="category", tickmode="array",
    tickvals=tickvals, ticktext=ticktext,
    tickangle=-45
)

# ── Layout tweaks ────────────────────────────────────────────────────────────
fig.update_layout(
    height=900, width=1000,
    title=dict(text=f"{ticker} — OHLC + RSI & MACD", x=0.5),
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(rangeslider_visible=False),
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
)

st.plotly_chart(fig, use_container_width=True)
