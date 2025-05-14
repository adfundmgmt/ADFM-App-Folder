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

Hover anywhere to inspect values—zoom/pan enabled.
""")
ticker   = st.sidebar.text_input("Ticker", "NVDA").upper()
period   = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Fetch with extra buffer for long MAs ────────────────────────────────────
base_days   = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"3y":1095,"5y":1825}
buffer_days = max(base_days.get(period,365), 500)

if period != "max":
    start = datetime.today() - timedelta(days=base_days[period] + buffer_days)
    df    = yf.Ticker(ticker).history(start=start, interval=interval)
else:
    df    = yf.Ticker(ticker).history(period="max", interval=interval)

if df.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# ── Prep: strip tz, trim to window, drop weekends ──────────────────────────
df.index = pd.to_datetime(df.index).tz_localize(None)
if period != "max":
    cutoff = df.index.max() - pd.Timedelta(days=base_days[period])
    df     = df.loc[df.index >= cutoff]
df = df[df.index.weekday < 5]

# categorical date label for perfect alignment
df["DateStr"] = df.index.strftime("%Y-%m-%d")

# ── Compute Indicators ───────────────────────────────────────────────────────
for w in (20,50,100,200):
    df[f"MA{w}"] = df["Close"].rolling(w).mean()

delta       = df["Close"].diff()
gain        = delta.clip(lower=0).rolling(14).mean()
loss        = -delta.clip(upper=0).rolling(14).mean()
rs          = gain / loss
df["RSI14"] = 100 - (100/(1+rs))

df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]   = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Hist"]   = df["MACD"] - df["Signal"]

available_mas = [w for w in (20,50,100,200) if len(df) >= w]

# ── Build Figure ────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.55,0.15,0.15,0.15],
    vertical_spacing=0.05,
    specs=[
        [{"type":"candlestick"}],
        [{"type":"bar"}],
        [{"type":"scatter"}],
        [{"type":"scatter"}],
    ]
)

# 1) Price + MAs
fig.add_trace(go.Candlestick(
    x=df["DateStr"],
    open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="green",
    decreasing_line_color="red",
    name="Price"
), row=1, col=1)
for w,color in zip(available_mas, ("purple","blue","orange","gray")):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1),
        name=f"MA{w}"
    ), row=1, col=1)

# 2) Volume (with legend entry)
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Volume"], width=1,
    marker_color=["green" if c>=o else "red" for c,o in zip(df["Close"],df["Open"])],
    name="Volume", showlegend=True
), row=2, col=1)

# 3) RSI (0–100, lines at 80/20)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["RSI14"],
    mode="lines", line=dict(color="purple", width=1),
    name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0,100], title_text="RSI", row=3, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Hist
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Hist"], marker_color="gray", name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["MACD"],
    mode="lines", line=dict(color="blue", width=1.5),
    name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["Signal"],
    mode="lines", line=dict(color="orange", width=1),
    name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# ── Monthly ticks on categorical axis ───────────────────────────────────────
month_starts = df.index.to_series().groupby(df.index.to_period("M")).first()
tickvals     = month_starts.dt.strftime("%Y-%m-%d").tolist()
ticktext     = month_starts.dt.strftime("%b-%y").tolist()

fig.update_xaxes(
    type="category",
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickangle=-45
)

# ── Layout tweaks ────────────────────────────────────────────────────────────
fig.update_layout(
    template="plotly_white",
    height=900, width=1000,
    title=dict(text=f"{ticker} — OHLC + RSI & MACD", x=0.5, xanchor="center", font_size=20),
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
)

# subtle grid styling
fig.update_xaxes(showgrid=True, gridcolor="LightGray", gridwidth=0.5, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="LightGray", gridwidth=0.5)

st.plotly_chart(fig, use_container_width=True)
