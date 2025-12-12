import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600
CAP_MAX_ROWS = 200_000
BB_WINDOW = 20
BB_MULT = 2.0
RSI_PERIOD = 14

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Quickly inspect price action and key technical indicators for any stock, ETF, or index.

    • Interactive OHLC candlesticks using Yahoo Finance data  
    • 8 / 20 / 50 / 100 / 200 day moving averages with full history continuity  
    • Volume bars colored by bar direction  
    • RSI (14) using Wilder smoothing  
    • MACD (12, 26, 9) with histogram  
    • Optional Bollinger Bands (20, 2.0)  

    Indicators are computed on full history with warmup to avoid edge artifacts.
    """,
    unsafe_allow_html=True,
)

ticker = st.sidebar.text_input("Ticker", "^GSPC").upper()
period = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def start_date_from_period(p: str):
    today = pd.Timestamp.today().normalize()
    if p == "max":
        return None
    if p.endswith("mo"):
        return today - pd.DateOffset(months=int(p[:-2]))
    if p.endswith("y"):
        return today - pd.DateOffset(years=int(p[:-1]))
    return None

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_full_history(tkr: str, intrvl: str) -> pd.DataFrame:
    df = yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=False)
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    return df

# ── Fetch full history ────────────────────────────────────────────────────
try:
    df_full = get_full_history(ticker, interval)
except Exception as e:
    st.error(str(e))
    st.stop()

if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

if len(df_full) > CAP_MAX_ROWS:
    st.warning("Dataset very large. Truncating to most recent data for performance.")
    df_full = df_full.tail(CAP_MAX_ROWS)

# ── Date window with warmup ───────────────────────────────────────────────
start_dt = start_date_from_period(period)
BUFFER_YEARS = 50

if start_dt is not None:
    warmup_start = start_dt - pd.DateOffset(years=BUFFER_YEARS)
    df_ind = df_full[df_full.index >= warmup_start].copy()
else:
    df_ind = df_full.copy()

# ── Indicators ───────────────────────────────────────────────────────────
for w in (8, 20, 50, 100, 200):
    df_ind[f"MA{w}"] = df_ind["Close"].rolling(w).mean()

# RSI (Wilder)
delta = df_ind["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
rs = avg_gain / avg_loss
df_ind["RSI14"] = 100 - (100 / (1 + rs))

# MACD
df_ind["EMA12"] = df_ind["Close"].ewm(span=12, adjust=False).mean()
df_ind["EMA26"] = df_ind["Close"].ewm(span=26, adjust=False).mean()
df_ind["MACD"] = df_ind["EMA12"] - df_ind["EMA26"]
df_ind["Signal"] = df_ind["MACD"].ewm(span=9, adjust=False).mean()
df_ind["Hist"] = df_ind["MACD"] - df_ind["Signal"]

# Bollinger Bands
df_ind["BB_MA"] = df_ind["Close"].rolling(BB_WINDOW).mean()
df_ind["BB_STD"] = df_ind["Close"].rolling(BB_WINDOW).std()
df_ind["BB_UPPER"] = df_ind["BB_MA"] + BB_MULT * df_ind["BB_STD"]
df_ind["BB_LOWER"] = df_ind["BB_MA"] - BB_MULT * df_ind["BB_STD"]

# ── Final slice ──────────────────────────────────────────────────────────
if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()
else:
    df_display = df_ind.copy()

df_display = df_display.dropna(subset=["Close", "RSI14", "MACD", "Signal"])

if df_display.empty:
    st.error("Not enough data after indicator warmup.")
    st.stop()

# ── Plotly Interactive Plot ───────────────────────────────────────────────
available_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) >= w]

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]]
)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=df_display.index,
    open=df_display["Open"],
    high=df_display["High"],
    low=df_display["Low"],
    close=df_display["Close"],
    increasing_line_color="#43A047",
    decreasing_line_color="#E53935",
    name="Price",
), row=1, col=1)

# Bollinger Bands
if show_bbands:
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display["BB_UPPER"],
        mode="lines", line=dict(width=1, color="#cfc61f"), name="BB Upper"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display["BB_LOWER"],
        mode="lines", line=dict(width=1, color="#cfc61f"),
        fill="tonexty", fillcolor="rgba(250,245,147,0.15)", name="BB Lower"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display["BB_MA"],
        mode="lines", line=dict(width=1, color="#faf593", dash="dot"), name="BB Mid"
    ), row=1, col=1)

# Moving Averages
for w, color in zip(available_mas, ("#7CB342", "#8E44AD", "#1565C0", "#F39C12", "#607D8B")):
    fig.add_trace(go.Scatter(
        x=df_display.index, y=df_display[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.3), name=f"MA{w}"
    ), row=1, col=1)

# Volume (bar direction)
vol_colors = ["#43A047" if c >= o else "#E53935"
              for c, o in zip(df_display["Close"], df_display["Open"])]
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Volume"],
    marker_color=vol_colors, opacity=0.7, name="Volume"
), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["RSI14"],
    mode="lines", line=dict(width=1.5, color="purple"), name="RSI (14)"
), row=3, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)
fig.update_yaxes(range=[0, 100], row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df_display["Hist"]]
fig.add_trace(go.Bar(
    x=df_display.index, y=df_display["Hist"],
    marker_color=hist_colors, opacity=0.6, name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["MACD"],
    mode="lines", line=dict(width=1.5, color="blue"), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df_display.index, y=df_display["Signal"],
    mode="lines", line=dict(width=1.2, color="orange"), name="Signal"
), row=4, col=1)

# Layout
fig.update_layout(
    height=950,
    title=dict(text=f"{ticker} – OHLC + RSI & MACD" + (" + Bollinger Bands" if show_bbands else ""), x=0.5),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(rangeslider_visible=False),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
)

fig.update_xaxes(dtick="M1", tickformat="%b %Y", showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed")

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
