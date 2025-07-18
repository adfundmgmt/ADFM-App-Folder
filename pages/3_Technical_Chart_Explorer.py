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
- 8/20/50/100/200-day moving averages (full length)  
- Volume bars color-coded by up/down days (vs prior close)  
- RSI (14-day) panel (0–100 scale) with Wilder smoothing  
- MACD (12,26,9) panel with colored histogram  
""")
ticker   = st.sidebar.text_input("Ticker", "^GSPC").upper()
period   = st.sidebar.selectbox(
    "Period", ["1mo","3mo","6mo","1y","2y","3y","5y","10y","max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)

# ── Define trading days for each period ────────────────────────────────────
# US market trading days: 1y ≈ 252, 3y ≈ 756, etc.
base_trading_days = {
    "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "3y": 756,
    "5y": 1260, "10y": 2520
}
CAP_MAX_DAYS = 252 * 25  # Cap 'max' at 25 years ≈ 6300 trading days

# ── Fetch & cache data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_data(tkr, period, intrvl):
    try:
        return yf.Ticker(tkr).history(period=period, interval=intrvl)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# For Yahoo's API, 'period' strings are best for fetching; we slice later
yahoo_period_map = {
    "1mo": "2mo", "3mo": "6mo", "6mo": "1y", "1y": "2y", "2y": "3y",
    "3y": "5y", "5y": "10y", "10y": "max", "max": "max"
}
fetch_period = yahoo_period_map.get(period, "max")
df_full = get_data(ticker, fetch_period, interval)
if df_full is None or df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

# ── Sanitize index, drop weekends for daily data ───────────────────────────
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)
if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

# ── Indicators on full history ─────────────────────────────────────────────
for w in (8, 20, 50, 100, 200):
    df_full[f"MA{w}"] = df_full["Close"].rolling(w).mean()

# RSI(14) — Wilder smoothing
delta    = df_full["Close"].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs       = avg_gain / avg_loss
df_full["RSI14"] = 100 - (100 / (1 + rs))

# MACD(12,26,9)
df_full["EMA12"]  = df_full["Close"].ewm(span=12, adjust=False).mean()
df_full["EMA26"]  = df_full["Close"].ewm(span=26, adjust=False).mean()
df_full["MACD"]   = df_full["EMA12"] - df_full["EMA26"]
df_full["Signal"] = df_full["MACD"].ewm(span=9, adjust=False).mean()
df_full["Hist"]   = df_full["MACD"] - df_full["Signal"]

# ── Trim to display window using trading days ──────────────────────────────
if period == "max":
    df = df_full.tail(CAP_MAX_DAYS).copy()
    effective_trading_days = CAP_MAX_DAYS
else:
    effective_trading_days = base_trading_days[period]
    if len(df_full) < effective_trading_days:
        df = df_full.copy()
    else:
        df = df_full.tail(effective_trading_days).copy()

df["DateStr"] = df.index.strftime("%Y-%m-%d")
available_mas = [w for w in (8, 20, 50, 100, 200) if len(df) >= w]

# ── Warn if window too short for certain MAs ──────────────────────────────
for w in (8, 20, 50, 100, 200):
    if len(df) < w:
        st.warning(f"Showing MA{w} with fewer than {w} data points may appear choppy.")

# ── Build Figure ──────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[
        [{"type": "candlestick"}],
        [{"type": "bar"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}],
    ]
)

# 1) Price + MAs
fig.add_trace(go.Candlestick(
    x=df["DateStr"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="#43A047", decreasing_line_color="#E53935",
    name="Price"
), row=1, col=1)
for w, color in zip(available_mas, ("green", "purple", "blue", "orange", "gray")):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"],
        mode="lines", line=dict(color=color, width=1.3),
        name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# 2) Volume (vs prior close)
vol_colors = [
    "#B0BEC5" if i == 0 else (
        "#43A047" if df["Close"].iat[i] > df["Close"].iat[i-1] else "#E53935"
    )
    for i in range(len(df))
]
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Volume"],
    width=0.4,
    marker_color=vol_colors, opacity=0.7,
    name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 3) RSI (14)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["RSI14"],
    mode="lines", line=dict(width=1.5, color="purple"),
    name="RSI (14)"
), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1, title_standoff=10)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + colored Hist
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df["Hist"]]
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Hist"],
    marker_color=hist_colors, opacity=0.6, width=0.4,
    name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["MACD"],
    mode="lines", line=dict(width=1.5, color="blue"),
    name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["Signal"],
    mode="lines", line=dict(width=1.2, color="orange"),
    name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# ── X‑Axis Monthly Ticks ──────────────────────────────────────────────────
month_starts = df.index.to_series().groupby(df.index.to_period("M")).first()
tickvals     = month_starts.dt.strftime("%Y-%m-%d").tolist()
ticktext     = month_starts.dt.strftime("%b-%y").tolist()
fig.update_xaxes(
    type="category", tickmode="array",
    tickvals=tickvals, ticktext=ticktext,
    tickangle=-45,
    showgrid=True, gridwidth=0.5, gridcolor="#e1e5ed"
)

# ── Layout Tweaks ─────────────────────────────────────────────────────────
fig.update_layout(
    height=950, width=1100,
    title=dict(text=f"{ticker} — OHLC + RSI & MACD", x=0.5),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(rangeslider_visible=False, showline=True, linewidth=1, linecolor='black'),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12)
)

st.plotly_chart(fig, use_container_width=True)

# ── Footnotes ─────────────────────────────────────────────────────────────
st.caption("© 2025 AD Fund Management LP")
