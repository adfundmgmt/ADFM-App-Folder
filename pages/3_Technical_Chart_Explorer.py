# ─────────────────────────────────────────────────────────────────────────────
#  Technical Chart Explorer  ·  OHLC  |  MAs  |  RSI-14  |  MACD (opt.)
#  AD Fund Management LP   ·  July-2025  (v3.1 – graceful on short history)
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Technical Chart Explorer", layout="wide")
st.title("Technical Chart Explorer")
st.subheader("Visualise price structure and momentum in one view")

# ── Sidebar  ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Overlay **price action** with:

- Moving Averages (8 / 20 / 50 / 100 / 200)  
- Volume bars (green / red)  
- RSI-14 (Wilder)  
- MACD 12-26-9 + histogram (if ≥ 35 bars)  

Data: **Yahoo Finance** via `yfinance` (split-adjusted).
"""
    )
    st.markdown("---")
    ticker   = st.text_input("Ticker", "NVDA").upper()
    period   = st.selectbox(
        "Period", ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"], index=3
    )
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# ── Look-back logic  ───────────────────────────────────────────────────────
MAX_DAYS  = 9_125
PERIOD_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
    "2y": 730, "3y": 1_095, "5y": 1_825, "10y": 3_652, "max": MAX_DAYS
}
days      = PERIOD_DAYS[period]
buffer    = max(days, 500)                    # for indicator warm-up
start_dt  = datetime.today() - timedelta(days=days + buffer)

# ── Robust fetch helper  ──────────────────────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def fetch(tkr: str, start, intrvl: str) -> pd.DataFrame:
    tries, delay = 0, 1
    while tries < 3:
        df = yf.Ticker(tkr).history(start=start, interval=intrvl,
                                    auto_adjust=True, actions=False)
        if not df.empty:
            break
        tries += 1
        time.sleep(delay); delay *= 2
    if df.empty:
        raise ValueError("Yahoo returned no data.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[df["Close"].notna()].copy()

# ── Download  ─────────────────────────────────────────────────────────────
try:
    hist = fetch(ticker, start_dt, interval)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

if interval == "1d":
    hist = hist[hist.index.weekday < 5]        # drop weekends

bars = len(hist)
if bars < 2:
    st.error("Yahoo has fewer than two price bars for this selection.")
    st.stop()

# ── Indicators (only if enough data)  ─────────────────────────────────────
ma_windows = (8, 20, 50, 100, 200)
have_ma    = [w for w in ma_windows if bars >= w]
skip_msgs  = []

for w in have_ma:
    hist[f"MA{w}"] = hist["Close"].rolling(w).mean()
if missing := [str(w) for w in ma_windows if w not in have_ma]:
    skip_msgs.append(f"MA{'/'.join(missing)} (need ≥ {missing[-1]} bars)")

# RSI-14 needs 15 bars
if bars >= 15:
    delta      = hist["Close"].diff()
    gain       = delta.clip(lower=0)
    loss       = -delta.clip(upper=0)
    avg_gain   = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss   = loss.ewm(alpha=1/14, adjust=False).mean()
    rs         = avg_gain / avg_loss
    hist["RSI14"] = 100 - (100 / (1 + rs))
else:
    skip_msgs.append("RSI-14 (need ≥ 15 bars)")

# MACD needs 35 bars
if bars >= 35:
    hist["EMA12"]  = hist["Close"].ewm(span=12, adjust=False).mean()
    hist["EMA26"]  = hist["Close"].ewm(span=26, adjust=False).mean()
    hist["MACD"]   = hist["EMA12"] - hist["EMA26"]
    hist["Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
    hist["Hist"]   = hist["MACD"] - hist["Signal"]
else:
    skip_msgs.append("MACD (need ≥ 35 bars)")

# ── Trim display window  ─────────────────────────────────────────────────
cutoff = hist.index.max() - timedelta(days=days)
df = hist.loc[hist.index >= cutoff].copy()
df["Date"] = df.index.strftime("%Y-%m-%d")

# ── Warning banner (single)  ─────────────────────────────────────────────
if skip_msgs:
    st.info("Unavailable this run: " + " · ".join(skip_msgs))

# ── Build figure  ────────────────────────────────────────────────────────
rows = 4 if "Hist" in df.columns else 3
row_heights = [0.60, 0.15, 0.25] if rows == 3 else [0.60, 0.15, 0.12, 0.13]
specs = [[{"type": "candlestick"}],
         [{"type": "bar"}],
         [{"type": "scatter"}]] + ([[{"type": "scatter"}]] if rows == 4 else [])

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    row_heights=row_heights, vertical_spacing=0.03, specs=specs
)

# 1) Price + MAs
fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"],  close=df["Close"],
    increasing_line_color="#16A34A", decreasing_line_color="#DC2626",
    name="Price"), row=1, col=1)

palette = ("#15803D", "#7E22CE", "#2563EB", "#EA580C", "#525252")
for w, c in zip(have_ma, palette):
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[f"MA{w}"], mode="lines",
        line=dict(color=c, width=1.2), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# 2) Volume
vol_col = np.where(df["Close"] >= df["Close"].shift(), "#16A34A", "#DC2626")
fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"],
                     marker_color=vol_col, opacity=0.75,
                     name="Volume"), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 3) RSI-14 or fallback empty line
if "RSI14" in df.columns:
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI14"],
        mode="lines", line=dict(width=1.3, color="#A21CAF"),
        name="RSI-14"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)
else:  # placeholder axis
    fig.update_yaxes(visible=False, row=3, col=1)

# 4) MACD if available
if "Hist" in df.columns:
    hist_colors = np.where(df["Hist"] >= 0, "#16A34A", "#DC2626")
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Hist"], marker_color=hist_colors,
        opacity=0.6, width=0.4, name="MACD Hist"
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD"],
        mode="lines", line=dict(color="#2563EB", width=1.2), name="MACD"
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Signal"],
        mode="lines", line=dict(color="#F59E0B", width=1.1), name="Signal"
    ), row=4, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

# Monthly ticks
mon = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    type="category", tickmode="array",
    tickvals=mon.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=mon.dt.strftime("%b-%y").tolist(),
    tickangle=-45, showgrid=True, gridcolor="#E5E7EB"
)

fig.update_layout(
    height=900, width=1100,
    title=dict(text=f"{ticker} · OHLC / MAs / RSI / MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.01, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    xaxis=dict(rangeslider_visible=False)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
