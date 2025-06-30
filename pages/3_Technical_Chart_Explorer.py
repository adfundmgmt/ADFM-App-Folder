# ─────────────────────────────────────────────────────────────────────────────
#  Stock-Lens  ·  OHLC + Moving Averages · RSI · MACD
#  AD Fund Management LP   (July-2025, v2.0 – complete rebuild)
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Streamlit page settings ────────────────────────────────────────────────
st.set_page_config(page_title="Stock-Lens", layout="wide")

LOGO_PATH = Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=70)

st.title("📈 Stock-Lens")
st.subheader("Visualise price structure & momentum in a single glance")

# ── Sidebar: About & inputs ───────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Quickly layer **price action** with key momentum gauges:

**Included**
- Interactive OHLC candlesticks  
- Rolling MAs: 8 / 20 / 50 / 100 / 200  
- Volume bars, colour-coded by up/down day  
- RSI-14 (Wilder) panel  
- MACD (12-26-9) with coloured histogram  

Data comes from **Yahoo Finance** (via `yfinance`).  
Figures auto-scale to the chosen period & interval.
"""
    )
    st.markdown("---")

    ticker = st.text_input("Ticker", "NVDA").upper()
    period = st.selectbox("Period",
                          ["1mo", "3mo", "6mo", "1y", "2y",
                           "3y", "5y", "10y", "max"],
                          index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# ── Period → look-back days mapping ───────────────────────────────────────
LOOKBACK_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
    "3y": 1_095, "5y": 1_825, "10y": 3_652
}
CAP_MAX_DAYS = 9_125         # cap “max” at 25 yrs

if period == "max":
    lookback_days = CAP_MAX_DAYS
    st.info("⚠️ ‘max’ capped at 25 years for performance.")
else:
    lookback_days = LOOKBACK_DAYS[period]

buffer_days = max(lookback_days, 500)          # for indicators needing history
start_date = datetime.today() - timedelta(days=lookback_days + buffer_days)

# ── Robust data fetch helper ──────────────────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_history(tkr: str, start, intrvl: str) -> pd.DataFrame:
    """
    Single-ticker fetch with three retries & auto-adjust fallback.
    Returns clean dataframe with tz-naive DatetimeIndex.
    """
    attempts, delay = 0, 1
    while attempts < 3:
        df = yf.Ticker(tkr).history(start=start, interval=intrvl, auto_adjust=False)
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay)
        delay *= 2

    if df.empty:
        raise ValueError("Yahoo returned no data.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Drop rows with no volume/close (holiday gaps)
    df = df.loc[df["Close"].notna()].copy()
    return df

# ── Download ─────────────────────────────────────────────────────────────
try:
    df_full = fetch_history(ticker, start_date, interval)
except Exception as err:
    st.error(f"Download failed – {err}")
    st.stop()

if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]    # strip weekends

# ── Indicator calculations (on full history) ─────────────────────────────
for w in (8, 20, 50, 100, 200):
    df_full[f"MA{w}"] = df_full["Close"].rolling(w).mean()

# RSI-14 (Wilder)
delta = df_full["Close"].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs       = avg_gain / avg_loss
df_full["RSI14"] = 100 - (100 / (1 + rs))

# MACD 12-26-9
df_full["EMA12"]  = df_full["Close"].ewm(span=12, adjust=False).mean()
df_full["EMA26"]  = df_full["Close"].ewm(span=26, adjust=False).mean()
df_full["MACD"]   = df_full["EMA12"] - df_full["EMA26"]
df_full["Signal"] = df_full["MACD"].ewm(span=9, adjust=False).mean()
df_full["Hist"]   = df_full["MACD"] - df_full["Signal"]

# ── Trim to requested display window ─────────────────────────────────────
cutoff = df_full.index.max() - timedelta(days=lookback_days)
df = df_full.loc[df_full.index >= cutoff].copy()

# guard: ensure we have at least two rows
if len(df) < 2:
    st.error("Not enough data for the chosen period/interval.")
    st.stop()

df["DateStr"] = df.index.strftime("%Y-%m-%d")
available_mas = [w for w in (8, 20, 50, 100, 200) if len(df) >= w]

# Warn on short history vs MA window
for w in (8, 20, 50, 100, 200):
    if len(df) < w:
        st.warning(f"MA{w} has < {w} points. It may look choppy.")

# ── Build Plotly figure ──────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.15, 0.12, 0.13],
    vertical_spacing=0.03,
    specs=[[{"type": "candlestick"}],
           [{"type": "bar"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

# 1) Candles + MAs
fig.add_trace(go.Candlestick(
    x=df["DateStr"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
    name="Price"
), row=1, col=1)

ma_palette = ("#388E3C", "#7B1FA2", "#1976D2", "#F57C00", "#616161")
for w, c in zip(available_mas, ma_palette):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"], mode="lines",
        line=dict(color=c, width=1.3), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# 2) Volume
vol_color = np.where(df["Close"] >= df["Close"].shift(), "#4CAF50", "#F44336")
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Volume"],
    marker_color=vol_color, opacity=0.75, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 3) RSI
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["RSI14"],
    mode="lines", line=dict(width=1.5, color="#9C27B0"),
    name="RSI-14"
), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Hist
hist_colors = np.where(df["Hist"] >= 0, "#4CAF50", "#F44336")
fig.add_trace(go.Bar(
    x=df["DateStr"], y=df["Hist"],
    marker_color=hist_colors, opacity=0.6, width=0.4, name="MACD Hist"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["MACD"],
    mode="lines", line=dict(color="#1976D2", width=1.3), name="MACD"
), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["DateStr"], y=df["Signal"],
    mode="lines", line=dict(color="#FF9800", width=1.1), name="Signal"
), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Monthly tick labels
month_starts = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    type="category",
    tickmode="array",
    tickvals=month_starts.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=month_starts.dt.strftime("%b-%y").tolist(),
    tickangle=-45,
    showgrid=True, gridcolor="#E0E0E0"
)

# Layout polish
fig.update_layout(
    height=950, width=1100,
    title=dict(text=f"{ticker} · OHLC + RSI + MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=60, r=20, t=60, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    xaxis=dict(rangeslider_visible=False, showline=True,
               linewidth=1, linecolor="black")
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
