# ─────────────────────────────────────────────────────────────────────────────
#  Technical Chart Explorer  ·  OHLC | MAs | RSI-14 | MACD
#  AD Fund Management LP   ·  July-2025
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Streamlit page settings ────────────────────────────────────────────────
st.set_page_config(page_title="Technical Chart Explorer", layout="wide")
st.title("Technical Chart Explorer")
st.subheader("Visualise price structure & momentum in one place")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Superimpose **price action** with:

- Moving Averages (8 / 20 / 50 / 100 / 200)  
- Volume bars (green / red)  
- RSI-14 (Wilder)  
- MACD 12-26-9 + histogram  

Powered by **Yahoo Finance** via `yfinance` (split-adjusted prices).
"""
    )
    st.markdown("---")
    ticker   = st.text_input("Ticker", "NVDA").upper()
    period   = st.selectbox("Period",
                            ["1mo", "3mo", "6mo", "1y",
                             "2y", "3y", "5y", "10y", "max"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# ── Period-to-lookback mapping ────────────────────────────────────────────
MAX_DAYS  = 365 * 25  # 25-year hard cap for daily data
LOOKBACK  = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
    "2y": 730, "3y": 1_095, "5y": 1_825, "10y": 3_652, "max": MAX_DAYS
}[period]

BUFFER    = max(LOOKBACK, 500)                    # extra for indicator warm-up
START     = datetime.today() - timedelta(days=LOOKBACK + BUFFER)

# ── Robust Yahoo fetch (split-ADJUSTED) ───────────────────────────────────
@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_history(tkr: str, start_dt, intrvl: str) -> pd.DataFrame:
    tries, delay = 0, 1
    while tries < 3:
        df = yf.Ticker(tkr).history(start=start_dt,
                                    interval=intrvl,
                                    auto_adjust=True,   # <<< split-adjust all OHLC
                                    actions=False)
        if not df.empty:
            break
        tries += 1
        time.sleep(delay); delay *= 2
    if df.empty:
        raise ValueError("Yahoo returned no data.")
    df = df[df["Close"].notna()].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

# ── Download & validate ──────────────────────────────────────────────────
try:
    data_full = fetch_history(ticker, START, interval)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

if interval == "1d":
    data_full = data_full[data_full.index.weekday < 5]   # drop weekends

if len(data_full) < 50:          # need enough bars for indicators
    st.error("Too little data for the selected settings.")
    st.stop()

# ── Indicators on full data ──────────────────────────────────────────────
for w in (8, 20, 50, 100, 200):
    data_full[f"MA{w}"] = data_full["Close"].rolling(w).mean()

chg        = data_full["Close"].diff()
gain       = chg.clip(lower=0)
loss       = -chg.clip(upper=0)
avg_gain   = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss   = loss.ewm(alpha=1/14, adjust=False).mean()
rs         = avg_gain / avg_loss
data_full["RSI14"] = 100 - (100 / (1 + rs))

data_full["EMA12"]  = data_full["Close"].ewm(span=12, adjust=False).mean()
data_full["EMA26"]  = data_full["Close"].ewm(span=26, adjust=False).mean()
data_full["MACD"]   = data_full["EMA12"] - data_full["EMA26"]
data_full["Signal"] = data_full["MACD"].ewm(span=9, adjust=False).mean()
data_full["Hist"]   = data_full["MACD"] - data_full["Signal"]

# ── Trim to display window ───────────────────────────────────────────────
cut = data_full.index.max() - timedelta(days=LOOKBACK)
df  = data_full.loc[data_full.index >= cut].copy()

df["Date"] = df.index.strftime("%Y-%m-%d")
valid_ma   = [w for w in (8, 20, 50, 100, 200) if len(df) >= w]

# ── Plotly Figure ────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.15, 0.12, 0.13], vertical_spacing=0.03,
    specs=[[{"type": "candlestick"}],
           [{"type": "bar"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

# 1) Candles + MAs
fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="#16A34A", decreasing_line_color="#DC2626",
    name="Price"), row=1, col=1)

palette = ("#15803D", "#7E22CE", "#2563EB", "#EA580C", "#525252")
for w, c in zip(valid_ma, palette):
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[f"MA{w}"], mode="lines",
        line=dict(color=c, width=1.2), name=f"MA{w}"
    ), row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# 2) Volume bars
vol_colors = np.where(df["Close"] >= df["Close"].shift(), "#16A34A", "#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Volume"],
    marker_color=vol_colors, opacity=0.75, name="Volume"
), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 3) RSI-14
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["RSI14"],
    mode="lines", line=dict(width=1.4, color="#A21CAF"),
    name="RSI-14"), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD + Hist
hist_color = np.where(df["Hist"] >= 0, "#16A34A", "#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Hist"],
    marker_color=hist_color, opacity=0.6, width=0.4, name="MACD Hist"
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
months = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    type="category",
    tickmode="array",
    tickvals=months.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=months.dt.strftime("%b-%y").tolist(),
    tickangle=-45, showgrid=True, gridcolor="#E5E7EB"
)

fig.update_layout(
    height=950, width=1100,
    title=dict(text=f"{ticker} · OHLC / MAs / RSI / MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    xaxis=dict(rangeslider_visible=False)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
