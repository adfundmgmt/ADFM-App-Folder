# ─────────────────────────────────────────────────────────────────────────────
#  Technical Chart Explorer – Yahoo-style price / MA / VPVR / RSI / MACD
#  AD Fund Management LP · July-2025
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Streamlit page & sidebar ───────────────────────────────────────────────
st.set_page_config(page_title="Technical Chart Explorer", layout="wide")
st.title("Technical Chart Explorer")
st.subheader("Yahoo-Finance style chart with MAs, Volume-Profile, RSI & MACD")

with st.sidebar:
    st.header("Inputs")
    ticker   = st.text_input("Ticker", "NVDA").upper()
    period   = st.selectbox(
        "Period", ["6mo", "1y", "2y", "3y", "5y", "10y", "max"], index=1
    )
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    bins     = st.slider("Volume-profile bins", 12, 60, 24)

# ── Period logic ───────────────────────────────────────────────────────────
CAP_YEARS   = 25
PERIOD_DAYS = {
    "6mo": 182, "1y": 365, "2y": 730, "3y": 1_095,
    "5y": 1_825, "10y": 3_652, "max": 365 * CAP_YEARS
}
lookback = PERIOD_DAYS[period]
buffer   = 400                       # warm-up for indicators
start_dt = datetime.today() - timedelta(days=lookback + buffer)

# ── Robust Yahoo download (split-adjusted) ─────────────────────────────────
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
        raise RuntimeError("Yahoo returned no data.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[df["Close"].notna()].copy()

try:
    data = fetch(ticker, start_dt, interval)
except Exception as err:
    st.error(f"Download failed – {err}")
    st.stop()

if interval == "1d":
    data = data[data.index.weekday < 5]

# ── Indicator calculations ────────────────────────────────────────────────
for w in (20, 50, 100, 200):
    data[f"MA{w}"] = data["Close"].rolling(w).mean()

delta  = data["Close"].diff()
gain   = delta.clip(lower=0)
loss   = -delta.clip(upper=0)
avg_g  = gain.ewm(alpha=1/14, adjust=False).mean()
avg_l  = loss.ewm(alpha=1/14, adjust=False).mean()
rs     = avg_g / avg_l
data["RSI14"] = 100 - (100 / (1 + rs))

data["EMA12"]  = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA26"]  = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"]   = data["EMA12"] - data["EMA26"]
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["Hist"]   = data["MACD"] - data["Signal"]

# ── Trim to visible window ────────────────────────────────────────────────
cutoff = data.index.max() - timedelta(days=lookback)
df     = data.loc[data.index >= cutoff].copy()
df["Date"] = df.index.strftime("%Y-%m-%d")

# ── Volume-profile (VPVR) calculation ─────────────────────────────────────
price_min, price_max = df["Low"].min(), df["High"].max()
edges = np.linspace(price_min, price_max, bins + 1)
volumes, _ = np.histogram(
    np.repeat(df["Close"].values, df["Volume"].astype(int) // df["Volume"].max()),
    bins=edges
)
volumes = volumes / volumes.max() * (price_max - price_min) * 0.20  # scale width

vp_y = 0.5 * (edges[:-1] + edges[1:])   # mid-price of each bin
vp_x = volumes

# ── Build figure ──────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=2,
    shared_xaxes=True,
    column_widths=[0.83, 0.17],                # right column for VPVR
    specs=[
        [{"rowspan":2, "type":"candlestick"}, {"rowspan":2, "type":"bar"}],
        [None, None],
        [{"type":"scatter"}, {"type":"bar"}],  # RSI row (empty right)
        [{"type":"scatter"}, {"type":"scatter"}]  # MACD row (empty right)
    ],
    vertical_spacing=0.02, horizontal_spacing=0.01
)

# 1) Candles
fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="#16A34A", decreasing_line_color="#DC2626",
    name="Price"), row=1, col=1)

# 1) MAs
ma_colors = {20:"#6366F1", 50:"#A855F7", 100:"#F59E0B", 200:"#525252"}
for w, c in ma_colors.items():
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[f"MA{w}"], mode="lines",
        line=dict(color=c, width=1.1), name=f"MA{w}"
    ), row=1, col=1)

# 1) Volume bars (under price)
vol_col = np.where(df["Close"] >= df["Close"].shift(), "#16A34A", "#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Volume"], marker_color=vol_col, opacity=0.75,
    name="Volume"), row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# 1-R) Volume profile histogram (horizontal bars)
fig.add_trace(go.Bar(
    y=vp_y, x=vp_x,
    orientation="h",
    marker_color="#9CA3AF", opacity=0.6,
    showlegend=False,
), row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)
fig.update_xaxes(showticklabels=False, row=1, col=2)

# 3) RSI-14
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["RSI14"],
    mode="lines", line=dict(color="#A21CAF", width=1.3),
    name="RSI-14"), row=3, col=1)
fig.update_yaxes(range=[0,100], title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD
hist_color = np.where(df["Hist"]>=0, "#16A34A", "#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Hist"],
    marker_color=hist_color, opacity=0.6, width=0.4,
    name="MACD Hist"), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["MACD"],
    mode="lines", line=dict(color="#2563EB", width=1.2),
    name="MACD"), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Signal"],
    mode="lines", line=dict(color="#F59E0B", width=1.1),
    name="Signal"), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Formatting
fig.update_layout(
    height=900, width=1200,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    margin=dict(l=60, r=20, t=40, b=40),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    xaxis=dict(rangeslider_visible=False)
)

# Monthly x-ticks (only on bottom)
months = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    row=4, col=1,
    tickmode="array",
    tickvals=months.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=months.dt.strftime("%b-%y").tolist(),
    tickangle=-45
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
