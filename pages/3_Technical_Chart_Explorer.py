# ─────────────────────────────────────────────────────────────────────────────
#  Technical Chart Explorer  ·  OHLC | MAs | VPVR | RSI | MACD
#  AD Fund Management LP   ·  July-2025  (bug-fixed layout)
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Page & sidebar ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Technical Chart Explorer", layout="wide")
st.title("Technical Chart Explorer")
st.subheader("Yahoo-style chart with MAs, Volume-Profile, RSI & MACD")

with st.sidebar:
    st.header("Inputs")
    ticker   = st.text_input("Ticker", "NVDA").upper()
    period   = st.selectbox("Period", ["6mo","1y","2y","3y","5y","10y","max"], index=1)
    interval = st.selectbox("Interval", ["1d","1wk"], index=0)
    bins     = st.slider("Volume-profile bins", 12, 60, 24)

# ── Look-back dates ─────────────────────────────────────────────────────────
CAP_YRS   = 25
PERIOD_DAYS = {"6mo":182, "1y":365, "2y":730, "3y":1095, "5y":1825,
               "10y":3650, "max":365*CAP_YRS}
lookback = PERIOD_DAYS[period]
start_dt = datetime.today() - timedelta(days=lookback + 400)  # warm-up

# ── Yahoo fetch (split-adjusted) ───────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(tkr, start, intrvl):
    tries, delay = 0, 1
    while tries < 3:
        df = yf.Ticker(tkr).history(start=start, interval=intrvl,
                                    auto_adjust=True, actions=False)
        if not df.empty:
            break
        tries += 1; time.sleep(delay); delay *= 2
    if df.empty:
        raise RuntimeError("No data from Yahoo.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[df["Close"].notna()].copy()

try:
    data = fetch(ticker, start_dt, interval)
except Exception as e:
    st.error(f"Download failed – {e}")
    st.stop()

if interval == "1d":
    data = data[data.index.weekday < 5]

# ── Indicators ─────────────────────────────────────────────────────────────
for w in (20, 50, 100, 200):
    data[f"MA{w}"] = data["Close"].rolling(w).mean()

chg = data["Close"].diff()
gain, loss = chg.clip(lower=0), -chg.clip(upper=0)
avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
rs    = avg_g / avg_l
data["RSI14"] = 100 - 100/(1+rs)

data["EMA12"]  = data["Close"].ewm(span=12, adjust=False).mean()
data["EMA26"]  = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"]   = data["EMA12"] - data["EMA26"]
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["Hist"]   = data["MACD"] - data["Signal"]

# ── Trim window ───────────────────────────────────────────────────────────
cutoff = data.index.max() - timedelta(days=lookback)
df     = data.loc[data.index >= cutoff].copy()
df["Date"] = df.index.strftime("%Y-%m-%d")

# ── Volume-profile (VPVR) --------------------------------------------------
lo, hi  = df["Low"].min(), df["High"].max()
edges   = np.linspace(lo, hi, bins+1)
vp_vol, _ = np.histogram(
    np.repeat(df["Close"].values, df["Volume"].astype(int)//df["Volume"].max()),
    bins=edges
)
vp_y = 0.5*(edges[:-1]+edges[1:])
vp_x = vp_vol / vp_vol.max() * (hi-lo)*0.18   # width ≈18 % of price range

# ── Subplot grid (no rowspan) ---------------------------------------------
fig = make_subplots(
    rows=4, cols=2,
    column_widths=[0.83,0.17],
    specs=[
        [{"type":"candlestick"}, {"type":"bar"}],  # price | VPVR
        [{"type":"bar"},           None],          # volume
        [{"type":"scatter"},       None],          # RSI
        [{"type":"scatter"},       None]           # MACD
    ],
    vertical_spacing=0.03, horizontal_spacing=0.02
)

# Candles + MAs
fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    increasing_line_color="#16A34A", decreasing_line_color="#DC2626",
    name="Price"), row=1, col=1)

for w,c in zip((20,50,100,200),
               ("#6366F1","#A855F7","#F59E0B","#525252")):
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df[f"MA{w}"],
        mode="lines", line=dict(color=c,width=1.1),
        name=f"MA{w}"), row=1, col=1)

fig.update_yaxes(title_text="Price", row=1, col=1)

# VPVR bars (horizontal)
fig.add_trace(go.Bar(
    y=vp_y, x=vp_x, orientation="h",
    marker_color="#9CA3AF", opacity=0.6,
    showlegend=False), row=1, col=2)
fig.update_xaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)

# Volume bars
vol_col = np.where(df["Close"]>=df["Close"].shift(),"#16A34A","#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Volume"],
    marker_color=vol_col, opacity=0.75,
    name="Volume"), row=2, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# RSI-14
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["RSI14"],
    mode="lines", line=dict(color="#A21CAF",width=1.3),
    name="RSI-14"), row=3, col=1)
fig.update_yaxes(range=[0,100], title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_col = np.where(df["Hist"]>=0,"#16A34A","#DC2626")
fig.add_trace(go.Bar(
    x=df["Date"], y=df["Hist"],
    marker_color=hist_col, opacity=0.6, width=0.4,
    name="MACD Hist"), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["MACD"],
    mode="lines", line=dict(color="#2563EB",width=1.2),
    name="MACD"), row=4, col=1)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Signal"],
    mode="lines", line=dict(color="#F59E0B",width=1.1),
    name="Signal"), row=4, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Monthly x-ticks only on bottom axis
months = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    row=4, col=1,
    tickmode="array",
    tickvals=months.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=months.dt.strftime("%b-%y").tolist(),
    tickangle=-45
)

fig.update_layout(
    height=900, width=1200,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.01, x=0.5, xanchor="center"),
    margin=dict(l=60, r=20, t=40, b=40),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12),
    xaxis=dict(rangeslider_visible=False)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
