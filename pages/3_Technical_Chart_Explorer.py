# ─────────────────────────────────────────────────────────────────────────────
#  Technical Chart Explorer  ·  OHLC · MAs · RSI · MACD
#  AD Fund Management LP   ·  July-2025  (v3.0)
# ─────────────────────────────────────────────────────────────────────────────
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Technical Chart Explorer", layout="wide")
st.title("Technical Chart Explorer")
st.subheader("Visualise price structure and momentum in one view")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
Use Yahoo Finance data to overlay **price action** with:

- Rolling MAs (8 | 20 | 50 | 100 | 200)  
- Volume colour-coded by up/down day  
- RSI-14 (Wilder)  
- MACD (12-26-9) + histogram  

Select any ticker, period, and interval. Figures auto-scale.
"""
    )
    st.markdown("---")

    ticker   = st.text_input("Ticker", "NVDA").upper()
    period   = st.selectbox("Period",
                            ["1mo", "3mo", "6mo", "1y", "2y",
                             "3y", "5y", "10y", "max"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# ── Helpers ───────────────────────────────────────────────────────────────
MAX_DAYS = 9_125                                   # 25-year hard cap
PERIOD_DAYS = {                                    # look-back for period
    "1mo": 30, "3mo": 90, "6mo": 180,  "1y": 365,
    "2y": 730, "3y": 1_095, "5y": 1_825, "10y": 3_652,
    "max": MAX_DAYS
}
lookback = PERIOD_DAYS[period]
buffer   = max(lookback, 500)                      # indicators need history
start    = datetime.today() - timedelta(days=lookback + buffer)

@st.cache_data(ttl=3_600, show_spinner=False)
def get_history(tkr: str, start_, intrvl: str) -> pd.DataFrame:
    attempts, delay = 0, 1
    while attempts < 3:
        df = yf.Ticker(tkr).history(start=start_, interval=intrvl,
                                    auto_adjust=False, actions=False)
        if not df.empty:
            break
        attempts += 1
        time.sleep(delay); delay *= 2
    if df.empty:
        raise ValueError("no data returned from Yahoo")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[df["Close"].notna()].copy()

# ── Download & basic validation ──────────────────────────────────────────
try:
    data_full = get_history(ticker, start, interval)
except Exception as err:
    st.error(f"Download failed – {err}")
    st.stop()

if interval == "1d":
    data_full = data_full[data_full.index.weekday < 5]   # strip weekends

if len(data_full) < 2:
    st.error("Not enough data for the chosen settings.")
    st.stop()

# ── Indicator calculations on full set ───────────────────────────────────
for w in (8, 20, 50, 100, 200):
    data_full[f"MA{w}"] = data_full["Close"].rolling(w).mean()

delta = data_full["Close"].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs        = avg_gain / avg_loss
data_full["RSI14"] = 100 - (100 / (1 + rs))

data_full["EMA12"]  = data_full["Close"].ewm(span=12, adjust=False).mean()
data_full["EMA26"]  = data_full["Close"].ewm(span=26, adjust=False).mean()
data_full["MACD"]   = data_full["EMA12"] - data_full["EMA26"]
data_full["Signal"] = data_full["MACD"].ewm(span=9, adjust=False).mean()
data_full["Hist"]   = data_full["MACD"] - data_full["Signal"]

# ── Trim to display window ───────────────────────────────────────────────
cut = data_full.index.max() - timedelta(days=lookback)
df  = data_full.loc[data_full.index >= cut].copy()

df["DateStr"] = df.index.strftime("%Y-%m-%d")
have_ma = [w for w in (8, 20, 50, 100, 200) if len(df) >= w]

# compact warning block (one message, not four)
short_mas = [str(w) for w in (8, 20, 50, 100, 200) if len(df) < w]
if short_mas:
    st.warning(
        f"MA{'/'.join(short_mas)} calculated with fewer than "
        f"{'/'.join(short_mas)} points — lines may look choppy."
    )

# ── Figure build ─────────────────────────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.15, 0.12, 0.13], vertical_spacing=0.03,
    specs=[[{"type": "candlestick"}],
           [{"type": "bar"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

# 1) Price + MAs
fig.add_trace(go.Candlestick(
    x=df["DateStr"], open=df["Open"], high=df["High"],
    low=df["Low"],  close=df["Close"],
    increasing_line_color="#4CAF50", decreasing_line_color="#F44336",
    name="Price"), row=1, col=1)

for w, c in zip(have_ma, ("#388E3C", "#7B1FA2", "#1976D2", "#F57C00", "#616161")):
    fig.add_trace(go.Scatter(
        x=df["DateStr"], y=df[f"MA{w}"], mode="lines",
        line=dict(color=c, width=1.25), name=f"MA{w}"
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
    mode="lines", line=dict(width=1.4, color="#9C27B0"),
    name="RSI-14"), row=3, col=1)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

# 4) MACD
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

# Monthly x-ticks
months = df.index.to_series().groupby(df.index.to_period("M")).first()
fig.update_xaxes(
    type="category",
    tickmode="array",
    tickvals=months.dt.strftime("%Y-%m-%d").tolist(),
    ticktext=months.dt.strftime("%b-%y").tolist(),
    tickangle=-45, showgrid=True, gridcolor="#E0E0E0"
)

fig.update_layout(
    height=950, width=1100,
    title=dict(text=f"{ticker} · OHLC, MAs, RSI & MACD", x=0.5),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=60, r=20, t=60, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
    xaxis=dict(rangeslider_visible=False)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("© 2025 AD Fund Management LP")
