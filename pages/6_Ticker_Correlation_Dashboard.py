# ──────────────────────────────────────────────────────────────────────────────
#  S&P Cyclicals vs. Defensives (Equal‑Weight) – Static Matplotlib Dashboard
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from dateutil.relativedelta import relativedelta
import datetime as dt

st.set_page_config(page_title="Cyc vs Def Static Dashboard", layout="wide")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight (Static)")

# ── Basket definitions ────────────────────────────────────────────────────────
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
ALL_TICKS  = CYCLICALS + DEFENSIVES

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Period")
    period_choice = st.selectbox(
        "Select look‑back window",
        ("3 M","6 M","9 M","YTD","1 Y","3 Y","5 Y","10 Y"),
        index=4     # default 1 Y
    )
    rsi_n   = st.slider("RSI window", 5, 30, 14)
    ma_fast = st.slider("Short MA", 10, 100, 50)
    ma_slow = st.slider("Long  MA", 100, 300, 200)

# ── Date handling ─────────────────────────────────────────────────────────────
today = dt.date.today()
if period_choice == "YTD":
    start_date = dt.date(today.year, 1, 1)
else:
    lookback = {
        "3 M":  {"months":3},  "6 M":  {"months":6},  "9 M": {"months":9},
        "1 Y":  {"years":1},   "3 Y":  {"years":3},   "5 Y": {"years":5},
        "10 Y": {"years":10}
    }[period_choice.replace(" ","")]  # remove thin‑space
    start_date = today - relativedelta(**lookback)

# ── Data download ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=12*3600)
def fetch_prices(tickers, start, end):
    px = yf.download(tickers, start=start, end=end, auto_adjust=True)["Adj Close"]
    return px.dropna(axis=1, how="all")

prices = fetch_prices(ALL_TICKS, start_date, today)
returns = prices.pct_change().dropna()

cyc_live = [t for t in CYCLICALS  if t in prices]
def_live = [t for t in DEFENSIVES if t in prices]

# ── Equal‑weight index (daily rebalanced) ─────────────────────────────────────
cyc_idx = (1 + returns[cyc_live].mean(axis=1)).cumprod()*100
def_idx = (1 + returns[def_live].mean(axis=1)).cumprod()*100
ratio   = cyc_idx / def_idx

# ── Moving averages & RSI ─────────────────────────────────────────────────────
ma_fast_series = ratio.rolling(ma_fast).mean()
ma_slow_series = ratio.rolling(ma_slow).mean()

def rsi(series, n):
    delta = series.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - 100/(1+rs)

rsi_series = rsi(ratio, rsi_n)

# ── Static Matplotlib figure ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios":[2,1], "hspace":0.05}
)

# Regime colouring: green above slow MA, red below
above = ratio >= ma_slow_series
ax1.plot(ratio.index, ratio.where(above), color="green", lw=2, label="Ratio > MA200")
ax1.plot(ratio.index, ratio.where(~above), color="red",   lw=2, label="Ratio < MA200")

# Moving averages
ax1.plot(ma_fast_series, color="blue",     lw=1.5, ls="--", label=f"{ma_fast}‑day MA")
ax1.plot(ma_slow_series, color="firebrick",lw=1.8, ls="-.", label=f"{ma_slow}‑day MA")

ax1.set_ylabel("Ratio")
ax1.set_title(f"Cyclicals vs Defensives — {period_choice} window", fontsize=13, pad=10)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, which="both", ls=":", lw=0.4)

# RSI pane
ax2.plot(rsi_series, color="black", lw=1.2)
ax2.axhline(70, color="red",   ls="--", lw=1)
ax2.axhline(30, color="green", ls="--", lw=1)
ax2.set_ylabel("RSI")
ax2.set_ylim(0,100)
ax2.grid(True, ls=":", lw=0.4)

# Nice date formatting
ax2.xaxis.set_major_formatter(DateFormatter('%b %Y'))
fig.autofmt_xdate()

st.pyplot(fig, clear_figure=False)

# ── Current reading table ─────────────────────────────────────────────────────
latest_rsi = rsi_series.dropna().iloc[-1]
status = "Overbought" if latest_rsi>70 else "Oversold" if latest_rsi<30 else "Neutral"

st.markdown("### Latest Readings")
st.table(pd.DataFrame({
    "Ratio" : [f"{ratio.iloc[-1]:.2f}"],
    f"RSI({rsi_n})":[f"{latest_rsi:.1f}"],
    "Status":[status]
}))
