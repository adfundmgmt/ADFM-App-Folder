import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from dateutil.relativedelta import relativedelta
import datetime as dt

st.set_page_config(page_title="Cyclicals vs Defensives — Bloomberg Style", layout="wide")
st.title("S&P Cyclicals Relative to Defensives — Equal-Weight (Bloomberg Replication)")

CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
ALL        = CYCLICALS + DEFENSIVES

with st.sidebar:
    period = st.selectbox("Look-back Window",
                          ("6 M","YTD","1 Y","3 Y"),
                          index=2)
    rsi_n   = st.slider("RSI Window", 5, 30, 14, 1)
    ma_fast = st.slider("Short MA (blue)", 10, 100, 50, 1)
    ma_slow = st.slider("Long  MA (red)", 100, 300, 200, 5)

today = dt.date.today()
if period == "YTD":
    start = dt.date(today.year, 1, 1)
elif period == "6 M":
    start = today - relativedelta(months=6)
elif period == "1 Y":
    start = today - relativedelta(years=1)
elif period == "3 Y":
    start = today - relativedelta(years=3)
else:
    start = today - relativedelta(years=1)

@st.cache_data(ttl=12*3600)
def fetch_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else \
             df.xs(df.columns.get_level_values(0).unique()[-1], axis=1, level=0)
    else:
        df = df.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
    df = df.replace(0, np.nan)
    return df

prices = fetch_prices(ALL, start, today)
prices = prices.ffill().bfill()

# -- Only use tickers with *complete* data for entire window
keep = prices.notna().all(axis=0)
prices = prices.loc[:, keep]
cyc_live = [t for t in CYCLICALS  if t in prices]
def_live = [t for t in DEFENSIVES if t in prices]

# -- Build rebased price-level equal-weighted indices
rebased = prices / prices.iloc[0] * 100
cyc_index = rebased[cyc_live].mean(axis=1)
def_index = rebased[def_live].mean(axis=1)
ratio_index = (cyc_index / def_index) * 100

# -- Moving averages
ma_fast_ser = ratio_index.rolling(ma_fast).mean()
ma_slow_ser = ratio_index.rolling(ma_slow).mean()

# -- RSI function
def rsi(series, window):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=window, min_periods=window).mean()
    avg_loss = down.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_ser = rsi(ratio_index, rsi_n)

# -- Chart (Bloomberg style, trend-colored)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1], "hspace":0.05})

trend_up = ratio_index >= ma_slow_ser
ax1.plot(ratio_index.index, ratio_index.where(trend_up),  color="green", lw=1.6, label="Trend Up")
ax1.plot(ratio_index.index, ratio_index.where(~trend_up), color="red",   lw=1.6, label="Trend Down")
ax1.plot(ma_fast_ser, color="blue",      lw=1.2, ls="--", label=f"{ma_fast}-day MA")
ax1.plot(ma_slow_ser, color="firebrick", lw=1.4, ls="-.", label=f"{ma_slow}-day MA")
ax1.set_ylabel("Ratio Index")
ax1.set_title(f"S&P Cyclicals vs Defensives — {period} Window", pad=10, fontsize=14)
ax1.legend(frameon=False, fontsize=10)
ax1.grid(True, ls=":", lw=0.3)
# Dynamic y-axis: slight padding for professional look
ymin, ymax = np.nanpercentile(ratio_index, 1), np.nanpercentile(ratio_index, 99)
ax1.set_ylim(ymin - (ymax-ymin)*0.06, ymax + (ymax-ymin)*0.10)

ax2.plot(rsi_ser, color="black", lw=1.2)
ax2.axhline(70, color="red",   ls="--", lw=1)
ax2.axhline(30, color="green", ls="--", lw=1)
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.grid(True, ls=":", lw=0.3)
ax2.xaxis.set_major_formatter(DateFormatter('%b %Y'))
fig.autofmt_xdate()

st.pyplot(fig, clear_figure=False)

# -- Diagnostics & Latest Table
st.markdown(f"**Cyclicals in basket:** {cyc_live}")
st.markdown(f"**Defensives in basket:** {def_live}")

latest_rsi = rsi_ser.dropna().iloc[-1]
state = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
st.markdown("### Latest Readings")
st.table(pd.DataFrame({
    "Ratio Index": [f"{ratio_index.iloc[-1]:.1f}"],
    f"RSI({rsi_n})": [f"{latest_rsi:.1f}"],
    "Status": [state]
}))
