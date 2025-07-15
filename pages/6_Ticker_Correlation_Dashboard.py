# ──────────────────────────────────────────────────────────────────────────────
#  Bloomberg‑style Cyclicals vs Defensives – Equal‑Weight, Price‑Level Ratio
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from dateutil.relativedelta import relativedelta
import datetime as dt

st.set_page_config(page_title="Cyc vs Def Dashboard", layout="wide")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight (Static)")

# ── Basket definitions ────────────────────────────────────────────────────────
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
ALL_TICKS  = CYCLICALS + DEFENSIVES

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    period = st.selectbox(
        "Look‑back window",
        ("3 M","6 M","9 M","YTD","1 Y","3 Y","5 Y","10 Y"),
        index=4                 # default 1 Y
    )
    rsi_n   = st.slider("RSI window", 5, 30, 14)
    ma_fast = st.slider("Short MA", 10, 100, 50)
    ma_slow = st.slider("Long  MA", 100, 300, 200)

# ── Date range ────────────────────────────────────────────────────────────────
today = dt.date.today()
if period == "YTD":
    start = dt.date(today.year, 1, 1)
else:
    key = period.replace(" ","").replace(" ","")   # remove thin / regular spaces
    start = today - relativedelta(**{
        "3M":dict(months=3),"6M":dict(months=6),"9M":dict(months=9),
        "1Y":dict(years=1), "3Y":dict(years=3), "5Y":dict(years=5),"10Y":dict(years=10)
    }[key])

# ── Robust downloader ─────────────────────────────────────────────────────────
@st.cache_data(ttl=12*3600)
def get_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else \
             df.xs(df.columns.get_level_values(0).unique()[-1], axis=1, level=0)
    else:
        df = df.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
    return df.dropna(axis=1, how="all")

prices = get_prices(ALL_TICKS, start, today)

# ── Rebase each ETF to 100 and build equal‑weight price indices ───────────────
rebased = prices / prices.iloc[0] * 100
cyc_idx = rebased[CYCLICALS ].mean(axis=1)
def_idx = rebased[DEFENSIVES].mean(axis=1)

# Ratio index starts at 100, like the screenshot
ratio = (cyc_idx / def_idx) * 100

# ── Moving averages & RSI ─────────────────────────────────────────────────────
ma_fast_ser = ratio.rolling(ma_fast).mean()
ma_slow_ser = ratio.rolling(ma_slow).mean()

def rsi(x, n):
    delta = x.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - 100/(1+rs)

rsi_ser = rsi(ratio, rsi_n)

# ── Matplotlib figure ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios":[2,1], "hspace":0.05}
)

# Trend‑colour the ratio: green when above long MA
trend_up = ratio >= ma_slow_ser
ax1.plot(ratio.index, ratio.where(trend_up),   color="green", lw=1.8)
ax1.plot(ratio.index, ratio.where(~trend_up),  color="red",   lw=1.8)

# Moving averages
ax1.plot(ma_fast_ser, color="blue",      lw=1.2, ls="--", label=f"{ma_fast}‑day MA")
ax1.plot(ma_slow_ser, color="firebrick", lw=1.4, ls="-.", label=f"{ma_slow}‑day MA")

ax1.set_ylim(100, 170)                          # mimic screenshot y‑axis band
ax1.set_ylabel("Ratio Index")
ax1.set_title(f"Cyclicals vs Defensives — {period} Window", fontsize=13, pad=10)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, ls=":", lw=0.4)

# RSI pane
ax2.plot(rsi_ser, color="black", lw=1.1)
ax2.axhline(70, color="red",   ls="--", lw=1)
ax2.axhline(30, color="green", ls="--", lw=1)
ax2.set_ylabel("RSI")
ax2.set_ylim(0,100)
ax2.grid(True, ls=":", lw=0.4)
ax2.xaxis.set_major_formatter(DateFormatter('%b %Y'))
fig.autofmt_xdate()

st.pyplot(fig, clear_figure=False)

# ── Status table ──────────────────────────────────────────────────────────────
latest_rsi = rsi_ser.dropna().iloc[-1]
state = "Overbought" if latest_rsi>70 else "Oversold" if latest_rsi<30 else "Neutral"
st.markdown("### Latest Readings")
st.table(pd.DataFrame({
    "Ratio Index":[f"{ratio.iloc[-1]:.1f}"],
    f"RSI({rsi_n})":[f"{latest_rsi:.1f}"],
    "Status":[state]
}))
