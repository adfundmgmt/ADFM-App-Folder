# ──────────────────────────────────────────────────────────────────────────────
#  S&P Cyclicals vs. Defensives (Equal‑Weight, Price‑Level) – Static Dashboard
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
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
        "Look‑back window",
        ("3 M","6 M","9 M","YTD","1 Y","3 Y","5 Y","10 Y"),
        index=4           # default 1 Y
    )
    rsi_n   = st.slider("RSI window", 5, 30, 14)
    ma_fast = st.slider("Short MA", 10, 100, 50)
    ma_slow = st.slider("Long  MA", 100, 300, 200)

# ── Date handling ─────────────────────────────────────────────────────────────
today = dt.date.today()
if period_choice == "YTD":
    start_date = dt.date(today.year, 1, 1)
else:
    key = period_choice.replace(" ", "").replace(" ", "")  # strip spaces/thin spaces
    start_date = today - relativedelta(**{
        "3M": dict(months=3), "6M":dict(months=6), "9M":dict(months=9),
        "1Y": dict(years=1),  "3Y":dict(years=3),  "5Y":dict(years=5),
        "10Y":dict(years=10)
    }[key])

# ── Downloader (robust to single vs multi‑index) ──────────────────────────────
@st.cache_data(ttl=12*3600)
def fetch_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            lvl0 = df.columns.get_level_values(0).unique()[-1]
            df = df.xs(lvl0, axis=1, level=0)
    else:
        df = df.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
    return df.dropna(axis=1, how="all")

prices = fetch_prices(ALL_TICKS, start_date, today)

# ── Build equal‑weight price‑level indices ────────────────────────────────────
rebased = prices / prices.iloc[0] * 100          # every ticker starts at 100
cyc_idx = rebased[CYCLICALS].mean(axis=1)
def_idx = rebased[DEFENSIVES].mean(axis=1)
ratio   = cyc_idx / def_idx

# ── Moving averages & RSI ─────────────────────────────────────────────────────
ma_fast_ser = ratio.rolling(ma_fast).mean()
ma_slow_ser = ratio.rolling(ma_slow).mean()

def rsi(series, n):
    delta = series.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - 100/(1+rs)

rsi_ser = rsi(ratio, rsi_n)

# ── Matplotlib figure ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios":[2,1], "hspace":0.05}
)

# Colour ratio by trend regime
above = ratio >= ma_slow_ser
ax1.plot(ratio.index, ratio.where(above), color="green", lw=2)
ax1.plot(ratio.index, ratio.where(~above), color="red",   lw=2)

# MAs
ax1.plot(ma_fast_ser, color="blue",      lw=1.4, ls="--", label=f"{ma_fast}‑day MA")
ax1.plot(ma_slow_ser, color="firebrick", lw=1.6, ls="-.", label=f"{ma_slow}‑day MA")

ax1.set_ylabel("Ratio")
ax1.set_title(f"Cyclicals vs Defensives — {period_choice} Window", fontsize=13, pad=10)
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, ls=":", lw=0.4)

# RSI pane
ax2.plot(rsi_ser, color="black", lw=1.2)
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
status = "Overbought" if latest_rsi>70 else "Oversold" if latest_rsi<30 else "Neutral"

st.markdown("### Latest Readings")
st.table(pd.DataFrame({
    "Ratio":[f"{ratio.iloc[-1]:.2f}"],
    f"RSI({rsi_n})":[f"{latest_rsi:.1f}"],
    "Status":[status]
}))
