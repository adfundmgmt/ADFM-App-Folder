import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# --- Sector ETF mapping
SECTOR_ETFS = {
    "S&P 500": "SPY",
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# --- Page Setup
st.set_page_config("S&P 500 Sector Breadth & Rotation", layout="wide")
st.title("📊 S&P 500 Sector Breadth & Rotation Monitor")
st.caption("Built by AD Fund Management LP. All data via Yahoo Finance. For informational use only.")

# --- Sidebar
with st.sidebar:
    st.header("How to Use This Tool")
    st.markdown(
        """
        - **Breadth Table:** % of sector/market above key moving averages (20/50/200-day).
        - **Performance Heatmap:** Recent returns for all sectors.
        - **Relative Strength Chart:** Plot any sector vs. SPY to see leadership trends.
        - **Sector Rotation Quadrant:** Visualize which sectors are leading/lagging (3M vs 1W return).
        ---
        """
    )
    lookback = st.slider("Lookback window (days)", 90, 365, 180, 15)
    st.markdown("Data is end-of-day. ETF proxies are used for sectors (e.g., XLK, XLV).")

# --- Data Download & Cleaning
@st.cache_data(ttl=21600)
def fetch_prices(tickers, period="max"):
    px = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):  # just one ticker
        px = px.to_frame()
    px = px.ffill().bfill()
    return px

# Pull prices
tickers = list(SECTOR_ETFS.values())
prices = fetch_prices(tickers, period="max")
prices = prices.iloc[-lookback:]

# --- 1. Sector Breadth Table (% Above MA)
def pct_above_ma(series, window):
    ma = series.rolling(window).mean()
    return (series > ma).mean() * 100

ma_windows = [20, 50, 200]
breadth = {}
for name, ticker in SECTOR_ETFS.items():
    breadth[name] = {
        f"%>{w}D": pct_above_ma(prices[ticker], w) for w in ma_windows
    }

breadth_df = pd.DataFrame(breadth).T[[f"%>{w}D" for w in ma_windows]].round(1)
st.markdown("### 🟩 Breadth Table: % Above Moving Averages")
st.dataframe(
    breadth_df.style.format("{:.1f}%")
        .background_gradient(cmap="YlGn", axis=None),
    use_container_width=True,
)

# --- 2. Sector Performance Heatmap
def perf_calc(s, d):
    if len(s) < d:
        return np.nan
    return (s.iloc[-1] / s.iloc[-d] - 1) * 100

def ytd_perf(s):
    ytd_idx = s.index.get_loc(s[s.index.year == datetime.now().year].index[0])
    return (s.iloc[-1] / s.iloc[ytd_idx] - 1) * 100

perf_windows = [("1W", 5), ("1M", 21), ("3M", 63)]
perf = {}
for name, ticker in SECTOR_ETFS.items():
    s = prices[ticker]
    perf[name] = {label: perf_calc(s, days) for label, days in perf_windows}
    # Add YTD
    try:
        perf[name]["YTD"] = ytd_perf(s)
    except Exception:
        perf[name]["YTD"] = np.nan

perf_df = pd.DataFrame(perf).T[["1W", "1M", "3M", "YTD"]].round(2)
st.markdown("### 🔥 Sector Performance Heatmap")
st.dataframe(
    perf_df.style.background_gradient(
        cmap="RdYlGn", axis=0
    ).format("{:+.2f}%"),
    use_container_width=True,
)

# --- 3. Sector Rotation Quadrant (Momentum vs Short-Term)
st.markdown("### 🔄 Sector Rotation Quadrant")
fig1, ax1 = plt.subplots(figsize=(7.5, 7))
x = perf_df["3M"]
y = perf_df["1W"]
colors = plt.cm.tab10(np.linspace(0, 1, len(perf_df)))
for i, name in enumerate(perf_df.index):
    ax1.scatter(x[name], y[name], s=170, label=name, color=colors[i])
    ax1.annotate(name, (x[name], y[name]), fontsize=10, ha='center', va='bottom')
ax1.axhline(0, color="gray", ls="--", lw=1)
ax1.axvline(0, color="gray", ls="--", lw=1)
ax1.set_xlabel("3M Return (%)", fontsize=12)
ax1.set_ylabel("1W Return (%)", fontsize=12)
ax1.set_title("Sector Rotation (Momentum vs Short-term)", fontsize=14, weight="bold")
plt.tight_layout()
st.pyplot(fig1)

# --- 4. Relative Strength Chart (Sector/Market Ratio)
st.markdown("### 📈 Sector Relative Strength vs. S&P 500")
sector = st.selectbox("Select a sector to chart relative to S&P 500 (SPY):", [k for k in SECTOR_ETFS if k != "S&P 500"])
ratio = prices[SECTOR_ETFS[sector]] / prices["SPY"]
fig2, ax2 = plt.subplots(figsize=(11, 4))
ax2.plot(ratio.index, ratio, color="#7d3c98", lw=2.4)
ax2.set_title(f"{sector} / SPY Ratio (Relative Strength)", fontsize=14, weight="bold")
ax2.set_ylabel("Ratio")
ax2.grid(alpha=0.25)
st.pyplot(fig2)

# --- 5. Mini Trend Panel: Best/Worst Sectors (YTD)
best = perf_df["YTD"].idxmax()
worst = perf_df["YTD"].idxmin()
st.markdown(
    f"<div style='margin-top:1em;font-size:1.1em'>"
    f"<b>🟢 Best YTD Sector:</b> <span style='color:green'>{best} ({perf_df.loc[best, 'YTD']:+.2f}%)</span><br>"
    f"<b>🔴 Worst YTD Sector:</b> <span style='color:red'>{worst} ({perf_df.loc[worst, 'YTD']:+.2f}%)</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.caption("Data: Yahoo Finance. Calculations: AD Fund Management LP. All rights reserved.")

