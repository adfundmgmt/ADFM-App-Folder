import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

# ------ Page config ------
st.set_page_config(layout="wide", page_title="Ratio Charts")

# ------ Sidebar: About & Inputs ------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "This dashboard visualizes regime shifts in US equities by comparing cyclical vs defensive sector performance and other key ratios.  \n"
    "- **Cyclical vs Defensive (Eq-Wt):** Ratio of cumulative returns for cyclical (XLK, XLI, XLF, XLC, XLY) vs defensive (XLP, XLE, XLV, XLRE, XLB, XLU) ETFs with 50/200-day MAs.  \n"
    "- **Preset Ratios:** SMH/IGV, QQQ/IWM, HYG/LQD, HYG/IEF.  \n"
    "- **Technicals:** 14-day RSI with 70/30 thresholds.  \n"
    "- **Custom Ratio:** Compare any two tickers over selected look-back."
)

st.sidebar.header("Look‑back")
spans = {"3 M": 90, "6 M": 180, "9 M": 270, "YTD": None,
         "1 Y": 365, "3 Y": 365*3, "5 Y": 365*5}
default_ix = list(spans.keys()).index("5 Y")
span_key = st.sidebar.selectbox("", list(spans.keys()), index=default_ix)

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
custom_t1 = st.sidebar.text_input("Ticker 1", "AAPL").strip().upper()
custom_t2 = st.sidebar.text_input("Ticker 2", "MSFT").strip().upper()

# ------ Main Title ------
st.title("Ratio Charts")

# ------ Date ranges ------
now = datetime.today()
# Full history (15 yrs) for MAs
hist_start = now - timedelta(days=365 * 15)
# Display window
if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    days = spans[span_key]
    disp_start = now - timedelta(days=days)

# ------ Data fetching ------
@st.cache_data(ttl=3600)
def fetch_closes(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # Prefer direct access; fall back to xs for MultiIndex
    try:
        closes = df['Close']
    except (KeyError, TypeError):
        closes = df.xs('Close', level=1, axis=1)
    # Ensure DataFrame
    if isinstance(closes, pd.Series):
        # single ticker
        closes = closes.to_frame(name=tickers[0])
    return closes.fillna(method='ffill').dropna()(method='ffill').dropna()

# Ticker lists
CYCLICALS = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
PRESETS = [("SMH","IGV"), ("QQQ","IWM"), ("HYG","LQD"), ("HYG","IEF")]

all_tickers = set(CYCLICALS + DEFENSIVES + [t for pair in PRESETS for t in pair])
if custom_t1 and custom_t2:
    all_tickers.update([custom_t1, custom_t2])

closes = fetch_closes(list(all_tickers), hist_start, now)

# ------ Computations ------
def compute_cumrets(df): return (1 + df.pct_change()).cumprod()

def compute_ratio(s1, s2, scale=100): return (s1/s2) * scale

def compute_rsi(s, window=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_dn = dn.rolling(window).mean()
    rs = ma_up/ma_dn
    return 100 - (100/(1+rs))

cumrets = compute_cumrets(closes)

# ------ Plot helper ------
def make_fig(ratio, title, ylab):
    ma50 = ratio.rolling(50).mean()
    ma200 = ratio.rolling(200).mean()
    view = ratio.loc[disp_start:]
    ma50_v = ma50.loc[disp_start:]
    ma200_v = ma200.loc[disp_start:]
    rsi = compute_rsi(ratio).loc[disp_start:]

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4), gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(view.index, view, 'k-', lw=1, label=title)
    ax1.plot(ma50.index, ma50, 'b-', lw=1, label='50-DMA')
    ax1.plot(ma200.index, ma200, 'r-', lw=1, label='200-DMA')
    ax1.set_xlim(disp_start, now)
    y = pd.concat([view, ma50.loc[disp_start:], ma200.loc[disp_start:]])
    mn, mx = y.min(), y.max(); pad=(mx-mn)*0.05
    ax1.set_ylim(mn-pad, mx+pad)
    ax1.set_ylabel(ylab); ax1.set_title(title)
    ax1.legend(fontsize=8); ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(rsi.index, rsi, 'k-', lw=1)
    ax2.axhline(70, color='red', ls=':', lw=1)
    ax2.axhline(30, color='green', ls=':', lw=1)
    ax2.set_xlim(disp_start, now); ax2.set_ylim(0,100)
    ax2.set_ylabel('RSI')
    if not rsi.empty:
        ax2.text(rsi.index[0], 72, 'Overbought', color='red', fontsize=7)
        ax2.text(rsi.index[0], 28, 'Oversold', color='green', fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

# ------ Rendering ------
# Cyclical vs Defensive
ratio_cd = compute_ratio(cumrets[CYCLICALS].mean(axis=1), cumrets[DEFENSIVES].mean(axis=1))
fig = make_fig(ratio_cd, 'Cyclicals / Defensives (Eq-Wt)', 'Ratio')
st.pyplot(fig, use_container_width=True)

st.markdown('---')
# Preset pairs
for t1, t2 in PRESETS:
    r = compute_ratio(cumrets[t1], cumrets[t2], scale=1)
    f = make_fig(r, f'{t1}/{t2}', f'{t1}/{t2}')
    st.pyplot(f, use_container_width=True)
    st.markdown('---')

# Custom
if custom_t1 and custom_t2:
    if custom_t1 in cumrets and custom_t2 in cumrets:
        r = compute_ratio(cumrets[custom_t1], cumrets[custom_t2])
        f = make_fig(r, f'{custom_t1}/{custom_t2}', f'{custom_t1}/{custom_t2}')
        st.pyplot(f, use_container_width=True)
    else:
        st.warning(f"Data not available for {custom_t1} or {custom_t2}.")
