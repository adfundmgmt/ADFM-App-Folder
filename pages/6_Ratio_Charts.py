import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------ Page config ------
st.set_page_config(layout="wide", page_title="Ratio Charts")

# ------ Sidebar: About & Inputs ------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "This dashboard visualizes regime shifts in US equities:\n\n"
    "- **Cyclical vs Defensive (Eq‑Wt):** Ratio of cum‑returns for cyclical (XLK, XLI, XLF, XLC, XLY) vs defensive (XLP, XLE, XLV, XLRE, XLB, XLU) ETFs, normalized to 100.\n"
    "- **Preset Ratios:** SMH/IGV, QQQ/IWM, HYG/LQD, HYG/IEF.\n"
    "- **Custom Ratio:** Compare any two tickers over your selected look‑back."
)

st.sidebar.header("Look‑back")
spans = {
    "3 Months": 90, "6 Months": 180, "9 Months": 270, "YTD": None,
    "1 Year": 365, "3 Years": 365*3, "5 Years": 365*5
}
default_ix = list(spans.keys()).index("5 Years")
span_key = st.sidebar.selectbox("", list(spans.keys()), index=default_ix)

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
custom_t1 = st.sidebar.text_input("Ticker 1", "NVDA").strip().upper()
custom_t2 = st.sidebar.text_input("Ticker 2", "SMH").strip().upper()

# ------ Main Title ------
st.title("Ratio Charts")

# ------ Date ranges ------
now = datetime.today()
hist_start = now - timedelta(days=365 * 15)  # for full MA history

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    days = spans[span_key]
    disp_start = now - timedelta(days=days)

# ------ Data fetching ------
@st.cache_data(ttl=3600)
def fetch_closes(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    try:
        closes = df['Close']
    except (KeyError, TypeError):
        closes = df.xs('Close', level=1, axis=1)
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    return closes.fillna(method='ffill').dropna()

# ------ Definitions ------
CYCLICALS   = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES  = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
PRESETS     = [("SMH","IGV"), ("QQQ","IWM"), ("HYG","LQD"), ("HYG","IEF")]
STATIC      = CYCLICALS + DEFENSIVES + [t for pair in PRESETS for t in pair]

def compute_cumrets(df):
    return (1 + df.pct_change()).cumprod()

def compute_ratio(s1, s2, scale=1):
    return (s1 / s2) * scale

def compute_rsi(s, window=14):
    d     = s.diff()
    up    = d.clip(lower=0)
    dn    = -d.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_dn = dn.rolling(window).mean()
    rs    = ma_up / ma_dn
    return 100 - (100 / (1 + rs))

def make_fig(ratio, title, ylab):
    ma50   = ratio.rolling(50).mean()
    ma200  = ratio.rolling(200).mean()
    view   = ratio.loc[disp_start:]
    rsi    = compute_rsi(ratio).loc[disp_start:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 4),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.plot(view.index, view,        'k-', lw=1, label=title)
    ax1.plot(ma50.index, ma50,        'b-', lw=1, label='50‑DMA')
    ax1.plot(ma200.index, ma200,      'r-', lw=1, label='200‑DMA')
    ax1.set_xlim(disp_start, now)

    y_all = pd.concat([view, ma50.loc[disp_start:], ma200.loc[disp_start:]])
    mn, mx = y_all.min(), y_all.max()
    pad    = (mx - mn) * 0.05
    ax1.set_ylim(mn - pad, mx + pad)

    ax1.set_ylabel(ylab)
    ax1.set_title(title)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.plot(rsi.index, rsi, 'k-', lw=1)
    ax2.axhline(70, ls=':', lw=1, color='red')
    ax2.axhline(30, ls=':', lw=1, color='green')
    ax2.set_xlim(disp_start, now)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    if not rsi.empty:
        ax2.text(rsi.index[0], 72, 'Overbought', fontsize=7, color='red')
        ax2.text(rsi.index[0], 28, 'Oversold',   fontsize=7, color='green')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig

# ------ 1) Static fetch & charts ------
closes_static   = fetch_closes(STATIC, hist_start, now)
cumrets_static  = compute_cumrets(closes_static)

# Cyclicals vs Defensives (×100 baseline)
ratio_cd = compute_ratio(
    cumrets_static[CYCLICALS].mean(axis=1),
    cumrets_static[DEFENSIVES].mean(axis=1),
    scale=100
)
st.pyplot(make_fig(ratio_cd, 'Cyclicals / Defensives (Eq‑Wt)', 'Ratio'),
           use_container_width=True)

st.markdown('---')
# Preset pairs (×1)
for t1, t2 in PRESETS:
    r = compute_ratio(cumrets_static[t1], cumrets_static[t2], scale=1)
    st.pyplot(make_fig(r, f'{t1}/{t2}', f'{t1}/{t2}'),
               use_container_width=True)
    st.markdown('---')

# ------ 2) Custom fetch & chart ------
if custom_t1 and custom_t2:
    try:
        closes_cust  = fetch_closes([custom_t1, custom_t2], hist_start, now)
        cumrets_cust = compute_cumrets(closes_cust)
        r_cust = compute_ratio(
            cumrets_cust[custom_t1],
            cumrets_cust[custom_t2],
            scale=1
        )
        st.pyplot(make_fig(r_cust, f'{custom_t1}/{custom_t2}', f'{custom_t1}/{custom_t2}'),
                   use_container_width=True)
    except Exception:
        st.warning(f"Data not available for {custom_t1}/{custom_t2}.")
