import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ------ CSS to tighten sidebar spacing ------
st.markdown("""
    <style>
    section[data-testid="stSidebar"] h2 {
        margin-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox {
        margin-top: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------ ETF groups ------
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

# ------ Page config ------
st.set_page_config(layout="wide", page_title="S&P Cyclicals vs Defensives Dashboard")
st.title("S&P Cyclicals Relative to Defensives — Equal‑Weight")

# ------ Sidebar: About section and lookback ------
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This dashboard tracks the **relative performance of S&P cyclical and defensive sector ETFs** (equal-weighted) to visualize risk-on/risk-off regime shifts in US equities.

    - **Cyclical basket:** XLK, XLI, XLF, XLC, XLY  
    - **Defensive basket:** XLP, XLE, XLV, XLRE, XLB, XLU  
    - Shows cumulative return ratio, 50/200-day moving averages, and RSI (14).

    **Other ratio charts below**: Semis/Software, QQQ/IWM, Credit Spreads.
    """)
    st.header("Look‑back")
    spans = {"3 M":90,"6 M":180,"9 M":270,"YTD":None,"1 Y":365,
             "3 Y":365*3,"5 Y":365*5,"10 Y":365*10}
    default_ix = list(spans.keys()).index("5 Y")
    span_key = st.selectbox("", list(spans.keys()), index=default_ix)

# ------ Date handling ------
today = datetime.today()
hist_start = today - timedelta(days=365*10+220)
disp_start = datetime(today.year,1,1) if span_key=="YTD" else today - timedelta(days=spans[span_key])

# ------ Data fetch (cached) ------
@st.cache_data(ttl=3600, show_spinner="Fetching ETF prices…")
def fetch_etfs(tickers, start, end):
    df = yf.download(tickers, start, end, group_by="ticker", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df.xs('Close', level=1, axis=1)
    else:
        closes = df[["Close"]]
    closes = closes.fillna(method='ffill').dropna()
    return closes

def calc_ratio(etfs1, etfs2):
    b1 = (1 + fetch_etfs(etfs1, hist_start, today).pct_change()).cumprod().mean(axis=1)
    b2 = (1 + fetch_etfs(etfs2, hist_start, today).pct_change()).cumprod().mean(axis=1)
    df = pd.DataFrame({'b1': b1, 'b2': b2}).dropna()
    ratio = (df['b1'] / df['b2']) * 100
    return ratio

def calc_ratio_simple(ticker1, ticker2, mult=1.0):
    closes = fetch_etfs([ticker1, ticker2], hist_start, today)
    df = closes[[ticker1, ticker2]].dropna()
    ratio = (df[ticker1] / df[ticker2]) * mult
    return ratio

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_dn = dn.rolling(n).mean()
    rs = ma_up / ma_dn
    return 100 - 100 / (1 + rs)

def plot_ratio_panel_static(ratio, disp_start, title, ylab="Ratio", y_margin=0.12):
    mask = ratio.index >= disp_start
    ratio_disp = ratio[mask]
    ma50 = ratio.rolling(50).mean()[mask]
    ma200 = ratio.rolling(200).mean()[mask]
    rsi_panel = rsi(ratio)[mask]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(16, 5),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Top: Ratio & MAs
    ax1.plot(ratio_disp.index, ratio_disp, color="black", label=title, linewidth=2)
    ax1.plot(ma50.index, ma50, color="blue", label="50-DMA", linewidth=1)
    ax1.plot(ma200.index, ma200, color="red", label="200-DMA", linewidth=1)
    ax1.set_ylabel(ylab, fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title(title, fontsize=15, pad=8)
    ax1.grid(True, which='both', linestyle='--', alpha=0.28)
    ax1.margins(x=0)

    # Expand y-axis (add margin top and bottom)
    if not ratio_disp.empty:
        y_min, y_max = ratio_disp.min(), ratio_disp.max()
        y_range = y_max - y_min
        pad = y_margin * y_range if y_range else 1
        ax1.set_ylim(y_min - pad, y_max + pad)

    # Bottom: RSI
    ax2.plot(rsi_panel.index, rsi_panel, color="black", linewidth=1)
    ax2.axhline(70, color="red", linestyle="dotted", linewidth=1)
    ax2.axhline(30, color="green", linestyle="dotted", linewidth=1)
    if not rsi_panel.empty:
        ax2.text(rsi_panel.index[0], 72, "Overbought", color="red", fontsize=9, va="bottom")
        ax2.text(rsi_panel.index[0], 32, "Oversold", color="green", fontsize=9, va="top")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", fontsize=11)
    ax2.grid(True, which='both', linestyle='--', alpha=0.28)
    ax2.margins(x=0)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.12, hspace=0.13)

    st.pyplot(fig)
    plt.close(fig)

# ------ Panels: Only the ratios you requested ------

cyc_def_ratio = calc_ratio(CYCLICALS, DEFENSIVES)
plot_ratio_panel_static(cyc_def_ratio, disp_start, "Cyclicals / Defensives (Equal-Weight)", ylab="Relative Ratio")

smh_igv_ratio = calc_ratio_simple("SMH", "IGV")
plot_ratio_panel_static(smh_igv_ratio, disp_start, "SMH / IGV Relative Strength & RSI", ylab="SMH / IGV")

qqq_iwm_ratio = calc_ratio_simple("QQQ", "IWM")
plot_ratio_panel_static(qqq_iwm_ratio, disp_start, "QQQ / IWM Relative Strength & RSI", ylab="QQQ / IWM")

hyg_lqd_ratio = calc_ratio_simple("HYG", "LQD")
plot_ratio_panel_static(hyg_lqd_ratio, disp_start, "HYG / LQD (High Yield vs Investment Grade)", ylab="HYG / LQD")

hyg_ief_ratio = calc_ratio_simple("HYG", "IEF")
plot_ratio_panel_static(hyg_ief_ratio, disp_start, "HYG / IEF (High Yield vs Treasuries)", ylab="HYG / IEF")
