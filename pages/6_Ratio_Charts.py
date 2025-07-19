import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

# ------ CSS to tighten sidebar spacing ------
st.markdown("""
    <style>
    section[data-testid=\"stSidebar\"] h2 {
        margin-bottom: 0.25rem !important;
    }
    section[data-testid=\"stSidebar\"] .stSelectbox {
        margin-top: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------ Configuration ------
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
OTHER_PAIRS = [("SMH","IGV"), ("QQQ","IWM"), ("HYG","LQD"), ("HYG","IEF")]
STATS_YEARS = 15  # lookback to ensure full MA history

st.set_page_config(layout="wide", page_title="Ratio Charts")
st.title("Ratio Charts")

# ------ Sidebar Inputs ------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        "Tracks relative performance of S&P cyclical vs defensive sector ETFs (equal-weighted), "
        "plus other ratio charts with RSI & moving averages."
    )

    st.header("Look‑back")
    spans = {"3 M": 90, "6 M": 180, "9 M": 270, "YTD": None,
             "1 Y": 365, "3 Y": 365*3, "5 Y": 365*5, "10 Y": 365*10}
    default_ix = list(spans.keys()).index("5 Y")
    span_key = st.selectbox("", list(spans.keys()), index=default_ix)

    st.markdown("---")
    st.subheader("Custom Ratio")
    custom_t1 = st.text_input("Ticker 1", "AAPL").strip().upper()
    custom_t2 = st.text_input("Ticker 2", "MSFT").strip().upper()

# ------ Date Ranges ------
now = datetime.today()
# Full history for moving averages
hist_start = now - timedelta(days=365 * STATS_YEARS)
# Display period
if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_days = spans[span_key]
    disp_start = now - timedelta(days=disp_days)

# ------ Data Fetch ------
@st.cache_data(ttl=3600)
def fetch_close_data(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)
    # Attempt to extract Close prices
    if 'Close' in df.columns:
        closes = df['Close']
    elif isinstance(df.columns, pd.MultiIndex) and 'Close' in df.columns.get_level_values(1):
        closes = df.xs('Close', level=1, axis=1)
    else:
        raise KeyError("Unable to find 'Close' in downloaded data columns")
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    return closes.fillna(method='ffill').dropna()

# Collect tickers
tickers = set(CYCLICALS + DEFENSIVES + [t for pair in OTHER_PAIRS for t in pair])
if custom_t1 and custom_t2:
    tickers |= {custom_t1, custom_t2}
closes = fetch_close_data(list(tickers), hist_start, now)

# ------ Analytics Helpers ------
def compute_cumrets(df: pd.DataFrame) -> pd.DataFrame:
    return (1 + df.pct_change()).cumprod()

def compute_ratio(s1: pd.Series, s2: pd.Series, scale: float = 100.0) -> pd.Series:
    return (s1 / s2) * scale


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_dn = dn.rolling(window).mean()
    rs = ma_up / ma_dn
    return 100 - (100 / (1 + rs))

cumrets = compute_cumrets(closes)

# ------ Plotting ------
def make_ratio_figure(ratio: pd.Series, title: str, ylab: str) -> plt.Figure:
    ma50_full = ratio.rolling(50).mean()
    ma200_full = ratio.rolling(200).mean()
    mask = ratio.index >= disp_start
    data = ratio.loc[mask]
    ma50 = ma50_full.loc[mask]
    ma200 = ma200_full.loc[mask]
    rsi_vals = compute_rsi(ratio).loc[mask]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 4),
        gridspec_kw={'height_ratios': [3, 1]}
    )
        # Top chart: plot MAs full history but clip view to display window
    ax1.plot(ma50_full.index, ma50_full, color='blue', linewidth=1.0, label='50-DMA')
    ax1.plot(ma200_full.index, ma200_full, color='red', linewidth=1.0, label='200-DMA')
    ax1.plot(data.index, data, color='black', linewidth=1.0, label=title)

    # Clip view to display window
    ax1.set_xlim(disp_start, now)
    # Y-limits based on full MA and data
    y_all = pd.concat([data, ma50_full.loc[data.index.min():], ma200_full.loc[data.index.min():]])
    y_min, y_max = y_all.min(), y_all.max()
    pad = (y_max - y_min) * 0.05
    ax1.set_ylim(y_min - pad, y_max + pad)

    ax1.set_title(title)
    ax1.set_ylabel(ylab)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.margins(x=0)

    # RSI panel
    ax2.plot(rsi_vals.index, rsi_vals, color='black', linewidth=1.0)
    ax2.axhline(70, color='red', linestyle=':', linewidth=1.0)
    ax2.axhline(30, color='green', linestyle=':', linewidth=1.0)
    # extend x-axis on RSI also
    ax2.set_xlim(hist_start, now)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    if not rsi_vals.empty:
        ax2.text(rsi_vals.index[0], 72, 'Overbought', color='red', fontsize=7, va='bottom')
        ax2.text(rsi_vals.index[0], 28, 'Oversold', color='green', fontsize=7, va='top')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.margins(x=0)

    plt.tight_layout()
    return fig

# ------ Render Chart Panels ------
ratio_cd = compute_ratio(
    cumrets[CYCLICALS].mean(axis=1),
    cumrets[DEFENSIVES].mean(axis=1)
)
fig1 = make_ratio_figure(ratio_cd, 'Cyclicals / Defensives (Eq-Wt)', 'Ratio')
st.pyplot(fig1, use_container_width=True)

st.markdown('---')
for t1, t2 in OTHER_PAIRS:
    if t1 in cumrets and t2 in cumrets:
        r = compute_ratio(cumrets[t1], cumrets[t2], scale=1.0)
        f = make_ratio_figure(r, f'{t1} / {t2}', f'{t1}/{t2}')
        st.pyplot(f, use_container_width=True)
        st.markdown('---')

if custom_t1 and custom_t2:
    if custom_t1 in cumrets and custom_t2 in cumrets:
        r_custom = compute_ratio(cumrets[custom_t1], cumrets[custom_t2])
        f_custom = make_ratio_figure(
            r_custom, f'{custom_t1} / {custom_t2}', f'{custom_t1}/{custom_t2}'
        )
        st.pyplot(f_custom, use_container_width=True)
    else:
        st.warning(f"Data not available for {custom_t1} or {custom_t2}.")

with st.expander("Download Chart"):
    buf = io.BytesIO()
    fig1.savefig(buf, format='png')
    st.download_button(
        "Download Cyclicals/Defensives", buf.getvalue(),
        "cyc_def.png", "image/png"
    )
