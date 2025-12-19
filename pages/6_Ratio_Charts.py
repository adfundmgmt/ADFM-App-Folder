import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from ui_helpers import render_page_header

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("default")

# -------------- Page config --------------
st.set_page_config(layout="wide", page_title="Ratio Charts")
render_page_header(
    "Ratio Charts",
    subtitle="Trace cyclical vs defensive leadership and custom pairs.",
    description="Unified header spacing keeps ratio controls and plots tidy across the suite.",
)

# -------------- Sidebar --------------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Track structural shifts in US equity leadership through ratio charts.

    What this dashboard shows  
    • Cyclical vs Defensive regime trend using equal weight baskets  
    • Preset macro ratios such as SMH/IGV, SMH/FXY, QQQ/IWM, HYG/LQD, HYG/IEF  
    • Dynamic custom ratio between any two tickers  
    • Rolling 50 and 200 day averages to highlight trend direction  
    • RSI panel to assess momentum and exhaustion points

    Data: Yahoo Finance, updated hourly.
    """
)

st.sidebar.header("Look back")
spans = {
    "3 Months": 90,
    "6 Months": 180,
    "9 Months": 270,
    "YTD": None,
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "20 Years": 365 * 20
}
span_key = st.sidebar.selectbox("", list(spans.keys()),
                                index=list(spans.keys()).index("5 Years"))

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
custom_t1 = st.sidebar.text_input("Ticker 1", "NVDA").strip().upper()
custom_t2 = st.sidebar.text_input("Ticker 2", "SMH").strip().upper()

# -------------- Dates --------------
now = datetime.today()
hist_start = now - timedelta(days=365 * 25)

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_start = pd.Timestamp(now - timedelta(days=spans[span_key]))

# -------------- Data fetch --------------
@st.cache_data(ttl=3600)
def fetch_closes(tickers, start, end):
    """
    Returns a DataFrame of adjusted closes with one column per ticker.
    Uses yf.download with auto_adjust=True and handles single vs multi output.
    Retries once with period if needed.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    def _normalize(df, tickers_list):
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            cols = {}
            for t in tickers_list:
                try:
                    s = df[(t, "Close")].dropna()
                    if not s.empty:
                        cols[t] = s
                except Exception:
                    continue
            out = pd.DataFrame(cols)
        else:
            if "Close" in df.columns:
                out = pd.DataFrame({tickers_list[0]: df["Close"].dropna()})
            else:
                out = pd.DataFrame()
        return out.sort_index().ffill().dropna(how="all")

    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, group_by="ticker", threads=True
        )
        out = _normalize(raw, tickers)
        if not out.empty:
            return out
    except Exception:
        pass

    try:
        raw2 = yf.download(
            tickers, period="max",
            auto_adjust=True, progress=False, group_by="ticker", threads=True
        )
        out2 = _normalize(raw2, tickers)
        return out2
    except Exception:
        return pd.DataFrame()

# -------------- Definitions --------------
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]

PRESETS = [
    ("SMH", "IGV"),
    ("SMH", "FXY"),
    ("QQQ", "IWM"),
    ("HYG", "LQD"),
    ("HYG", "IEF"),
    ("SPHB", "SPLV"),
]

STATIC = CYCLICALS + DEFENSIVES + [t for a, b in PRESETS for t in (a, b)]

def compute_cumrets(df: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + df.pct_change()).cumprod()

def last_on_or_before(s: pd.Series, ts: pd.Timestamp) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    try:
        return float(s.loc[:ts].iloc[-1])
    except Exception:
        return float(s.iloc[0])

def compute_ratio(s1: pd.Series, s2: pd.Series, base_date: pd.Timestamp, base=100.0, scale=1.0) -> pd.Series:
    if s1.dropna().empty or s2.dropna().empty:
        return pd.Series(dtype=float)
    s1, s2 = s1.align(s2, join="inner")
    if s1.dropna().empty or s2.dropna().empty:
        return pd.Series(dtype=float)
    b1 = last_on_or_before(s1, base_date)
    b2 = last_on_or_before(s2, base_date)
    if not np.isfinite(b1) or not np.isfinite(b2):
        return pd.Series(dtype=float)
    n1 = s1 / b1 * base
    n2 = s2 / b2 * base
    return (n1 / n2) * scale

def rsi_wilder(s: pd.Series, window=14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    ma_dn = dn.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_fig(ratio: pd.Series, title: str, ylab: str):
    ratio = ratio.dropna()
    if ratio.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    ma50  = ratio.rolling(50, min_periods=1).mean()
    ma200 = ratio.rolling(200, min_periods=1).mean()
    view = ratio.loc[disp_start:]
    rsi  = rsi_wilder(ratio).loc[disp_start:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 4),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    ax1.plot(view.index, view, color="black", lw=1.2, label=title)
    ax1.plot(ma50.index, ma50, color="#1f77b4", lw=1.0, label="50 DMA")
    ax1.plot(ma200.index, ma200, color="#d62728", lw=1.0, label="200 DMA")
    ax1.set_xlim(view.index.min(), pd.Timestamp(now))
    ax1.margins(x=0)

    y_all = pd.concat([view, ma50.loc[disp_start:], ma200.loc[disp_start:]]).replace([np.inf, -np.inf], np.nan).dropna()
    mn, mx = y_all.min(), y_all.max()
    pad = (mx - mn) * 0.05 if mx != mn else (abs(mn) * 0.05 if mn != 0 else 1.0)
    ax1.set_ylim(mn - pad, mx + pad)

    ax1.set_ylabel(ylab)
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(rsi.index, rsi, color="black", lw=1.0)
    ax2.axhline(70, ls=":", lw=1, color="red")
    ax2.axhline(30, ls=":", lw=1, color="green")
    ax2.set_xlim(view.index.min(), pd.Timestamp(now))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    if not rsi.empty:
        ax2.text(rsi.index[0], 72, "Overbought", fontsize=7, color="red")
        ax2.text(rsi.index[0], 28, "Oversold", fontsize=7, color="green")
    ax2.grid(True, linestyle="--", alpha=0.3)

    return fig

# -------------- Static block: Cyclicals vs Defensives --------------
closes_static = fetch_closes(STATIC, hist_start, now)
if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

cumrets_stat = compute_cumrets(closes_static)

eq_cyc = cumrets_stat[CYCLICALS].mean(axis=1)
eq_def = cumrets_stat[DEFENSIVES].mean(axis=1)
r_cd   = compute_ratio(eq_cyc, eq_def, base_date=disp_start, base=100.0)

st.pyplot(make_fig(r_cd, "Cyclicals/Defensives (Eq Wt)", "Ratio"), use_container_width=True)

st.markdown("---")

# -------------- Preset ratios --------------
for a, b in PRESETS:
    if a in cumrets_stat.columns and b in cumrets_stat.columns:
        ratio = compute_ratio(cumrets_stat[a], cumrets_stat[b], base_date=disp_start, base=100.0)
        st.pyplot(make_fig(ratio, f"{a}/{b}", f"{a}/{b}"), use_container_width=True)
        st.markdown('---')

# -------------- Custom ratio --------------
if custom_t1 and custom_t2:
    closes_c = fetch_closes([custom_t1, custom_t2], hist_start, now)
    if not closes_c.empty and all(t in closes_c.columns for t in [custom_t1, custom_t2]):
        cum_c = compute_cumrets(closes_c)
        r_c = compute_ratio(cum_c[custom_t1], cum_c[custom_t2], base_date=disp_start, base=100.0)
        st.pyplot(make_fig(r_c, f"{custom_t1}/{custom_t2}", f"{custom_t1}/{custom_t2}"), use_container_width=True)
    else:
        st.warning(f"Data not available for {custom_t1}/{custom_t2}.")
