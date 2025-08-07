import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("default")

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Ratio Charts")
st.title("Ratio Charts")

# ---------------- Sidebar ----------------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    "This dashboard visualizes regime shifts in US equities:\n\n"
    "- **Cyclical vs Defensive (Eq Wt):** Ratio of cumulative returns for cyclical (XLK, XLI, XLF, XLC, XLY)"
    " vs defensive (XLP, XLE, XLV, XLRE, XLB, XLU) ETFs, normalized to 100.\n"
    "- **Preset Ratios:** SMH/IGV, QQQ/IWM, HYG/LQD, HYG/IEF.\n"
    "- **Custom Ratio:** Compare any two tickers over your selected look back period."
)

st.sidebar.header("Look back")
spans = {
    "3 Months": 90, "6 Months": 180, "9 Months": 270, "YTD": None,
    "1 Year": 365, "3 Years": 365 * 3, "5 Years": 365 * 5
}
default_ix = list(spans.keys()).index("5 Years")
span_key = st.sidebar.selectbox("", list(spans.keys()), index=default_ix)

st.sidebar.markdown("---")
st.sidebar.header("Custom Ratio")
custom_t1 = st.sidebar.text_input("Ticker 1", "NVDA").strip().upper()
custom_t2 = st.sidebar.text_input("Ticker 2", "SMH").strip().upper()

# ---------------- Dates ----------------
now = datetime.today()
hist_start = now - timedelta(days=365 * 15)  # enough history for long MAs

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_start = pd.Timestamp(now - timedelta(days=spans[span_key]))

# ---------------- Data fetch ----------------
@st.cache_data(ttl=3600)
def fetch_closes(tickers, start, end):
    """
    Return Adjusted Close prices as a DataFrame with columns per ticker.
    Uses auto_adjust=True, so Close is already adjusted.
    Robust to single or multi index outputs.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker"
    )

    # Normalize to flat columns of Close (which is adjusted)
    closes = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = df[(t, "Close")].dropna()
                if not s.empty:
                    closes[t] = s
            except Exception:
                continue
    else:
        if "Close" in df:
            colname = tickers[0] if len(tickers) == 1 else "Close"
            s = df["Close"].dropna()
            if not s.empty:
                closes[colname] = s

    if not closes:
        return pd.DataFrame()

    out = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
    return out

# ---------------- Definitions ----------------
CYCLICALS  = ["XLK", "XLI", "XLF", "XLC", "XLY"]
DEFENSIVES = ["XLP", "XLE", "XLV", "XLRE", "XLB", "XLU"]
PRESETS    = [("SMH", "IGV"), ("QQQ", "IWM"), ("HYG", "LQD"), ("HYG", "IEF")]
STATIC     = CYCLICALS + DEFENSIVES + [t for pair in PRESETS for t in pair]

def compute_cumrets(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative return index starting at 1.0 per column."""
    return (1.0 + df.pct_change()).cumprod()

def last_on_or_before(s: pd.Series, ts: pd.Timestamp) -> float:
    """Value at the last index on or before ts."""
    s = s.dropna()
    if s.empty:
        return np.nan
    # s.loc[:ts] handles no exact match; take last available
    try:
        return float(s.loc[:ts].iloc[-1])
    except Exception:
        return float(s.iloc[0])

def compute_ratio(s1: pd.Series, s2: pd.Series, base_date: pd.Timestamp, base=100.0, scale=1.0) -> pd.Series:
    """
    Normalize both series to base at base_date using the last available value
    on or before base_date, then return the ratio series scaled to `base`.
    Works even if indices do not perfectly align.
    """
    if s1.dropna().empty or s2.dropna().empty:
        return pd.Series(dtype=float)

    # Align indices
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
    # Moving averages
    ma50  = ratio.rolling(50, min_periods=1).mean()
    ma200 = ratio.rolling(200, min_periods=1).mean()

    view = ratio.loc[disp_start:]
    rsi  = rsi_wilder(ratio).loc[disp_start:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 4),
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )

    ax1.plot(view.index, view, color="black", lw=1.2, label=title)
    ax1.plot(ma50.index, ma50, color="#1f77b4", lw=1.0, label="50 DMA")
    ax1.plot(ma200.index, ma200, color="#d62728", lw=1.0, label="200 DMA")

    ax1.set_xlim(view.index.min() if len(view) else disp_start, pd.Timestamp(now))
    ax1.margins(x=0)

    y_all = pd.concat([view, ma50.loc[disp_start:], ma200.loc[disp_start:]]).replace([np.inf, -np.inf], np.nan).dropna()
    if not y_all.empty:
        mn, mx = y_all.min(), y_all.max()
        pad = (mx - mn) * 0.05 if mx != mn else (abs(mn) * 0.05 if mn != 0 else 1)
        ax1.set_ylim(mn - pad, mx + pad)

    ax1.set_ylabel(ylab)
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(rsi.index, rsi, color="black", lw=1.0)
    ax2.axhline(70, ls=":", lw=1, color="red")
    ax2.axhline(30, ls=":", lw=1, color="green")
    ax2.set_xlim(view.index.min() if len(view) else disp_start, pd.Timestamp(now))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    if not rsi.empty:
        ax2.text(rsi.index[0], 72, "Overbought", fontsize=7, color="red")
        ax2.text(rsi.index[0], 28, "Oversold", fontsize=7, color="green")
    ax2.grid(True, linestyle="--", alpha=0.3)

    return fig

# ---------------- Static block: Cyclicals vs Defensives ----------------
closes_static = fetch_closes(STATIC, hist_start, now)
if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

cumrets_stat = compute_cumrets(closes_static)

# Equal weight aggregates built from cu
