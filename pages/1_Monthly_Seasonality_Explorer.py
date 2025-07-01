# seasonality_dashboard.py
# Streamlit app – hardened download logic for SPY/Yahoo hiccups
import datetime as dt
import io
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import PercentFormatter, MaxNLocator

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

FALLBACK_MAP = {
    "^GSPC": "SP500",
    "^DJI": "DJIA",
    "^IXIC": "NASDAQCOM",
}

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ────────────────────────── Streamlit UI ────────────────────────── #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Explore the seasonal patterns behind any stock, index, or commodity.**

        • Pulls data from Yahoo Finance (plus FRED fallback for deep index history)  
        • Median/mean monthly return, hit rate, min/max, volatility bands  
        • Green = positive month, red = negative, black diamonds = hit rate  
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────── Data helpers ────────────────────────── #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> pd.Series | None:
    """
    Robust yfinance fetch with exponential back-off and no caching.
    Returns a Close-price Series or None.
    """
    for n in range(retries):
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,  # single thread avoids extra 429s
        )
        if not df.empty:
            return df["Close"]
        time.sleep(2 * (n + 1))  # simple back-off
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> pd.Series | None:
    """
    Wrapper with fallback logic.  Cached for 1 h to avoid hammering YF.
    """
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")

    # 1) Try Yahoo directly
    series = _yf_download(symbol, start, end)
    if series is not None:
        return series

    # 2) Special-case SPY → use underlying index
    if symbol == "SPY":
        series = _yf_download("^GSPC", start, end)
        if series is not None:
            return series

    # 3) Fallback to FRED mappings for ^GSPC, ^DJI, ^IXIC
    if pdr and symbol in FALLBACK_MAP:
        try:
            fred_tk = FALLBACK_MAP[symbol]
            df_fred = pdr.DataReader(fred_tk, "fred", start, end)
            if not df_fred.empty:
                return df_fred[fred_tk].rename("Close")
        except Exception:
            pass

    # 4) Give up
    return None


def seasonal_stats(prices: pd.Series) -> pd.DataFrame:
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")
    grouped = monthly_ret.groupby(monthly_ret.index.month)

    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["median_ret"] = grouped.median()
    stats["mean_ret"] = grouped.mean()
    stats["hit_rate"] = grouped.apply(lambda x: (x > 0).mean() * 100)
    stats["min_ret"] = grouped.min()
    stats["max_ret"] = grouped.max()
    stats["years_observed"] = grouped.apply(lambda x: x.index.year.nunique())
    stats["label"] = MONTH_LABELS
    return stats


def plot_seasonality(stats: pd.DataFrame, title: str, return_metric: str = "Median") -> io.BytesIO:
    col_map = {"Median": "median_ret", "Mean": "mean_ret"}
    ret_col = col_map[return_metric]

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=200)
    labels = stats["label"]
    ret = stats[ret_col].to_numpy(float)
    hit = stats["hit_rate"].to_numpy(float)
    min_ret = stats["min_ret"].to_numpy(float)
    max_ret = stats["max_ret"].to_numpy(float)

    bar_cols = ["mediumseagreen" if v >= 0 else "indianred" for v in ret]
    edge_cols = ["darkgreen" if v >= 0 else "darkred" for v in ret]
    yerr = np.abs(np.vstack([ret - min_ret, max_ret - ret]))

    ax1.bar(
        labels, ret, width=0.8,
        color=bar_cols, edgecolor=edge_cols, linewidth=1.2,
        yerr=yerr, capsize=6, alpha=0.85, zorder=2,
        error_kw=dict(ecolor="gray", lw=1.6, alpha=0.7),
    )
    ax1.set_ylabel(f"{return_metric} return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter())
    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max()))
    ax1.set_ylim(min(min_ret.min(), 0) - pad, max(max_ret.max(), 0) + pad)
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.6, alpha=0.7, zorder=1)

    ax2 = ax1.twinx()
    ax2.scatter(labels, hit, marker="D", s=90, color="black", zorder=3)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter())

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(pad=2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# ────────────────────────── Main controls ────────────────────────── #
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC").upper()
with col2:
    start_year = st.number_input("Start year", value=2020, min_value=1900, max_value=dt.datetime.today().year)
with col3:
    end_year = st.number_input("End year", value=dt.datetime.today().year,
                               min_value=int(start_year), max_value=dt.datetime.today().year)

start_date = f"{int(start_year)}-01-01"
end_date = f"{int(end_year)}-12-31"

metric = st.radio("Select return metric for chart:", ["Median", "Mean"], horizontal=True)

with st.spinner("Fetching and analyzing data…"):
    prices = fetch_prices(symbol, start_date, end_date)

if prices is None or prices.empty:
    st.error(f"No data found for '{symbol}' in the given date range. "
             "Try a different symbol or adjust the years.")
    st.stop()

stats = seasonal_stats(prices)
first_year, last_year = prices.index[0].year, prices.index[-1].year
ret_col = {"Median": "median_ret", "Mean": "mean_ret"}[metric]
best = stats.loc[stats[ret_col].idxmax()]
worst = stats.loc[stats[ret_col].idxmin()]

st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            ⬆️ Best month: {best['label']} ({best[ret_col]:.2f}% | High {best['max_ret']:.2f}% | Low {best['min_ret']:.2f}%)
        </span>&nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            ⬇️ Worst month: {worst['label']} ({worst[ret_col]:.2f}% | High {worst['max_ret']:.2f}% | Low {worst['min_ret']:.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

buf = plot_seasonality(stats, f"{symbol} seasonality ({first_year}–{last_year})", metric)
st.image(buf, use_container_width=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("Download chart as PNG", buf.getvalue(),
                       file_name=f"{symbol}_seasonality_{first_year}_{last_year}.png")
with dl2:
    csv_df = stats[["label", "median_ret", "mean_ret", "hit_rate", "min_ret", "max_ret", "years_observed"]].copy()
    csv_df.columns = ["Month", "Median %", "Mean %", "Hit-Rate %", "Min %", "Max %", "Years"]
    st.download_button("Download stats (CSV)", csv_df.to_csv(index=False),
                       file_name=f"{symbol}_monthly_stats_{first_year}_{last_year}.csv")

st.caption("© 2025 AD Fund Management LP")
