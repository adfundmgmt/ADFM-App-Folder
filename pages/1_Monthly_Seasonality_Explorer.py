# seasonality_dashboard.py
# Streamlit app: robust downloads, correct Jan counts, clean CSV
# Adds first-half vs second-half overlay inside monthly bars

import datetime as dt
import io
import time
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
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

# -------------------------- Streamlit UI -------------------------- #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Explore seasonal patterns for any stock, index, or commodity.

        • Yahoo Finance primary source, FRED fallback for deep index history  
        • Median or mean monthly return, hit rate, min and max, error bars  
        • Inner stacked bars show 1H vs 2H of each month  
        """,
        unsafe_allow_html=True,
    )

# -------------------------- Data helpers -------------------------- #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
    """Robust yfinance fetch with exponential backoff. Returns Close series or None."""
    for n in range(retries):
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,  # single thread reduces 429s
        )
        if not df.empty and "Close" in df:
            return df["Close"]
        time.sleep(2 * (n + 1))
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    """
    Wrapper with fallback logic. Pads the start to ensure prior month-end exists
    so January in the first requested year is computable.
    """
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")

    # pad start by 45 days to include prior month-end
    start_pad_dt = pd.Timestamp(start) - pd.DateOffset(days=45)
    start_pad = start_pad_dt.strftime("%Y-%m-%d")

    # 1) Try Yahoo directly
    series = _yf_download(symbol, start_pad, end)
    if series is not None:
        return series

    # 2) Special-case SPY -> try the index
    if symbol == "SPY":
        series = _yf_download("^GSPC", start_pad, end)
        if series is not None:
            return series

    # 3) FRED fallbacks for major indices
    if pdr and symbol in FALLBACK_MAP:
        try:
            fred_tk = FALLBACK_MAP[symbol]
            df_fred = pdr.DataReader(fred_tk, "fred", start_pad, end)
            if df_fred is not None and not df_fred.empty and fred_tk in df_fred:
                return df_fred[fred_tk].rename("Close")
        except Exception:
            pass

    # 4) Give up
    return None


def _snap_on_or_after(dts: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the first index date on or after target. None if not found."""
    pos = dts.searchsorted(target)
    if pos < len(dts):
        return dts[pos]
    return None


def _month_boundaries(date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    For any date within a month, return:
    - prev month end (last trading day of prior month)
    - mid snap date for H1 end (first trading day on or after the 15th)
    - month end (last trading day of this month)
    These require the caller to provide a trading calendar (the prices index).
    """
    y, m = date.year, date.month
    first_of_month = pd.Timestamp(y, m, 1)
    fifteenth = pd.Timestamp(y, m, 15)
    next_month = (first_of_month + pd.offsets.MonthEnd(1)) + pd.offsets.Day(1)
    # placeholders; actual snaps are performed in half_month_returns
    return first_of_month, fifteenth, next_month - pd.offsets.Day(1)


def half_month_returns(prices: pd.Series) -> pd.DataFrame:
    """
    Compute monthly, first-half, and second-half arithmetic returns in percent.
    H1: from last trading day of prior month close to first trading day on/after the 15th.
    H2: from that mid snap to month-end.
    """
    px = prices.dropna().copy()
    if px.empty:
        return pd.DataFrame()

    dindex = px.index

    # Monthly endpoints
    month_end = px.resample("M").last()
    month_start_px = month_end.shift(1)
    month_ret = (month_end / month_start_px - 1.0).dropna() * 100.0  # percent
    month_period = month_ret.index.to_period("M")

    # Build H1/H2 endpoints by walking each month
    rows = []
    for per in month_period:
        y, m = per.year, per.month
        first_of_m = pd.Timestamp(y, m, 1)
        fifteenth = pd.Timestamp(y, m, 15)
        last_of_m = (first_of_m + pd.offsets.MonthEnd(1)).normalize()

        # prior month last trading day price
        prior_end_date = (first_of_m - pd.offsets.Day(1))
        prior_end_trade = dindex[dindex <= prior_end_date]
        if len(prior_end_trade) == 0:
            continue  # cannot compute January of very first year without padding
        prev_eom = prior_end_trade[-1]

        # mid snap: first trading day on or after the 15th
        mid = _snap_on_or_after(dindex, fifteenth)
        if mid is None or mid > last_of_m:
            continue

        # month end trading day
        eom_trade = dindex[dindex <= last_of_m]
        if len(eom_trade) == 0:
            continue
        eom = eom_trade[-1]

        # compute returns
        p0 = px.loc[prev_eom]
        p_mid = px.loc[mid]
        p_end = px.loc[eom]

        r_h1 = (p_mid / p0 - 1.0) * 100.0
        r_h2 = (p_end / p_mid - 1.0) * 100.0
        r_full = (p_end / p0 - 1.0) * 100.0

        rows.append(
            {
                "period": per,
                "month": m,
                "h1_ret": r_h1,
                "h2_ret": r_h2,
                "month_ret": r_full,
            }
        )

    hf = pd.DataFrame(rows)
    if hf.empty:
        return hf

    # aggregate by calendar month across years
    grouped = hf.groupby("month")
    out = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    out["median_ret"] = grouped["month_ret"].median()
    out["mean_ret"] = grouped["month_ret"].mean()
    out["hit_rate"] = grouped["month_ret"].apply(lambda x: (x > 0).mean() * 100)
    out["min_ret"] = grouped["month_ret"].min()
    out["max_ret"] = grouped["month_ret"].max()
    out["years_observed"] = grouped["month_ret"].count()
    out["h1_median"] = grouped["h1_ret"].median()
    out["h2_median"] = grouped["h2_ret"].median()
    out["label"] = MONTH_LABELS
    return out


def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute monthly seasonality stats, filtered to the requested [start_year, end_year],
    and include H1/H2 medians for overlay.
    """
    # filter to the explicit window first (we still need prior month-end; fetch_prices pads)
    px = prices[(prices.index.year >= start_year - 1) & (prices.index.year <= end_year)].copy()

    # main monthly stats
    monthly_end = px.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret = monthly_ret[(monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)]
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

    # half-month overlay stats computed on same constrained window
    half = half_month_returns(px)
    # keep only months fully inside [start_year, end_year] by recomputing using mask on input ensured above
    if not half.empty:
        stats["h1_median"] = half["h1_median"]
        stats["h2_median"] = half["h2_median"]

    return stats


def plot_seasonality(stats: pd.DataFrame, title: str, return_metric: str = "Median", show_halves: bool = True) -> io.BytesIO:
    col_map = {"Median": "median_ret", "Mean": "mean_ret"}
    ret_col = col_map[return_metric]

    # Drop months with insufficient data
    plot_df = stats.dropna(subset=[ret_col, "hit_rate", "min_ret", "max_ret"]).copy()
    labels = plot_df["label"].to_list()
    ret = plot_df[ret_col].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=200)

    # Main monthly bars with error bars
    bar_cols = ["mediumseagreen" if v >= 0 else "indianred" for v in ret]
    edge_cols = ["darkgreen" if v >= 0 else "darkred" for v in ret]
    yerr = np.abs(np.vstack([ret - min_ret, max_ret - ret]))

    x = np.arange(len(labels))
    ax1.bar(
        x, ret, width=0.78,
        color=bar_cols, edgecolor=edge_cols, linewidth=1.2,
        yerr=yerr, capsize=6, alpha=0.85, zorder=2,
        error_kw=dict(ecolor="gray", lw=1.6, alpha=0.7),
    )
    ax1.set_xticks(x, labels)

    ax1.set_ylabel(f"{return_metric} return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    pad = 0.1 * max(abs(np.nanmin(min_ret)), abs(np.nanmax(max_ret)))
    ax1.set_ylim(min(np.nanmin(min_ret), 0) - pad, max(np.nanmax(max_ret), 0) + pad)
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.6, alpha=0.7, zorder=1)

    # Half-month overlay: a centered, narrower stacked bar drawn on top
    if show_halves and "h1_median" in plot_df and "h2_median" in plot_df:
        h1 = plot_df["h1_median"].to_numpy(float)
        h2 = plot_df["h2_median"].to_numpy(float)

        # Inner stacked bars: width narrower so they appear "inside"
        inner_w = 0.48
        # Colors: darker for H1, lighter for H2; sign-aware
        def cpos(i): return ["#2e8b57", "#66cdaa"][i]  # seagreen shades
        def cneg(i): return ["#8b2e2e", "#e07a7a"][i]  # red shades

        inner_bottom = np.zeros_like(h1)
        for i in range(len(x)):
            color_h1 = cpos(0) if h1[i] >= 0 else cneg(0)
            color_h2 = cpos(1) if h2[i] >= 0 else cneg(1)

            # Draw H1 segment
            ax1.bar(
                x[i], h1[i], width=inner_w, bottom=0.0,
                color=color_h1, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3
            )
            # Draw H2 segment stacked atop H1
            ax1.bar(
                x[i], h2[i], width=inner_w, bottom=h1[i],
                color=color_h2, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3
            )

        # Legend patches
        legend_patches = [
            Patch(facecolor="#2e8b57", edgecolor="white", label="H1 (1–15)"),
            Patch(facecolor="#66cdaa", edgecolor="white", label="H2 (16–EOM)"),
        ]
        ax1.legend(handles=legend_patches, loc="upper left", frameon=False)

    # Hit rate on twin axis (unchanged)
    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=90, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(pad=2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# -------------------------- Main controls -------------------------- #
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
show_halves = st.toggle("Show 1H vs 2H inside the bars", value=True)

with st.spinner("Fetching and analyzing data..."):
    used_symbol = symbol
    prices = fetch_prices(symbol, start_date, end_date)
    if prices is None and symbol == "SPY":
        prices = fetch_prices("^GSPC", start_date, end_date)
        if prices is not None:
            used_symbol = "^GSPC"

if prices is None or prices.empty:
    st.error(f"No data found for '{symbol}' in the given date range. Try a different symbol or adjust the years.")
    st.stop()

# Clamp to first valid date to avoid empty early years in the fetched series
first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index (^GSPC) fallback for seasonality.")

stats = seasonal_stats(prices, int(start_year), int(end_year))
first_year, last_year = int(start_year), int(end_year)
ret_col = {"Median": "median_ret", "Mean": "mean_ret"}[metric]

# Guard for all-NaN rows when window is very short
valid_rows = stats.dropna(subset=[ret_col])
if valid_rows.empty:
    st.error("Insufficient data in the selected window to compute statistics.")
    st.stop()

best = stats.loc[valid_rows[ret_col].idxmax()]
worst = stats.loc[valid_rows[ret_col].idxmin()]

st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight
