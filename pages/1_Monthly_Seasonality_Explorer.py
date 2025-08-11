# seasonality_dashboard.py
# Streamlit app: robust downloads, correct Jan counts, clean CSV
# Adds intra-month split (1H vs 2H) inside each bar

import datetime as dt
import io
import time
import warnings

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
        • Green = positive month, red = negative, black diamonds = hit rate  
        • New: In-bar split shows average 1st-half vs 2nd-half contributions  
        """,
        unsafe_allow_html=True,
    )

# -------------------------- Data helpers -------------------------- #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> pd.Series | None:
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
def fetch_prices(symbol: str, start: str, end: str) -> pd.Series | None:
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


def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    """
    Build a table of per-month returns and their 1H / 2H decomposition using trading days.

    Returns DataFrame indexed by month period (YYYY-MM) with columns:
      total_ret, h1_ret, h2_ret
    All returns in percent.
    """
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret"])

    # Work with daily closes
    daily = prices.asfreq("B")  # business day index; NaNs for non-trading days
    daily = daily.ffill()        # forward-fill to handle non-trading business days

    out_rows = []
    # iterate over calendar months present in the series
    months = pd.period_range(prices.index.min().to_period("M"), prices.index.max().to_period("M"), freq="M")
    for m in months:
        # slice actual trading days belonging to this calendar month based on original index
        mask = (prices.index.to_period("M") == m)
        month_days = prices.loc[mask]
        if month_days.shape[0] < 3:
            continue

        first = month_days.iloc[0]
        last = month_days.iloc[-1]

        # midpoint based on trading-day count
        n = month_days.shape[0]
        mid_idx = (n // 2) - 1  # end of first half (0-based)
        if mid_idx < 0:
            continue
        mid_close = month_days.iloc[mid_idx]

        h1 = (mid_close / first - 1.0) * 100.0
        h2 = (last / mid_close - 1.0) * 100.0
        tot = (last / first - 1.0) * 100.0

        out_rows.append(
            {
                "period": m,              # Period[M]
                "total_ret": tot,
                "h1_ret": h1,
                "h2_ret": h2,
                "year": m.year,
                "month": m.month,
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df


def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute monthly seasonality stats, filtered to the requested [start_year, end_year].
    Also compute average first-half and second-half contribution shares per month.
    """
    # Monthly totals (for min/max, hit-rate, etc.)
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")

    # Intra-month halves
    halves = _intra_month_halves(prices)

    # Filter to requested window
    sel = (monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)
    monthly_ret = monthly_ret[sel]

    if halves.empty:
        grouped = monthly_ret.groupby(monthly_ret.index.month)
        stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
        stats["median_ret"] = grouped.median()
        stats["mean_ret"] = grouped.mean()
        stats["hit_rate"] = grouped.apply(lambda x: (x > 0).mean() * 100)
        stats["min_ret"] = grouped.min()
        stats["max_ret"] = grouped.max()
        stats["years_observed"] = grouped.apply(lambda x: x.index.year.nunique())
        stats["label"] = MONTH_LABELS
        # no halves available
        stats["h1_share"] = np.nan
        stats["h2_share"] = np.nan
        return stats

    halves = halves[(halves["year"] >= start_year) & (halves["year"] <= end_year)]

    # Shares are based on MEAN contributions to ensure additivity when scaling the bar
    # Avoid divide-by-zero; if avg total near 0, split 50/50.
    halves["total_epsilon"] = halves["h1_ret"] + halves["h2_ret"]
    grouped_total = monthly_ret.groupby(monthly_ret.index.month)
    grouped_halves = halves.groupby("month")

    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["median_ret"] = grouped_total.median()
    stats["mean_ret"] = grouped_total.mean()
    stats["hit_rate"] = grouped_total.apply(lambda x: (x > 0).mean() * 100)
    stats["min_ret"] = grouped_total.min()
    stats["max_ret"] = grouped_total.max()
    stats["years_observed"] = grouped_total.apply(lambda x: x.index.year.nunique())
    stats["label"] = MONTH_LABELS

    mean_h1 = grouped_halves["h1_ret"].mean()
    mean_h2 = grouped_halves["h2_ret"].mean()
    mean_tot_halves = mean_h1.add(mean_h2, fill_value=0)

    # contribution shares used to split the bar height (summing to 1)
    h1_share = pd.Series(0.5, index=stats.index)  # default
    valid = (mean_tot_halves.abs() > 1e-9)
    h1_share.loc[valid] = (mean_h1.loc[valid] / mean_tot_halves.loc[valid]).clip(-2, 2)  # keep sane bounds
    # if total mean near zero or invalid, keep 0.5/0.5
    stats["h1_share"] = h1_share
    stats["h2_share"] = 1.0 - h1_share

    return stats


def plot_seasonality(stats: pd.DataFrame, title: str, return_metric: str = "Median") -> io.BytesIO:
    """
    Plot monthly bars with error bars and hit rate.
    Bars are internally split into average 1H and 2H contributions (scaled to bar height).
    """
    col_map = {"Median": "median_ret", "Mean": "mean_ret"}
    ret_col = col_map[return_metric]

    # Drop months with insufficient data
    plot_df = stats.dropna(subset=[ret_col, "hit_rate", "min_ret", "max_ret"]).copy()
    labels = plot_df["label"].tolist()
    ret = plot_df[ret_col].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)
    h1_share = plot_df["h1_share"].fillna(0.5).to_numpy(float)
    h2_share = plot_df["h2_share"].fillna(0.5).to_numpy(float)

    # main figure
    fig, ax1 = plt.subplots(figsize=(11.5, 6.3), dpi=200)

    # Compute stacked heights so they always sum to the chosen bar metric
    h1_height = ret * h1_share
    h2_height = ret - h1_height  # ensures exact sum

    # choose colors by sign of each segment; two shades for halves
    def seg_colors(values):
        base_pos = np.array(["#62c38e"] * len(values))   # light green
        base_neg = np.array(["#e07a73"] * len(values))   # light red
        return np.where(values >= 0, base_pos, base_neg)

    def edge_colors(values)_
