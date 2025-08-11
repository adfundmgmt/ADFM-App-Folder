# seasonality_dashboard.py
# Streamlit app: robust downloads, correct Jan counts, clean CSV
# Adds intra-month split (1H vs 2H) inside each bar

import datetime as dt
import io
import time
import warnings
from typing import Optional

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

FALLBACK_MAP = {"^GSPC": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM"}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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
        • New: In-bar split shows average first-half vs second-half contributions  
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
            threads=False,
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

    return None


def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    """
    Build a table of per-month returns and their 1H / 2H decomposition using trading days.
    Returns DataFrame indexed by month period (YYYY-MM) with columns: total_ret, h1_ret, h2_ret (percent).
    """
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret"])

    out_rows = []
    months = pd.period_range(prices.index.min().to_period("M"),
                             prices.index.max().to_period("M"), freq="M")
    for m in months:
        mask = (prices.index.to_period("M") == m)
        month_days = prices.loc[mask]
        if month_days.shape[0] < 3:
            continue

        first = month_days.iloc[0]
        last = month_days.iloc[-1]

        n = month_days.shape[0]
        mid_idx = (n // 2) - 1  # end of first half (0-based)
        if mid_idx < 0:
            continue
        mid_close = month_days.iloc[mid_idx]

        h1 = (mid_close / first - 1.0) * 100.0
        h2 = (last / mid_close - 1.0) * 100.0
        tot = (last / first - 1.0) * 100.0

        out_rows.append(
            {"period": m, "total_ret": tot, "h1_ret": h1, "h2_ret": h2, "year": m.year, "month": m.month}
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df


def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute monthly seasonality stats in [start_year, end_year].
    Also compute average first-half and second-half contribution shares per month.
    """
    # Monthly totals
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")

    # Intra-month halves
    halves = _intra_month_halves(prices)

    sel = (monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)
    monthly_ret = monthly_ret[sel]

    grouped_total = monthly_ret.groupby(monthly_ret.index.month)
    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["median_ret"] = grouped_total.median()
    stats["mean_ret"] = grouped_total.mean()
    stats["hit_rate"] = grouped_total.apply(lambda x: (x > 0).mean() * 100)
    stats["min_ret"] = grouped_total.min()
    stats["max_ret"] = grouped_total.max()
    stats["years_observed"] = grouped_total.apply(lambda x: x.index.year.nunique())
    stats["label"] = MONTH_LABELS

    if halves.empty:
        stats["h1_share"] = np.nan
        stats["h2_share"] = np.nan
        return stats

    halves = halves[(halves["year"] >= start_year) & (halves["year"] <= end_year)]
    grouped_halves = halves.groupby("month")
    mean_h1 = grouped_halves["h1_ret"].mean()
    mean_h2 = grouped_halves["h2_ret"].mean()
    mean_tot = mean_h1.add(mean_h2, fill_value=0)

    h1_share = pd.Series(0.5, index=stats.index)
    valid = (mean_tot.abs() > 1e-9) & mean_tot.index.to_series().isin(stats.index)
    h1_share.loc[valid] = (mean_h1.loc[valid] / mean_tot.loc[valid]).clip(-2, 2)
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

    plot_df = stats.dropna(subset=[ret_col, "hit_rate", "min_ret", "max_ret"]).copy()
    labels = plot_df["label"].tolist()
    ret = plot_df[ret_col].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)
    h1_share = plot_df["h1_share"].fillna(0.5).to_numpy(float)
    h2_share = plot_df["h2_share"].fillna(0.5).to_numpy(float)

    fig, ax1 = plt.subplots(figsize=(11.5, 6.3), dpi=200)

    h1_height = ret * h1_share
    h2_height = ret - h1_height
    x = np.arange(len(labels))

    def seg_colors(values):
        pos = np.array(["#62c38e"] * len(values))
        neg = np.array(["#e07a73"] * len(values))
        return np.where(values >= 0, pos, neg)

    def edge_colors(values):
        pos = np.array(["#1f7a4f"] * len(values))
        neg = np.array(["#8b1e1a"] * len(values))
        return np.where(values >= 0, pos, neg)

    ax1.bar(x, h1_height, width=0.8, color=seg_colors(h1_height),
            edgecolor=edge_colors(h1_height), linewidth=1.0, zorder=2, alpha=0.95)
    ax1.bar(x, h2_height, width=0.8, bottom=h1_height, color=seg_colors(h2_height),
            edgecolor=edge_colors(h2_height), linewidth=1.0, zorder=2, alpha=0.95)

    yerr = np.abs(np.vstack([ret - min_ret, max_ret - ret]))
    ax1.errorbar(x, ret, yerr=yerr, fmt="none", ecolor="gray",
                 elinewidth=1.6, alpha=0.7, capsize=6, zorder=3)

    ax1.set_xticks(x, labels)
    ax1.set_ylabel(f"{return_metric} return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max()))
    ax1.set_ylim(min(min_ret.min(), 0) - pad, max(max_ret.max(), 0) + pad)
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.6, alpha=0.7, zorder=1)

    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=90, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    legend = [
        Patch(facecolor="#62c38e", edgecolor="#1f7a4f", label="1st half"),
        Patch(facecolor="#3aa369", edgecolor="#1f7a4f", label="2nd half"),
    ]
    ax1.legend(handles=legend, loc="upper left", frameon=False)

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
    end_year = st.number_input(
        "End year",
        value=dt.datetime.today().year,
        min_value=int(start_year),
        max_value=dt.datetime.today().year,
    )

start_date = f"{int(start_year)}-01-01"
end_date = f"{int(end_year)}-12-31"

metric = st.radio("Select return metric for chart:", ["Median", "Mean"], horizontal=True)

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

first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index (^GSPC) fallback for seasonality.")

stats = seasonal_stats(prices, int(start_year), int(end_year))
first_year, last_year = int(start_year), int(end_year)
ret_col = {"Median": "median_ret", "Mean": "mean_ret"}[metric]

valid_rows = stats.dropna(subset=[ret_col])
if valid_rows.empty:
    st.error("Insufficient data in the selected window to compute statistics.")
    st.stop()

best = stats.loc[valid_rows[ret_col].idxmax()]
worst = stats.loc[valid_rows[ret_col].idxmin()]

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

buf = plot_seasonality(stats, f"{used_symbol} Seasonality ({first_year}-{last_year})", metric)
st.image(buf, use_container_width=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download chart as PNG",
        buf.getvalue(),
        file_name=f"{used_symbol}_seasonality_{first_year}_{last_year}.png"
    )
with dl2:
    csv_df = stats[["label","median_ret","mean_ret","hit_rate","min_ret","max_ret","years_observed","h1_share","h2_share"]].copy()
    csv_df.columns = ["Month","Median %","Mean %","Hit-Rate %","Min %","Max %","Years","Avg 1H Share","Avg 2H Share"]
    st.download_button(
        "Download stats (CSV)",
        csv_df.to_csv(index=False),
        file_name=f"{used_symbol}_monthly_stats_{first_year}_{last_year}.csv"
    )

st.caption("Note: In-bar split uses the average 1H and 2H contribution shares for that calendar month, scaled to the selected metric so segments sum exactly to the bar height. Shares are based on trading-day halves.")
st.caption("© 2025 AD Fund Management LP")
