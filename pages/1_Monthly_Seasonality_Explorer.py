# seasonality_dashboard.py
# Monthly seasonality with accurate 1H vs 2H contributions
# Bars = mean(1H) + mean(2H). Colors by each half's sign. 2H is hatched.
# If "Median" is selected, a thin marker shows the monthly median total.

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
        • Bars = mean of first-half + mean of second-half returns (accurate)  
        • First half solid, second half hatched; each half colored by its own sign  
        • Error bars show min and max monthly total returns; black diamonds = hit rate  
        • If 'Median' is chosen, a thin marker shows the monthly median total
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
    Returns DataFrame indexed by month period with columns: total_ret, h1_ret, h2_ret (percent).
    """
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

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
    Bars are built from mean first-half and mean second-half, summed exactly.
    """
    # Monthly totals
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")

    # Intra-month halves
    halves = _intra_month_halves(prices)

    sel = (monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)
    monthly_ret = monthly_ret[sel]

    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    grouped_total = monthly_ret.groupby(monthly_ret.index.month)
    stats["mean_total"] = grouped_total.mean()
    stats["median_total"] = grouped_total.median()
    stats["hit_rate"] = grouped_total.apply(lambda x: (x > 0).mean() * 100)
    stats["min_ret"] = grouped_total.min()
    stats["max_ret"] = grouped_total.max()
    stats["years_observed"] = grouped_total.apply(lambda x: x.index.year.nunique())
    stats["label"] = MONTH_LABELS

    # Halves averaged by calendar month
    if halves.empty:
        stats["mean_h1"] = np.nan
        stats["mean_h2"] = np.nan
        return stats

    halves = halves[(halves["year"] >= start_year) & (halves["year"] <= end_year)]
    grouped_halves = halves.groupby("month")
    stats["mean_h1"] = grouped_halves["h1_ret"].mean()
    stats["mean_h2"] = grouped_halves["h2_ret"].mean()

    # Ensure total consistency from halves
    stats["mean_total_from_halves"] = stats["mean_h1"].fillna(0) + stats["mean_h2"].fillna(0)

    return stats


def plot_seasonality(stats: pd.DataFrame, title: str, main_metric: str = "Mean",
                     show_labels: bool = True, label_threshold: float = 0.15) -> io.BytesIO:
    """
    Plot monthly bars with error bars and hit rate.
    Bars are the sum of mean_h1 and mean_h2. Each half is colored by its own sign.
    If main_metric == "Median", draw a thin marker at the median total for transparency.
    """
    # Clean rows
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "hit_rate", "min_ret", "max_ret"]).copy()

    labels = plot_df["label"].tolist()
    mean_h1 = plot_df["mean_h1"].to_numpy(float)
    mean_h2 = plot_df["mean_h2"].to_numpy(float)
    mean_total_from_halves = plot_df["mean_total_from_halves"].to_numpy(float)
    median_total = plot_df["median_total"].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    # Center for error bars and annotations
    if main_metric == "Mean":
        center_total = mean_total_from_halves
    else:
        center_total = median_total

    fig, ax1 = plt.subplots(figsize=(11.8, 6.5), dpi=200)

    x = np.arange(len(labels))
    # first half heights and second half heights (stacked)
    h1 = mean_h1
    h2 = mean_h2
    totals = mean_total_from_halves

    # segment colors by sign
    def seg_face(values):
        return np.where(values >= 0, "#62c38e", "#e07a73")

    def seg_edge(values):
        return np.where(values >= 0, "#1f7a4f", "#8b1e1a")

    # Draw first half (solid)
    ax1.bar(
        x, h1, width=0.8,
        color=seg_face(h1), edgecolor=seg_edge(h1), linewidth=1.0, zorder=2, alpha=0.95
    )

    # Draw second half (hatched), stacked on top of first half
    bars2 = ax1.bar(
        x, h2, width=0.8, bottom=h1,
        color=seg_face(h2), edgecolor=seg_edge(h2), linewidth=1.0, zorder=2, alpha=0.95,
        hatch="///"
    )
    for rect in bars2:
        rect.set_hatch("///")

    # Error bars: min/max relative to selected center_total
    yerr = np.abs(np.vstack([center_total - min_ret, max_ret - center_total]))
    ax1.errorbar(x, center_total, yerr=yerr, fmt="none", ecolor="gray",
                 elinewidth=1.6, alpha=0.7, capsize=6, zorder=3)

    # If main metric is Median, draw a thin marker at the median total for each bar
    if main_metric == "Median":
        ax1.hlines(median_total, x - 0.35, x + 0.35, colors="black", linestyles="-", linewidth=1.0, zorder=4)

    # Axes cosmetics
    ax1.set_xticks(x, labels)
    ax1.set_ylabel(f"{main_metric} return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max()))
    ymin = min(min_ret.min(), totals.min(), 0) - pad
    ymax = max(max_ret.max(), totals.max(), 0) + pad
    ax1.set_ylim(ymin, ymax)
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.6, alpha=0.7, zorder=1)

    # Hit rate on twin axis
    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=90, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Numeric labels for clarity
    if show_labels:
        for xi, a, b, t in zip(x, h1, h2, totals):
            # 1H label at the top of the first segment
            if abs(a) >= label_threshold:
                ax1.text(xi, a - (0.02 if a < 0 else 0) , f"1H {a:+.1f}%", ha="center",
                         va="bottom" if a >= 0 else "top", fontsize=8)
            # 2H label at the top of the second segment
            top_of_2h = a + b
            if abs(b) >= label_threshold:
                ax1.text(xi, top_of_2h - (0.02 if top_of_2h < 0 else 0), f"2H {b:+.1f}%",
                         ha="center", va="bottom" if top_of_2h >= 0 else "top", fontsize=8)
            # Total label slightly above the center metric marker
            ax1.text(xi, t, f"Tot {t:+.1f}%", ha="center",
                     va="bottom" if t >= 0 else "top", fontsize=8, color="dimgray")

    # Legend
    legend = [
        Patch(facecolor="#62c38e", edgecolor="#1f7a4f", label="Positive segment"),
        Patch(facecolor="#e07a73", edgecolor="#8b1e1a", label="Negative segment"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Second half (hatched)"),
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

metric = st.radio("Select main metric for center and label:", ["Mean", "Median"], horizontal=True)
show_labels = st.checkbox("Show numeric 1H, 2H, Total labels", value=True)

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

# Guard
if stats.dropna(subset=["mean_h1", "mean_h2"]).empty:
    st.error("Insufficient data in the selected window to compute statistics.")
    st.stop()

# Best / worst by mean totals from halves
stats_valid = stats.dropna(subset=["mean_total_from_halves"])
best_idx = stats_valid["mean_total_from_halves"].idxmax()
worst_idx = stats_valid["mean_total_from_halves"].idxmin()
best = stats.loc[best_idx]
worst = stats.loc[worst_idx]

st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            ⬆️ Best month (by mean): {best['label']} ({best['mean_total_from_halves']:.2f}% | High {best['max_ret']:.2f}% | Low {best['min_ret']:.2f}%)
        </span>&nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            ⬇️ Worst month (by mean): {worst['label']} ({worst['mean_total_from_halves']:.2f}% | High {worst['max_ret']:.2f}% | Low {worst['min_ret']:.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

buf = plot_seasonality(
    stats,
    f"{used_symbol} Seasonality ({int(start_year)}-{int(end_year)})",
    main_metric=metric,
    show_labels=show_labels,
)
st.image(buf, use_container_width=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "Download chart as PNG",
        buf.getvalue(),
        file_name=f"{used_symbol}_seasonality_{int(start_year)}_{int(end_year)}.png"
    )
with dl2:
    csv_df = stats[[
        "label","mean_h1","mean_h2","mean_total_from_halves",
        "median_total","hit_rate","min_ret","max_ret","years_observed"
    ]].copy()
    csv_df.columns = [
        "Month","Mean 1H %","Mean 2H %","Mean Total %",
        "Median Total %","Hit-Rate %","Min %","Max %","Years"
    ]
    st.download_button(
        "Download stats (CSV)",
        csv_df.to_csv(index=False),
        file_name=f"{used_symbol}_monthly_stats_{int(start_year)}_{int(end_year)}.csv"
    )

st.caption("Bars equal mean(1H) + mean(2H). First half is solid; second half is hatched. Each half is green if positive, red if negative. Error bars use min and max of total monthly returns. If 'Median' is selected, the thin marker shows the monthly median total.")
st.caption("© 2025 AD Fund Management LP")
