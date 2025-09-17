# seasonality_dashboard.py
# Mean-based seasonality with accurate 1H vs 2H contributions
# Partial months excluded using last BUSINESS day logic.

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
from matplotlib import gridspec

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
        • Bars = mean of first-half + mean of second-half returns  
        • First half solid, second half hatched; each half colored by its own sign  
        • Error bars show min and max of total monthly returns; black diamonds show hit rate  
        • Partial months excluded using last business day to avoid MTD bleed
        """,
        unsafe_allow_html=True,
    )

# -------------------------- Data helpers -------------------------- #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
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
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")

    # pad start by 45 days to include prior month-end
    start_pad_dt = pd.Timestamp(start) - pd.DateOffset(days=45)
    start_pad = start_pad_dt.strftime("%Y-%m-%d")

    series = _yf_download(symbol, start_pad, end)
    if series is not None:
        return series

    if symbol == "SPY":
        series = _yf_download("^GSPC", start_pad, end)
        if series is not None:
            return series

    if pdr and symbol in FALLBACK_MAP:
        try:
            fred_tk = FALLBACK_MAP[symbol]
            df_fred = pdr.DataReader(fred_tk, "fred", start_pad, end)
            if df_fred is not None and not df_fred.empty and fred_tk in df_fred:
                return df_fred[fred_tk].rename("Close")
        except Exception:
            pass

    return None


def last_bmonth_end(period: pd.Period) -> pd.Timestamp:
    """Last business day for a period like 2025-08."""
    # Take calendar month-end then roll to last business day
    return (period.to_timestamp("M") + pd.offsets.BMonthEnd(0))


def drop_partial_months(prices: pd.Series, end_year: int) -> pd.Series:
    """Keep only fully completed months up to end_year using last business day rule."""
    if prices.empty:
        return prices

    hard_end = min(pd.Timestamp(f"{int(end_year)}-12-31"), pd.Timestamp.today())
    prices = prices.loc[:hard_end]
    if prices.empty:
        return prices

    max_dt = prices.index.max()
    current_period = max_dt.to_period("M")
    # If we already have the last business day of the current period, the current month is complete
    is_current_complete = max_dt >= last_bmonth_end(current_period)
    last_complete_period = current_period if is_current_complete else current_period - 1

    # Keep only observations up to and including last complete month
    prices = prices[prices.index.to_period("M") <= last_complete_period]
    return prices


def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    """
    Per-month returns and 1H / 2H decomposition using trading days.
    Month is considered valid if we have data through that month's last business day.
    """
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

    out_rows = []
    months = pd.period_range(prices.index.min().to_period("M"),
                             prices.index.max().to_period("M"), freq="M")
    for m in months:
        month_days = prices[prices.index.to_period("M") == m]
        if month_days.shape[0] < 3:
            continue

        # Require coverage through last business day
        if month_days.index.max() < last_bmonth_end(m):
            continue

        first = month_days.iloc[0]
        last = month_days.iloc[-1]

        n = month_days.shape[0]
        mid_idx = (n // 2) - 1  # end of 1H
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
    """Compute monthly seasonality stats in [start_year, end_year], excluding partial months."""
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")

    halves = _intra_month_halves(prices)

    sel = (monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)
    monthly_ret = monthly_ret[sel]

    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    if not monthly_ret.empty:
        grouped_total = monthly_ret.groupby(monthly_ret.index.month)
        stats["hit_rate"] = grouped_total.apply(lambda x: (x > 0).mean() * 100)
        stats["min_ret"] = grouped_total.min()
        stats["max_ret"] = grouped_total.max()
        stats["years_observed"] = grouped_total.apply(lambda x: x.index.year.nunique())
    else:
        stats["hit_rate"] = np.nan
        stats["min_ret"] = np.nan
        stats["max_ret"] = np.nan
        stats["years_observed"] = np.nan
    stats["label"] = MONTH_LABELS

    if halves.empty:
        stats["mean_h1"] = np.nan
        stats["mean_h2"] = np.nan
        stats["mean_total"] = np.nan
        return stats

    halves = halves[(halves["year"] >= start_year) & (halves["year"] <= end_year)]
    grouped_halves = halves.groupby("month")
    stats["mean_h1"] = grouped_halves["h1_ret"].mean()
    stats["mean_h2"] = grouped_halves["h2_ret"].mean()
    stats["mean_total"] = stats["mean_h1"].fillna(0) + stats["mean_h2"].fillna(0)

    return stats


def _format_pct(x: float) -> str:
    return f"{x:+.1f}%" if pd.notna(x) else "n/a"


def _cell_colors(values: list[list[float]]) -> list[list[str]]:
    pos = "#d9f2e4"
    neg = "#f7d9d7"
    neu = "#f2f2f2"
    colors = []
    for row in values:
        colors.append([pos if (pd.notna(v) and v > 0) else neg if (pd.notna(v) and v < 0) else neu for v in row])
    return colors


def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]).copy()
    if plot_df.empty:
        fig = plt.figure(figsize=(12.5, 7.8), dpi=200)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No complete months in the selected range", ha="center", va="center", fontsize=14)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)
        return buf

    labels = plot_df["label"].tolist()
    mean_h1 = plot_df["mean_h1"].to_numpy(float)
    mean_h2 = plot_df["mean_h2"].to_numpy(float)
    totals = plot_df["mean_total"].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    fig = plt.figure(figsize=(12.5, 7.8), dpi=200)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3.2, 1.1], hspace=0.25)
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(labels))
    h1 = mean_h1
    h2 = mean_h2

    def seg_face(values):
        return np.where(values >= 0, "#62c38e", "#e07a73")

    def seg_edge(values):
        return np.where(values >= 0, "#1f7a4f", "#8b1e1a")

    ax1.bar(x, h1, width=0.8, color=seg_face(h1), edgecolor=seg_edge(h1), linewidth=1.0, zorder=2, alpha=0.95)
    bars2 = ax1.bar(x, h2, width=0.8, bottom=h1, color=seg_face(h2),
                    edgecolor=seg_edge(h2), linewidth=1.0, zorder=2, alpha=0.95, hatch="///")
    for r in bars2:
        r.set_hatch("///")

    yerr = np.abs(np.vstack([totals - min_ret, max_ret - totals]))
    ax1.errorbar(x, totals, yerr=yerr, fmt="none", ecolor="gray", elinewidth=1.6, alpha=0.7, capsize=6, zorder=3)

    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Mean return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max()))
    ymin = min(min_ret.min(), totals.min(), 0) - pad
    ymax = max(max_ret.max(), totals.max(), 0) + pad
    ax1.set_ylim(ymin, ymax)
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.6, alpha=0.7, zorder=1)

    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=90, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    legend = [Patch(facecolor="white", edgecolor="black", hatch="///", label="Second half (hatched)")]
    ax1.legend(handles=legend, loc="upper left", frameon=False)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    row_labels = ["1H %", "2H %", "Total %"]
    row1 = [float(v) if pd.notna(v) else np.nan for v in mean_h1]
    row2 = [float(v) if pd.notna(v) else np.nan for v in mean_h2]
    row3 = [float(v) if pd.notna(v) else np.nan for v in totals]
    cell_vals = [row1, row2, row3]
    cell_txt = [[_format_pct(v) for v in row] for row in cell_vals]
    cell_colors = _cell_colors(cell_vals)

    table = ax_tbl.table(
        cellText=cell_txt, rowLabels=row_labels, colLabels=labels,
        loc="center", cellLoc="center", rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        elif col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor(cell_colors[row - 1][col])
            cell.set_edgecolor("black")
            cell.set_linewidth(1)

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(pad=1.5)
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

prices = drop_partial_months(prices, int(end_year))

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index (^GSPC) fallback for seasonality.")

stats = seasonal_stats(prices, int(start_year), int(end_year))

if stats.dropna(subset=["mean_h1", "mean_h2"]).empty:
    st.error("Insufficient complete months in the selected window to compute statistics.")
    st.stop()

stats_valid = stats.dropna(subset=["mean_total"])
best_idx = stats_valid["mean_total"].idxmax()
worst_idx = stats_valid["mean_total"].idxmin()
best = stats.loc[best_idx]
worst = stats.loc[worst_idx]

st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            ⬆️ Best month (by mean): {best['label']} ({best['mean_total']:.2f}% | High {best['max_ret']:.2f}% | Low {best['min_ret']:.2f}%)
        </span>&nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            ⬇️ Worst month (by mean): {worst['label']} ({worst['mean_total']:.2f}% | High {worst['max_ret']:.2f}% | Low {worst['min_ret']:.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

buf = plot_seasonality(stats, f"{used_symbol} Seasonality ({int(start_year)}-{int(end_year)})")
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
        "label","mean_h1","mean_h2","mean_total",
        "hit_rate","min_ret","max_ret","years_observed"
    ]].copy()
    csv_df.columns = [
        "Month","Mean 1H %","Mean 2H %","Mean Total %",
        "Hit-Rate %","Min %","Max %","Years"
    ]
    st.download_button(
        "Download stats (CSV)",
        csv_df.to_csv(index=False),
        file_name=f"{used_symbol}_monthly_stats_{int(start_year)}_{int(end_year)}.csv"
    )

st.caption("Bars equal mean(1H) plus mean(2H). First half solid; second half hatched. Each half is green if positive, red if negative. Error bars use min and max of total monthly returns. Table cells are color coded by sign. Partial months excluded using last business day.")
st.caption("© 2025 AD Fund Management LP")
