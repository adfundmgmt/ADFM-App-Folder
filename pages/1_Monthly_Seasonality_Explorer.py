# seasonality_dashboard.py
# Monthly Seasonality Explorer
# - Original bar+table unchanged
# - Intra-month curve (white, black lines) anchored to prior month-end
# - Optional overlay of the current year's path
# - Forward-to-EOM mini-stats when the selected month is the current month
# - Robust trading-day ordinal (no crash), no stray Streamlit outputs
# - FIX: _month_paths_prev_eom uses px.index for masks (prevents IndexError)

import datetime as dt
import io
import time
import warnings
from typing import Optional, Tuple, List

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
        • Error bars show min and max monthly total returns; black diamonds = hit rate  
        • Bottom table consolidates 1H, 2H, Total per month  
        • Intra-month curve shows the average path by trading day relative to the prior month-end
        """,
        unsafe_allow_html=True,
    )

# -------------------------- Data helpers -------------------------- #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
    for n in range(retries):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if not df.empty and "Close" in df:
            return df["Close"]
        time.sleep(2 * (n + 1))
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")

    # pad start to capture prior month-end
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

def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

    out_rows = []
    months = pd.period_range(prices.index.min().to_period("M"),
                             prices.index.max().to_period("M"), freq="M")
    for m in months:
        m_mask = (prices.index.to_period("M") == m)
        month_days = prices.loc[m_mask]
        if month_days.shape[0] < 3:
            continue

        prev_month_days = prices.loc[(prices.index.to_period("M") == (m - 1))]
        if prev_month_days.empty:
            continue
        prev_eom = prev_month_days.iloc[-1]

        last = month_days.iloc[-1]
        n = month_days.shape[0]
        mid_idx = (n // 2) - 1
        if mid_idx < 0:
            continue
        mid_close = month_days.iloc[mid_idx]

        h1 = (mid_close / prev_eom - 1.0) * 100.0
        h2 = (last / mid_close - 1.0) * 100.0
        tot = (last / prev_eom - 1.0) * 100.0

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
    monthly_end = prices.resample("M").last()
    monthly_ret = monthly_end.pct_change().dropna() * 100
    monthly_ret.index = monthly_ret.index.to_period("M")

    halves = _intra_month_halves(prices)

    sel = (monthly_ret.index.year >= start_year) & (monthly_ret.index.year <= end_year)
    monthly_ret = monthly_ret[sel]

    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    grouped_total = monthly_ret.groupby(monthly_ret.index.month)
    stats["hit_rate"] = grouped_total.apply(lambda x: (x > 0).mean() * 100)
    stats["min_ret"] = grouped_total.min()
    stats["max_ret"] = grouped_total.max()
    stats["years_observed"] = grouped_total.apply(lambda x: x.index.year.nunique())
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
    stats["mean_total"] = grouped_halves["total_ret"].mean()  # equals mean_h1 + mean_h2
    return stats

def _format_pct(x: float) -> str:
    return f"{x:+.1f}%" if pd.notna(x) else "n/a"

def _cell_colors(values: List[List[float]]) -> List[List[str]]:
    pos = "#d9f2e4"
    neg = "#f7d9d7"
    neu = "#f2f2f2"
    colors = []
    for row in values:
        colors.append([pos if (pd.notna(v) and v > 0) else neg if (pd.notna(v) and v < 0) else neu for v in row])
    return colors

def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]).copy()

    labels = plot_df["label"].tolist()
    mean_h1 = plot_df["mean_h1"].to_numpy(float)
    mean_h2 = plot_df["mean_h2"].to_numpy(float)
    totals  = plot_df["mean_total"].to_numpy(float)
    hit     = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    fig = plt.figure(figsize=(12.5, 7.8), dpi=200)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3.2, 1.1], hspace=0.25)
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(labels))
    def seg_face(values): return np.where(values >= 0, "#62c38e", "#e07a73")
    def seg_edge(values): return np.where(values >= 0, "#1f7a4f", "#8b1e1a")

    ax1.bar(x, mean_h1, width=0.8, color=seg_face(mean_h1), edgecolor=seg_edge(mean_h1), linewidth=1.0, zorder=2, alpha=0.95)
    bars2 = ax1.bar(x, mean_h2, width=0.8, bottom=mean_h1, color=seg_face(mean_h2), edgecolor=seg_edge(mean_h2),
                    linewidth=1.0, zorder=2, alpha=0.95, hatch="///")
    for r in bars2: r.set_hatch("///")

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

    ax_tbl = fig.add_subplot(gs[1]); ax_tbl.axis("off")
    row_labels = ["1H %", "2H %", "Total %"]
    row1, row2, row3 = list(map(list, [mean_h1, mean_h2, totals]))
    cell_vals = [row1, row2, row3]
    cell_txt = [[_format_pct(v) for v in row] for row in cell_vals]
    cell_colors = _cell_colors(cell_vals)
    table = ax_tbl.table(cellText=cell_txt, rowLabels=row_labels, colLabels=labels,
                         loc="center", cellLoc="center", rowLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#f0f0f0")
        elif col == -1:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor(cell_colors[row - 1][col]); cell.set_edgecolor("black"); cell.set_linewidth(1)

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(pad=1.5)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=200); plt.close(fig); buf.seek(0)
    return buf

# -------- Intra-month path anchored to prior month-end -------- #
def _month_paths_prev_eom(prices: pd.Series, month_int: int, start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.Series]:
    px = prices[(prices.index.year >= start_year) & (prices.index.year <= end_year)].copy()
    if px.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    paths = {}
    for y in sorted(px.index.year.unique()):
        # IMPORTANT: build masks on px.index, not prices.index
        m = px.loc[(px.index.year == y) & (px.index.month == month_int)]
        if m.shape[0] < 3:
            continue

        prev_year = y if month_int > 1 else y - 1
        prev_month_num = month_int - 1 if month_int > 1 else 12
        prev_mask = (px.index.year == prev_year) & (px.index.month == prev_month_num)
        prev_month = px.loc[prev_mask]
        if prev_month.empty:
            continue

        prev_eom = float(prev_month.iloc[-1])
        norm = (m / prev_eom) * 100.0
        norm.index = pd.RangeIndex(start=1, stop=1 + len(norm), step=1)
        paths[y] = norm

    if not paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_days = max(len(s) for s in paths.values())
    df = pd.DataFrame(index=pd.RangeIndex(1, max_days))
    for y, s in paths.items():
        df[y] = s

    avg_path = df.mean(axis=1, skipna=True)
    return df, avg_path

def _avg_calendar_day_for_ordinal(prices: pd.Series, month_int: int, start_year: int, end_year: int, ordinal: int) -> Optional[int]:
    px = prices[(prices.index.year >= start_year) & (prices.index.year <= end_year)]
    days = []
    for y in sorted(px.index.year.unique()):
        m = px.loc[(px.index.year == y) & (px.index.month == month_int)]
        if len(m) >= ordinal:
            days.append(m.index[ordinal - 1].day)
    if not days:
        return None
    return int(round(np.mean(days)))

# ---- Robust trading-day ordinal (handles weekends/holidays/out-of-range) ---- #
def _trading_day_ordinal(month_index: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    """
    Return 1-based trading-day ordinal for `when` within `month_index`.
    Works if `when` is a weekend/holiday or outside the month’s range.
    """
    if month_index.empty:
        return 1
    idx = month_index.sort_values()
    d = pd.Timestamp(when).normalize()
    ord_ = int(np.searchsorted(idx.values, np.datetime64(d), side="right"))
    return max(1, min(ord_, len(idx)))

def plot_intra_month_curve(
    prices: pd.Series, month_int: int, start_year: int, end_year: int, symbol_shown: str,
    overlay_current_year: bool = True
) -> io.BytesIO:
    df, avg_path = _month_paths_prev_eom(prices, month_int, start_year, end_year)
    fig = plt.figure(figsize=(12.5, 7.2), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    if df.empty or avg_path.empty:
        ax.text(0.5, 0.5, "Not enough data to compute intra-month curve", ha="center", va="center", fontsize=12, color="black")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="white"); plt.close(fig); buf.seek(0); return buf

    x = avg_path.index.values; y = avg_path.values
    ax.plot(x, y, linewidth=2.4, color="black", label="Average")

    # Optional overlay of current year
    today = pd.Timestamp.today()
    cur_year = today.year
    if overlay_current_year and (start_year <= cur_year <= end_year):
        m = prices.loc[(prices.index.year == cur_year) & (prices.index.month == month_int)]
        prev_mask = (prices.index.year == (cur_year if month_int > 1 else cur_year - 1)) & \
                    (prices.index.month == (month_int - 1 if month_int > 1 else 12))
        prev_month = prices.loc[prev_mask]
        if not m.empty and not prev_month.empty:
            prev_eom = float(prev_month.iloc[-1])
            cur_norm = (m / prev_eom) * 100.0
            cur_norm.index = pd.RangeIndex(start=1, stop=1 + len(cur_norm), step=1)
            if len(cur_norm) >= 2:
                ax.plot(cur_norm.index.values, cur_norm.values, linewidth=1.8, color="black",
                        alpha=0.4, linestyle="--", label=str(cur_year))

    low_idx, low_val = int(avg_path.idxmin()), float(avg_path.min())
    high_idx, high_val = int(avg_path.idxmax()), float(avg_path.max())
    end_val = float(avg_path.iloc[-1])
    ax.scatter([low_idx, high_idx], [low_val, high_val], s=45, color="black", zorder=3)

    ax.set_title(f"{symbol_shown} {MONTH_LABELS[month_int-1]} Performance by Trading Day ({int(start_year)}–{int(end_year)})",
                 color="black", fontsize=16, weight="bold", pad=8)
    ax.set_xlabel(f"{MONTH_LABELS[month_int-1]} trading days", color="black", fontsize=10, weight="bold")
    ax.set_ylabel("Cumulative index (prev month-end = 100)", color="black", fontsize=10, weight="bold")

    ax.grid(axis="y", linestyle="--", color="#d9d9d9", alpha=1.0)
    ax.tick_params(colors="black");  [sp.set_color("black") for sp in ax.spines.values()]
    ax.legend(frameon=False, loc="upper left")

    low_dom = _avg_calendar_day_for_ordinal(prices, month_int, start_year, end_year, low_idx)
    high_dom = _avg_calendar_day_for_ordinal(prices, month_int, start_year, end_year, high_idx)

    def _box(txt, xy, offset_xy):
        ax.annotate(txt, xy=xy, xytext=(xy[0] + offset_xy[0], xy[1] + offset_xy[1]), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.0, shrinkA=2, shrinkB=2),
                    fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
    if low_dom is not None:
        _box(f"Avg {MONTH_LABELS[month_int-1]} Low\nDay {low_idx} (~{MONTH_LABELS[month_int-1]} {low_dom})\n{low_val:.2f}",
             (low_idx, low_val), (0.6, -0.35))
    if high_dom is not None:
        _box(f"Avg {MONTH_LABELS[month_int-1]} High\nDay {high_idx} (~{MONTH_LABELS[month_int-1]} {high_dom})\n{high_val:.2f}",
             (high_idx, high_val), (-2.0, 0.35))

    # actionable mini-stats only if selected month is current month and current year is in range
    if (today.month == month_int) and (start_year <= today.year <= end_year):
        m_cur = prices.loc[(prices.index.year == today.year) & (prices.index.month == month_int)]
        if not m_cur.empty:
            tday_ord = _trading_day_ordinal(m_cur.index, today)
            if tday_ord in df.index:
                lvl_t = df.loc[tday_ord]
                lvl_end = df.iloc[-1]
                fwd = (lvl_end - lvl_t).dropna()
                if not fwd.empty:
                    stats_text = (
                        f"From today (Day {tday_ord}) to EOM across years:\n"
                        f"• Mean: {fwd.mean():+.2f}\n"
                        f"• Median: {fwd.median():+.2f}\n"
                        f"• Hit rate >0: {(fwd > 0).mean()*100:0.0f}%  | N={fwd.shape[0]}"
                    )
                    ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, fontsize=9.0, color="black",
                            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8))

    fig.tight_layout(pad=1.0)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="white"); plt.close(fig); buf.seek(0)
    return buf

# -------------------------- Main controls -------------------------- #
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("Ticker symbol", value="^GSPC").upper()
with col2:
    start_year = st.number_input("Start year", value=2008, min_value=1900, max_value=dt.datetime.today().year)
with col3:
    end_year = st.number_input("End year", value=dt.datetime.today().year, min_value=int(start_year), max_value=dt.datetime.today().year)

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

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index (^GSPC) fallback for seasonality.")

stats = seasonal_stats(prices, int(start_year), int(end_year))
if stats.dropna(subset=["mean_h1", "mean_h2"]).empty:
    st.error("Insufficient data in the selected window to compute statistics.")
    st.stop()

# Best/worst
stats_valid = stats.dropna(subset=["mean_total"])
best_idx = stats_valid["mean_total"].idxmax()
worst_idx = stats_valid["mean_total"].idxmin()
best = stats.loc[best_idx]; worst = stats.loc[worst_idx]

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

# Original chart + table
buf = plot_seasonality(stats, f"{used_symbol} Seasonality ({int(start_year)}-{int(end_year)})")
st.image(buf, use_container_width=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("Download chart as PNG", buf.getvalue(),
                       file_name=f"{used_symbol}_seasonality_{int(start_year)}_{int(end_year)}.png")
with dl2:
    csv_df = stats[["label","mean_h1","mean_h2","mean_total","hit_rate","min_ret","max_ret","years_observed"]].copy()
    csv_df.columns = ["Month","Mean 1H %","Mean 2H %","Mean Total %","Hit-Rate %","Min %","Max %","Years"]
    st.download_button("Download stats (CSV)", csv_df.to_csv(index=False),
                       file_name=f"{used_symbol}_monthly_stats_{int(start_year)}_{int(end_year)}.csv")

st.caption("Bars equal mean(1H) + mean(2H). First half solid; second half hatched. Error bars use min and max of total monthly returns.")

# -------------------------- Intra-month curve below -------------------------- #
st.subheader("Intra-Month Seasonality Curve")
col_m1, col_m2 = st.columns([2,1])
with col_m1:
    month_choice = st.selectbox("Month", options=list(range(1,13)), index=9, format_func=lambda m: MONTH_LABELS[m-1])
with col_m2:
    overlay = st.checkbox("Overlay current year", value=True)

curve_buf = plot_intra_month_curve(prices, month_choice, int(start_year), int(end_year), used_symbol,
                                   overlay_current_year=overlay)
st.image(curve_buf, use_container_width=True)

# Clean export only, no stray prints
df_all, avg_series = _month_paths_prev_eom(prices, month_choice, int(start_year), int(end_year))
if not avg_series.empty:
    avg_df = avg_series.to_frame(name="Avg Path (Prev EOM=100)"); avg_df.index.name = "Trading Day"
    st.download_button("Download average path (CSV)", avg_df.to_csv(),
                       file_name=f"{used_symbol}_{MONTH_LABELS[month_choice-1]}_avg_path_{int(start_year)}_{int(end_year)}.csv")

st.caption("© 2025 AD Fund Management LP")
