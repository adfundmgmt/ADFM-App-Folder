import io
import time
import warnings
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

FALLBACK_MAP = {
    "^SPX": "SP500",
    "^DJI": "DJIA",
    "^IXIC": "NASDAQCOM",
}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# -------------------------- Streamlit UI -------------------------- #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

st.markdown(
    """
    <style>
    .adfm-card-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 0.35rem 0 0.85rem 0;
    }
    .adfm-card {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        background: #fafaf8;
        padding: 12px 14px;
    }
    .adfm-card-label {
        font-size: 0.78rem;
        color: #666666;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    .adfm-card-value {
        font-size: 1.12rem;
        font-weight: 700;
        color: #111111;
        line-height: 1.15;
    }
    .adfm-card-sub {
        font-size: 0.80rem;
        color: #666666;
        margin-top: 6px;
        line-height: 1.25;
    }
    .adfm-note {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        background: #fcfcfb;
        padding: 14px 16px;
        margin: 0.35rem 0 1rem 0;
        line-height: 1.55;
        color: #161616;
    }
    .adfm-note-title {
        font-size: 0.80rem;
        color: #666666;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 7px;
    }
    @media (max-width: 900px) {
        .adfm-card-row {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------- Data helpers -------------------------- #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
    for n in range(retries):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if not df.empty and "Close" in df:
            ser = df["Close"]
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[:, 0]
            ser = pd.to_numeric(ser, errors="coerce").dropna()
            if not ser.empty:
                return ser.rename("Close")
        time.sleep(2 * (n + 1))
    return None


def _fred_series(series_code: str, start: str, end: str) -> Optional[pd.Series]:
    if pdr is None:
        return None
    try:
        df = pdr.DataReader(series_code, "fred", start, end)
        if df is None or df.empty or series_code not in df.columns:
            return None
        ser = pd.to_numeric(df[series_code], errors="coerce").dropna()
        if ser.empty:
            return None
        return ser.rename(series_code)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")

    start_pad_dt = pd.Timestamp(start) - pd.DateOffset(days=45)
    start_pad = start_pad_dt.strftime("%Y-%m-%d")

    series = _yf_download(symbol, start_pad, end)
    if series is not None:
        return series

    if symbol == "SPY":
        series = _yf_download("^SPX", start_pad, end)
        if series is not None:
            return series

    if symbol in FALLBACK_MAP:
        fred_tk = FALLBACK_MAP[symbol]
        series = _fred_series(fred_tk, start_pad, end)
        if series is not None:
            return series.rename("Close")

    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_regime_data(start: str, end: str) -> pd.DataFrame:
    """
    Returns a monthly regime table indexed by PeriodIndex(freq='M') with:
      - is_recession
      - regime_cycle
      - fed_regime
      - fedfunds
    """
    start_dt = pd.Timestamp(start) - pd.DateOffset(years=2)
    end_dt = min(pd.Timestamp(end), pd.Timestamp.today()) + pd.DateOffset(days=31)

    usrec = _fred_series("USREC", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    fedfunds = _fred_series("FEDFUNDS", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    idx_start = start_dt.to_period("M")
    idx_end = end_dt.to_period("M")
    monthly_index = pd.period_range(idx_start, idx_end, freq="M")

    regime = pd.DataFrame(index=monthly_index)

    if usrec is not None:
        usrec_m = usrec.resample("M").last()
        usrec_m.index = usrec_m.index.to_period("M")
        regime["is_recession"] = usrec_m.reindex(monthly_index)
        regime["is_recession"] = regime["is_recession"].fillna(0).astype(int)
    else:
        regime["is_recession"] = 0

    if fedfunds is not None:
        ff_m = fedfunds.resample("M").last()
        ff_m.index = ff_m.index.to_period("M")
        regime["fedfunds"] = ff_m.reindex(monthly_index)
        delta = regime["fedfunds"].diff()
        regime["fed_regime"] = np.where(
            delta > 0,
            "Hiking",
            np.where(delta < 0, "Cutting", "Steady"),
        )
        regime.loc[regime["fedfunds"].isna(), "fed_regime"] = "Unknown"
    else:
        regime["fedfunds"] = np.nan
        regime["fed_regime"] = "Unknown"

    regime["regime_cycle"] = np.where(regime["is_recession"] == 1, "Recession", "Expansion")
    return regime


def _presidential_cycle_bucket(year: int) -> str:
    mod = year % 4
    if mod == 0:
        return "Election years"
    if mod == 1:
        return "Post-election years"
    if mod == 2:
        return "Midterm years"
    return "Pre-election years"


def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    """
    For each month:
      prev_eom   = previous month-end close
      mid_close  = close at mid-point of trading days
      last       = month-end close

    total_ret (%) = (last / prev_eom - 1)
    h1_ret   (%)  = (mid_close / prev_eom - 1)
    h2_ret   (%)  = total_ret - h1_ret
    """
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

    out_rows = []
    months = pd.period_range(
        prices.index.min().to_period("M"),
        prices.index.max().to_period("M"),
        freq="M",
    )

    for m in months:
        m_mask = prices.index.to_period("M") == m
        month_days = prices.loc[m_mask]
        if month_days.shape[0] < 3:
            continue

        prev_month_days = prices.loc[prices.index.to_period("M") == (m - 1)]
        if prev_month_days.empty:
            continue

        prev_eom = float(prev_month_days.iloc[-1])
        last = float(month_days.iloc[-1])

        n = month_days.shape[0]
        mid_idx = (n // 2) - 1
        if mid_idx < 0:
            continue

        mid_close = float(month_days.iloc[mid_idx])
        tot = (last / prev_eom - 1.0) * 100.0
        h1 = (mid_close / prev_eom - 1.0) * 100.0
        h2 = tot - h1

        out_rows.append(
            {
                "period": m,
                "year": m.year,
                "month": m.month,
                "total_ret": tot,
                "h1_ret": h1,
                "h2_ret": h2,
                "month_obs": int(month_days.shape[0]),
                "is_complete_month": bool(m < pd.Timestamp.today().to_period("M")),
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df


def build_filter_table(prices: pd.Series, regime_df: pd.DataFrame) -> pd.DataFrame:
    halves = _intra_month_halves(prices)
    if halves.empty:
        return halves

    df = halves.copy()
    df["pres_cycle_bucket"] = df["year"].apply(_presidential_cycle_bucket)
    df["is_post_gfc"] = df["year"] >= 2009
    df["is_post_covid"] = df.index >= pd.Period("2020-01", freq="M")

    if regime_df is not None and not regime_df.empty:
        keep_cols = ["is_recession", "regime_cycle", "fed_regime", "fedfunds"]
        df = df.join(regime_df[keep_cols], how="left")
    else:
        df["is_recession"] = 0
        df["regime_cycle"] = "Expansion"
        df["fed_regime"] = "Unknown"
        df["fedfunds"] = np.nan

    df["regime_cycle"] = df["regime_cycle"].fillna("Unknown")
    df["fed_regime"] = df["fed_regime"].fillna("Unknown")
    return df


def _latest_complete_year(prices: pd.Series) -> int:
    if prices.empty:
        return pd.Timestamp.today().year - 1

    latest_period = prices.index.max().to_period("M")
    current_period = pd.Timestamp.today().to_period("M")
    if latest_period >= current_period:
        return current_period.year - 1
    return latest_period.year


def resolve_year_window(
    preset: str,
    custom_start_year: int,
    custom_end_year: int,
    latest_complete_year: int,
) -> Tuple[int, int]:
    if preset == "Custom":
        return int(custom_start_year), int(custom_end_year)
    if preset == "All history":
        return 1900, int(latest_complete_year)
    if preset == "Last 10 years":
        return int(latest_complete_year - 9), int(latest_complete_year)
    if preset == "Last 20 years":
        return int(latest_complete_year - 19), int(latest_complete_year)
    if preset == "Post-GFC (2009+)":
        return 2009, int(latest_complete_year)
    if preset == "Post-COVID (2020+)":
        return 2020, int(latest_complete_year)
    return int(custom_start_year), int(custom_end_year)


def apply_filters(
    filter_table: pd.DataFrame,
    start_year: int,
    end_year: int,
    cycle_filter: str,
    regime_filter: str,
    fed_filter: str,
    complete_months_only: bool,
) -> pd.DataFrame:
    df = filter_table.copy()

    df = df[(df["year"] >= int(start_year)) & (df["year"] <= int(end_year))]

    if complete_months_only:
        df = df[df["is_complete_month"]]

    if cycle_filter != "All years":
        df = df[df["pres_cycle_bucket"] == cycle_filter]

    if regime_filter != "All regimes":
        df = df[df["regime_cycle"] == regime_filter]

    if fed_filter != "All Fed regimes":
        df = df[df["fed_regime"] == fed_filter]

    return df.sort_index()


def seasonal_stats_from_filtered(filtered_halves: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["label"] = MONTH_LABELS

    if filtered_halves.empty:
        stats["hit_rate"] = np.nan
        stats["min_ret"] = np.nan
        stats["max_ret"] = np.nan
        stats["years_observed"] = np.nan
        stats["mean_h1"] = np.nan
        stats["mean_h2"] = np.nan
        stats["mean_total"] = np.nan
        stats["observations"] = np.nan
        return stats

    grouped = filtered_halves.groupby("month")

    stats["hit_rate"] = grouped["total_ret"].apply(lambda x: (x > 0).mean() * 100.0)
    stats["min_ret"] = grouped["total_ret"].min()
    stats["max_ret"] = grouped["total_ret"].max()
    stats["years_observed"] = grouped["year"].nunique()
    stats["mean_h1"] = grouped["h1_ret"].mean()
    stats["mean_h2"] = grouped["h2_ret"].mean()
    stats["mean_total"] = grouped["total_ret"].mean()
    stats["observations"] = grouped["total_ret"].count()

    return stats


def _format_pct(x: float) -> str:
    return f"{x:+.1f}%" if pd.notna(x) else "n/a"


def _cell_colors(values: List[List[float]]) -> List[List[str]]:
    pos = "#d9f2e4"
    neg = "#f7d9d7"
    neu = "#f2f2f2"
    colors = []
    for row in values:
        colors.append(
            [
                pos if (pd.notna(v) and v > 0) else neg if (pd.notna(v) and v < 0) else neu
                for v in row
            ]
        )
    return colors


def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]).copy()

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

    def seg_face(values):
        return np.where(values >= 0, "#62c38e", "#e07a73")

    def seg_edge(values):
        return np.where(values >= 0, "#1f7a4f", "#8b1e1a")

    ax1.bar(
        x,
        mean_h1,
        width=0.8,
        color=seg_face(mean_h1),
        edgecolor=seg_edge(mean_h1),
        linewidth=1.0,
        zorder=2,
        alpha=0.95,
    )
    bars2 = ax1.bar(
        x,
        mean_h2,
        width=0.8,
        bottom=mean_h1,
        color=seg_face(mean_h2),
        edgecolor=seg_edge(mean_h2),
        linewidth=1.0,
        zorder=2,
        alpha=0.95,
        hatch="///",
    )
    for r in bars2:
        r.set_hatch("///")

    yerr = np.abs(np.vstack([totals - min_ret, max_ret - totals]))
    ax1.errorbar(
        x,
        totals,
        yerr=yerr,
        fmt="none",
        ecolor="gray",
        elinewidth=1.6,
        alpha=0.7,
        capsize=6,
        zorder=3,
    )

    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Mean return (%)", weight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max())) if len(min_ret) > 0 else 0.5
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
    row1, row2, row3 = list(map(list, [mean_h1, mean_h2, totals]))
    cell_vals = [row1, row2, row3]
    cell_txt = [[_format_pct(v) for v in row] for row in cell_vals]
    cell_colors = _cell_colors(cell_vals)
    table = ax_tbl.table(
        cellText=cell_txt,
        rowLabels=row_labels,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
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


def _month_paths_prev_eom_equal_weight_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if filtered_halves.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    target_periods = filtered_halves.loc[filtered_halves["month"] == month_int].index.tolist()
    if not target_periods:
        return pd.DataFrame(), pd.Series(dtype=float)

    raw_paths = {}

    for p in target_periods:
        y = p.year
        m_num = p.month

        month_days = prices.loc[(prices.index.year == y) & (prices.index.month == m_num)]
        if month_days.shape[0] < 3:
            continue

        prev_p = p - 1
        prev_month_days = prices.loc[(prices.index.year == prev_p.year) & (prices.index.month == prev_p.month)]
        if prev_month_days.empty:
            continue

        prev_eom = float(prev_month_days.iloc[-1])
        cum_ret = (month_days / prev_eom - 1.0) * 100.0
        cum_ret.index = pd.RangeIndex(start=1, stop=1 + len(cum_ret), step=1)
        cum_ret.loc[0] = 0.0
        cum_ret = cum_ret.sort_index()
        raw_paths[str(p)] = cum_ret

    if not raw_paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_days = max(int(s.index.max()) for s in raw_paths.values())
    full_index = pd.RangeIndex(0, max_days + 1)

    df = pd.DataFrame(index=full_index)
    for k, s in raw_paths.items():
        df[k] = s.reindex(full_index).ffill()

    avg_path = df.mean(axis=1)
    return df, avg_path


def _avg_calendar_day_for_ordinal_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    ordinal: int,
) -> Optional[int]:
    month_periods = filtered_halves.loc[filtered_halves["month"] == month_int].index.tolist()
    if not month_periods or ordinal <= 0:
        return None

    days = []
    for p in month_periods:
        m = prices.loc[(prices.index.year == p.year) & (prices.index.month == p.month)]
        if len(m) >= ordinal:
            days.append(m.index[ordinal - 1].day)

    if not days:
        return None
    return int(round(np.mean(days)))


def _trading_day_ordinal(month_index: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    if month_index.empty:
        return 1
    idx = month_index.sort_values()
    d = pd.Timestamp(when).normalize()
    ord_ = int(np.searchsorted(idx.values, np.datetime64(d), side="right"))
    return max(1, min(ord_, len(idx)))


def build_intra_month_summary(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    symbol_shown: str,
) -> Dict[str, Any]:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(prices, filtered_halves, month_int)

    summary: Dict[str, Any] = {
        "ok": False,
        "symbol": symbol_shown,
        "month_int": month_int,
        "month_label": MONTH_LABELS[month_int - 1],
        "sample_months": 0,
        "sample_years": 0,
        "avg_month_end": np.nan,
        "avg_low_day": None,
        "avg_low_val": np.nan,
        "avg_low_dom": None,
        "avg_high_day": None,
        "avg_high_val": np.nan,
        "avg_high_dom": None,
        "day5_val": np.nan,
        "day10_val": np.nan,
        "day15_val": np.nan,
        "day20_val": np.nan,
        "front_loaded": None,
        "recovery_strength": np.nan,
        "current_year": pd.Timestamp.today().year,
        "current_year_month_available": False,
        "today_ord": None,
        "today_avg_level": np.nan,
        "today_cur_level": np.nan,
        "live_gap_vs_avg": np.nan,
        "fwd_mean": np.nan,
        "fwd_median": np.nan,
        "fwd_hit_rate": np.nan,
        "fwd_n": 0,
        "current_year_end": np.nan,
        "current_vs_hist_end_gap": np.nan,
        "avg_path": avg_sel,
        "df_paths": df_sel,
    }

    month_filtered = filtered_halves.loc[filtered_halves["month"] == month_int]
    if not month_filtered.empty:
        summary["sample_months"] = int(month_filtered.shape[0])
        summary["sample_years"] = int(month_filtered["year"].nunique())

    if avg_sel.empty or df_sel.empty:
        return summary

    summary["ok"] = True
    summary["avg_month_end"] = float(avg_sel.iloc[-1])

    avg_ex = avg_sel.copy()
    if 0 in avg_ex.index:
        avg_ex = avg_ex.drop(index=0)

    if not avg_ex.empty:
        low_idx = int(avg_ex.idxmin())
        high_idx = int(avg_ex.idxmax())
        summary["avg_low_day"] = low_idx
        summary["avg_low_val"] = float(avg_ex.loc[low_idx])
        summary["avg_low_dom"] = _avg_calendar_day_for_ordinal_from_filtered(prices, filtered_halves, month_int, low_idx)
        summary["avg_high_day"] = high_idx
        summary["avg_high_val"] = float(avg_ex.loc[high_idx])
        summary["avg_high_dom"] = _avg_calendar_day_for_ordinal_from_filtered(prices, filtered_halves, month_int, high_idx)

    for k in [5, 10, 15, 20]:
        if k in avg_sel.index:
            summary[f"day{k}_val"] = float(avg_sel.loc[k])

    d10 = summary["day10_val"]
    month_end = summary["avg_month_end"]
    summary["front_loaded"] = bool(pd.notna(d10) and pd.notna(month_end) and abs(d10) >= 0.6 * abs(month_end))
    if summary["avg_low_day"] is not None and pd.notna(month_end):
        summary["recovery_strength"] = float(month_end - summary["avg_low_val"])

    today = pd.Timestamp.today()
    cur_year = today.year
    m = prices.loc[(prices.index.year == cur_year) & (prices.index.month == month_int)]
    prev_mask = (prices.index.year == (cur_year if month_int > 1 else cur_year - 1)) & (
        prices.index.month == (month_int - 1 if month_int > 1 else 12)
    )
    prev_month = prices.loc[prev_mask]

    if not m.empty and not prev_month.empty:
        summary["current_year_month_available"] = True
        prev_eom = float(prev_month.iloc[-1])

        cur_cum = (m / prev_eom - 1.0) * 100.0
        cur_cum.index = pd.RangeIndex(start=1, stop=1 + len(cur_cum), step=1)
        cur_cum.loc[0] = 0.0
        cur_cum = cur_cum.sort_index()

        summary["current_year_end"] = float(cur_cum.iloc[-1])
        if pd.notna(summary["avg_month_end"]):
            summary["current_vs_hist_end_gap"] = float(summary["current_year_end"] - summary["avg_month_end"])

        if today.month == month_int:
            tday_ord = _trading_day_ordinal(m.index, today)
            summary["today_ord"] = tday_ord

            if tday_ord in avg_sel.index:
                summary["today_avg_level"] = float(avg_sel.loc[tday_ord])
            if tday_ord in cur_cum.index:
                summary["today_cur_level"] = float(cur_cum.loc[tday_ord])

            if pd.notna(summary["today_avg_level"]) and pd.notna(summary["today_cur_level"]):
                summary["live_gap_vs_avg"] = float(summary["today_cur_level"] - summary["today_avg_level"])

            if tday_ord in df_sel.index:
                lvl_t = df_sel.loc[tday_ord]
                lvl_end = df_sel.loc[df_sel.index.max()]
                fwd = (lvl_end - lvl_t).dropna()
                if not fwd.empty:
                    summary["fwd_mean"] = float(fwd.mean())
                    summary["fwd_median"] = float(fwd.median())
                    summary["fwd_hit_rate"] = float((fwd > 0).mean() * 100.0)
                    summary["fwd_n"] = int(fwd.shape[0])

    return summary


def render_intra_month_cards(summary: Dict[str, Any]) -> None:
    if not summary.get("ok", False):
        return

    month_label = summary["month_label"]
    low_day = summary["avg_low_day"]
    high_day = summary["avg_high_day"]
    low_dom = summary["avg_low_dom"]
    high_dom = summary["avg_high_dom"]

    low_txt = "n/a"
    if low_day is not None and pd.notna(summary["avg_low_val"]):
        low_txt = f"Day {low_day} | {summary['avg_low_val']:+.2f}%"
        if low_dom is not None:
            low_txt += f"<div class='adfm-card-sub'>~ {month_label} {low_dom}</div>"
        else:
            low_txt += "<div class='adfm-card-sub'>Average trough</div>"

    high_txt = "n/a"
    if high_day is not None and pd.notna(summary["avg_high_val"]):
        high_txt = f"Day {high_day} | {summary['avg_high_val']:+.2f}%"
        if high_dom is not None:
            high_txt += f"<div class='adfm-card-sub'>~ {month_label} {high_dom}</div>"
        else:
            high_txt += "<div class='adfm-card-sub'>Average peak</div>"

    if summary.get("today_ord") is not None and summary.get("fwd_n", 0) > 0:
        live_title = f"From Day {summary['today_ord']} to month-end"
        live_val = f"{summary['fwd_mean']:+.2f}%"
        live_sub = (
            f"Median {summary['fwd_median']:+.2f}% | "
            f"Hit rate {summary['fwd_hit_rate']:.0f}% | "
            f"N={summary['fwd_n']}"
        )
    else:
        live_title = "Path profile"
        if summary.get("front_loaded") is True:
            live_val = "Front-half weighted"
            live_sub = "A large share of the month is usually realized by Day 10."
        elif summary.get("front_loaded") is False:
            live_val = "Back-half weighted"
            live_sub = "A meaningful share of the move tends to arrive after Day 10."
        else:
            live_val = "n/a"
            live_sub = "Not enough data."

    html = f"""
    <div class="adfm-card-row">
        <div class="adfm-card">
            <div class="adfm-card-label">Average month-end return</div>
            <div class="adfm-card-value">{summary['avg_month_end']:+.2f}%</div>
            <div class="adfm-card-sub">Equal-weighted across {summary['sample_months']} month observations and {summary['sample_years']} distinct years</div>
        </div>
        <div class="adfm-card">
            <div class="adfm-card-label">Average trough</div>
            <div class="adfm-card-value">{low_txt}</div>
        </div>
        <div class="adfm-card">
            <div class="adfm-card-label">Average peak</div>
            <div class="adfm-card-value">{high_txt}</div>
        </div>
        <div class="adfm-card">
            <div class="adfm-card-label">{live_title}</div>
            <div class="adfm-card-value">{live_val}</div>
            <div class="adfm-card-sub">{live_sub}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_intra_month_commentary(summary: Dict[str, Any]) -> None:
    if not summary.get("ok", False):
        return

    month_label = summary["month_label"]
    avg_end = summary["avg_month_end"]
    low_day = summary["avg_low_day"]
    high_day = summary["avg_high_day"]
    low_val = summary["avg_low_val"]
    high_val = summary["avg_high_val"]
    d5 = summary["day5_val"]
    d10 = summary["day10_val"]
    recovery = summary["recovery_strength"]

    if pd.isna(d5):
        open_shape = f"{month_label} does not have enough history in the chosen filtered sample to say much about the opening stretch."
    elif d5 > 0.20:
        open_shape = f"{month_label} has historically opened firm, with the average path already at {d5:+.2f}% by Day 5."
    elif d5 < -0.20:
        open_shape = f"{month_label} has historically opened soft, with the average path at {d5:+.2f}% by Day 5."
    else:
        open_shape = f"{month_label} has historically opened close to flat, with the average path at {d5:+.2f}% by Day 5."

    if low_day is not None and high_day is not None:
        if low_day < high_day and pd.notna(recovery) and recovery > 0.25:
            shape_core = (
                f"The average tape usually finds its low around Day {low_day} at {low_val:+.2f}% and then works higher "
                f"into a peak around Day {high_day} at {high_val:+.2f}%, which gives the month a dip-then-recovery shape."
            )
        elif high_day < low_day:
            shape_core = (
                f"The average path tends to peak early around Day {high_day} at {high_val:+.2f}% and then fade toward a later "
                f"low near Day {low_day} at {low_val:+.2f}%, so strength has historically arrived earlier than the close."
            )
        else:
            shape_core = (
                f"The average path clusters its key turning points between Day {min(low_day, high_day)} and Day {max(low_day, high_day)}, "
                f"which suggests a choppier month than the endpoint alone would imply."
            )
    else:
        shape_core = "The average path is available, but the turning points are not stable enough in the selected filtered sample to describe cleanly."

    if pd.notna(d10) and pd.notna(avg_end):
        if abs(d10) >= 0.6 * abs(avg_end):
            timing = (
                f"By Day 10, the average path has already reached {d10:+.2f}%, which means a large share of the typical "
                f"monthly move is realized relatively early."
            )
        else:
            timing = (
                f"By Day 10, the average path sits at {d10:+.2f}% against a month-end average of {avg_end:+.2f}%, which suggests "
                f"the back half matters more than the opening stretch."
            )
    else:
        timing = "The month-end profile is clearer than the first-ten-day profile in this sample."

    live_read = ""
    if summary.get("today_ord") is not None and summary.get("fwd_n", 0) > 0:
        gap = summary.get("live_gap_vs_avg", np.nan)
        if pd.notna(gap):
            if gap > 0.20:
                rel = f"The current year is running ahead of the historical average by {gap:+.2f}% at the same trading-day mark."
            elif gap < -0.20:
                rel = f"The current year is running behind the historical average by {gap:+.2f}% at the same trading-day mark."
            else:
                rel = "The current year is tracking close to the historical average path at the same trading-day mark."
        else:
            rel = "The current year overlay is available, though the live gap versus average is not cleanly measurable."

        live_read = (
            f"{rel} From Day {summary['today_ord']} into month-end, the historical path has averaged "
            f"{summary['fwd_mean']:+.2f}% with a median of {summary['fwd_median']:+.2f}% and a positive hit rate of "
            f"{summary['fwd_hit_rate']:.0f}% across {summary['fwd_n']} observations."
        )
    elif summary.get("current_year_month_available", False) and pd.notna(summary.get("current_vs_hist_end_gap", np.nan)):
        gap = summary["current_vs_hist_end_gap"]
        if gap > 0.20:
            live_read = (
                f"The current year's completed {month_label} finished {gap:+.2f}% above the long-run average month-end path, "
                f"so realized strength exceeded the historical template."
            )
        elif gap < -0.20:
            live_read = (
                f"The current year's completed {month_label} finished {gap:+.2f}% below the long-run average month-end path, "
                f"so realized performance lagged the historical template."
            )
        else:
            live_read = (
                f"The current year's completed {month_label} finished broadly in line with the long-run average month-end path."
            )

    commentary = f"""
    <div class="adfm-note">
        <div class="adfm-note-title">INTRA-MONTH READ</div>
        <div>{open_shape} {shape_core} {timing} {live_read}</div>
    </div>
    """
    st.markdown(commentary, unsafe_allow_html=True)


def plot_intra_month_curve(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    symbol_shown: str,
) -> io.BytesIO:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(prices, filtered_halves, month_int)

    fig = plt.figure(figsize=(12.5, 7.2), dpi=200, facecolor="#fcfcfb")
    ax = fig.add_subplot(111, facecolor="#fcfcfb")

    if avg_sel.empty or df_sel.empty:
        ax.text(
            0.5,
            0.5,
            "Not enough data to compute intra-month curve",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#fcfcfb")
        plt.close(fig)
        buf.seek(0)
        return buf

    std_sel = df_sel.std(axis=1)

    x_vals = avg_sel.index.values
    y_vals = avg_sel.values

    ax.axhline(0.0, color="#9b9b9b", linestyle=":", linewidth=1.0, zorder=1)

    ax.fill_between(
        x_vals,
        (avg_sel - std_sel).values,
        (avg_sel + std_sel).values,
        color="#d8d8d8",
        alpha=0.35,
        zorder=1.3,
        label="1σ range",
    )

    ax.plot(
        x_vals,
        y_vals,
        linewidth=2.8,
        color="#111111",
        linestyle="-",
        label="Average path",
        zorder=2.4,
    )

    today = pd.Timestamp.today()
    cur_year = today.year

    cur_path = None
    m = prices.loc[(prices.index.year == cur_year) & (prices.index.month == month_int)]
    prev_mask = (prices.index.year == (cur_year if month_int > 1 else cur_year - 1)) & (
        prices.index.month == (month_int - 1 if month_int > 1 else 12)
    )
    prev_month = prices.loc[prev_mask]

    if not m.empty and not prev_month.empty:
        prev_eom = float(prev_month.iloc[-1])
        cur_cum = (m / prev_eom - 1.0) * 100.0
        cur_cum.index = pd.RangeIndex(start=1, stop=1 + len(cur_cum), step=1)
        cur_cum.loc[0] = 0.0
        cur_cum = cur_cum.sort_index()
        cur_path = cur_cum

        if len(cur_path) >= 3:
            ax.plot(
                cur_path.index.values,
                cur_path.values,
                linewidth=2.0,
                color="#333333",
                alpha=0.55,
                linestyle="--",
                label=str(cur_year),
                zorder=2.2,
            )

    avg_ex = avg_sel.copy()
    if 0 in avg_ex.index:
        avg_ex = avg_ex.drop(index=0)

    low_idx = int(avg_ex.idxmin())
    low_val = float(avg_ex.loc[low_idx])
    high_idx = int(avg_ex.idxmax())
    high_val = float(avg_ex.loc[high_idx])

    ax.scatter([low_idx, high_idx], [low_val, high_val], s=34, color="#111111", zorder=3.2)

    low_dom = _avg_calendar_day_for_ordinal_from_filtered(prices, filtered_halves, month_int, low_idx)
    high_dom = _avg_calendar_day_for_ordinal_from_filtered(prices, filtered_halves, month_int, high_idx)

    for vline in [5, 10, 15, 20]:
        if vline <= int(max(x_vals)):
            ax.axvline(vline, color="#e5e5e5", linestyle="-", linewidth=0.8, zorder=0.8)

    summary_note = [
        f"Avg low: Day {low_idx} ({low_val:+.2f}%)",
        f"Avg high: Day {high_idx} ({high_val:+.2f}%)",
    ]
    if low_dom is not None:
        summary_note[0] += f" ~ {MONTH_LABELS[month_int - 1]} {low_dom}"
    if high_dom is not None:
        summary_note[1] += f" ~ {MONTH_LABELS[month_int - 1]} {high_dom}"

    ax.text(
        0.985,
        0.97,
        "\n".join(summary_note),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.1,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d6d6d6", lw=0.9),
    )

    ax.set_title(
        f"{symbol_shown} {MONTH_LABELS[month_int - 1]} | Intra-Month Seasonality Curve",
        color="black",
        fontsize=16,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Trading day since prior month-end anchor", color="black", fontsize=10, weight="bold")
    ax.set_ylabel("Return from prior month-end (%)", color="black", fontsize=10, weight="bold")
    ax.grid(axis="y", linestyle="--", color="#d9d9d9", alpha=1.0, linewidth=0.8)
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_color("#111111")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if (today.month == month_int) and not df_sel.empty:
        m_cur = prices.loc[(prices.index.year == today.year) & (prices.index.month == month_int)]
        if not m_cur.empty:
            tday_ord = _trading_day_ordinal(m_cur.index, today)
            ax.axvline(tday_ord, color="#9b9b9b", linestyle=":", linewidth=1.25, zorder=2.0)

            if tday_ord in avg_sel.index:
                ax.scatter(tday_ord, avg_sel.loc[tday_ord], s=28, color="#111111", zorder=4)
            if cur_path is not None and tday_ord in cur_path.index:
                ax.scatter(
                    tday_ord,
                    cur_path.loc[tday_ord],
                    s=36,
                    facecolors="white",
                    edgecolors="#111111",
                    linewidths=1.0,
                    zorder=4,
                )

    y_min = float((avg_sel - std_sel).min())
    y_max = float((avg_sel + std_sel).max())
    pad_y = 0.12 * max(abs(y_min), abs(y_max)) if np.isfinite(y_min) and np.isfinite(y_max) else 0.0
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    buf.seek(0)
    return buf


# -------------------------- Main controls -------------------------- #
today_dt = pd.Timestamp.today()
this_year = int(today_dt.year)

col1, col2, col3 = st.columns([2.0, 1.0, 1.0])
with col1:
    symbol = st.text_input("Ticker symbol", value="^SPX").upper()
with col2:
    custom_start_year = st.number_input("Custom start year", value=2020, min_value=1900, max_value=this_year)
with col3:
    custom_end_year = st.number_input(
        "Custom end year",
        value=this_year,
        min_value=int(custom_start_year),
        max_value=this_year,
    )

start_fetch_date = f"{int(custom_start_year) - 2}-01-01"
end_fetch_date = f"{this_year}-12-31"

with st.spinner("Fetching and analyzing data..."):
    used_symbol = symbol
    prices = fetch_prices(symbol, start_fetch_date, end_fetch_date)
    if prices is None and symbol == "SPY":
        prices = fetch_prices("^SPX", start_fetch_date, end_fetch_date)
        if prices is not None:
            used_symbol = "^SPX"

if prices is None or prices.empty:
    st.error(f"No data found for '{symbol}' in the given date range. Try a different symbol or adjust the years.")
    st.stop()

first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index (^SPX) fallback for seasonality.")

latest_complete_year = _latest_complete_year(prices)
regime_df = fetch_regime_data(str(prices.index.min().date()), str(prices.index.max().date()))
filter_table = build_filter_table(prices, regime_df)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Seasonality explorer for monthly return tendencies, hit-rate patterns, and intra-month path behavior across filtered historical samples.

        Methodology
        • Monthly return is measured from prior month-end close to current month-end close.
        • First-half and second-half contributions are split using the midpoint trading day of each month.
        • Historical stats are computed only after the sample is explicitly filtered.
        • Presidential cycle buckets are calendar-based and exact.
        • Recession / expansion uses FRED USREC monthly recession indicator.
        • Fed regime uses monthly change in FRED FEDFUNDS:
          Hiking if month-over-month change > 0
          Cutting if month-over-month change < 0
          Steady otherwise
        • Complete months only excludes the current partial month from the historical sample.

        Accuracy guardrails
        • Filter membership is applied before any aggregation.
        • Observation count and distinct year count are shown below.
        • Audit panel lists the exact included months in the sample.
        """
    )

    st.subheader("Sample Filters")

    sample_preset = st.selectbox(
        "Sample preset",
        [
            "Custom",
            "All history",
            "Last 10 years",
            "Last 20 years",
            "Post-GFC (2009+)",
            "Post-COVID (2020+)",
        ],
        index=0,
    )

    cycle_filter = st.selectbox(
        "Presidential cycle filter",
        [
            "All years",
            "Election years",
            "Midterm years",
            "Pre-election years",
            "Post-election years",
        ],
        index=0,
    )

    regime_filter = st.selectbox(
        "Macro regime filter",
        ["All regimes", "Expansion", "Recession"],
        index=0,
    )

    fed_filter = st.selectbox(
        "Fed regime filter",
        ["All Fed regimes", "Hiking", "Cutting", "Steady"],
        index=0,
    )

    complete_months_only = st.checkbox("Complete months only", value=True)

start_year, end_year = resolve_year_window(
    sample_preset,
    int(custom_start_year),
    int(custom_end_year),
    latest_complete_year,
)

filtered = apply_filters(
    filter_table=filter_table,
    start_year=start_year,
    end_year=end_year,
    cycle_filter=cycle_filter,
    regime_filter=regime_filter,
    fed_filter=fed_filter,
    complete_months_only=complete_months_only,
)

if filtered.empty:
    st.error("No observations match the selected filters.")
    st.stop()

stats = seasonal_stats_from_filtered(filtered)
if stats.dropna(subset=["mean_h1", "mean_h2"]).empty:
    st.error("Insufficient data in the selected filtered window to compute statistics.")
    st.stop()

obs_months = int(filtered.shape[0])
obs_years = int(filtered["year"].nunique())
min_period = str(filtered.index.min())
max_period = str(filtered.index.max())

meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
meta_c1.metric("Filtered month observations", f"{obs_months}")
meta_c2.metric("Distinct years", f"{obs_years}")
meta_c3.metric("Window", f"{start_year}-{end_year}")
meta_c4.metric("Included months", f"{min_period} to {max_period}")

if obs_months < 24 or obs_years < 5:
    st.warning(
        "Thin sample warning: the current filter set leaves a small historical base. "
        "Read the output as directional rather than stable."
    )

stats_valid = stats.dropna(subset=["mean_total"])
best_idx = stats_valid["mean_total"].idxmax()
worst_idx = stats_valid["mean_total"].idxmin()
best = stats.loc[best_idx]
worst = stats.loc[worst_idx]

st.markdown(
    f"""
    <div style='text-align:center'>
        <span style='font-size:1.18em; font-weight:600; color:#218739'>
            Best month: {best['label']} ({best['mean_total']:.2f}% | High {best['max_ret']:.2f}% | Low {best['min_ret']:.2f}%)
        </span>&nbsp;&nbsp;&nbsp;
        <span style='font-size:1.18em; font-weight:600; color:#c93535'>
            Worst month: {worst['label']} ({worst['mean_total']:.2f}% | High {worst['max_ret']:.2f}% | Low {worst['min_ret']:.2f}%)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_caption_parts = [
    f"{used_symbol}",
    f"{start_year}-{end_year}",
    cycle_filter,
    regime_filter,
    fed_filter,
    "Complete months only" if complete_months_only else "Partial current month allowed",
]
chart_title = " | ".join(filter_caption_parts)

buf = plot_seasonality(stats, f"{used_symbol} Seasonality ({chart_title})")
st.image(buf, use_container_width=True)

st.caption(
    "Bars equal mean(1H) + mean(2H) contributions. First half solid; second half hatched. "
    "Error bars use min and max of total monthly returns within the filtered sample."
)

st.subheader("Intra-Month Seasonality Curve")

month_mode_col1, month_mode_col2 = st.columns([1.15, 1.85])
with month_mode_col1:
    month_mode = st.radio(
        "View preset",
        options=["Current", "Best", "Worst", "Manual"],
        horizontal=True,
        index=0,
    )

default_month_idx = max(0, min(today_dt.month - 1, 11))

if month_mode == "Current":
    month_choice = today_dt.month
elif month_mode == "Best":
    month_choice = int(best_idx)
elif month_mode == "Worst":
    month_choice = int(worst_idx)
else:
    with month_mode_col2:
        month_choice = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=default_month_idx,
            format_func=lambda m: MONTH_LABELS[m - 1],
        )

if month_mode != "Manual":
    st.caption(f"Selected month: {MONTH_LABELS[month_choice - 1]}")

summary = build_intra_month_summary(prices, filtered, month_choice, used_symbol)
render_intra_month_cards(summary)
render_intra_month_commentary(summary)

curve_buf = plot_intra_month_curve(prices, filtered, month_choice, used_symbol)
st.image(curve_buf, use_container_width=True)

if summary.get("ok", False):
    extra_caption = (
        f"Average path is equal-weighted across {summary['sample_months']} filtered month observations "
        f"and {summary['sample_years']} distinct years, anchored to the prior month-end. "
        f"Shaded band shows ±1 standard deviation. Current year is shown when available."
    )
else:
    extra_caption = "Not enough history in the selected filtered sample to build the intra-month path."

st.caption(extra_caption)

with st.expander("Audit included observations"):
    audit_df = filtered.copy()
    audit_df = audit_df.reset_index().rename(columns={"index": "period"})
    audit_df["period"] = audit_df["period"].astype(str)
    audit_df = audit_df[
        [
            "period",
            "year",
            "month",
            "pres_cycle_bucket",
            "regime_cycle",
            "fed_regime",
            "is_complete_month",
            "total_ret",
            "h1_ret",
            "h2_ret",
        ]
    ].sort_values(["year", "month"])

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

st.caption("© 2026 AD Fund Management LP")
