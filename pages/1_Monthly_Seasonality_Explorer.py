import datetime as dt
import io
import time
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

st.set_page_config(page_title="Seasonality Dashboard", layout="wide")

FALLBACK_MAP = {"^SPX": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM"}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PRESET_WINDOWS = {
    "Custom": None,
    "10Y": 10,
    "20Y": 20,
    "30Y": 30,
    "Full history": "full",
}
POPULAR_SYMBOLS = ["^SPX", "QQQ", "IWM", "TLT", "GLD", "USO", "CL=F", "GC=F"]

COLORS: Dict[str, str] = {
    "green_fill": "#62c38e",
    "green_edge": "#1f7a4f",
    "red_fill": "#e07a73",
    "red_edge": "#8b1e1a",
    "neutral_fill": "#f0f0f0",
    "grid": "#d9d9d9",
    "line": "#111111",
    "muted": "#7a7a7a",
    "band": "#d0d0d0",
    "current": "#111111",
    "today": "#b0b0b0",
}


def _meta_dict(
    requested_symbol: str,
    source_symbol: str,
    source_name: str,
    proxy_used: bool,
    adjusted: bool,
    coverage_start: Optional[pd.Timestamp],
    coverage_end: Optional[pd.Timestamp],
    rows: int,
    notes: List[str],
) -> Dict[str, object]:
    return {
        "requested_symbol": requested_symbol,
        "source_symbol": source_symbol,
        "source_name": source_name,
        "proxy_used": proxy_used,
        "adjusted": adjusted,
        "coverage_start": coverage_start.isoformat() if coverage_start is not None else None,
        "coverage_end": coverage_end.isoformat() if coverage_end is not None else None,
        "rows": int(rows),
        "notes": list(notes),
    }


def _meta_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.Timestamp(value)


# ----------------------------- Data layer ----------------------------- #
def _clean_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", "") in df.columns:
            series = df[("Close", "")]
        elif "Close" in df.columns.get_level_values(0):
            close_cols = [col for col in df.columns if col[0] == "Close"]
            if not close_cols:
                return None
            series = df[close_cols[0]]
        else:
            return None
    else:
        if "Close" not in df.columns:
            return None
        series = df["Close"]

    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return None
        series = series.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None

    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.sort_index()
    series.name = "Close"
    return series


def _yf_download(symbol: str, start: str, end: str, retries: int = 3, pause_seconds: float = 1.5) -> Optional[pd.Series]:
    for attempt in range(retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            series = _clean_close_series(df)
            if series is not None and not series.empty:
                return series
        except Exception:
            pass
        time.sleep(pause_seconds * (attempt + 1))
    return None


def _fred_download(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    if pdr is None or symbol not in FALLBACK_MAP:
        return None

    try:
        fred_symbol = FALLBACK_MAP[symbol]
        df = pdr.DataReader(fred_symbol, "fred", start, end)
        if df is None or df.empty or fred_symbol not in df.columns:
            return None
        series = pd.to_numeric(df[fred_symbol], errors="coerce").dropna()
        if series.empty:
            return None
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()
        series.name = "Close"
        return series
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices_with_meta(symbol: str, start: str, end: str) -> Tuple[Optional[pd.Series], Dict[str, object]]:
    symbol = symbol.strip().upper()
    end_ts = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
    start_pad = (pd.Timestamp(start) - pd.DateOffset(days=45)).strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")
    notes: List[str] = []

    series = _yf_download(symbol, start_pad, end_str)
    if series is not None:
        return series, _meta_dict(
            requested_symbol=symbol,
            source_symbol=symbol,
            source_name="Yahoo Finance",
            proxy_used=False,
            adjusted=True,
            coverage_start=series.index.min(),
            coverage_end=series.index.max(),
            rows=series.shape[0],
            notes=notes,
        )

    notes.append(f"Yahoo Finance returned no usable close series for {symbol}.")

    if symbol == "SPY":
        proxy = _yf_download("^SPX", start_pad, end_str)
        if proxy is not None:
            notes.append("SPY was unavailable, so the app used ^SPX as a price index proxy.")
            return proxy, _meta_dict(
                requested_symbol=symbol,
                source_symbol="^SPX",
                source_name="Yahoo Finance",
                proxy_used=True,
                adjusted=True,
                coverage_start=proxy.index.min(),
                coverage_end=proxy.index.max(),
                rows=proxy.shape[0],
                notes=notes,
            )

    fred_series = _fred_download(symbol, start_pad, end_str)
    if fred_series is not None:
        notes.append(f"Yahoo Finance was unavailable, so the app used the FRED fallback for {symbol}.")
        return fred_series, _meta_dict(
            requested_symbol=symbol,
            source_symbol=symbol,
            source_name="FRED",
            proxy_used=False,
            adjusted=False,
            coverage_start=fred_series.index.min(),
            coverage_end=fred_series.index.max(),
            rows=fred_series.shape[0],
            notes=notes,
        )

    return None, _meta_dict(
        requested_symbol=symbol,
        source_symbol=symbol,
        source_name="Unavailable",
        proxy_used=False,
        adjusted=False,
        coverage_start=None,
        coverage_end=None,
        rows=0,
        notes=notes,
    )


# ----------------------------- Domain layer ----------------------------- #
def _month_is_complete(prices: pd.Series, period: pd.Period) -> bool:
    month_slice = prices.loc[prices.index.to_period("M") == period]
    if month_slice.empty:
        return False
    last_obs = month_slice.index.max().normalize()
    calendar_month_end = period.to_timestamp(how="end").normalize()
    return last_obs >= calendar_month_end


def _intra_month_halves(prices: pd.Series, include_partial_last_month: bool = False) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month", "complete"])

    months = pd.period_range(prices.index.min().to_period("M"), prices.index.max().to_period("M"), freq="M")
    out_rows: List[Dict[str, object]] = []

    for period in months:
        month_days = prices.loc[prices.index.to_period("M") == period]
        if month_days.shape[0] < 3:
            continue

        prev_month_days = prices.loc[prices.index.to_period("M") == (period - 1)]
        if prev_month_days.empty:
            continue

        is_complete = _month_is_complete(prices, period)
        if not include_partial_last_month and not is_complete:
            continue

        prev_eom = float(prev_month_days.iloc[-1])
        last_close = float(month_days.iloc[-1])
        n_days = month_days.shape[0]
        mid_idx = (n_days // 2) - 1
        if mid_idx < 0:
            continue
        mid_close = float(month_days.iloc[mid_idx])

        total_ret = (last_close / prev_eom - 1.0) * 100.0
        h1_ret = (mid_close / prev_eom - 1.0) * 100.0
        h2_ret = total_ret - h1_ret

        out_rows.append(
            {
                "period": period,
                "total_ret": total_ret,
                "h1_ret": h1_ret,
                "h2_ret": h2_ret,
                "year": period.year,
                "month": period.month,
                "complete": is_complete,
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def seasonal_stats(prices: pd.Series, start_year: int, end_year: int, include_partial_last_month: bool) -> pd.DataFrame:
    halves = _intra_month_halves(prices, include_partial_last_month=include_partial_last_month)
    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["label"] = MONTH_LABELS

    base_cols = ["hit_rate", "min_ret", "max_ret", "years_observed", "mean_h1", "mean_h2", "mean_total", "median_total"]
    if halves.empty:
        for col in base_cols:
            stats[col] = np.nan
        return stats

    halves = halves[(halves["year"] >= start_year) & (halves["year"] <= end_year)]
    if halves.empty:
        for col in base_cols:
            stats[col] = np.nan
        return stats

    grouped = halves.groupby("month")
    stats["hit_rate"] = grouped["total_ret"].apply(lambda x: (x > 0).mean() * 100.0)
    stats["min_ret"] = grouped["total_ret"].min()
    stats["max_ret"] = grouped["total_ret"].max()
    stats["years_observed"] = grouped["year"].nunique()
    stats["mean_h1"] = grouped["h1_ret"].mean()
    stats["mean_h2"] = grouped["h2_ret"].mean()
    stats["mean_total"] = grouped["total_ret"].mean()
    stats["median_total"] = grouped["total_ret"].median()
    return stats


def _month_paths_prev_eom_equal_weight(prices: pd.Series, month_int: int, start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.Series]:
    px = prices[(prices.index.year >= start_year) & (prices.index.year <= end_year)].copy()
    if px.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    raw_paths: Dict[int, pd.Series] = {}
    for year in sorted(px.index.year.unique()):
        month_slice = px.loc[(px.index.year == year) & (px.index.month == month_int)]
        if month_slice.shape[0] < 3:
            continue

        prev_year = year if month_int > 1 else year - 1
        prev_month_num = month_int - 1 if month_int > 1 else 12
        prev_month_slice = px.loc[(px.index.year == prev_year) & (px.index.month == prev_month_num)]
        if prev_month_slice.empty:
            continue

        prev_eom = float(prev_month_slice.iloc[-1])
        cum_ret = (month_slice / prev_eom - 1.0) * 100.0
        cum_ret.index = pd.RangeIndex(start=1, stop=1 + len(cum_ret), step=1)
        cum_ret.loc[0] = 0.0
        cum_ret = cum_ret.sort_index()
        raw_paths[year] = cum_ret

    if not raw_paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_days = max(int(series.index.max()) for series in raw_paths.values())
    full_index = pd.RangeIndex(0, max_days + 1)
    df = pd.DataFrame(index=full_index)

    for year, series in raw_paths.items():
        df[year] = series.reindex(full_index).ffill()

    avg_path = df.mean(axis=1)
    return df, avg_path


def _avg_calendar_day_for_ordinal(prices: pd.Series, month_int: int, start_year: int, end_year: int, ordinal: int) -> Optional[int]:
    px = prices[(prices.index.year >= start_year) & (prices.index.year <= end_year)]
    if px.empty or ordinal <= 0:
        return None

    day_values: List[int] = []
    for year in sorted(px.index.year.unique()):
        month_slice = px.loc[(px.index.year == year) & (px.index.month == month_int)]
        if len(month_slice) >= ordinal:
            day_values.append(int(month_slice.index[ordinal - 1].day))

    if not day_values:
        return None
    return int(round(np.mean(day_values)))


def _trading_day_ordinal(month_index: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    if month_index.empty:
        return 1
    idx = month_index.sort_values()
    day = pd.Timestamp(when).normalize()
    ordinal = int(np.searchsorted(idx.values, np.datetime64(day), side="right"))
    return max(1, min(ordinal, len(idx)))


def build_month_table(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy().reset_index(drop=True)
    return pd.DataFrame(
        {
            "Month": df["label"],
            "1H %": df["mean_h1"],
            "2H %": df["mean_h2"],
            "Total %": df["mean_total"],
            "Median %": df["median_total"],
            "Hit Rate %": df["hit_rate"],
            "Low %": df["min_ret"],
            "High %": df["max_ret"],
            "Years": df["years_observed"],
        }
    )


# ----------------------------- Chart layer ----------------------------- #
def _format_pct(x: float, decimals: int = 1) -> str:
    return f"{x:+.{decimals}f}%" if pd.notna(x) else "n/a"


def _segment_fill(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, COLORS["green_fill"], COLORS["red_fill"])


def _segment_edge(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, COLORS["green_edge"], COLORS["red_edge"])


def plot_seasonality_figure(stats: pd.DataFrame, title: str) -> plt.Figure:
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]).copy()

    labels = plot_df["label"].tolist()
    mean_h1 = plot_df["mean_h1"].to_numpy(float)
    mean_h2 = plot_df["mean_h2"].to_numpy(float)
    totals = plot_df["mean_total"].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    fig = plt.figure(figsize=(13.2, 7.2), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")

    x = np.arange(len(labels))

    ax.bar(
        x,
        mean_h1,
        width=0.78,
        color=_segment_fill(mean_h1),
        edgecolor=_segment_edge(mean_h1),
        linewidth=1.0,
        zorder=2,
    )

    ax.bar(
        x,
        mean_h2,
        width=0.78,
        bottom=mean_h1,
        color=_segment_fill(mean_h2),
        edgecolor=_segment_edge(mean_h2),
        linewidth=1.0,
        hatch="///",
        zorder=2,
    )

    yerr = np.abs(np.vstack([totals - min_ret, max_ret - totals]))
    ax.errorbar(
        x,
        totals,
        yerr=yerr,
        fmt="none",
        ecolor=COLORS["muted"],
        elinewidth=1.4,
        alpha=0.75,
        capsize=5,
        zorder=3,
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean return (%)", fontsize=11, weight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="y", linestyle="--", color=COLORS["grid"], linewidth=0.7, alpha=0.9, zorder=1)
    ax.set_axisbelow(True)

    lower = min(np.nanmin(min_ret), np.nanmin(totals), 0.0)
    upper = max(np.nanmax(max_ret), np.nanmax(totals), 0.0)
    pad = 0.10 * max(abs(lower), abs(upper), 1.0)
    ax.set_ylim(lower - pad, upper + pad)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax2 = ax.twinx()
    ax2.scatter(x, hit, marker="D", s=48, color=COLORS["line"], zorder=4)
    ax2.set_ylabel("Hit rate", fontsize=11, weight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax2.spines["top"].set_visible(False)

    legend_handles = [Patch(facecolor="white", edgecolor=COLORS["line"], hatch="///", label="Second half")]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.suptitle(title, fontsize=16, weight="bold", y=0.97)
    fig.tight_layout()
    return fig


def plot_intra_month_curve_figure(prices: pd.Series, month_int: int, start_year: int, end_year: int, symbol_shown: str) -> Tuple[plt.Figure, Optional[pd.Series]]:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight(prices, month_int, start_year, end_year)

    fig = plt.figure(figsize=(13.2, 7.2), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, facecolor="white")
    stats_box = None

    if avg_sel.empty or df_sel.empty:
        ax.text(0.5, 0.5, "Not enough data to compute intra-month curve", ha="center", va="center", fontsize=12, color=COLORS["line"])
        ax.axis("off")
        fig.tight_layout()
        return fig, stats_box

    std_sel = df_sel.std(axis=1)
    x_vals = avg_sel.index.values
    y_vals = avg_sel.values

    ax.axhline(0.0, color=COLORS["muted"], linestyle=":", linewidth=1.0, zorder=1)
    ax.fill_between(
        x_vals,
        (avg_sel - std_sel).values,
        (avg_sel + std_sel).values,
        color=COLORS["band"],
        alpha=0.35,
        zorder=1.5,
        label="±1σ",
    )

    ax.plot(x_vals, y_vals, linewidth=2.5, color=COLORS["line"], linestyle="-", label=f"Average {MONTH_LABELS[month_int - 1]} path")

    today = pd.Timestamp.today().normalize()
    current_year = today.year
    cur_path = None

    current_month_slice = prices.loc[(prices.index.year == current_year) & (prices.index.month == month_int)]
    prev_mask = (
        (prices.index.year == (current_year if month_int > 1 else current_year - 1))
        & (prices.index.month == (month_int - 1 if month_int > 1 else 12))
    )
    prev_month_slice = prices.loc[prev_mask]

    if not current_month_slice.empty and not prev_month_slice.empty:
        prev_eom = float(prev_month_slice.iloc[-1])
        cur_cum = (current_month_slice / prev_eom - 1.0) * 100.0
        cur_cum.index = pd.RangeIndex(start=1, stop=1 + len(cur_cum), step=1)
        cur_cum.loc[0] = 0.0
        cur_cum = cur_cum.sort_index()
        cur_path = cur_cum

        if len(cur_path) >= 3:
            ax.plot(cur_path.index.values, cur_path.values, linewidth=2.0, color=COLORS["current"], alpha=0.45, linestyle="--", label=str(current_year))

    avg_ex = avg_sel.drop(index=0, errors="ignore")
    if not avg_ex.empty:
        low_idx = int(avg_ex.idxmin())
        high_idx = int(avg_ex.idxmax())
        low_val = float(avg_ex.loc[low_idx])
        high_val = float(avg_ex.loc[high_idx])

        ax.scatter([low_idx, high_idx], [low_val, high_val], s=38, color=COLORS["line"], zorder=3)

        low_dom = _avg_calendar_day_for_ordinal(prices, month_int, start_year, end_year, low_idx)
        high_dom = _avg_calendar_day_for_ordinal(prices, month_int, start_year, end_year, high_idx)

        def _annotate_point(text: str, xy: Tuple[float, float], offset: Tuple[float, float]) -> None:
            ax.annotate(
                text,
                xy=xy,
                xytext=(xy[0] + offset[0], xy[1] + offset[1]),
                textcoords="data",
                arrowprops=dict(arrowstyle="-", color=COLORS["line"], lw=0.8, shrinkA=2, shrinkB=2),
                fontsize=9.2,
                color=COLORS["line"],
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["line"], lw=0.8),
            )

        if low_dom is not None:
            _annotate_point(
                f"Avg low\nDay {low_idx} | ~{MONTH_LABELS[month_int - 1]} {low_dom} | {low_val:+.2f}%",
                (low_idx, low_val),
                (0.6, -0.9),
            )
        if high_dom is not None:
            _annotate_point(
                f"Avg high\nDay {high_idx} | ~{MONTH_LABELS[month_int - 1]} {high_dom} | {high_val:+.2f}%",
                (high_idx, high_val),
                (-3.0, 0.9),
            )

    ax.set_title(f"{symbol_shown} {MONTH_LABELS[month_int - 1]} intra-month path", fontsize=16, weight="bold", pad=10)
    ax.set_xlabel(f"Trading day of {MONTH_LABELS[month_int - 1]} (Day 0 = prior month-end)", fontsize=10, weight="bold")
    ax.set_ylabel("Return from prior month-end (%)", fontsize=10, weight="bold")
    ax.grid(axis="y", linestyle="--", color=COLORS["grid"], alpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if today.month == month_int and not current_month_slice.empty:
        tday_ord = _trading_day_ordinal(current_month_slice.index, today)
        ax.axvline(tday_ord, color=COLORS["today"], linestyle=":", linewidth=1.2, zorder=2)

        if tday_ord in avg_sel.index:
            ax.scatter(tday_ord, avg_sel.loc[tday_ord], s=34, color=COLORS["line"], zorder=4)
        if cur_path is not None and tday_ord in cur_path.index:
            ax.scatter(
                tday_ord,
                cur_path.loc[tday_ord],
                s=42,
                facecolors="white",
                edgecolors=COLORS["line"],
                linewidths=1.0,
                zorder=4,
            )

        level_t = df_sel.loc[tday_ord]
        level_end = df_sel.loc[df_sel.index.max()]
        forward = (level_end - level_t).dropna()
        if not forward.empty:
            stats_box = pd.Series(
                {
                    "Mean": forward.mean(),
                    "Median": forward.median(),
                    "Hit Rate": (forward > 0).mean() * 100.0,
                    "N": float(forward.shape[0]),
                }
            )

    y_min = float((avg_sel - std_sel).min())
    y_max = float((avg_sel + std_sel).max())
    pad_y = 0.10 * max(abs(y_min), abs(y_max), 1.0)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig, stats_box


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ----------------------------- UI helpers ----------------------------- #
def style_month_table(df: pd.DataFrame):
    numeric_cols = ["1H %", "2H %", "Total %", "Median %", "Hit Rate %", "Low %", "High %"]

    def color_returns(val: float) -> str:
        if pd.isna(val):
            return ""
        if val > 0:
            return "background-color: #d9f2e4;"
        if val < 0:
            return "background-color: #f7d9d7;"
        return "background-color: #f2f2f2;"

    styler = df.style.format(
        {
            "1H %": "{:+.2f}%",
            "2H %": "{:+.2f}%",
            "Total %": "{:+.2f}%",
            "Median %": "{:+.2f}%",
            "Hit Rate %": "{:.0f}%",
            "Low %": "{:+.2f}%",
            "High %": "{:+.2f}%",
            "Years": "{:.0f}",
        }
    )

    for col in numeric_cols:
        styler = styler.map(color_returns, subset=[col])

    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#f0f0f0"), ("font-weight", "600"), ("text-align", "center")]},
            {"selector": "td", "props": [("padding", "6px 8px")]},
        ]
    )
    return styler


def coverage_text(meta: Dict[str, object]) -> str:
    start = _meta_ts(meta.get("coverage_start"))
    end = _meta_ts(meta.get("coverage_end"))
    if start is None or end is None:
        return "Coverage unavailable"
    return f"{start.date()} to {end.date()}"


def years_from_preset(prices: pd.Series, preset: str, custom_start_year: int, custom_end_year: int) -> Tuple[int, int]:
    latest_year = int(min(pd.Timestamp.today().year, custom_end_year))
    if prices.empty:
        return custom_start_year, custom_end_year

    earliest_year = int(prices.index.min().year)
    if preset == "Full history":
        return earliest_year, latest_year

    window = PRESET_WINDOWS.get(preset)
    if isinstance(window, int):
        return max(earliest_year, latest_year - window + 1), latest_year

    return custom_start_year, custom_end_year


# ----------------------------- App ----------------------------- #
if "seasonality_symbol" not in st.session_state:
    st.session_state["seasonality_symbol"] = "^SPX"
if "seasonality_start_year" not in st.session_state:
    st.session_state["seasonality_start_year"] = 2020
if "seasonality_end_year" not in st.session_state:
    st.session_state["seasonality_end_year"] = dt.datetime.today().year
if "seasonality_preset" not in st.session_state:
    st.session_state["seasonality_preset"] = "Custom"
if "seasonality_month_choice" not in st.session_state:
    st.session_state["seasonality_month_choice"] = pd.Timestamp.today().month
if "seasonality_include_partial" not in st.session_state:
    st.session_state["seasonality_include_partial"] = False

st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.subheader("About This Tool")
    st.markdown(
        """
This dashboard shows how a symbol tends to behave by calendar month and within each month.

The top chart splits every month into first-half and second-half return contributions, overlays the full observed monthly range, and shows hit rate so you can see both direction and consistency.

The intra-month chart anchors each year to the prior month-end and builds an equal-weight average path by trading day, which helps frame where a month usually strengthens or weakens.

The app uses Yahoo Finance as the primary source, falls back to FRED for selected index history, flags proxy usage when needed, and excludes incomplete months from historical averages unless you explicitly include them.
        """
    )
    st.subheader("Quick symbols")
    st.caption("Click a preset to update the dashboard.")
    sidebar_cols = st.columns(2)
    for idx, preset_symbol in enumerate(POPULAR_SYMBOLS):
        if sidebar_cols[idx % 2].button(preset_symbol, use_container_width=True):
            st.session_state["seasonality_symbol"] = preset_symbol

current_year = dt.datetime.today().year

control_1, control_2, control_3, control_4 = st.columns([2.3, 1.2, 1.0, 1.0])
with control_1:
    symbol = st.text_input("Ticker symbol", key="seasonality_symbol").strip().upper()
with control_2:
    preset = st.selectbox(
        "History window",
        options=list(PRESET_WINDOWS.keys()),
        key="seasonality_preset",
    )
with control_3:
    start_year_input = st.number_input(
        "Start year",
        min_value=1900,
        max_value=current_year,
        key="seasonality_start_year",
        disabled=st.session_state["seasonality_preset"] != "Custom",
    )
with control_4:
    end_year_input = st.number_input(
        "End year",
        min_value=int(st.session_state["seasonality_start_year"]),
        max_value=current_year,
        key="seasonality_end_year",
        disabled=st.session_state["seasonality_preset"] != "Custom",
    )

sub_1, sub_2, sub_3 = st.columns([1.25, 1.1, 2.65])
with sub_1:
    include_partial_last_month = st.checkbox(
        "Include incomplete current month in history",
        key="seasonality_include_partial",
    )
with sub_2:
    month_choice = st.selectbox(
        "Intra-month chart",
        options=list(range(1, 13)),
        key="seasonality_month_choice",
        format_func=lambda m: MONTH_LABELS[m - 1],
    )
with sub_3:
    st.caption("Changing any input reruns the dashboard. Press Enter in the ticker box after editing the symbol.")

requested_start = f"{int(start_year_input)}-01-01"
requested_end = f"{int(end_year_input)}-12-31"

with st.spinner("Fetching and analyzing data..."):
    prices, meta = fetch_prices_with_meta(symbol, requested_start, requested_end)

if prices is None or prices.empty:
    st.error(f"No usable data found for '{symbol}' in the requested range.")
    notes = meta.get("notes", [])
    if notes:
        for note in notes:
            st.warning(note)
    st.stop()

first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

start_year, end_year = years_from_preset(prices, preset, int(start_year_input), int(end_year_input))
stats = seasonal_stats(prices, start_year, end_year, include_partial_last_month)
valid_stats = stats.dropna(subset=["mean_total"])

if valid_stats.empty:
    st.error("Insufficient data in the selected window to compute seasonality.")
    st.stop()

best_idx = valid_stats["mean_total"].idxmax()
worst_idx = valid_stats["mean_total"].idxmin()
best = stats.loc[best_idx]
worst = stats.loc[worst_idx]

info_1, info_2, info_3, info_4 = st.columns(4)
info_1.metric("Source", f"{meta['source_name']} | {meta['source_symbol']}")
info_2.metric("Coverage", coverage_text(meta))
info_3.metric("Best month", f"{best['label']} {_format_pct(best['mean_total'], 2)}")
info_4.metric("Worst month", f"{worst['label']} {_format_pct(worst['mean_total'], 2)}")

info_5, info_6, info_7, info_8 = st.columns(4)
info_5.metric("Best hit rate", f"{stats['hit_rate'].max():.0f}%")
info_6.metric("Worst hit rate", f"{stats['hit_rate'].min():.0f}%")
info_7.metric("Years used", f"{start_year} to {end_year}")
info_8.metric("Rows", f"{int(meta['rows'])}")

if bool(meta.get("proxy_used", False)):
    st.warning(
        f"You asked for {meta['requested_symbol']}, but the app used {meta['source_symbol']} as a proxy. "
        "Cross-instrument comparisons need extra care."
    )

notes = meta.get("notes", [])
if notes:
    with st.expander("Data source notes", expanded=False):
        for note in notes:
            st.write(f"- {note}")

overview_tab, intramonth_tab, data_tab, methodology_tab = st.tabs(["Overview", "Intra-month", "Data table", "Methodology"])

with overview_tab:
    seasonality_fig = plot_seasonality_figure(stats, f"{meta['source_symbol']} seasonality | {start_year}-{end_year}")
    seasonality_png = figure_to_png_bytes(seasonality_fig)
    st.image(seasonality_png, use_container_width=True)
    st.caption(
        "Bars show average first-half and second-half contributions. Whiskers show the full observed monthly range. Diamonds show hit rate."
    )
    st.download_button(
        "Download overview PNG",
        data=seasonality_png,
        file_name=f"{meta['source_symbol']}_seasonality_{start_year}_{end_year}.png",
        mime="image/png",
    )

with intramonth_tab:
    curve_fig, forward_stats = plot_intra_month_curve_figure(
        prices=prices,
        month_int=month_choice,
        start_year=start_year,
        end_year=end_year,
        symbol_shown=str(meta["source_symbol"]),
    )
    curve_png = figure_to_png_bytes(curve_fig)
    st.image(curve_png, use_container_width=True)

    if forward_stats is not None:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Mean fwd to month-end", _format_pct(float(forward_stats["Mean"]), 2))
        f2.metric("Median fwd to month-end", _format_pct(float(forward_stats["Median"]), 2))
        f3.metric("Hit rate", f"{float(forward_stats['Hit Rate']):.0f}%")
        f4.metric("Observations", f"{int(forward_stats['N'])}")

    st.download_button(
        "Download intra-month PNG",
        data=curve_png,
        file_name=f"{meta['source_symbol']}_{MONTH_LABELS[month_choice - 1].lower()}_intramonth_{start_year}_{end_year}.png",
        mime="image/png",
    )

with data_tab:
    month_table = build_month_table(stats)
    st.dataframe(style_month_table(month_table), use_container_width=True, hide_index=True)
    csv_bytes = month_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{meta['source_symbol']}_seasonality_table_{start_year}_{end_year}.csv",
        mime="text/csv",
    )

with methodology_tab:
    st.markdown(
        f"""
**Window used:** {start_year} to {end_year}

**Seasonality logic**
- Monthly returns are anchored to the prior month-end.
- First half uses the midpoint trading day of the month.
- Second half equals total month return less first-half return.
- Historical averages exclude incomplete months unless you include them.

**Intra-month curve**
- Each year starts at Day 0, which is the prior month-end.
- Paths are aligned to a common trading-day grid.
- Shorter months are forward-filled so the endpoint matches the equal-weight average month return across years.

**Caveats**
- A proxy can be used if the requested instrument is unavailable.
- A price index and a total-return ETF are different instruments.
- Min and max whiskers show extremes, so outliers can matter a lot.
        """
    )

st.caption("© 2026 AD Fund Management LP")
