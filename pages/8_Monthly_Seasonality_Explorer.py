import io
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from adfm_core.monthly_returns_matrix import (
    build_monthly_returns_frame,
    render_monthly_returns_matrix,
)
from adfm_core.ui import render_footer

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    _yf_cache_dir = Path(tempfile.gettempdir()) / "adfm-yfinance-cache"
    _yf_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_yf_cache_dir))
except Exception:
    pass

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None


# =========================
# CONFIG
# =========================

FALLBACK_MAP = {
    "^SPX": "SP500",
    "^GSPC": "SP500",
    "^DJI": "DJIA",
    "^IXIC": "NASDAQCOM",
}

DXY_FALLBACKS = ["DX=F", "UUP"]

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

POS_GREEN = "#52b788"
NEG_RED = "#e85d5d"
NEUTRAL_GREY = "#8b949e"
LIGHT_GREEN = "#d9f2e4"
LIGHT_RED = "#f7d9d7"
LIGHT_GREY = "#f2f2f2"
BACKGROUND = "#fcfcfb"

CACHE_TTL_SECONDS = 3600


# =========================
# STREAMLIT SETUP
# =========================

st.set_page_config(page_title="Monthly Seasonality Explorer", layout="wide")
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
    .adfm-small-note {
        color: #666666;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: -0.35rem;
        margin-bottom: 0.65rem;
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


# =========================
# DATA HELPERS
# =========================


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _yf_download(
    symbol: str, start: str, end: str, retries: int = 3
) -> Optional[pd.Series]:
    for n in range(retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty and "Close" in df:
                ser = df["Close"]
                if isinstance(ser, pd.DataFrame):
                    ser = ser.iloc[:, 0]
                ser = pd.to_numeric(ser, errors="coerce").dropna()
                ser.index = pd.to_datetime(ser.index).tz_localize(None)
                if not ser.empty:
                    return ser.rename("Close")
        except Exception:
            pass

        time.sleep(1.25 * (n + 1))

    return None


def _fred_series(series_code: str, start: str, end: str) -> Optional[pd.Series]:
    if pdr is None:
        return None

    try:
        df = pdr.DataReader(series_code, "fred", start, end)
        if df is None or df.empty or series_code not in df.columns:
            return None

        ser = pd.to_numeric(df[series_code], errors="coerce").dropna()
        ser.index = pd.to_datetime(ser.index).tz_localize(None)

        if ser.empty:
            return None

        return ser.rename(series_code)

    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    symbol = _clean_symbol(symbol)
    end = min(pd.Timestamp(end), _today()).strftime("%Y-%m-%d")

    start_pad_dt = pd.Timestamp(start) - pd.DateOffset(days=45)
    start_pad = start_pad_dt.strftime("%Y-%m-%d")

    series = _yf_download(symbol, start_pad, end)
    if series is not None:
        return series

    if symbol == "SPY":
        series = _yf_download("^GSPC", start_pad, end)
        if series is not None:
            return series

    if symbol in FALLBACK_MAP:
        fred_tk = FALLBACK_MAP[symbol]
        series = _fred_series(fred_tk, start_pad, end)
        if series is not None:
            return series.rename("Close")

    return None


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_regime_market_series(start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start) - pd.DateOffset(years=1)
    end_dt = min(pd.Timestamp(end), _today()) + pd.DateOffset(days=5)

    out = pd.DataFrame()

    vix = _yf_download(
        "^VIX", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )
    if vix is not None:
        out["vix"] = vix

    tnx = _yf_download(
        "^TNX", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )
    if tnx is not None:
        out["tnx"] = tnx / 10.0

    dxy = _yf_download(
        "DX-Y.NYB", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )
    if dxy is None:
        for fallback in DXY_FALLBACKS:
            dxy = _yf_download(
                fallback, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            )
            if dxy is not None:
                break

    if dxy is not None:
        out["dxy"] = dxy

    if out.empty:
        return out

    out = out.sort_index()
    return out


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_regime_data(start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start) - pd.DateOffset(years=2)
    end_dt = min(pd.Timestamp(end), _today()) + pd.DateOffset(days=31)

    usrec = _fred_series(
        "USREC", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )
    fedfunds = _fred_series(
        "FEDFUNDS", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    )

    idx_start = start_dt.to_period("M")
    idx_end = end_dt.to_period("M")
    monthly_index = pd.period_range(idx_start, idx_end, freq="M")

    regime = pd.DataFrame(index=monthly_index)

    if usrec is not None:
        usrec_m = usrec.resample("ME").last()
        usrec_m.index = usrec_m.index.to_period("M")
        regime["is_recession"] = usrec_m.reindex(monthly_index)
        regime["is_recession"] = regime["is_recession"].fillna(0).astype(int)
    else:
        regime["is_recession"] = 0

    if fedfunds is not None:
        ff_m = fedfunds.resample("ME").last()
        ff_m.index = ff_m.index.to_period("M")
        regime["fedfunds"] = ff_m.reindex(monthly_index)

        delta_3m = regime["fedfunds"].diff(3)

        regime["fed_regime"] = np.where(
            delta_3m > 0.05,
            "Hiking",
            np.where(delta_3m < -0.05, "Cutting", "Steady"),
        )
        regime.loc[regime["fedfunds"].isna(), "fed_regime"] = "Unknown"
    else:
        regime["fedfunds"] = np.nan
        regime["fed_regime"] = "Unknown"

    regime["regime_cycle"] = np.where(
        regime["is_recession"] == 1, "Recession", "Expansion"
    )
    return regime


def build_monthly_regime_features(regime_daily: pd.DataFrame) -> pd.DataFrame:
    if regime_daily is None or regime_daily.empty:
        return pd.DataFrame()

    monthly = pd.DataFrame()
    daily = regime_daily.copy().sort_index()

    if "vix" in daily:
        vix_m = daily["vix"].resample("ME").mean()
        monthly["avg_vix"] = vix_m
        monthly["vix_bucket"] = pd.cut(
            monthly["avg_vix"],
            bins=[-np.inf, 15, 20, 25, np.inf],
            labels=["VIX <15", "VIX 15-20", "VIX 20-25", "VIX >25"],
        ).astype(str)

    if "tnx" in daily:
        tnx_m = daily["tnx"].resample("ME").last()
        tnx_delta = tnx_m.diff(3)
        monthly["teny"] = tnx_m
        monthly["teny_3m_chg"] = tnx_delta
        monthly["teny_trend"] = np.where(
            tnx_delta > 0.15,
            "10Y rising",
            np.where(tnx_delta < -0.15, "10Y falling", "10Y flat"),
        )

    if "dxy" in daily:
        dxy_m = daily["dxy"].resample("ME").last()
        dxy_delta = dxy_m.pct_change(3) * 100.0
        monthly["dxy"] = dxy_m
        monthly["dxy_3m_chg_pct"] = dxy_delta
        monthly["dxy_trend"] = np.where(
            dxy_delta > 1.5,
            "Dollar rising",
            np.where(dxy_delta < -1.5, "Dollar falling", "Dollar flat"),
        )

    if monthly.empty:
        return monthly

    monthly.index = monthly.index.to_period("M")
    return monthly


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
    if prices.empty:
        return pd.DataFrame(columns=["total_ret", "h1_ret", "h2_ret", "year", "month"])

    out_rows = []
    months = pd.period_range(
        prices.index.min().to_period("M"),
        prices.index.max().to_period("M"),
        freq="M",
    )

    current_month = _today().to_period("M")

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
                "is_complete_month": bool(m < current_month),
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df.set_index(pd.PeriodIndex(df["period"], freq="M"), inplace=True)
    df.drop(columns=["period"], inplace=True)
    return df


def build_filter_table(
    prices: pd.Series,
    regime_df: pd.DataFrame,
    market_regime_df: pd.DataFrame,
) -> pd.DataFrame:
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

    if market_regime_df is not None and not market_regime_df.empty:
        df = df.join(market_regime_df, how="left")

    df["regime_cycle"] = df["regime_cycle"].fillna("Unknown")
    df["fed_regime"] = df["fed_regime"].fillna("Unknown")
    df["vix_bucket"] = df.get(
        "vix_bucket", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")
    df["teny_trend"] = df.get(
        "teny_trend", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")
    df["dxy_trend"] = df.get(
        "dxy_trend", pd.Series(index=df.index, dtype=object)
    ).fillna("Unknown")

    return df


def _latest_complete_year(prices: pd.Series) -> int:
    if prices.empty:
        return _today().year - 1

    latest_period = prices.index.max().to_period("M")
    current_period = _today().to_period("M")
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
    if preset == "Last 5 years":
        return int(latest_complete_year - 4), int(latest_complete_year)
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
    complete_months_only: bool,
    fed_filter: str,
    vix_filter: str,
    teny_filter: str,
    dxy_filter: str,
) -> pd.DataFrame:
    df = filter_table.copy()

    df = df[(df["year"] >= int(start_year)) & (df["year"] <= int(end_year))]

    if complete_months_only:
        df = df[df["is_complete_month"]]

    if cycle_filter != "All years":
        df = df[df["pres_cycle_bucket"] == cycle_filter]

    if fed_filter != "All Fed regimes":
        df = df[df["fed_regime"] == fed_filter]

    if vix_filter != "All VIX regimes":
        df = df[df["vix_bucket"] == vix_filter]

    if teny_filter != "All 10Y regimes":
        df = df[df["teny_trend"] == teny_filter]

    if dxy_filter != "All dollar regimes":
        df = df[df["dxy_trend"] == dxy_filter]

    return df.sort_index()


def seasonal_stats_from_filtered(filtered_halves: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=pd.Index(range(1, 13), name="month"))
    stats["label"] = MONTH_LABELS

    if filtered_halves.empty:
        for col in [
            "hit_rate",
            "min_ret",
            "max_ret",
            "years_observed",
            "mean_h1",
            "mean_h2",
            "mean_total",
            "median_total",
            "p25_total",
            "p75_total",
            "downside_freq",
            "observations",
        ]:
            stats[col] = np.nan
        return stats

    grouped = filtered_halves.groupby("month")

    stats["hit_rate"] = grouped["total_ret"].apply(lambda x: (x > 0).mean() * 100.0)
    stats["min_ret"] = grouped["total_ret"].min()
    stats["max_ret"] = grouped["total_ret"].max()
    stats["years_observed"] = grouped["year"].nunique()
    stats["mean_h1"] = grouped["h1_ret"].mean()
    stats["mean_h2"] = grouped["h2_ret"].mean()
    stats["mean_total"] = grouped["total_ret"].mean()
    stats["median_total"] = grouped["total_ret"].median()
    stats["p25_total"] = grouped["total_ret"].quantile(0.25)
    stats["p75_total"] = grouped["total_ret"].quantile(0.75)
    stats["downside_freq"] = grouped["total_ret"].apply(
        lambda x: (x < -2.0).mean() * 100.0
    )
    stats["observations"] = grouped["total_ret"].count()

    return stats


def _format_pct(x: float) -> str:
    return f"{x:+.1f}%" if pd.notna(x) else "n/a"


def _cell_colors(values: List[List[float]]) -> List[List[str]]:
    colors = []
    for row in values:
        row_colors = []
        for v in row:
            if pd.notna(v) and v > 0:
                row_colors.append(LIGHT_GREEN)
            elif pd.notna(v) and v < 0:
                row_colors.append(LIGHT_RED)
            else:
                row_colors.append(LIGHT_GREY)
        colors.append(row_colors)
    return colors


# =========================
# ORIGINAL CHART 1
# =========================


def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(
        subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]
    ).copy()

    labels = plot_df["label"].tolist()
    mean_h1 = plot_df["mean_h1"].to_numpy(float)
    mean_h2 = plot_df["mean_h2"].to_numpy(float)
    totals = plot_df["mean_total"].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    min_ret = plot_df["min_ret"].to_numpy(float)
    max_ret = plot_df["max_ret"].to_numpy(float)

    fig = plt.figure(figsize=(18, 9.6), dpi=220)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[6.2, 1.55], hspace=0.03)
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(labels))

    def seg_face(values):
        return np.where(values >= 0, POS_GREEN, NEG_RED)

    def seg_edge(values):
        return np.where(values >= 0, "#1f7a4f", "#8b1e1a")

    ax1.bar(
        x,
        mean_h1,
        width=0.84,
        color=seg_face(mean_h1),
        edgecolor=seg_edge(mean_h1),
        linewidth=1.2,
        zorder=2,
        alpha=0.95,
    )

    bars2 = ax1.bar(
        x,
        mean_h2,
        width=0.84,
        bottom=mean_h1,
        color=seg_face(mean_h2),
        edgecolor=seg_edge(mean_h2),
        linewidth=1.2,
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
        elinewidth=1.8,
        alpha=0.75,
        capsize=7,
        zorder=3,
    )

    ax1.set_xticks(x, labels)
    ax1.tick_params(axis="x", labelsize=13)
    ax1.tick_params(axis="y", labelsize=13)
    ax1.set_ylabel("Mean return (%)", weight="bold", fontsize=15)
    ax1.margins(x=0.02)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    pad = 0.1 * max(abs(min_ret.min()), abs(max_ret.max())) if len(min_ret) > 0 else 0.5
    ymin = min(min_ret.min(), totals.min(), 0) - pad
    ymax = max(max_ret.max(), totals.max(), 0) + pad
    ax1.set_ylim(ymin, ymax)
    ax1.grid(
        axis="y", linestyle="--", color="lightgrey", linewidth=0.7, alpha=0.75, zorder=1
    )

    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=110, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold", fontsize=15)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    legend = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label="Second half (hatched)",
        )
    ]
    ax1.legend(handles=legend, loc="upper left", frameon=False, fontsize=13)

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
        bbox=[0.035, 0.18, 0.93, 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)

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
            cell.set_linewidth(1.0)

    fig.suptitle(title, fontsize=24, weight="bold", y=0.965)
    fig.subplots_adjust(left=0.055, right=0.94, top=0.88, bottom=0.09)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_monthly_profile(
    stats: pd.DataFrame,
    title: str,
    selected_month: int,
) -> io.BytesIO:
    """Render the compact monthly profile paired with the selectable matrix."""

    plot_df = stats.reindex(range(1, 13)).copy()
    totals = plot_df["mean_total"].to_numpy(float)
    hit_rate = plot_df["hit_rate"].to_numpy(float)
    p25 = plot_df["p25_total"].to_numpy(float)
    p75 = plot_df["p75_total"].to_numpy(float)
    x = np.arange(12)

    colors = np.where(
        totals > 0, POS_GREEN, np.where(totals < 0, NEG_RED, NEUTRAL_GREY)
    )
    edges = np.array(
        [
            "#1f7a4f" if value > 0 else "#8b1e1a" if value < 0 else "#6b7280"
            for value in totals
        ]
    )
    linewidths = np.full(12, 1.0)
    selected_index = max(0, min(int(selected_month) - 1, 11))
    edges[selected_index] = "#111111"
    linewidths[selected_index] = 2.8

    fig, ax1 = plt.subplots(figsize=(9.0, 5.7), dpi=190, facecolor=BACKGROUND)
    ax1.set_facecolor(BACKGROUND)
    ax1.axvspan(
        selected_index - 0.48,
        selected_index + 0.48,
        color="#111111",
        alpha=0.045,
        zorder=0,
    )
    ax1.axhline(0.0, color="#9ca3af", linewidth=1.0, zorder=1)
    bars = ax1.bar(
        x,
        totals,
        width=0.72,
        color=colors,
        edgecolor=edges,
        linewidth=linewidths,
        alpha=0.92,
        label="Average return",
        zorder=2,
    )

    lower = np.maximum(totals - p25, 0.0)
    upper = np.maximum(p75 - totals, 0.0)
    ax1.errorbar(
        x,
        totals,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#6b7280",
        elinewidth=1.2,
        capsize=3,
        alpha=0.72,
        zorder=3,
    )

    for index, bar in enumerate(bars):
        if pd.notna(totals[index]):
            offset = 0.16 if totals[index] >= 0 else -0.16
            va = "bottom" if totals[index] >= 0 else "top"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                totals[index] + offset,
                f"{totals[index]:+.1f}%",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold" if index == selected_index else "normal",
            )

    ax2 = ax1.twinx()
    hit_line = ax2.plot(
        x,
        hit_rate,
        color="#111111",
        marker="D",
        markersize=4.2,
        linewidth=1.35,
        label="Positive-return hit rate",
        zorder=4,
    )[0]
    ax2.scatter(
        [selected_index],
        [hit_rate[selected_index]],
        s=54,
        facecolor="#111111",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )

    ax1.set_xticks(x, MONTH_LABELS)
    ax1.set_ylabel("Average return (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Hit rate", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.grid(axis="y", linestyle="--", color="#d9d9d9", linewidth=0.7, zorder=0)
    ax1.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)
    ax1.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax1.legend(
        [bars, hit_line],
        ["Average return", "Hit rate"],
        frameon=False,
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout(pad=1.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=190, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================
# INTRA-MONTH HELPERS
# =========================


def _month_paths_prev_eom_equal_weight_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if filtered_halves.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    target_periods = filtered_halves.loc[
        filtered_halves["month"] == month_int
    ].index.tolist()
    if not target_periods:
        return pd.DataFrame(), pd.Series(dtype=float)

    raw_paths = {}

    for p in target_periods:
        y = p.year
        m_num = p.month

        month_days = prices.loc[
            (prices.index.year == y) & (prices.index.month == m_num)
        ]
        if month_days.shape[0] < 3:
            continue

        prev_p = p - 1
        prev_month_days = prices.loc[
            (prices.index.year == prev_p.year) & (prices.index.month == prev_p.month)
        ]
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


def _year_month_path(prices: pd.Series, year: int, month_int: int) -> pd.Series:
    month_prices = prices.loc[
        (prices.index.year == int(year)) & (prices.index.month == int(month_int))
    ]
    previous_period = pd.Period(year=int(year), month=int(month_int), freq="M") - 1
    previous_prices = prices.loc[
        (prices.index.year == previous_period.year)
        & (prices.index.month == previous_period.month)
    ]
    if month_prices.empty or previous_prices.empty:
        return pd.Series(dtype=float)
    path = (month_prices / float(previous_prices.iloc[-1]) - 1.0) * 100.0
    path.index = pd.RangeIndex(start=1, stop=1 + len(path), step=1)
    path.loc[0] = 0.0
    return path.sort_index()


def _avg_calendar_day_for_ordinal_from_filtered(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    ordinal: int,
) -> Optional[int]:
    month_periods = filtered_halves.loc[
        filtered_halves["month"] == month_int
    ].index.tolist()
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
    comparison_year: Optional[int] = None,
) -> Dict[str, Any]:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(
        prices, filtered_halves, month_int
    )

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
        "comparison_year": int(comparison_year or _today().year),
        "comparison_year_month_available": False,
        "comparison_year_end": np.nan,
        "comparison_vs_hist_end_gap": np.nan,
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
        summary["avg_low_dom"] = _avg_calendar_day_for_ordinal_from_filtered(
            prices, filtered_halves, month_int, low_idx
        )

        summary["avg_high_day"] = high_idx
        summary["avg_high_val"] = float(avg_ex.loc[high_idx])
        summary["avg_high_dom"] = _avg_calendar_day_for_ordinal_from_filtered(
            prices, filtered_halves, month_int, high_idx
        )

    for k in [5, 10, 15, 20]:
        if k in avg_sel.index:
            summary[f"day{k}_val"] = float(avg_sel.loc[k])

    d10 = summary["day10_val"]
    month_end = summary["avg_month_end"]
    summary["front_loaded"] = bool(
        pd.notna(d10) and pd.notna(month_end) and abs(d10) >= 0.6 * abs(month_end)
    )

    if summary["avg_low_day"] is not None and pd.notna(month_end):
        summary["recovery_strength"] = float(month_end - summary["avg_low_val"])

    comparison_path = _year_month_path(
        prices, int(summary["comparison_year"]), month_int
    )
    if not comparison_path.empty:
        summary["comparison_year_month_available"] = True
        summary["comparison_year_end"] = float(comparison_path.iloc[-1])
        if pd.notna(summary["avg_month_end"]):
            summary["comparison_vs_hist_end_gap"] = float(
                summary["comparison_year_end"] - summary["avg_month_end"]
            )

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

    comparison_year = int(summary.get("comparison_year", _today().year))
    if summary.get("comparison_year_month_available", False):
        path_title = f"{comparison_year} vs filtered path"
        path_val = f"{summary['comparison_year_end']:+.2f}%"
        path_sub = f"{summary['comparison_vs_hist_end_gap']:+.2f}pp versus the filtered average month-end return."
    elif summary.get("front_loaded") is True:
        path_title = "Path profile"
        path_val = "Front-half weighted"
        path_sub = "A large share of the month is usually realized by Day 10."
    elif summary.get("front_loaded") is False:
        path_title = "Path profile"
        path_val = "Back-half weighted"
        path_sub = "A meaningful share of the move tends to arrive after Day 10."
    else:
        path_title = "Path profile"
        path_val = "n/a"
        path_sub = "Not enough data."

    html = f"""
    <div class="adfm-card-row">
        <div class="adfm-card">
            <div class="adfm-card-label">Average month-end return</div>
            <div class="adfm-card-value">{summary["avg_month_end"]:+.2f}%</div>
            <div class="adfm-card-sub">Equal-weighted across {summary["sample_months"]} month observations and {summary["sample_years"]} distinct years</div>
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
            <div class="adfm-card-label">{path_title}</div>
            <div class="adfm-card-value">{path_val}</div>
            <div class="adfm-card-sub">{path_sub}</div>
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
    comparison_year = int(summary.get("comparison_year", _today().year))
    if summary.get("comparison_year_month_available", False) and pd.notna(
        summary.get("comparison_vs_hist_end_gap", np.nan)
    ):
        gap = summary["comparison_vs_hist_end_gap"]
        if gap > 0.20:
            live_read = (
                f"{comparison_year} {month_label} finished {gap:+.2f}% above the filtered average month-end path, "
                f"so realized strength exceeded the historical template."
            )
        elif gap < -0.20:
            live_read = (
                f"{comparison_year} {month_label} finished {gap:+.2f}% below the filtered average month-end path, "
                f"so realized performance lagged the historical template."
            )
        else:
            live_read = f"{comparison_year} {month_label} finished broadly in line with the filtered average month-end path."

    commentary = f"""
    <div class="adfm-note">
        <div class="adfm-note-title">INTRA-MONTH READ</div>
        <div>{open_shape} {shape_core} {timing} {live_read}</div>
    </div>
    """
    st.markdown(commentary, unsafe_allow_html=True)


# =========================
# ORIGINAL CHART 2
# =========================


def plot_intra_month_curve(
    prices: pd.Series,
    filtered_halves: pd.DataFrame,
    month_int: int,
    symbol_shown: str,
    comparison_year: Optional[int] = None,
) -> io.BytesIO:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(
        prices, filtered_halves, month_int
    )

    fig = plt.figure(figsize=(14.5, 8.3), dpi=220, facecolor=BACKGROUND)
    ax = fig.add_subplot(111, facecolor=BACKGROUND)

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
        fig.savefig(
            buf, format="png", bbox_inches="tight", dpi=220, facecolor=BACKGROUND
        )
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

    comparison_year = int(comparison_year or _today().year)
    comparison_path = _year_month_path(prices, comparison_year, month_int)
    if len(comparison_path) >= 3:
        ax.plot(
            comparison_path.index.values,
            comparison_path.values,
            linewidth=2.0,
            color="#333333",
            alpha=0.55,
            linestyle="--",
            label=str(comparison_year),
            zorder=2.2,
        )

    avg_ex = avg_sel.copy()
    if 0 in avg_ex.index:
        avg_ex = avg_ex.drop(index=0)

    low_idx = int(avg_ex.idxmin())
    low_val = float(avg_ex.loc[low_idx])
    high_idx = int(avg_ex.idxmax())
    high_val = float(avg_ex.loc[high_idx])

    ax.scatter(
        [low_idx, high_idx], [low_val, high_val], s=34, color="#111111", zorder=3.2
    )

    low_dom = _avg_calendar_day_for_ordinal_from_filtered(
        prices, filtered_halves, month_int, low_idx
    )
    high_dom = _avg_calendar_day_for_ordinal_from_filtered(
        prices, filtered_halves, month_int, high_idx
    )

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
        fontsize=10.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#d6d6d6", lw=0.9),
    )

    ax.set_title(
        f"{symbol_shown} {MONTH_LABELS[month_int - 1]} | Intra-Month Seasonality Curve",
        color="black",
        fontsize=18,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel(
        "Trading day since prior month-end anchor",
        color="black",
        fontsize=11,
        weight="bold",
    )
    ax.set_ylabel(
        "Return from prior month-end (%)", color="black", fontsize=11, weight="bold"
    )
    ax.grid(axis="y", linestyle="--", color="#d9d9d9", alpha=1.0, linewidth=0.8)
    ax.tick_params(colors="black", labelsize=11)

    for sp in ax.spines.values():
        sp.set_color("#111111")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.01)

    y_min = float((avg_sel - std_sel).min())
    y_max = float((avg_sel + std_sel).max())
    pad_y = (
        0.12 * max(abs(y_min), abs(y_max))
        if np.isfinite(y_min) and np.isfinite(y_max)
        else 0.0
    )
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.09)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BACKGROUND)
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================
# MAIN CONTROLS
# =========================

today_dt = _today()
this_year = int(today_dt.year)

st.caption(
    "One shared sample, one active month, and one comparison year drive the matrix and both companion charts."
)


def _reset_seasonality_view() -> None:
    keys = [
        "seasonality_symbol",
        "seasonality_lookback",
        "seasonality_custom_start",
        "seasonality_custom_end",
        "seasonality_cycle",
        "seasonality_complete_only",
        "seasonality_fed",
        "seasonality_vix",
        "seasonality_teny",
        "seasonality_dxy",
        "seasonality_month_picker",
        "seasonality_year_picker",
        "seasonality_matrix",
    ]
    for key in keys:
        st.session_state.pop(key, None)


control_symbol, control_lookback, control_reset = st.columns(
    [2.2, 2.4, 0.75], vertical_alignment="bottom"
)
with control_symbol:
    symbol = st.text_input(
        "Ticker symbol", value="^SPX", key="seasonality_symbol"
    ).upper()
with control_lookback:
    lookback = st.segmented_control(
        "Global lookback",
        options=["5Y", "10Y", "20Y", "All", "Custom"],
        default="10Y",
        key="seasonality_lookback",
        width="stretch",
    )
with control_reset:
    st.button("Reset", on_click=_reset_seasonality_view, width="stretch")

custom_start_year = max(1900, this_year - 9)
custom_end_year = this_year
if lookback == "Custom":
    custom_start_col, custom_end_col = st.columns(2)
    with custom_start_col:
        custom_start_year = int(
            st.number_input(
                "Custom start year",
                min_value=1900,
                max_value=this_year,
                value=max(1900, this_year - 9),
                key="seasonality_custom_start",
            )
        )
    with custom_end_col:
        custom_end_year = int(
            st.number_input(
                "Custom end year",
                min_value=custom_start_year,
                max_value=this_year,
                value=this_year,
                key="seasonality_custom_end",
            )
        )

preset_map = {
    "5Y": "Last 5 years",
    "10Y": "Last 10 years",
    "20Y": "Last 20 years",
    "All": "All history",
    "Custom": "Custom",
}
sample_preset = preset_map.get(str(lookback or "10Y"), "Last 10 years")

if sample_preset == "All history":
    start_fetch_year = 1900
elif sample_preset == "Custom":
    start_fetch_year = max(1900, custom_start_year - 2)
else:
    start_fetch_year = max(1900, this_year - 22)

start_fetch_date = f"{start_fetch_year}-01-01"
end_fetch_date = f"{this_year}-12-31"

with st.spinner("Fetching and analyzing data..."):
    used_symbol = symbol
    prices = fetch_prices(symbol, start_fetch_date, end_fetch_date)
    if prices is None and symbol == "SPY":
        prices = fetch_prices("^GSPC", start_fetch_date, end_fetch_date)
        if prices is not None:
            used_symbol = "^GSPC"

if prices is None or prices.empty:
    st.error(f"No data found for '{symbol}'. Try a different symbol or lookback.")
    st.stop()

first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index fallback for seasonality.")

latest_price_date = prices.index.max().strftime("%Y-%m-%d")
latest_complete_year = _latest_complete_year(prices)
regime_df = fetch_regime_data(
    str(prices.index.min().date()), str(prices.index.max().date())
)
market_regime_daily = fetch_regime_market_series(
    str(prices.index.min().date()), str(prices.index.max().date())
)
market_regime_monthly = build_monthly_regime_features(market_regime_daily)
filter_table = build_filter_table(prices, regime_df, market_regime_monthly)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** A coordinated seasonality workspace for monthly returns, hit rates, intra-month paths, and regime-conditioned behavior.

        **Interaction model**
        - The global lookback defines the sample for every output.
        - Select a matrix month column to update both charts.
        - Select a matrix year row to overlay that year on the intra-month path.
        - Advanced filters change the matrix average row and both charts together.

        **Data source**
        - Yahoo Finance where available.
        - FRED fallbacks and macro regime series.
        """
    )

fed_options = ["All Fed regimes"] + sorted(
    [
        value
        for value in filter_table["fed_regime"].dropna().unique()
        if value != "Unknown"
    ]
)
vix_options = ["All VIX regimes"] + sorted(
    [
        value
        for value in filter_table["vix_bucket"].dropna().unique()
        if value != "Unknown"
    ]
)
teny_options = ["All 10Y regimes"] + sorted(
    [
        value
        for value in filter_table["teny_trend"].dropna().unique()
        if value != "Unknown"
    ]
)
dxy_options = ["All dollar regimes"] + sorted(
    [
        value
        for value in filter_table["dxy_trend"].dropna().unique()
        if value != "Unknown"
    ]
)

for key, options in [
    ("seasonality_fed", fed_options),
    ("seasonality_vix", vix_options),
    ("seasonality_teny", teny_options),
    ("seasonality_dxy", dxy_options),
]:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]

with st.expander("Advanced sample and regime filters", expanded=False):
    filter_row_1 = st.columns([1.45, 1.0, 1.35])
    with filter_row_1[0]:
        cycle_filter = st.selectbox(
            "Presidential cycle",
            [
                "All years",
                "Election years",
                "Midterm years",
                "Pre-election years",
                "Post-election years",
            ],
            key="seasonality_cycle",
        )
    with filter_row_1[1]:
        complete_months_only = st.checkbox(
            "Complete months only",
            value=True,
            key="seasonality_complete_only",
        )
    with filter_row_1[2]:
        fed_filter = st.selectbox("Fed regime", fed_options, key="seasonality_fed")

    filter_row_2 = st.columns(3)
    with filter_row_2[0]:
        vix_filter = st.selectbox("VIX regime", vix_options, key="seasonality_vix")
    with filter_row_2[1]:
        teny_filter = st.selectbox("10Y trend", teny_options, key="seasonality_teny")
    with filter_row_2[2]:
        dxy_filter = st.selectbox("Dollar trend", dxy_options, key="seasonality_dxy")

start_year, end_year = resolve_year_window(
    sample_preset,
    custom_start_year,
    custom_end_year,
    latest_complete_year,
)
filtered = apply_filters(
    filter_table=filter_table,
    start_year=start_year,
    end_year=end_year,
    cycle_filter=cycle_filter,
    complete_months_only=complete_months_only,
    fed_filter=fed_filter,
    vix_filter=vix_filter,
    teny_filter=teny_filter,
    dxy_filter=dxy_filter,
)

if filtered.empty:
    st.error("No observations match the selected lookback and advanced filters.")
    st.stop()

stats = seasonal_stats_from_filtered(filtered)
stats_valid = stats.dropna(subset=["mean_total"])
if stats.dropna(subset=["mean_h1", "mean_h2"]).empty or stats_valid.empty:
    st.error(
        "Insufficient data in the selected sample to compute seasonality statistics."
    )
    st.stop()

current_period = prices.index.max().to_period("M")
sample_years = sorted(
    filtered["year"].dropna().astype(int).unique().tolist(), reverse=True
)
matrix_years = [int(current_period.year)] + [
    year for year in sample_years if year != int(current_period.year)
]

if "seasonality_month_picker" not in st.session_state:
    st.session_state["seasonality_month_picker"] = MONTH_LABELS[
        int(current_period.month) - 1
    ]
if st.session_state["seasonality_month_picker"] not in MONTH_LABELS:
    st.session_state["seasonality_month_picker"] = MONTH_LABELS[
        int(current_period.month) - 1
    ]

if (
    "seasonality_year_picker" not in st.session_state
    or st.session_state["seasonality_year_picker"] not in matrix_years
):
    st.session_state["seasonality_year_picker"] = matrix_years[0]

selected_month_label = str(st.session_state["seasonality_month_picker"])
selected_month = MONTH_LABELS.index(selected_month_label) + 1
selected_year = int(st.session_state["seasonality_year_picker"])

active_filters = [
    f"{lookback} lookback",
    f"{start_year}-{end_year} completed-year sample",
    cycle_filter,
    fed_filter,
    vix_filter,
    teny_filter,
    dxy_filter,
]
active_filters = [item for item in active_filters if not str(item).startswith("All ")]
st.caption("Active sample · " + " · ".join(active_filters))

if int(filtered.shape[0]) < 24 or int(filtered["year"].nunique()) < 5:
    st.warning(
        "Thin sample warning: the current filter set leaves a small historical base. "
        "Read the output as directional rather than stable."
    )

st.subheader("Monthly Returns Matrix")
st.caption(
    "FILTER AVG uses the active sample; calendar rows show realized close-to-close returns. "
    "The highlighted month and year coordinate the charts below."
)

current_observation = filter_table.loc[filter_table.index == current_period]
matrix_observations = pd.concat([filtered, current_observation]).sort_index()
matrix_observations = matrix_observations.loc[
    ~matrix_observations.index.duplicated(keep="last")
]
matrix_frame = build_monthly_returns_frame(matrix_observations, stats, matrix_years)
render_monthly_returns_matrix(
    matrix_observations,
    stats,
    matrix_years,
    current_period,
    selected_month=selected_month,
    selected_year=selected_year,
    average_label=f"{lookback} AVG" if lookback in {"5Y", "10Y", "20Y"} else "FILTER AVG",
)

selection_month_col, selection_year_col = st.columns(
    [4.2, 1.3], vertical_alignment="bottom"
)
with selection_month_col:
    st.segmented_control(
        "Selected month",
        options=MONTH_LABELS,
        key="seasonality_month_picker",
        width="stretch",
    )
with selection_year_col:
    st.selectbox(
        "Comparison year",
        options=matrix_years,
        key="seasonality_year_picker",
    )

selected_month_label = str(st.session_state["seasonality_month_picker"])
selected_month = MONTH_LABELS.index(selected_month_label) + 1
selected_year = int(st.session_state["seasonality_year_picker"])
selected_stats = stats.loc[selected_month]
comparison_value = matrix_frame.loc[str(selected_year), selected_month_label]
comparison_text = (
    f"{comparison_value:+.2f}%" if pd.notna(comparison_value) else "not available"
)
st.markdown(
    f"""
    <div class="adfm-note">
        <div class="adfm-note-title">ACTIVE SELECTION</div>
        <div><b>{selected_month_label}</b> averages {selected_stats["mean_total"]:+.2f}% with a
        positive-return hit rate of {selected_stats["hit_rate"]:.0f}% in the filtered sample.
        <b>{selected_year} {selected_month_label}</b> is {comparison_text}.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_caption_parts = [
    used_symbol,
    f"{start_year}-{end_year}",
    cycle_filter,
    fed_filter,
    vix_filter,
    teny_filter,
    dxy_filter,
]
filter_caption_parts = [
    part for part in filter_caption_parts if not str(part).startswith("All ")
]
chart_context = " | ".join(filter_caption_parts)

profile_buf = plot_monthly_profile(
    stats,
    f"{used_symbol} | Monthly Profile",
    selected_month,
)
curve_buf = plot_intra_month_curve(
    prices,
    filtered,
    selected_month,
    used_symbol,
    comparison_year=selected_year,
)

profile_col, curve_col = st.columns(2, gap="large")
with profile_col:
    st.subheader("Monthly Profile")
    st.image(profile_buf, width="stretch")
    st.caption(
        "Bars show filtered average monthly returns; whiskers show the interquartile range. "
        "The line shows hit rate, and the active month is outlined."
    )
with curve_col:
    st.subheader(f"{selected_month_label} Intra-Month Path")
    st.image(curve_buf, width="stretch")
    st.caption(
        f"Filtered average path with ±1 standard deviation and the {selected_year} path overlaid when available."
    )

summary = build_intra_month_summary(
    prices,
    filtered,
    selected_month,
    used_symbol,
    comparison_year=selected_year,
)
st.subheader("Selected-Month Read")
render_intra_month_cards(summary)
render_intra_month_commentary(summary)

with st.expander("Methodology and active sample"):
    st.markdown(
        f"""
        **Chart context:** {chart_context}

        - Global lookback and advanced filters feed the matrix average row, monthly profile, and intra-month path.
        - Calendar rows remain realized returns; FILTER AVG is recalculated from the active filtered sample.
        - Intra-month paths are anchored to the previous month-end and equal-weighted across included observations.
        - Missing observations remain unavailable rather than being fabricated.
        - Latest price date: **{latest_price_date}**
        - Filtered observations: **{int(filtered.shape[0])} months across {int(filtered["year"].nunique())} years**
        """
    )

with st.expander("Audit included observations"):
    audit_df = filtered.copy().reset_index().rename(columns={"index": "period"})
    audit_df["period"] = audit_df["period"].astype(str)
    audit_cols = [
        "period",
        "year",
        "month",
        "pres_cycle_bucket",
        "regime_cycle",
        "fed_regime",
        "vix_bucket",
        "teny_trend",
        "dxy_trend",
        "is_complete_month",
        "total_ret",
        "h1_ret",
        "h2_ret",
    ]
    audit_cols = [column for column in audit_cols if column in audit_df.columns]
    st.dataframe(
        audit_df[audit_cols].sort_values(["year", "month"]),
        width="stretch",
        hide_index=True,
    )

render_footer()

