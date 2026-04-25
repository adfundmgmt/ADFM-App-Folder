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

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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


def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
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

    vix = _yf_download("^VIX", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if vix is not None:
        out["vix"] = vix

    tnx = _yf_download("^TNX", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if tnx is not None:
        out["tnx"] = tnx / 10.0

    dxy = _yf_download("DX-Y.NYB", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if dxy is None:
        for fallback in DXY_FALLBACKS:
            dxy = _yf_download(fallback, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
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

    usrec = _fred_series("USREC", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    fedfunds = _fred_series("FEDFUNDS", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

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

    regime["regime_cycle"] = np.where(regime["is_recession"] == 1, "Recession", "Expansion")
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
    df["vix_bucket"] = df.get("vix_bucket", pd.Series(index=df.index, dtype=object)).fillna("Unknown")
    df["teny_trend"] = df.get("teny_trend", pd.Series(index=df.index, dtype=object)).fillna("Unknown")
    df["dxy_trend"] = df.get("dxy_trend", pd.Series(index=df.index, dtype=object)).fillna("Unknown")

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
    stats["downside_freq"] = grouped["total_ret"].apply(lambda x: (x < -2.0).mean() * 100.0)
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
    plot_df = stats.dropna(subset=["mean_h1", "mean_h2", "min_ret", "max_ret", "hit_rate"]).copy()

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
    ax1.grid(axis="y", linestyle="--", color="lightgrey", linewidth=0.7, alpha=0.75, zorder=1)

    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="D", s=110, color="black", zorder=4)
    ax2.set_ylabel("Hit rate of positive returns", weight="bold", fontsize=15)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    legend = [Patch(facecolor="white", edgecolor="black", hatch="///", label="Second half (hatched)")]
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
        "current_year": _today().year,
        "current_year_month_available": False,
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

    today = _today()
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

    if summary.get("front_loaded") is True:
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
    if summary.get("current_year_month_available", False) and pd.notna(summary.get("current_vs_hist_end_gap", np.nan)):
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
            live_read = f"The current year's completed {month_label} finished broadly in line with the long-run average month-end path."

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
) -> io.BytesIO:
    df_sel, avg_sel = _month_paths_prev_eom_equal_weight_from_filtered(prices, filtered_halves, month_int)

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
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=220, facecolor=BACKGROUND)
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

    today = _today()
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
    ax.set_xlabel("Trading day since prior month-end anchor", color="black", fontsize=11, weight="bold")
    ax.set_ylabel("Return from prior month-end (%)", color="black", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", color="#d9d9d9", alpha=1.0, linewidth=0.8)
    ax.tick_params(colors="black", labelsize=11)

    for sp in ax.spines.values():
        sp.set_color("#111111")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.01)

    y_min = float((avg_sel - std_sel).min())
    y_max = float((avg_sel + std_sel).max())
    pad_y = 0.12 * max(abs(y_min), abs(y_max)) if np.isfinite(y_min) and np.isfinite(y_max) else 0.0
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.09)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BACKGROUND)
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================
# ADDITIONAL CHARTS
# =========================

def plot_median_iqr_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    plot_df = stats.dropna(subset=["median_total", "p25_total", "p75_total", "hit_rate"]).copy()

    labels = plot_df["label"].tolist()
    x = np.arange(len(labels))

    med = plot_df["median_total"].to_numpy(float)
    p25 = plot_df["p25_total"].to_numpy(float)
    p75 = plot_df["p75_total"].to_numpy(float)
    mean = plot_df["mean_total"].to_numpy(float)
    hit = plot_df["hit_rate"].to_numpy(float)
    downside = plot_df["downside_freq"].to_numpy(float)

    fig = plt.figure(figsize=(18, 9.6), dpi=220, facecolor=BACKGROUND)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[6.2, 1.55], hspace=0.03)
    ax1 = fig.add_subplot(gs[0], facecolor=BACKGROUND)

    bar_colors = np.where(med >= 0, POS_GREEN, NEG_RED)

    ax1.bar(
        x,
        med,
        width=0.78,
        color=bar_colors,
        edgecolor="#111111",
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
        label="Median return",
    )

    yerr = np.vstack([med - p25, p75 - med])
    ax1.errorbar(
        x,
        med,
        yerr=yerr,
        fmt="none",
        ecolor="#111111",
        elinewidth=1.6,
        alpha=0.75,
        capsize=6,
        zorder=3,
        label="IQR",
    )

    ax1.scatter(
        x,
        mean,
        marker="D",
        s=72,
        color="black",
        zorder=4,
        label="Mean",
    )

    ax1.axhline(0, color="#9b9b9b", linestyle=":", linewidth=1.0)
    ax1.set_xticks(x, labels)
    ax1.tick_params(axis="x", labelsize=13)
    ax1.tick_params(axis="y", labelsize=13)
    ax1.set_ylabel("Return (%)", weight="bold", fontsize=15)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.grid(axis="y", linestyle="--", color="#d9d9d9", linewidth=0.75, alpha=0.9, zorder=1)

    ax2 = ax1.twinx()
    ax2.scatter(x, hit, marker="o", s=92, color="#333333", zorder=4, label="Hit rate")
    ax2.set_ylabel("Hit rate of positive returns", weight="bold", fontsize=15)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc="upper left", frameon=False, fontsize=12)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    row_labels = ["Median %", "P25 %", "P75 %", "Hit %", "Downside %"]
    cell_vals = [med, p25, p75, hit, downside]

    cell_txt = []
    for i, row in enumerate(cell_vals):
        if row_labels[i] in ["Hit %", "Downside %"]:
            cell_txt.append([f"{v:.0f}%" if pd.notna(v) else "n/a" for v in row])
        else:
            cell_txt.append([_format_pct(v) for v in row])

    color_basis = [med, p25, p75, hit - 50, -downside]
    cell_colors = _cell_colors([list(row) for row in color_basis])

    table = ax_tbl.table(
        cellText=cell_txt,
        rowLabels=row_labels,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.035, 0.15, 0.93, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
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
            cell.set_linewidth(0.85)

    fig.suptitle(title, fontsize=24, weight="bold", y=0.965)
    fig.subplots_adjust(left=0.055, right=0.94, top=0.88, bottom=0.09)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BACKGROUND)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_regime_comparison_table(
    base_filtered: pd.DataFrame,
    selected_month: int,
) -> pd.DataFrame:
    if base_filtered.empty:
        return pd.DataFrame()

    month_df = base_filtered[base_filtered["month"] == selected_month].copy()
    if month_df.empty:
        return pd.DataFrame()

    rows = []

    def add_row(label: str, sub: pd.DataFrame) -> None:
        if sub.empty:
            return

        rows.append(
            {
                "Regime": label,
                "N": int(sub.shape[0]),
                "Mean": sub["total_ret"].mean(),
                "Median": sub["total_ret"].median(),
                "Hit Rate": (sub["total_ret"] > 0).mean() * 100.0,
                "P25": sub["total_ret"].quantile(0.25),
                "P75": sub["total_ret"].quantile(0.75),
                "Worst": sub["total_ret"].min(),
                "Best": sub["total_ret"].max(),
                "Downside Freq < -2%": (sub["total_ret"] < -2.0).mean() * 100.0,
            }
        )

    add_row("All filtered years", month_df)

    for col in ["fed_regime", "vix_bucket", "teny_trend", "dxy_trend", "regime_cycle"]:
        if col in month_df.columns:
            vals = sorted([x for x in month_df[col].dropna().unique() if str(x) != "Unknown"])
            for val in vals:
                add_row(str(val), month_df[month_df[col] == val])

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["N", "Median"], ascending=[False, False]).reset_index(drop=True)
    return out


def plot_regime_comparison(regime_table: pd.DataFrame, title: str) -> io.BytesIO:
    fig = plt.figure(figsize=(15.5, 9.2), dpi=220, facecolor=BACKGROUND)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5.6, 2.0], hspace=0.08)

    ax = fig.add_subplot(gs[0], facecolor=BACKGROUND)

    if regime_table.empty:
        ax.text(0.5, 0.5, "Not enough data for regime comparison", ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=220, facecolor=BACKGROUND)
        plt.close(fig)
        buf.seek(0)
        return buf

    plot_df = regime_table.copy()
    plot_df = plot_df[plot_df["N"] >= 3].copy()
    plot_df = plot_df.sort_values("Median", ascending=True).tail(10)

    if plot_df.empty:
        ax.text(0.5, 0.5, "No regime bucket has at least 3 observations", ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=220, facecolor=BACKGROUND)
        plt.close(fig)
        buf.seek(0)
        return buf

    y = np.arange(len(plot_df))
    med = plot_df["Median"].to_numpy(float)
    mean = plot_df["Mean"].to_numpy(float)
    p25 = plot_df["P25"].to_numpy(float)
    p75 = plot_df["P75"].to_numpy(float)
    n_vals = plot_df["N"].to_numpy(int)

    colors = np.where(med >= 0, POS_GREEN, NEG_RED)

    ax.barh(
        y,
        med,
        color=colors,
        edgecolor="#111111",
        linewidth=0.85,
        alpha=0.95,
        zorder=2,
    )

    xerr = np.vstack([med - p25, p75 - med])
    ax.errorbar(
        med,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#111111",
        elinewidth=1.35,
        capsize=4,
        alpha=0.75,
        zorder=3,
    )

    ax.scatter(mean, y, marker="D", s=46, color="black", zorder=4)

    ax.axvline(0, color="#9b9b9b", linestyle=":", linewidth=1.0)
    ax.set_yticks(y, plot_df["Regime"].tolist())
    ax.set_xlabel("Median return (%) with IQR range", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=20, weight="bold", pad=10)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", linestyle="--", color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax.tick_params(labelsize=10.5)

    for i, v in enumerate(med):
        ax.text(
            v + (0.08 if v >= 0 else -0.08),
            i,
            f"{v:+.2f}% | N={n_vals[i]}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9.5,
            color="#111111",
        )

    for sp in ax.spines.values():
        sp.set_color("#111111")

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    table_df = plot_df.sort_values("Median", ascending=False).head(8).copy()
    col_labels = table_df["Regime"].tolist()

    rows = [
        ("N", table_df["N"].tolist()),
        ("Median", table_df["Median"].tolist()),
        ("Mean", table_df["Mean"].tolist()),
        ("Hit %", table_df["Hit Rate"].tolist()),
        ("Worst", table_df["Worst"].tolist()),
    ]

    cell_text = []
    cell_colors = []

    for row_name, vals in rows:
        if row_name == "N":
            cell_text.append([str(int(v)) for v in vals])
            cell_colors.append([LIGHT_GREY for _ in vals])
        elif row_name == "Hit %":
            cell_text.append([f"{v:.0f}%" if pd.notna(v) else "n/a" for v in vals])
            cell_colors.append([LIGHT_GREEN if v >= 50 else LIGHT_RED for v in vals])
        else:
            cell_text.append([_format_pct(v) for v in vals])
            cell_colors.append(
                [
                    LIGHT_GREEN if pd.notna(v) and v > 0 else LIGHT_RED if pd.notna(v) and v < 0 else LIGHT_GREY
                    for v in vals
                ]
            )

    table = ax_tbl.table(
        cellText=cell_text,
        rowLabels=[r[0] for r in rows],
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.02, 0.12, 0.96, 0.78],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)

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
            cell.set_linewidth(0.75)

    fig.subplots_adjust(left=0.20, right=0.96, top=0.91, bottom=0.08)

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

col1, col2, col3 = st.columns([2.0, 1.0, 1.0])

with col1:
    symbol = st.text_input("Ticker symbol", value="^SPX").upper()

with col2:
    custom_start_year = st.number_input(
        "Custom start year",
        value=2020,
        min_value=1900,
        max_value=this_year,
    )

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
        prices = fetch_prices("^GSPC", start_fetch_date, end_fetch_date)
        if prices is not None:
            used_symbol = "^GSPC"

if prices is None or prices.empty:
    st.error(f"No data found for '{symbol}' in the given date range. Try a different symbol or adjust the years.")
    st.stop()

first_valid = prices.first_valid_index()
if first_valid is not None:
    prices = prices.loc[first_valid:]

if used_symbol != symbol:
    st.info("SPY data unavailable. Using S&P 500 index fallback for seasonality.")

latest_price_date = prices.index.max().strftime("%Y-%m-%d")
latest_complete_year = _latest_complete_year(prices)

regime_df = fetch_regime_data(str(prices.index.min().date()), str(prices.index.max().date()))
market_regime_daily = fetch_regime_market_series(str(prices.index.min().date()), str(prices.index.max().date()))
market_regime_monthly = build_monthly_regime_features(market_regime_daily)
filter_table = build_filter_table(prices, regime_df, market_regime_monthly)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Seasonality explorer for monthly tendencies, hit-rate patterns, intra-month behavior, and regime-conditioned behavior.

        **What this tab shows**
        - Month-level return tendencies after explicit sample filtering.
        - First-half versus second-half contribution splits within each month.
        - Intra-month average path versus current-year behavior when available.
        - Regime-conditioned month behavior using Fed, VIX, 10Y, and dollar context.

        **Data source**
        - Historical daily market price series from Yahoo Finance where available.
        - FRED fallback for selected major indices.
        - FRED recession and Fed Funds data for macro regime tagging.
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

    complete_months_only = st.checkbox("Complete months only", value=True)

    st.subheader("Regime Filters")

    fed_options = ["All Fed regimes"] + sorted([x for x in filter_table["fed_regime"].dropna().unique() if x != "Unknown"])
    fed_filter = st.selectbox("Fed regime", fed_options, index=0)

    vix_options = ["All VIX regimes"] + sorted([x for x in filter_table["vix_bucket"].dropna().unique() if x != "Unknown"])
    vix_filter = st.selectbox("VIX regime", vix_options, index=0)

    teny_options = ["All 10Y regimes"] + sorted([x for x in filter_table["teny_trend"].dropna().unique() if x != "Unknown"])
    teny_filter = st.selectbox("10Y trend", teny_options, index=0)

    dxy_options = ["All dollar regimes"] + sorted([x for x in filter_table["dxy_trend"].dropna().unique() if x != "Unknown"])
    dxy_filter = st.selectbox("Dollar trend", dxy_options, index=0)


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
    complete_months_only=complete_months_only,
    fed_filter=fed_filter,
    vix_filter=vix_filter,
    teny_filter=teny_filter,
    dxy_filter=dxy_filter,
)

if filtered.empty:
    st.error("No observations match the selected filters.")
    st.stop()

stats = seasonal_stats_from_filtered(filtered)

if stats.dropna(subset=["mean_h1", "mean_h2"]).empty:
    st.error("Insufficient data in the selected filtered window to compute statistics.")
    st.stop()


# =========================
# TOP METADATA
# =========================

obs_months = int(filtered.shape[0])
obs_years = int(filtered["year"].nunique())

meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
meta_c1.metric("Filtered month observations", f"{obs_months}")
meta_c2.metric("Distinct years", f"{obs_years}")
meta_c3.metric("Window", f"{start_year}-{end_year}")
meta_c4.metric("Latest price date", latest_price_date)

if obs_months < 24 or obs_years < 5:
    st.warning(
        "Thin sample warning: the current filter set leaves a small historical base. "
        "Read the output as directional rather than stable."
    )


# =========================
# BEST / WORST SUMMARY
# =========================

stats_valid = stats.dropna(subset=["mean_total"])

if stats_valid.empty:
    st.error("No valid monthly seasonality statistics after filtering.")
    st.stop()

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


# =========================
# ORIGINAL OUTPUT 1
# =========================

filter_caption_parts = [
    f"{used_symbol}",
    f"{start_year}-{end_year}",
    cycle_filter,
    fed_filter,
    vix_filter,
    teny_filter,
    dxy_filter,
    "Complete months only" if complete_months_only else "Partial current month allowed",
]
filter_caption_parts = [x for x in filter_caption_parts if not str(x).startswith("All ")]
chart_title = " | ".join(filter_caption_parts)

buf = plot_seasonality(stats, f"{used_symbol} Seasonality ({chart_title})")
st.image(buf, use_container_width=True)

st.caption(
    "Bars equal mean(1H) + mean(2H) contributions. First half solid, second half hatched. "
    "Error bars use min and max of total monthly returns within the filtered sample."
)


# =========================
# ORIGINAL OUTPUT 2
# =========================

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


# =========================
# ADDITIONAL OUTPUTS
# =========================

st.markdown("---")
st.subheader("Additional Decision Views")

st.markdown(
    """
    <div class="adfm-small-note">
    These charts sit below the original two outputs and are meant to make the seasonality read more robust and easier to interpret.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Median and IQR Seasonality")

median_buf = plot_median_iqr_seasonality(
    stats,
    f"{used_symbol} Median Seasonality With IQR ({chart_title})",
)
st.image(median_buf, use_container_width=True)

st.caption(
    "Bars show median monthly return, whiskers show the interquartile range, black diamonds show mean return, and the embedded table shows the month-level distribution."
)

st.markdown(f"### Regime Comparison for {MONTH_LABELS[month_choice - 1]}")

regime_comp = build_regime_comparison_table(filtered, month_choice)
regime_buf = plot_regime_comparison(
    regime_comp,
    f"{used_symbol} {MONTH_LABELS[month_choice - 1]} | Regime-Conditioned Return Profile",
)
st.image(regime_buf, use_container_width=True)

st.caption(
    "Bars show median return by regime bucket. Whiskers show the interquartile range, black diamonds show mean return, and the table below the chart keeps the key regime stats in image format."
)


# =========================
# AUDIT
# =========================

with st.expander("Audit included observations"):
    audit_df = filtered.copy()
    audit_df = audit_df.reset_index().rename(columns={"index": "period"})
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

    audit_cols = [c for c in audit_cols if c in audit_df.columns]
    audit_df = audit_df[audit_cols].sort_values(["year", "month"])

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

st.caption("© 2026 AD Fund Management LP")
