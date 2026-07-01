import datetime as dt
import io
import time
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter, MultipleLocator


# =========================
# CONFIG
# =========================

plt.style.use("default")

CACHE_TTL_SECONDS = 3600
MIN_DAYS_REQUIRED = 30
MIN_DAYS_CURRENT_YEAR = 5
MIN_DAYS_FOR_CORR = 10
NY_TZ = ZoneInfo("America/New_York")

DEFAULT_HORIZONS = [21, 63, 126, 252]
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_FORWARD_HORIZON = 252
DEFAULT_TOP_N = 7

TICKER_ALIASES = {
    "^SPX": "^GSPC",
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "S&P": "^GSPC",
    "SPY": "SPY",
    "NDX": "^NDX",
    "NASDAQ": "^IXIC",
    "CCMP": "^IXIC",
    "RUT": "^RUT",
    "IWM": "IWM",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

REGIME_SYMBOLS = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
    "hyg": "HYG",
    "ief": "IEF",
}

CHART_FACE = "white"
GRID_COLOR = "#d9dee7"
TEXT_COLOR = "#2f3542"
MUTED_TEXT = "#7a8497"
GOOD = "#178f55"
BAD = "#c9413a"
BLUE = "#276ef1"
LIGHT_FILL = "#e8edf5"
MID_FILL = "#cfd7e6"

st.set_page_config(page_title="Market Memory Explorer", layout="wide")


# =========================
# FORMATTING / UI HELPERS
# =========================

def fmt_pct(x, digits: int = 2, signed: bool = True) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        sign = "+" if signed and x >= 0 else ""
        return f"{sign}{x:.{digits}%}"
    except Exception:
        return "N/A"


def fmt_num(x, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"


def fmt_int(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{int(x):,}"
    except Exception:
        return "N/A"


def fmt_bps(x, digits: int = 0) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.{digits}f} bp"
    except Exception:
        return "N/A"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.15rem;
            padding-bottom: 2.25rem;
        }
        .clean-metric {
            min-height: 72px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: #f7f9fc;
            border: 1px solid #dde3ed;
            border-radius: 14px;
            padding: 0.78rem 1.00rem;
            box-shadow: inset 0 1px 1px rgba(16, 24, 40, 0.02);
        }
        .clean-metric-label {
            color: #8b97a8;
            font-size: 0.72rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            white-space: nowrap;
        }
        .clean-metric-value {
            color: #202733;
            font-size: 1.30rem;
            font-weight: 760;
            letter-spacing: -0.03em;
            white-space: nowrap;
            padding-top: 0.25rem;
        }
        .section-caption {
            color: #6f7888;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-top: -0.25rem;
            margin-bottom: 0.85rem;
        }
        div[data-testid="stTextInput"] label p,
        div[data-testid="stNumberInput"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stSlider"] label p,
        div[data-testid="stMultiSelect"] label p {
            color: #7f8a9b;
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: #f7f9fc;
            border: 1px solid #dde3ed;
            border-radius: 12px;
        }
        hr {
            border: none;
            border-top: 1px solid #dfe4ec;
            margin: 1.15rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_logo() -> None:
    candidate_paths = [
        Path("assets/logo.png"),
        Path("logo.png"),
        Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png"),
    ]
    for path in candidate_paths:
        if path.exists():
            st.image(str(path), width=68)
            return


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="clean-metric">
            <div class="clean-metric-label">{label}</div>
            <div class="clean-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_axes(ax) -> None:
    ax.set_facecolor(CHART_FACE)
    ax.grid(True, ls=":", lw=0.75, color=GRID_COLOR, alpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d0dc")
    ax.spines["bottom"].set_color("#c9d0dc")
    ax.tick_params(axis="both", colors=MUTED_TEXT, labelsize=10)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.title.set_color(TEXT_COLOR)


def set_percent_axis(ax, values, force_zero: bool = False) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if force_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)

    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.02
    ax.set_ylim(ymin - pad, ymax + pad)

    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    raw_step = max(span / 10, 0.0025)
    candidates = np.array([0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00])
    step = float(candidates[np.argmin(np.abs(candidates - raw_step))])

    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))


def get_palette(n: int):
    cmap_names = ["tab10", "Dark2", "Set1", "tab20", "Paired", "tab20b", "tab20c"]
    colors = []
    seen = set()
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        raw = cmap.colors if hasattr(cmap, "colors") else [cmap(x) for x in np.linspace(0, 1, 20)]
        for color in raw:
            rgb = tuple(color[:3])
            key = tuple(round(v, 4) for v in rgb)
            if key not in seen:
                seen.add(key)
                colors.append(rgb)
    if n <= len(colors):
        return colors[:n]
    fallback = [plt.get_cmap("hsv")(x)[:3] for x in np.linspace(0, 1, n)]
    return (colors + fallback)[:n]


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


def dataframe_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# =========================
# DATA LOADING
# =========================

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str) -> pd.DataFrame:
    symbol = str(symbol).strip()
    attempts = 0
    delay = 1.0
    last_error = None

    while attempts < 4:
        try:
            df = yf.download(
                symbol,
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", symbol) in df.columns:
                    close = df[("Close", symbol)]
                elif "Close" in df.columns.get_level_values(0):
                    close = df.xs("Close", level=0, axis=1).iloc[:, 0]
                else:
                    close = pd.Series(dtype=float)
            else:
                close = df["Close"] if "Close" in df.columns else pd.Series(dtype=float)

            out = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()

            if out.empty:
                hist = yf.Ticker(symbol).history(period="max", auto_adjust=True)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if "Close" in hist.columns:
                    out = pd.DataFrame({"Close": pd.to_numeric(hist["Close"], errors="coerce")}).dropna()

            if not out.empty:
                out.index = pd.to_datetime(out.index).tz_localize(None)
                out = out[~out.index.duplicated(keep="last")].sort_index()
                out["Year"] = out.index.year
                return out

            last_error = "Yahoo returned no usable Close data."
        except Exception as exc:
            last_error = str(exc)

        attempts += 1
        time.sleep(delay)
        delay *= 2

    raise ValueError(last_error or "Yahoo returned no usable data.")


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_regime_proxy(symbol: str) -> pd.Series:
    hist = load_history(symbol)
    return hist["Close"].dropna().copy()


def load_regime_data(main_close: pd.Series, enabled: bool) -> tuple[dict, list]:
    data = {"asset": main_close.dropna().copy()}
    missing = []
    if not enabled:
        return data, missing

    for key, symbol in REGIME_SYMBOLS.items():
        try:
            data[key] = load_regime_proxy(symbol)
        except Exception:
            missing.append(symbol)

    if "hyg" in data and "ief" in data:
        ratio = (data["hyg"] / data["ief"]).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            data["credit"] = ratio

    return data, missing


def business_days_behind(last_date: pd.Timestamp, reference_date: dt.date) -> int:
    last_day = pd.Timestamp(last_date).date()
    if last_day >= reference_date:
        return 0
    return len(pd.bdate_range(pd.Timestamp(last_day) + pd.Timedelta(days=1), pd.Timestamp(reference_date)))


# =========================
# PATH / RETURN MATH
# =========================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def normalized_price_path(series: pd.Series) -> pd.Series | None:
    series = series.dropna().astype(float)
    if len(series) < 2:
        return None
    first = float(series.iloc[0])
    if not np.isfinite(first) or first <= 0:
        return None
    out = series / first - 1.0
    out.index = np.arange(0, len(out))
    return out.astype(float)


def daily_returns_from_cum_path(path: pd.Series) -> pd.Series:
    path = path.dropna().astype(float)
    if path.empty:
        return pd.Series(dtype=float)
    ret = (1.0 + path).div((1.0 + path).shift(1)).sub(1.0)
    ret.iloc[0] = path.iloc[0]
    return ret.dropna()


def max_drawdown_from_path(path: pd.Series) -> float:
    path = path.dropna().astype(float)
    if path.empty:
        return np.nan
    wealth = 1.0 + path
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def annualized_vol_from_path(path: pd.Series) -> float:
    daily = daily_returns_from_cum_path(path)
    if len(daily) < 2:
        return np.nan
    return float(daily.std(ddof=0) * np.sqrt(252))


def trailing_move(path: pd.Series, days: int) -> float:
    path = path.dropna().astype(float)
    if len(path) < max(2, days):
        return np.nan
    return float((1.0 + path.iloc[-1]) / (1.0 + path.iloc[-days]) - 1.0)


def rolling_max_drawdown_array(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or x[0] <= 0:
        return np.nan
    running_peak = np.maximum.accumulate(x)
    dd = x / running_peak - 1.0
    return float(np.min(dd))


def forward_path_from_loc(close_px: pd.Series, loc: int, horizon: int) -> pd.Series | None:
    if loc < 0 or loc + horizon >= len(close_px):
        return None
    window = close_px.iloc[loc: loc + horizon + 1].dropna().copy()
    if len(window) != horizon + 1:
        return None
    first = float(window.iloc[0])
    if not np.isfinite(first) or first <= 0:
        return None
    out = window / first - 1.0
    out.index = np.arange(0, len(out))
    return out.astype(float)


def forward_matrix_from_locs(close_px: pd.Series, locs: list[int], horizon: int) -> np.ndarray:
    paths = []
    for loc in locs:
        path = forward_path_from_loc(close_px, int(loc), horizon)
        if path is not None and len(path) == horizon + 1:
            paths.append(path.values)
    if not paths:
        return np.empty((0, horizon + 1))
    return np.vstack(paths)


def similarity_from_gap(gap: float, tolerance: float) -> float:
    if not np.isfinite(gap) or tolerance <= 0:
        return 0.0
    return float(np.clip(1.0 - abs(gap) / tolerance, 0.0, 1.0))


def build_true_ytd_paths(raw: pd.DataFrame, current_year: int) -> tuple[dict, dict]:
    close = raw["Close"].dropna().copy()
    paths = {}
    date_maps = {}

    for yr in sorted(close.index.year.unique()):
        year_closes = close.loc[close.index.year == yr].dropna()
        if year_closes.empty:
            continue
        if yr == current_year:
            if len(year_closes) < MIN_DAYS_CURRENT_YEAR:
                continue
        elif len(year_closes) < MIN_DAYS_REQUIRED:
            continue

        prev_closes = close.loc[close.index < year_closes.index[0]].dropna()
        if prev_closes.empty:
            continue

        base = float(prev_closes.iloc[-1])
        if not np.isfinite(base) or base <= 0:
            continue

        ytd = year_closes / base - 1.0
        day_index = np.arange(1, len(ytd) + 1)
        ytd.index = day_index
        paths[int(yr)] = ytd.astype(float)
        date_maps[int(yr)] = pd.Series(year_closes.index, index=day_index)

    return paths, date_maps


# =========================
# REGIME / FEATURE ENGINE
# =========================

def bucket_vix_value(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x < 15:
        return "calm"
    if x < 25:
        return "normal"
    return "stressed"


def bucket_bps_change(x: float, threshold: float = 25.0) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x >= threshold:
        return "rising"
    if x <= -threshold:
        return "falling"
    return "flat"


def bucket_pct_change(x: float, threshold: float = 0.02, up_label: str = "rising", down_label: str = "falling") -> str:
    if not np.isfinite(x):
        return "unknown"
    if x >= threshold:
        return up_label
    if x <= -threshold:
        return down_label
    return "flat"


def current_feature_snapshot(feature_df: pd.DataFrame) -> dict:
    row = feature_df.iloc[-1]
    return {
        "Close": row.get("Close", np.nan),
        "YTD": row.get("ytd", np.nan),
        "21D": row.get("ret_21", np.nan),
        "63D": row.get("ret_63", np.nan),
        "126D": row.get("ret_126", np.nan),
        "252D": row.get("ret_252", np.nan),
        "Vol 63D": row.get("vol_63", np.nan),
        "Trail DD 63D": row.get("trail_dd_63", np.nan),
        "Trail DD 252D": row.get("trail_dd_252", np.nan),
        "Distance vs 200D": row.get("dist_200d", np.nan),
        "Above 200D": row.get("above_200d", False),
        "VIX": row.get("vix", np.nan),
        "10Y 63D": row.get("tnx_63_bps", np.nan),
        "DXY 63D": row.get("dxy_63", np.nan),
        "Credit 63D": row.get("credit_63", np.nan),
    }


def build_feature_frame(close_px: pd.Series, regime_data: dict) -> pd.DataFrame:
    close_px = close_px.dropna().astype(float).copy()
    df = pd.DataFrame({"Close": close_px})
    df["Date"] = df.index
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["_loc"] = np.arange(len(df))

    prev_year_close = close_px.groupby(close_px.index.year).transform("first")
    year_start_base = []
    for date_value in close_px.index:
        y = date_value.year
        prev = close_px.loc[close_px.index.year < y]
        year_start_base.append(float(prev.iloc[-1]) if not prev.empty else np.nan)
    df["prior_year_close"] = year_start_base
    df["ytd"] = df["Close"] / df["prior_year_close"] - 1.0

    daily = df["Close"].pct_change()
    for n in [5, 10, 21, 42, 63, 126, 252]:
        df[f"ret_{n}"] = df["Close"] / df["Close"].shift(n) - 1.0

    for n in [21, 63, 126, 252]:
        df[f"vol_{n}"] = daily.rolling(n).std(ddof=0) * np.sqrt(252)
        df[f"trail_dd_{n}"] = df["Close"].rolling(n + 1).apply(rolling_max_drawdown_array, raw=True)

    for n in [50, 100, 200]:
        df[f"ma_{n}"] = df["Close"].rolling(n).mean()
        df[f"above_{n}d"] = df["Close"] >= df[f"ma_{n}"]
        df[f"dist_{n}d"] = df["Close"] / df[f"ma_{n}"] - 1.0

    df["ma_50_gt_200"] = df["ma_50"] >= df["ma_200"]

    # Forward returns and forward maximum drawdowns, point-in-time anchored.
    for h in DEFAULT_HORIZONS:
        df[f"next_{h}"] = df["Close"].shift(-h) / df["Close"] - 1.0
        df[f"fwd_dd_{h}"] = df["Close"].rolling(h + 1).apply(rolling_max_drawdown_array, raw=True).shift(-h)

    # Optional macro/regime proxies aligned to the asset calendar.
    if "vix" in regime_data:
        vix = regime_data["vix"].reindex(df.index).ffill()
        df["vix"] = vix
        df["vix_bucket"] = df["vix"].apply(bucket_vix_value)
    else:
        df["vix"] = np.nan
        df["vix_bucket"] = "unknown"

    if "tnx" in regime_data:
        tnx = regime_data["tnx"].reindex(df.index).ffill()
        df["tnx"] = tnx
        df["tnx_63_bps"] = (tnx - tnx.shift(63)) * 10.0
        df["tnx_trend"] = df["tnx_63_bps"].apply(bucket_bps_change)
    else:
        df["tnx"] = np.nan
        df["tnx_63_bps"] = np.nan
        df["tnx_trend"] = "unknown"

    if "dxy" in regime_data:
        dxy = regime_data["dxy"].reindex(df.index).ffill()
        df["dxy"] = dxy
        df["dxy_63"] = dxy / dxy.shift(63) - 1.0
        df["dxy_trend"] = df["dxy_63"].apply(lambda x: bucket_pct_change(x, 0.02, "rising", "falling"))
    else:
        df["dxy"] = np.nan
        df["dxy_63"] = np.nan
        df["dxy_trend"] = "unknown"

    if "credit" in regime_data:
        credit = regime_data["credit"].reindex(df.index).ffill()
        df["credit"] = credit
        df["credit_63"] = credit / credit.shift(63) - 1.0
        df["credit_trend"] = df["credit_63"].apply(lambda x: bucket_pct_change(x, 0.02, "improving", "worsening"))
    else:
        df["credit"] = np.nan
        df["credit_63"] = np.nan
        df["credit_trend"] = "unknown"

    # Keep Date only as a normal column. yfinance often names the index "Date",
    # which makes pandas treat "Date" as both an index level and a column label.
    # Resetting here prevents ambiguous sort/filter operations downstream.
    return df.reset_index(drop=True)


def historical_universe(feature_df: pd.DataFrame, start_year: int, min_history_days: int = 252, max_horizon: int = 252) -> pd.DataFrame:
    df = feature_df.copy()
    df = df[df["Year"] >= int(start_year)]
    df = df[df["_loc"] >= int(min_history_days)]
    df = df[df["_loc"] <= int(feature_df["_loc"].max()) - int(max_horizon)]
    return df.dropna(subset=["Close", "ret_21", "ret_63", "ret_252", f"next_{max_horizon}"]).copy()


# =========================
# TABLE / SUMMARY FUNCTIONS
# =========================

def numeric_series_from_first_available(df: pd.DataFrame, candidate_cols: list[str]) -> pd.Series:
    """Return a numeric Series from the first available column name.

    The app uses machine-friendly names in the base-rate universe
    (for example, ``next_63`` / ``fwd_dd_63``) and presentation-friendly
    names in analog tables (for example, ``Next 63D`` / ``Fwd DD 63D``).
    This helper lets comparison tables work with either shape.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for col in candidate_cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()

    return pd.Series(dtype=float)


def summarize_forward_distribution(df: pd.DataFrame, horizons: list[int], label: str = "Cohort") -> pd.DataFrame:
    rows = []
    for h in horizons:
        vals = numeric_series_from_first_available(df, [f"next_{h}", f"Next {h}D"])
        dds = numeric_series_from_first_available(df, [f"fwd_dd_{h}", f"Fwd DD {h}D"])
        if vals.empty:
            continue
        rows.append(
            {
                "Sample": label,
                "Horizon": f"{h}D",
                "n": len(vals),
                "Hit Rate": float((vals > 0).mean()),
                "Median": float(vals.median()),
                "Mean": float(vals.mean()),
                "10th %ile": float(vals.quantile(0.10)),
                "25th %ile": float(vals.quantile(0.25)),
                "75th %ile": float(vals.quantile(0.75)),
                "90th %ile": float(vals.quantile(0.90)),
                "Worst": float(vals.min()),
                "Best": float(vals.max()),
                "Median Fwd DD": float(dds.median()) if not dds.empty else np.nan,
                "Worst Fwd DD": float(dds.min()) if not dds.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_to_base(cohort_df: pd.DataFrame, base_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    cohort_stats = summarize_forward_distribution(cohort_df, horizons, "Cohort")
    base_stats = summarize_forward_distribution(base_df, horizons, "Base")
    if cohort_stats.empty or base_stats.empty:
        return pd.DataFrame()

    merged = cohort_stats.merge(base_stats, on="Horizon", suffixes=("", " Base"))
    out = pd.DataFrame(
        {
            "Horizon": merged["Horizon"],
            "Cohort n": merged["n"],
            "Base n": merged["n Base"],
            "Cohort Hit": merged["Hit Rate"],
            "Base Hit": merged["Hit Rate Base"],
            "Hit Spread": merged["Hit Rate"] - merged["Hit Rate Base"],
            "Cohort Median": merged["Median"],
            "Base Median": merged["Median Base"],
            "Median Spread": merged["Median"] - merged["Median Base"],
            "Cohort Worst": merged["Worst"],
            "Base Worst": merged["Worst Base"],
            "Cohort Worst DD": merged["Worst Fwd DD"],
            "Base Worst DD": merged["Worst Fwd DD Base"],
        }
    )
    return out


def format_percent_table(df: pd.DataFrame, pct_cols: list[str], int_cols: list[str] | None = None, bps_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    int_cols = int_cols or []
    bps_cols = bps_cols or []
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: fmt_pct(x))
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt_int)
    for col in bps_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt_bps)
    return out


def percentile_rank(series: pd.Series, value: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0 or not np.isfinite(value):
        return np.nan
    return float((vals <= value).mean())


def current_percentile_table(feature_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    current = feature_df.iloc[-1]
    specs = [
        ("YTD", "ytd"),
        ("21D Return", "ret_21"),
        ("63D Return", "ret_63"),
        ("126D Return", "ret_126"),
        ("252D Return", "ret_252"),
        ("63D Vol", "vol_63"),
        ("63D Drawdown", "trail_dd_63"),
        ("Distance vs 200D", "dist_200d"),
    ]
    rows = []
    for label, col in specs:
        value = current.get(col, np.nan)
        pctile = percentile_rank(base_df[col], value) if col in base_df.columns else np.nan
        rows.append({"Metric": label, "Current": value, "Historical Percentile": pctile})
    return pd.DataFrame(rows)


# =========================
# ANALOG SCORING
# =========================

def ytd_composite_score(current_path: pd.Series, hist_path: pd.Series, rho: float) -> dict:
    hist_path = hist_path.iloc[:len(current_path)].dropna().astype(float)
    current_path = current_path.dropna().astype(float)

    current_endpoint = float(current_path.iloc[-1])
    hist_endpoint = float(hist_path.iloc[-1])
    current_vol = annualized_vol_from_path(current_path)
    hist_vol = annualized_vol_from_path(hist_path)
    current_dd = max_drawdown_from_path(current_path)
    hist_dd = max_drawdown_from_path(hist_path)
    current_slope_21 = trailing_move(current_path, min(21, len(current_path)))
    hist_slope_21 = trailing_move(hist_path, min(21, len(hist_path)))

    endpoint_score = similarity_from_gap(current_endpoint - hist_endpoint, 0.20)
    vol_score = similarity_from_gap(current_vol - hist_vol, max(0.10, current_vol, hist_vol)) if np.isfinite(current_vol) and np.isfinite(hist_vol) else 0.0
    dd_score = similarity_from_gap(current_dd - hist_dd, 0.20)
    slope_score = similarity_from_gap(current_slope_21 - hist_slope_21, 0.12)

    score = 0.50 * float(rho) + 0.20 * endpoint_score + 0.12 * vol_score + 0.10 * dd_score + 0.08 * slope_score

    return {
        "Score": float(score),
        "Endpoint Gap": float(hist_endpoint - current_endpoint),
        "Vol Gap": float(hist_vol - current_vol) if np.isfinite(hist_vol) and np.isfinite(current_vol) else np.nan,
        "Drawdown Gap": float(hist_dd - current_dd),
        "21D Slope Gap": float(hist_slope_21 - current_slope_21),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
        "Slope Score": float(slope_score),
    }


def rolling_composite_score(current_path: pd.Series, hist_path: pd.Series, rho: float) -> dict:
    current_endpoint = float(current_path.iloc[-1])
    hist_endpoint = float(hist_path.iloc[-1])
    current_vol = annualized_vol_from_path(current_path)
    hist_vol = annualized_vol_from_path(hist_path)
    current_dd = max_drawdown_from_path(current_path)
    hist_dd = max_drawdown_from_path(hist_path)

    endpoint_score = similarity_from_gap(current_endpoint - hist_endpoint, 0.25)
    vol_score = similarity_from_gap(current_vol - hist_vol, max(0.10, current_vol, hist_vol)) if np.isfinite(current_vol) and np.isfinite(hist_vol) else 0.0
    dd_score = similarity_from_gap(current_dd - hist_dd, 0.25)
    score = 0.58 * float(rho) + 0.18 * endpoint_score + 0.12 * vol_score + 0.12 * dd_score

    return {
        "Score": float(score),
        "Endpoint Gap": float(hist_endpoint - current_endpoint),
        "Vol Gap": float(hist_vol - current_vol) if np.isfinite(hist_vol) and np.isfinite(current_vol) else np.nan,
        "Drawdown Gap": float(hist_dd - current_dd),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
    }


def select_clustered_matches(df: pd.DataFrame, gap_days: int, max_one_per_year: bool) -> pd.DataFrame:
    if df.empty:
        return df
    chosen_rows = []
    chosen_locs = []
    used_years = set()
    candidates = df.sort_values(["Score", "Correlation"], ascending=[False, False]).reset_index(drop=True)

    for _, row in candidates.iterrows():
        signal_loc = int(row["_signal_loc"])
        year = int(row["Year"])
        if max_one_per_year and year in used_years:
            continue
        if any(abs(signal_loc - loc) < gap_days for loc in chosen_locs):
            continue
        chosen_rows.append(row)
        chosen_locs.append(signal_loc)
        used_years.add(year)

    if not chosen_rows:
        return pd.DataFrame(columns=df.columns)
    selected = pd.DataFrame(chosen_rows).reset_index(drop=True)
    selected["Cluster Windows"] = [int((df["_signal_loc"].sub(int(row["_signal_loc"])).abs() < gap_days).sum()) for _, row in selected.iterrows()]
    return selected.sort_values(["Score", "Correlation"], ascending=[False, False]).reset_index(drop=True)


# =========================
# CHARTS
# =========================

def plot_ytd_analogs(ticker_label: str, current_year: int, current: pd.Series, ytd_df: pd.DataFrame, analog_df: pd.DataFrame, n_days: int):
    palette = get_palette(len(analog_df))
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    fig.patch.set_facecolor(CHART_FACE)
    all_values = [current.values]

    for idx, row in analog_df.reset_index(drop=True).iterrows():
        yr = int(row["Year"])
        ser = ytd_df[yr].dropna()
        all_values.append(ser.values)
        ax.plot(
            ser.index,
            ser.values,
            "--",
            lw=2.15,
            alpha=0.92,
            color=palette[idx],
            label=f"{yr} | score {row['Score']:.2f} | ρ {row['Correlation']:.2f}",
        )

    ax.plot(current.index, current.values, color="black", lw=3.4, label=f"{current_year} YTD", zorder=10)
    ax.axvline(n_days, color="#6b7280", ls=":", lw=1.25, alpha=0.85)
    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)
    xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
    ax.set_xlim(1, xmax)
    ax.set_title(f"{ticker_label} | Current YTD Path vs Historical Analogs", fontsize=15, weight="bold", loc="left", pad=12)
    ax.set_xlabel("Trading Day of Year", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    set_percent_axis(ax, np.hstack(all_values), force_zero=True)
    clean_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def plot_forward_cone(ticker_label: str, matrix: np.ndarray, horizon: int, title: str, subtitle: str = "", overlay_label: str | None = None, overlay_path: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    fig.patch.set_facecolor(CHART_FACE)

    if matrix.size == 0:
        ax.text(0.5, 0.5, "No forward paths available", ha="center", va="center", transform=ax.transAxes)
        clean_axes(ax)
        return fig

    x = np.arange(0, matrix.shape[1])
    median = np.nanmedian(matrix, axis=0)
    p10 = np.nanpercentile(matrix, 10, axis=0)
    p25 = np.nanpercentile(matrix, 25, axis=0)
    p75 = np.nanpercentile(matrix, 75, axis=0)
    p90 = np.nanpercentile(matrix, 90, axis=0)

    ax.fill_between(x, p10, p90, color=LIGHT_FILL, alpha=0.90, label="10th to 90th %ile")
    ax.fill_between(x, p25, p75, color=MID_FILL, alpha=0.85, label="25th to 75th %ile")
    ax.plot(x, median, color="black", lw=3.0, label="Median forward path", zorder=10)

    all_values = [p10, p90, median]
    if overlay_path is not None and len(overlay_path) > 1:
        ax.plot(overlay_path.index, overlay_path.values, color=BLUE, lw=2.7, label=overlay_label or "Current / live path", zorder=11)
        all_values.append(overlay_path.values)

    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)
    ax.axvline(0, color="#6b7280", ls=":", lw=1.0, alpha=0.85)
    ax.set_xlim(0, horizon)
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=12)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=MUTED_TEXT)
    ax.set_xlabel("Trading Days After Signal", fontsize=12)
    ax.set_ylabel("Forward Cumulative Return", fontsize=12)
    set_percent_axis(ax, np.hstack(all_values), force_zero=True)
    clean_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def plot_setup_overlay(ticker_label: str, current_path: pd.Series, close_px: pd.Series, selected_df: pd.DataFrame, window: int):
    palette = get_palette(len(selected_df))
    fig, ax = plt.subplots(figsize=(14.5, 7.25))
    fig.patch.set_facecolor(CHART_FACE)
    all_values = [current_path.values]

    for idx, row in selected_df.reset_index(drop=True).iterrows():
        start_loc = int(row["_start_loc"])
        end_loc = int(row["_end_loc"])
        hist_path = normalized_price_path(close_px.iloc[start_loc:end_loc])
        if hist_path is None:
            continue
        all_values.append(hist_path.values)
        ax.plot(
            hist_path.index,
            hist_path.values,
            "--",
            lw=2.0,
            alpha=0.90,
            color=palette[idx],
            label=f"{int(row['Year'])} | score {row['Score']:.2f} | ρ {row['Correlation']:.2f}",
        )

    ax.plot(current_path.index, current_path.values, color="black", lw=3.4, label="Current trailing setup", zorder=10)
    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)
    ax.set_xlim(0, window - 1)
    ax.set_title(f"{ticker_label} | Current {window}D Setup vs Prior Matched Setups", fontsize=15, weight="bold", loc="left", pad=12)
    ax.set_xlabel("Trading Days in Setup Window", fontsize=12)
    ax.set_ylabel("Normalized Return", fontsize=12)
    set_percent_axis(ax, np.hstack(all_values), force_zero=True)
    clean_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def plot_distribution_bars(df: pd.DataFrame, value_col: str, title: str, current_value: float | None = None):
    vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    fig.patch.set_facecolor(CHART_FACE)
    if vals.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        clean_axes(ax)
        return fig

    colors = [GOOD if x >= 0 else BAD for x in vals]
    x_labels = df.loc[vals.index, "Year"].astype(str).values if "Year" in df.columns else np.arange(len(vals))
    ax.bar(np.arange(len(vals)), vals.values, color=colors, alpha=0.88, width=0.82)
    ax.axhline(0, color="#6b7280", lw=1.0)
    ax.axhline(vals.median(), color="#6b7280", ls=":", lw=1.3, label=f"Median {fmt_pct(vals.median())}")
    if current_value is not None and np.isfinite(current_value):
        ax.axhline(current_value, color=BLUE, ls="--", lw=1.5, label=f"Current {fmt_pct(current_value)}")
    if len(vals) <= 40:
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        ax.set_xticks([])
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=12)
    ax.set_ylabel("Return", fontsize=12)
    set_percent_axis(ax, vals.values, force_zero=True)
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# =========================
# SETUP FILTERS
# =========================

def apply_setup_filters(
    base_df: pd.DataFrame,
    ret_21_min: float | None,
    ret_63_min: float | None,
    ret_126_min: float | None,
    ret_252_min: float | None,
    max_trail_dd_63: float | None,
    max_trail_dd_252: float | None,
    above_200d: str,
    ma_50_vs_200: str,
    vix_max: float | None,
    vix_bucket: str,
    tnx_trend: str,
    dxy_trend: str,
    credit_trend: str,
) -> pd.DataFrame:
    df = base_df.copy()

    if ret_21_min is not None:
        df = df[df["ret_21"] >= ret_21_min]
    if ret_63_min is not None:
        df = df[df["ret_63"] >= ret_63_min]
    if ret_126_min is not None:
        df = df[df["ret_126"] >= ret_126_min]
    if ret_252_min is not None:
        df = df[df["ret_252"] >= ret_252_min]
    if max_trail_dd_63 is not None:
        df = df[df["trail_dd_63"] >= max_trail_dd_63]
    if max_trail_dd_252 is not None:
        df = df[df["trail_dd_252"] >= max_trail_dd_252]

    if above_200d == "Above 200D only":
        df = df[df["above_200d"] == True]
    elif above_200d == "Below 200D only":
        df = df[df["above_200d"] == False]

    if ma_50_vs_200 == "50D > 200D only":
        df = df[df["ma_50_gt_200"] == True]
    elif ma_50_vs_200 == "50D < 200D only":
        df = df[df["ma_50_gt_200"] == False]

    if vix_max is not None and "vix" in df.columns:
        df = df[df["vix"] <= vix_max]
    if vix_bucket != "Any" and "vix_bucket" in df.columns:
        df = df[df["vix_bucket"] == vix_bucket.lower()]
    if tnx_trend != "Any" and "tnx_trend" in df.columns:
        df = df[df["tnx_trend"] == tnx_trend.lower()]
    if dxy_trend != "Any" and "dxy_trend" in df.columns:
        df = df[df["dxy_trend"] == dxy_trend.lower()]
    if credit_trend != "Any" and "credit_trend" in df.columns:
        df = df[df["credit_trend"] == credit_trend.lower()]

    return df.copy()


# =========================
# APP START
# =========================

inject_css()
add_logo()

st.title("Market Memory Explorer")
st.caption("Historical analogs, setup cohorts, base rates, and forward drawdown math from adjusted close data.")

with st.sidebar:
    st.header("Controls")
    ticker_in = st.text_input("Ticker", "^SPX").strip().upper()
    ticker = TICKER_ALIASES.get(ticker_in, ticker_in)
    ticker_label = ticker_in if ticker_in else ticker

    st.markdown("---")
    start_year = st.slider("Historical sample start", 1950, dt.datetime.now(NY_TZ).year, 1950, 1)
    top_n = st.slider("Top matches shown", 1, 15, DEFAULT_TOP_N, 1)
    load_regime = st.checkbox("Load macro regime proxies", value=True)

    st.markdown("---")
    st.subheader("YTD Analog Settings")
    min_ytd_corr = st.slider("Minimum YTD correlation", -1.00, 1.00, 0.00, 0.05, format="%.2f")

    st.markdown("---")
    st.subheader("Rolling Analog Settings")
    rolling_window = st.select_slider("Trailing setup window", options=[63, 126, 189, 252, 378, 504], value=DEFAULT_ROLLING_WINDOW)
    rolling_horizon = st.select_slider("Forward horizon", options=[21, 42, 63, 126, 252], value=DEFAULT_FORWARD_HORIZON)
    rolling_min_corr = st.slider("Minimum rolling correlation", 0.00, 1.00, 0.75, 0.05, format="%.2f")
    rolling_step = st.slider("Rolling scan step", 1, 10, 1, 1)
    cluster_gap = st.slider("Minimum spacing between matches", 21, 252, 63, 21)
    max_one_per_year = st.checkbox("Max one rolling match per year", value=True)

try:
    raw = load_history(ticker)
except Exception as exc:
    st.error(f"Download failed for {ticker}: {exc}")
    st.stop()

close_px = raw["Close"].dropna().copy()
if close_px.empty:
    st.error("No usable price history found.")
    st.stop()

last_data_date = pd.Timestamp(close_px.index[-1])
ny_now = dt.datetime.now(NY_TZ)
stale_bdays = business_days_behind(last_data_date, ny_now.date())
if stale_bdays > 1:
    st.warning(f"Latest close for {ticker_label} is {last_data_date:%Y-%m-%d}. That is {stale_bdays} business days behind New York time.")

regime_data, missing_regime = load_regime_data(close_px, load_regime)
if missing_regime:
    st.caption(f"Some optional regime proxies failed to load: {', '.join(missing_regime)}")

feature_df = build_feature_frame(close_px, regime_data)
base_df = historical_universe(feature_df, start_year=start_year, min_history_days=252, max_horizon=252)
latest_year = int(last_data_date.year)
current = current_feature_snapshot(feature_df)

metric_cols = st.columns(6)
with metric_cols[0]:
    render_metric("Latest Close", f"{current['Close']:,.2f}" if np.isfinite(current["Close"]) else "N/A")
with metric_cols[1]:
    render_metric("YTD", fmt_pct(current["YTD"]))
with metric_cols[2]:
    render_metric("63D", fmt_pct(current["63D"]))
with metric_cols[3]:
    render_metric("252D", fmt_pct(current["252D"]))
with metric_cols[4]:
    render_metric("63D Vol", fmt_pct(current["Vol 63D"], signed=False))
with metric_cols[5]:
    render_metric("Vs 200D", fmt_pct(current["Distance vs 200D"]))

st.caption(
    f"Data through {last_data_date:%Y-%m-%d}. Adjusted close from Yahoo Finance. "
    f"Base-rate universe: {len(base_df):,} eligible historical signal dates since {start_year}."
)


tab_ytd, tab_snapshot, tab_rolling, tab_scanner, tab_base = st.tabs(
    ["YTD Analogs", "Snapshot", "Rolling Analogs", "Setup Scanner", "Base Rates"]
)


# =========================
# TAB 1: SNAPSHOT
# =========================

with tab_snapshot:
    st.subheader("Current Market State")
    st.markdown(
        "<div class='section-caption'>Current return, volatility, drawdown, trend, and macro proxy readings placed against the historical sample.</div>",
        unsafe_allow_html=True,
    )

    pct_table = current_percentile_table(feature_df, base_df)
    display_pct = pct_table.copy()
    display_pct["Current"] = display_pct.apply(
        lambda r: fmt_pct(r["Current"], signed=(r["Metric"] != "63D Vol")) if r["Metric"] != "63D Vol" else fmt_pct(r["Current"], signed=False), axis=1
    )
    display_pct["Historical Percentile"] = display_pct["Historical Percentile"].map(lambda x: fmt_pct(x, signed=False))
    st.dataframe(display_pct, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns([0.58, 0.42], gap="large")
    with c1:
        base_stats = summarize_forward_distribution(base_df, DEFAULT_HORIZONS, "Base")
        display_base = format_percent_table(
            base_stats,
            ["Hit Rate", "Median", "Mean", "10th %ile", "25th %ile", "75th %ile", "90th %ile", "Worst", "Best", "Median Fwd DD", "Worst Fwd DD"],
            int_cols=["n"],
        )
        st.markdown("**Unconditional forward base rates**")
        st.dataframe(display_base, use_container_width=True, hide_index=True)
    with c2:
        snapshot_rows = [
            {"Regime Item": "Above 200D", "Current": "Yes" if bool(current["Above 200D"]) else "No"},
            {"Regime Item": "VIX", "Current": fmt_num(current["VIX"], 1)},
            {"Regime Item": "10Y 63D Change", "Current": fmt_bps(current["10Y 63D"])},
            {"Regime Item": "DXY 63D", "Current": fmt_pct(current["DXY 63D"])},
            {"Regime Item": "Credit 63D", "Current": fmt_pct(current["Credit 63D"])},
        ]
        st.markdown("**Macro proxy snapshot**")
        st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)


# =========================
# TAB 2: YTD ANALOGS
# =========================

with tab_ytd:
    st.subheader("Calendar-Year Analog Map")
    st.markdown(
        "<div class='section-caption'>True YTD paths are anchored to the prior year-end close. The score combines path correlation, endpoint gap, volatility gap, drawdown gap, and 21-day slope gap.</div>",
        unsafe_allow_html=True,
    )

    paths, date_maps = build_true_ytd_paths(raw, latest_year)
    if not paths or latest_year not in paths:
        st.warning(f"No usable YTD data for {latest_year}.")
    else:
        ytd_df = pd.DataFrame(paths)
        current_ytd = ytd_df[latest_year].dropna()
        n_days = len(current_ytd)

        if n_days < MIN_DAYS_FOR_CORR:
            st.info(f"{latest_year} has only {n_days} trading days. Correlation analogs are unstable before {MIN_DAYS_FOR_CORR} trading days.")
        else:
            records = []
            for yr in ytd_df.columns:
                yr = int(yr)
                if yr == latest_year or yr < start_year:
                    continue
                ser = ytd_df[yr].dropna()
                if len(ser) < n_days:
                    continue
                hist_slice = ser.iloc[:n_days]
                rho = safe_corr(current_ytd.values, hist_slice.values)
                if not np.isfinite(rho) or rho < min_ytd_corr:
                    continue

                components = ytd_composite_score(current_ytd, hist_slice, rho)
                match_date = pd.Timestamp(date_maps[yr].loc[n_days])
                full_year_return = float(ser.iloc[-1])
                ytd_at_match = float(hist_slice.iloc[-1])
                return_after_match = float((1.0 + full_year_return) / (1.0 + ytd_at_match) - 1.0)

                row = feature_df.loc[feature_df.index == match_date]
                loc = int(row["_loc"].iloc[0]) if not row.empty else None
                nexts = {}
                if loc is not None:
                    for h in DEFAULT_HORIZONS:
                        nexts[f"Next {h}D"] = feature_df.iloc[loc].get(f"next_{h}", np.nan)
                        nexts[f"Fwd DD {h}D"] = feature_df.iloc[loc].get(f"fwd_dd_{h}", np.nan)

                records.append(
                    {
                        "Year": yr,
                        "Match Date": match_date,
                        "Score": components["Score"],
                        "Correlation": float(rho),
                        "YTD at Match": ytd_at_match,
                        "Full-Year Return": full_year_return,
                        "Return After Match": return_after_match,
                        "Realized Vol": annualized_vol_from_path(hist_slice),
                        "Max Drawdown": max_drawdown_from_path(hist_slice),
                        **nexts,
                        **components,
                    }
                )

            if not records:
                st.warning("No historical years meet the current YTD filter.")
            else:
                calendar_df = pd.DataFrame(records).sort_values(["Score", "Correlation"], ascending=[False, False]).reset_index(drop=True)
                top_calendar = calendar_df.head(top_n).copy()
                fig_ytd = plot_ytd_analogs(ticker_label, latest_year, current_ytd, ytd_df, top_calendar, n_days)
                st.pyplot(fig_ytd, clear_figure=True)
                st.download_button("Download YTD analog chart", fig_to_png_bytes(fig_ytd), file_name=f"{ticker_label}_ytd_analogs.png", mime="image/png")
                plt.close(fig_ytd)

                display_cols = [
                    "Year", "Match Date", "Score", "Correlation", "YTD at Match", "Full-Year Return", "Return After Match",
                    "Next 21D", "Next 63D", "Next 126D", "Next 252D", "Fwd DD 252D",
                    "Endpoint Gap", "Vol Gap", "Drawdown Gap", "21D Slope Gap",
                ]

                # Some tickers / sparse histories can produce calendar-year analog rows
                # without complete forward-horizon columns. Pandas raises a KeyError
                # if we select columns that do not exist, so create the missing display
                # columns explicitly and show them as N/A rather than breaking the app.
                for col in display_cols:
                    if col not in top_calendar.columns:
                        top_calendar[col] = np.nan

                table = top_calendar[display_cols].copy()
                table["Match Date"] = pd.to_datetime(table["Match Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                table = format_percent_table(
                    table,
                    ["YTD at Match", "Full-Year Return", "Return After Match", "Next 21D", "Next 63D", "Next 126D", "Next 252D", "Fwd DD 252D", "Endpoint Gap", "Vol Gap", "Drawdown Gap", "21D Slope Gap"],
                )
                for col in ["Score", "Correlation"]:
                    table[col] = top_calendar[col].map(lambda x: fmt_num(x, 3)) if col in top_calendar.columns else "N/A"
                st.markdown("**Top calendar-year analogs**")
                st.dataframe(table, use_container_width=True, hide_index=True)
                dataframe_download_button(top_calendar, "Download YTD analog table", f"{ticker_label}_ytd_analogs.csv")

                compare_df = compare_to_base(top_calendar, base_df, DEFAULT_HORIZONS)
                if not compare_df.empty:
                    display_compare = format_percent_table(
                        compare_df,
                        ["Cohort Hit", "Base Hit", "Hit Spread", "Cohort Median", "Base Median", "Median Spread", "Cohort Worst", "Base Worst", "Cohort Worst DD", "Base Worst DD"],
                        int_cols=["Cohort n", "Base n"],
                    )
                    st.markdown("**Top analog forward results versus base rates**")
                    st.dataframe(display_compare, use_container_width=True, hide_index=True)


# =========================
# TAB 3: ROLLING ANALOGS
# =========================

with tab_rolling:
    st.subheader("Rolling Setup Analogs")
    st.markdown(
        "<div class='section-caption'>This scans every prior rolling window of the selected length, clusters overlapping matches, and measures what happened after the signal date.</div>",
        unsafe_allow_html=True,
    )

    if len(close_px) < rolling_window * 3:
        st.info(f"Need at least {rolling_window * 3} trading days of history for a clean rolling setup plus forward window.")
    else:
        current_start_loc = len(close_px) - rolling_window
        current_trailing = close_px.iloc[-rolling_window:].copy()
        current_trailing_path = normalized_price_path(current_trailing)

        rolling_matches = []
        if current_trailing_path is not None:
            max_start = len(close_px) - rolling_window - rolling_horizon
            for start_loc in range(0, max_start + 1, int(rolling_step)):
                end_loc = start_loc + rolling_window
                signal_loc = end_loc - 1
                if end_loc > current_start_loc:
                    continue
                hist_window = close_px.iloc[start_loc:end_loc].copy()
                hist_path = normalized_price_path(hist_window)
                if hist_path is None or len(hist_path) != len(current_trailing_path):
                    continue
                rho = safe_corr(current_trailing_path.values, hist_path.values)
                if not np.isfinite(rho) or rho < rolling_min_corr:
                    continue
                fwd_path = forward_path_from_loc(close_px, signal_loc, rolling_horizon)
                if fwd_path is None:
                    continue
                components = rolling_composite_score(current_trailing_path, hist_path, rho)
                signal_date = pd.Timestamp(hist_window.index[-1])
                row = feature_df.iloc[signal_loc]
                rolling_matches.append(
                    {
                        "Year": int(signal_date.year),
                        "Match Start": pd.Timestamp(hist_window.index[0]),
                        "Match End": signal_date,
                        "Signal Date": signal_date,
                        "Score": components["Score"],
                        "Correlation": float(rho),
                        "Setup Return": float(hist_path.iloc[-1]),
                        "Next 21D": row.get("next_21", np.nan),
                        "Next 63D": row.get("next_63", np.nan),
                        "Next 126D": row.get("next_126", np.nan),
                        "Next 252D": row.get("next_252", np.nan),
                        "Fwd DD 21D": row.get("fwd_dd_21", np.nan),
                        "Fwd DD 63D": row.get("fwd_dd_63", np.nan),
                        "Fwd DD 126D": row.get("fwd_dd_126", np.nan),
                        "Fwd DD 252D": row.get("fwd_dd_252", np.nan),
                        "VIX": row.get("vix", np.nan),
                        "10Y 63D": row.get("tnx_63_bps", np.nan),
                        "DXY 63D": row.get("dxy_63", np.nan),
                        "Credit 63D": row.get("credit_63", np.nan),
                        "_start_loc": int(start_loc),
                        "_end_loc": int(end_loc),
                        "_signal_loc": int(signal_loc),
                        **components,
                    }
                )

        if not rolling_matches:
            st.warning(f"No rolling {rolling_window}D windows found with correlation above {rolling_min_corr:.2f}.")
        else:
            rolling_df = pd.DataFrame(rolling_matches)
            selected_rolling = select_clustered_matches(rolling_df, int(cluster_gap), bool(max_one_per_year))
            overlay_rolling = selected_rolling.head(top_n).copy()

            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig_setup = plot_setup_overlay(ticker_label, current_trailing_path, close_px, overlay_rolling, rolling_window)
                st.pyplot(fig_setup, clear_figure=True)
                st.download_button("Download setup overlay", fig_to_png_bytes(fig_setup), file_name=f"{ticker_label}_rolling_setup_overlay.png", mime="image/png")
                plt.close(fig_setup)
            with c2:
                locs = selected_rolling["_signal_loc"].astype(int).tolist()
                matrix = forward_matrix_from_locs(close_px, locs, rolling_horizon)
                fig_cone = plot_forward_cone(
                    ticker_label,
                    matrix,
                    rolling_horizon,
                    f"{ticker_label} | Forward Distribution After Similar {rolling_window}D Setups",
                    subtitle=f"n={len(locs)} clustered matches, min ρ={rolling_min_corr:.2f}",
                )
                st.pyplot(fig_cone, clear_figure=True)
                st.download_button("Download forward cone", fig_to_png_bytes(fig_cone), file_name=f"{ticker_label}_rolling_forward_cone.png", mime="image/png")
                plt.close(fig_cone)

            display_cols = [
                "Year", "Match Start", "Match End", "Score", "Correlation", "Cluster Windows", "Setup Return",
                "Next 21D", "Next 63D", "Next 126D", "Next 252D", "Fwd DD 252D",
                "VIX", "10Y 63D", "DXY 63D", "Credit 63D",
            ]
            table = selected_rolling[display_cols].copy()
            for col in ["Match Start", "Match End"]:
                table[col] = pd.to_datetime(table[col]).dt.strftime("%Y-%m-%d")
            table = format_percent_table(
                table,
                ["Setup Return", "Next 21D", "Next 63D", "Next 126D", "Next 252D", "Fwd DD 252D", "DXY 63D", "Credit 63D"],
                int_cols=["Cluster Windows"],
                bps_cols=["10Y 63D"],
            )
            for col in ["Score", "Correlation", "VIX"]:
                table[col] = selected_rolling[col].map(lambda x: fmt_num(x, 3 if col != "VIX" else 1))
            st.markdown("**Clustered rolling matches**")
            st.dataframe(table, use_container_width=True, hide_index=True)
            dataframe_download_button(selected_rolling, "Download rolling analog table", f"{ticker_label}_rolling_analogs.csv")

            compare_df = compare_to_base(selected_rolling, base_df, DEFAULT_HORIZONS)
            if not compare_df.empty:
                display_compare = format_percent_table(
                    compare_df,
                    ["Cohort Hit", "Base Hit", "Hit Spread", "Cohort Median", "Base Median", "Median Spread", "Cohort Worst", "Base Worst", "Cohort Worst DD", "Base Worst DD"],
                    int_cols=["Cohort n", "Base n"],
                )
                st.markdown("**Rolling analog cohort versus base rates**")
                st.dataframe(display_compare, use_container_width=True, hide_index=True)

            failures = selected_rolling.sort_values(f"Next {rolling_horizon}D", ascending=True).head(5)
            if not failures.empty:
                fail_cols = ["Year", "Match End", f"Next {rolling_horizon}D", f"Fwd DD {rolling_horizon}D", "Score", "Correlation"]
                fail = failures[fail_cols].copy()
                fail["Match End"] = pd.to_datetime(fail["Match End"]).dt.strftime("%Y-%m-%d")
                fail = format_percent_table(fail, [f"Next {rolling_horizon}D", f"Fwd DD {rolling_horizon}D"])
                fail["Score"] = failures["Score"].map(lambda x: fmt_num(x, 3))
                fail["Correlation"] = failures["Correlation"].map(lambda x: fmt_num(x, 3))
                st.markdown("**Failure cases inside the matched set**")
                st.dataframe(fail, use_container_width=True, hide_index=True)


# =========================
# TAB 4: SETUP SCANNER
# =========================

with tab_scanner:
    st.subheader("Setup Scanner")
    st.markdown(
        "<div class='section-caption'>Build a condition-based historical cohort, then compare its forward return and drawdown profile against the normal base rate.</div>",
        unsafe_allow_html=True,
    )

    scan_left, scan_right = st.columns([0.50, 0.50], gap="large")
    with scan_left:
        st.markdown("**Price and trend filters**")
        ret_21_min_raw = st.slider("Minimum 21D return", -30, 50, -100 if False else -5, 1, format="%d%%")
        ret_63_min_raw = st.slider("Minimum 63D return", -40, 80, 0, 1, format="%d%%")
        ret_126_min_raw = st.slider("Minimum 126D return", -50, 120, -50, 1, format="%d%%")
        ret_252_min_raw = st.slider("Minimum 252D return", -70, 200, -70, 1, format="%d%%")
        max_dd_63_raw = st.slider("Worst allowed trailing 63D drawdown", -60, 0, -20, 1, format="%d%%")
        max_dd_252_raw = st.slider("Worst allowed trailing 252D drawdown", -80, 0, -40, 1, format="%d%%")
        above_200d_filter = st.selectbox("200D trend filter", ["Any", "Above 200D only", "Below 200D only"], index=1)
        ma_50_filter = st.selectbox("50D / 200D filter", ["Any", "50D > 200D only", "50D < 200D only"], index=0)

    with scan_right:
        st.markdown("**Macro proxy filters**")
        use_vix_cap = st.checkbox("Use VIX cap", value=False)
        vix_cap = st.slider("Maximum VIX", 8, 80, 25, 1) if use_vix_cap else None
        vix_bucket_filter = st.selectbox("VIX bucket", ["Any", "Calm", "Normal", "Stressed"], index=0)
        tnx_trend_filter = st.selectbox("10Y 63D trend", ["Any", "Rising", "Flat", "Falling"], index=0)
        dxy_trend_filter = st.selectbox("Dollar 63D trend", ["Any", "Rising", "Flat", "Falling"], index=0)
        credit_trend_filter = st.selectbox("Credit 63D trend", ["Any", "Improving", "Flat", "Worsening"], index=0)
        scanner_horizon = st.select_slider("Chart horizon", options=[21, 42, 63, 126, 252], value=252)
        max_rows_to_show = st.slider("Rows shown", 10, 200, 50, 10)

    cohort = apply_setup_filters(
        base_df=base_df,
        ret_21_min=ret_21_min_raw / 100.0,
        ret_63_min=ret_63_min_raw / 100.0,
        ret_126_min=ret_126_min_raw / 100.0,
        ret_252_min=ret_252_min_raw / 100.0,
        max_trail_dd_63=max_dd_63_raw / 100.0,
        max_trail_dd_252=max_dd_252_raw / 100.0,
        above_200d=above_200d_filter,
        ma_50_vs_200=ma_50_filter,
        vix_max=vix_cap,
        vix_bucket=vix_bucket_filter,
        tnx_trend=tnx_trend_filter,
        dxy_trend=dxy_trend_filter,
        credit_trend=credit_trend_filter,
    )

    st.markdown("---")
    if cohort.empty:
        st.warning("No historical observations match the selected setup. Loosen one or two filters.")
    else:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric("Cohort n", f"{len(cohort):,}")
        with metric_cols[1]:
            render_metric(f"Median Next {scanner_horizon}D", fmt_pct(cohort[f"next_{scanner_horizon}"].median()))
        with metric_cols[2]:
            render_metric(f"Hit Rate {scanner_horizon}D", fmt_pct((cohort[f"next_{scanner_horizon}"] > 0).mean(), signed=False))
        with metric_cols[3]:
            render_metric(f"Worst DD {scanner_horizon}D", fmt_pct(cohort[f"fwd_dd_{scanner_horizon}"].min()))

        locs = cohort["_loc"].astype(int).tolist()
        matrix = forward_matrix_from_locs(close_px, locs, scanner_horizon)
        fig_scanner = plot_forward_cone(
            ticker_label,
            matrix,
            scanner_horizon,
            f"{ticker_label} | Forward Distribution After Custom Setup",
            subtitle=f"n={len(cohort):,} observations since {start_year}",
        )
        st.pyplot(fig_scanner, clear_figure=True)
        st.download_button("Download setup scanner chart", fig_to_png_bytes(fig_scanner), file_name=f"{ticker_label}_setup_scanner_forward_cone.png", mime="image/png")
        plt.close(fig_scanner)

        compare_df = compare_to_base(cohort, base_df, DEFAULT_HORIZONS)
        display_compare = format_percent_table(
            compare_df,
            ["Cohort Hit", "Base Hit", "Hit Spread", "Cohort Median", "Base Median", "Median Spread", "Cohort Worst", "Base Worst", "Cohort Worst DD", "Base Worst DD"],
            int_cols=["Cohort n", "Base n"],
        )
        st.markdown("**Setup versus base rate**")
        st.dataframe(display_compare, use_container_width=True, hide_index=True)

        cohort_table_cols = [
            "Date", "Year", "ret_21", "ret_63", "ret_126", "ret_252", "trail_dd_63", "trail_dd_252",
            "above_200d", "vix", "tnx_63_bps", "dxy_63", "credit_63",
            "next_21", "next_63", "next_126", "next_252", "fwd_dd_252",
        ]
        cohort_for_table = cohort.reset_index(drop=True).copy()
        available_cohort_cols = [col for col in cohort_table_cols if col in cohort_for_table.columns]
        table_raw = cohort_for_table.sort_values("Date", ascending=False).head(max_rows_to_show)[available_cohort_cols].copy()
        table_raw["Date"] = pd.to_datetime(table_raw["Date"]).dt.strftime("%Y-%m-%d")
        table_raw = table_raw.rename(
            columns={
                "ret_21": "21D", "ret_63": "63D", "ret_126": "126D", "ret_252": "252D",
                "trail_dd_63": "Trail DD 63D", "trail_dd_252": "Trail DD 252D", "above_200d": "Above 200D",
                "vix": "VIX", "tnx_63_bps": "10Y 63D", "dxy_63": "DXY 63D", "credit_63": "Credit 63D",
                "next_21": "Next 21D", "next_63": "Next 63D", "next_126": "Next 126D", "next_252": "Next 252D", "fwd_dd_252": "Fwd DD 252D",
            }
        )
        table_display = format_percent_table(
            table_raw,
            ["21D", "63D", "126D", "252D", "Trail DD 63D", "Trail DD 252D", "DXY 63D", "Credit 63D", "Next 21D", "Next 63D", "Next 126D", "Next 252D", "Fwd DD 252D"],
            bps_cols=["10Y 63D"],
        )
        table_display["VIX"] = table_raw["VIX"].map(lambda x: fmt_num(x, 1))
        table_display["Above 200D"] = table_raw["Above 200D"].map(lambda x: "Yes" if bool(x) else "No")
        st.markdown("**Most recent matching observations**")
        st.dataframe(table_display, use_container_width=True, hide_index=True)
        dataframe_download_button(cohort, "Download setup cohort", f"{ticker_label}_setup_cohort.csv")

        worst_cases = cohort.reset_index(drop=True).sort_values(f"next_{scanner_horizon}", ascending=True).head(8)
        worst_display = worst_cases[["Date", "Year", f"next_{scanner_horizon}", f"fwd_dd_{scanner_horizon}", "ret_63", "ret_252", "vix", "tnx_63_bps", "dxy_63", "credit_63"]].copy()
        worst_display["Date"] = pd.to_datetime(worst_display["Date"]).dt.strftime("%Y-%m-%d")
        worst_display = worst_display.rename(
            columns={
                f"next_{scanner_horizon}": f"Next {scanner_horizon}D",
                f"fwd_dd_{scanner_horizon}": f"Fwd DD {scanner_horizon}D",
                "ret_63": "63D", "ret_252": "252D", "vix": "VIX", "tnx_63_bps": "10Y 63D", "dxy_63": "DXY 63D", "credit_63": "Credit 63D",
            }
        )
        worst_display = format_percent_table(
            worst_display,
            [f"Next {scanner_horizon}D", f"Fwd DD {scanner_horizon}D", "63D", "252D", "DXY 63D", "Credit 63D"],
            bps_cols=["10Y 63D"],
        )
        worst_display["VIX"] = worst_cases["vix"].map(lambda x: fmt_num(x, 1))
        st.markdown("**Failure cases for this setup**")
        st.dataframe(worst_display, use_container_width=True, hide_index=True)


# =========================
# TAB 5: BASE RATES
# =========================

with tab_base:
    st.subheader("Base Rates and Historical Distribution")
    st.markdown(
        "<div class='section-caption'>The base-rate table is the reference point for every setup. A conditional result is useful only if it improves either return, hit rate, or drawdown profile versus this distribution.</div>",
        unsafe_allow_html=True,
    )

    base_stats = summarize_forward_distribution(base_df, DEFAULT_HORIZONS, "Base")
    display_base = format_percent_table(
        base_stats,
        ["Hit Rate", "Median", "Mean", "10th %ile", "25th %ile", "75th %ile", "90th %ile", "Worst", "Best", "Median Fwd DD", "Worst Fwd DD"],
        int_cols=["n"],
    )
    st.dataframe(display_base, use_container_width=True, hide_index=True)
    dataframe_download_button(base_df, "Download full base-rate universe", f"{ticker_label}_base_rate_universe.csv")

    dist_horizon = st.select_slider("Distribution horizon", options=[21, 63, 126, 252], value=63)
    by_year = base_df.groupby("Year")[f"next_{dist_horizon}"].median().reset_index()
    fig_dist = plot_distribution_bars(
        by_year,
        f"next_{dist_horizon}",
        f"{ticker_label} | Median Forward {dist_horizon}D Return by Signal Year",
    )
    st.pyplot(fig_dist, clear_figure=True)
    st.download_button("Download base-rate chart", fig_to_png_bytes(fig_dist), file_name=f"{ticker_label}_base_rate_{dist_horizon}d_by_year.png", mime="image/png")
    plt.close(fig_dist)

st.caption("© 2026 AD Fund Management LP")
