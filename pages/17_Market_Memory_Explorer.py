
import datetime as dt
import time
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

MIN_DAYS_REQUIRED = 30
MIN_DAYS_CURRENT_YEAR = 5
MIN_DAYS_FOR_CORR = 10
CACHE_TTL_SECONDS = 3600

TRAILING_DAYS = 252
ROLLING_MIN_CORR_DEFAULT = 0.75
ROLLING_STEP_DEFAULT = 1
ROLLING_CLUSTER_GAP_DEFAULT = 63

NY_TZ = ZoneInfo("America/New_York")

TICKER_ALIASES = {
    "^SPX": "^GSPC",
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "NDX": "^NDX",
    "NASDAQ": "^IXIC",
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

st.set_page_config(page_title="Market Memory Explorer", layout="wide")


# =========================
# SMALL UI HELPERS
# =========================

def fmt_pct(x, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}%}"
    except Exception:
        return "N/A"


def fmt_num(x, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"


def add_logo() -> None:
    candidate_paths = [
        Path("assets/logo.png"),
        Path("logo.png"),
        Path("/mnt/data/0ea02e99-f067-4315-accc-0d2bbd3ee87d.png"),
    ]
    for path in candidate_paths:
        if path.exists():
            st.image(str(path), width=70)
            return



def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.25rem;
        }
        .hero-wrap {
            margin: 0.15rem 0 1.10rem 0;
        }
        .hero-title {
            color: #202733;
            font-size: 2.35rem;
            font-weight: 700;
            line-height: 1.08;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #667085;
            font-size: 1.02rem;
            line-height: 1.45;
            max-width: 980px;
        }
        .inline-stat-wrap {
            margin-top: 1.95rem;
        }
        .inline-stat {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #fafbfd 100%);
            border: 1px solid #e4e9f1;
            border-radius: 14px;
            padding: 0.95rem 1.15rem;
            min-height: 54px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .inline-stat-label {
            color: #8b97a8;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        .inline-stat-value {
            color: #202733;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.03em;
            white-space: nowrap;
        }
        div[data-testid="stTextInput"] label p {
            color: #8b97a8;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        div[data-baseweb="input"] > div {
            background: #f7f9fc;
            border: 1px solid #dde3ed;
            border-radius: 14px;
            min-height: 54px;
            box-shadow: inset 0 1px 1px rgba(16, 24, 40, 0.02);
        }
        div[data-baseweb="input"] input {
            color: #202733;
            font-size: 1.08rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">Market Memory Explorer</div>
            <div class="hero-subtitle">Compare the current market path with history, then measure what happened next.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inline_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="inline-stat-wrap">
            <div class="inline-stat">
                <div class="inline-stat-label">{label}</div>
                <div class="inline-stat-value">{value}</div>
            </div>
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


def set_percent_axis(ax, values) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    pad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02

    ax.set_ylim(ymin - pad, ymax + pad)

    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    raw_step = max(span / 12, 0.0025)
    candidates = np.array([
        0.0025, 0.005, 0.01, 0.02, 0.025, 0.05,
        0.10, 0.20, 0.25, 0.50, 1.00
    ])
    step = float(candidates[np.argmin(np.abs(candidates - raw_step))])

    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))


def get_palette(n: int):
    cmap_names = ["tab10", "Dark2", "Set1", "tab20", "tab20b", "tab20c", "Paired"]
    colors = []
    seen = set()

    for name in cmap_names:
        cmap = plt.get_cmap(name)
        if hasattr(cmap, "colors"):
            raw_colors = cmap.colors
        else:
            raw_colors = [cmap(x) for x in np.linspace(0, 1, 20)]

        for color in raw_colors:
            rgb = tuple(color[:3])
            key = tuple(round(v, 4) for v in rgb)
            if key not in seen:
                seen.add(key)
                colors.append(rgb)

    if n <= len(colors):
        return colors[:n]

    fallback = [plt.get_cmap("hsv")(x)[:3] for x in np.linspace(0, 1, n)]
    return (colors + fallback)[:n]


# =========================
# DATA LOADING
# =========================

@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def load_history(symbol: str) -> pd.DataFrame:
    """
    Robust Yahoo loader with retries. Returns a clean Close-only frame.
    """
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


def load_optional_regime_data(raw_close: pd.Series):
    """
    Regime data is optional. The app should still work if one proxy fails.
    """
    data = {"asset": raw_close.dropna().copy()}
    missing = []

    for key, symbol in REGIME_SYMBOLS.items():
        try:
            hist = load_history(symbol)
            data[key] = hist["Close"].dropna().copy()
        except Exception:
            missing.append(symbol)

    if "hyg" in data and "ief" in data:
        ratio = (data["hyg"] / data["ief"]).dropna()
        if not ratio.empty:
            data["credit"] = ratio

    return data, missing


# =========================
# PATH MATH
# =========================

def business_days_behind(last_date: pd.Timestamp, reference_date: dt.date) -> int:
    last_day = pd.Timestamp(last_date).date()
    if last_day >= reference_date:
        return 0
    return len(pd.bdate_range(pd.Timestamp(last_day) + pd.Timedelta(days=1), pd.Timestamp(reference_date)))


def build_true_ytd_paths(raw: pd.DataFrame, current_year: int):
    """
    True YTD return path uses the prior year's final close as the base.
    This preserves the first trading day's gap instead of forcing day one to 0%.
    """
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
        else:
            if len(year_closes) < MIN_DAYS_REQUIRED:
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

        dates = pd.Series(year_closes.index, index=day_index)

        paths[int(yr)] = ytd.astype(float)
        date_maps[int(yr)] = dates

    return paths, date_maps


def daily_returns_from_cum_path(path: pd.Series) -> pd.Series:
    path = path.dropna().astype(float)
    if path.empty:
        return pd.Series(dtype=float)

    ret = (1.0 + path).div((1.0 + path).shift(1)).sub(1.0)
    ret.iloc[0] = path.iloc[0]
    return ret.dropna()


def normalized_price_path(series: pd.Series):
    series = series.dropna().astype(float)
    if len(series) < 2:
        return None
    first = float(series.iloc[0])
    if not np.isfinite(first) or first <= 0:
        return None
    out = series / first - 1.0
    out.index = np.arange(0, len(out))
    return out


def forward_path_from_signal(series: pd.Series, signal_loc: int, horizon: int):
    end_loc = signal_loc + horizon
    if signal_loc < 0 or end_loc >= len(series):
        return None

    window = series.iloc[signal_loc:end_loc + 1].copy()
    if len(window) != horizon + 1 or window.isna().any():
        return None

    first = float(window.iloc[0])
    if not np.isfinite(first) or first <= 0:
        return None

    out = window / first - 1.0
    out.index = np.arange(0, len(out))
    return out.astype(float)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


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


def similarity_from_gap(gap: float, tolerance: float) -> float:
    if not np.isfinite(gap) or tolerance <= 0:
        return 0.0
    return float(np.clip(1.0 - abs(gap) / tolerance, 0.0, 1.0))


def ytd_composite_score(current_path: pd.Series, hist_path: pd.Series, rho: float, regime_score=None):
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

    if np.isfinite(current_vol) and np.isfinite(hist_vol):
        vol_score = similarity_from_gap(current_vol - hist_vol, max(0.10, current_vol, hist_vol))
    else:
        vol_score = 0.0

    dd_score = similarity_from_gap(current_dd - hist_dd, 0.20)
    slope_score = similarity_from_gap(current_slope_21 - hist_slope_21, 0.12)

    score = (
        0.50 * float(rho)
        + 0.20 * endpoint_score
        + 0.12 * vol_score
        + 0.10 * dd_score
        + 0.08 * slope_score
    )

    if regime_score is not None and np.isfinite(regime_score):
        score = 0.88 * score + 0.12 * float(regime_score)

    return {
        "Score": float(score),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
        "Slope Score": float(slope_score),
    }


def rolling_composite_score(current_path: pd.Series, hist_path: pd.Series, rho: float, regime_score=None):
    current_endpoint = float(current_path.iloc[-1])
    hist_endpoint = float(hist_path.iloc[-1])

    current_vol = annualized_vol_from_path(current_path)
    hist_vol = annualized_vol_from_path(hist_path)

    current_dd = max_drawdown_from_path(current_path)
    hist_dd = max_drawdown_from_path(hist_path)

    endpoint_score = similarity_from_gap(current_endpoint - hist_endpoint, 0.25)

    if np.isfinite(current_vol) and np.isfinite(hist_vol):
        vol_score = similarity_from_gap(current_vol - hist_vol, max(0.10, current_vol, hist_vol))
    else:
        vol_score = 0.0

    dd_score = similarity_from_gap(current_dd - hist_dd, 0.25)

    score = 0.58 * float(rho) + 0.18 * endpoint_score + 0.12 * vol_score + 0.12 * dd_score

    if regime_score is not None and np.isfinite(regime_score):
        score = 0.88 * score + 0.12 * float(regime_score)

    return {
        "Score": float(score),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
    }


# =========================
# REGIME FEATURES
# =========================

def value_on_or_before(series: pd.Series, date_value: pd.Timestamp):
    s = series.loc[:pd.Timestamp(date_value)].dropna()
    if s.empty:
        return None
    value = float(s.iloc[-1])
    return value if np.isfinite(value) else None


def window_on_or_before(series: pd.Series, date_value: pd.Timestamp, n: int) -> pd.Series:
    return series.loc[:pd.Timestamp(date_value)].dropna().tail(n)


def bucket_trend(pct_change: float, up_threshold: float = 0.02, down_threshold: float = -0.02) -> str:
    if not np.isfinite(pct_change):
        return "unknown"
    if pct_change >= up_threshold:
        return "rising"
    if pct_change <= down_threshold:
        return "falling"
    return "flat"


def bucket_yield_change(change_in_tnx_points: float) -> str:
    """
    Yahoo ^TNX is quoted as yield x 10. A 2.5 point move is roughly 25 bps.
    """
    if not np.isfinite(change_in_tnx_points):
        return "unknown"
    if change_in_tnx_points >= 2.5:
        return "rising"
    if change_in_tnx_points <= -2.5:
        return "falling"
    return "flat"


def bucket_vix(value) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < 15:
        return "calm"
    if value < 25:
        return "normal"
    return "stressed"


def regime_features_for_date(regime_data: dict, date_value: pd.Timestamp) -> dict:
    features = {}

    asset = regime_data.get("asset")
    if asset is not None:
        asset_window = window_on_or_before(asset, date_value, 220)
        if len(asset_window) >= 200:
            ma200 = float(asset_window.tail(200).mean())
            px = float(asset_window.iloc[-1])
            features["Asset vs 200D"] = "above" if px >= ma200 else "below"

    vix = regime_data.get("vix")
    if vix is not None:
        features["VIX"] = bucket_vix(value_on_or_before(vix, date_value))

    tnx = regime_data.get("tnx")
    if tnx is not None:
        tnx_window = window_on_or_before(tnx, date_value, 64)
        if len(tnx_window) >= 64:
            features["10Y Yield 63D"] = bucket_yield_change(float(tnx_window.iloc[-1] - tnx_window.iloc[0]))

    dxy = regime_data.get("dxy")
    if dxy is not None:
        dxy_window = window_on_or_before(dxy, date_value, 64)
        if len(dxy_window) >= 64 and float(dxy_window.iloc[0]) != 0:
            features["Dollar 63D"] = bucket_trend(float(dxy_window.iloc[-1] / dxy_window.iloc[0] - 1.0))

    credit = regime_data.get("credit")
    if credit is not None:
        credit_window = window_on_or_before(credit, date_value, 64)
        if len(credit_window) >= 64 and float(credit_window.iloc[0]) != 0:
            raw_trend = float(credit_window.iloc[-1] / credit_window.iloc[0] - 1.0)
            trend = bucket_trend(raw_trend)
            if trend == "rising":
                trend = "improving"
            elif trend == "falling":
                trend = "worsening"
            features["Credit Risk 63D"] = trend

    return features


def compare_regimes(current_features: dict, hist_features: dict):
    keys = sorted(set(current_features).intersection(hist_features))
    valid_keys = [
        k for k in keys
        if current_features.get(k) not in (None, "unknown") and hist_features.get(k) not in (None, "unknown")
    ]

    if not valid_keys:
        return np.nan, "N/A"

    matches = [k for k in valid_keys if current_features.get(k) == hist_features.get(k)]
    score = len(matches) / len(valid_keys)

    detail = ", ".join([f"{k}: {hist_features[k]}" for k in valid_keys])
    return float(score), detail


def current_regime_text(features: dict) -> str:
    if not features:
        return "No regime features available."
    return "; ".join([f"{k}: {v}" for k, v in features.items()])


# =========================
# TABLE HELPERS
# =========================

def make_forward_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("Next 21D", "Next 21D"),
        ("Next 63D", "Next 63D"),
        ("Next 126D", "Next 126D"),
        ("Next 252D", "Next 252D"),
    ]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "Horizon": label,
                "Median": vals.median(),
                "Mean": vals.mean(),
                "Hit Rate": (vals > 0).mean(),
                "10th %ile": vals.quantile(0.10),
                "25th %ile": vals.quantile(0.25),
                "75th %ile": vals.quantile(0.75),
                "90th %ile": vals.quantile(0.90),
                "Worst": vals.min(),
                "Best": vals.max(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    for col in out.columns:
        if col != "Horizon":
            out[col] = out[col].map(lambda x: fmt_pct(x))
    return out


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

    cluster_counts = []
    for _, row in selected.iterrows():
        loc = int(row["_signal_loc"])
        count = int((df["_signal_loc"].sub(loc).abs() < gap_days).sum())
        cluster_counts.append(count)

    selected["Cluster Windows"] = cluster_counts
    return selected.sort_values(["Score", "Correlation"], ascending=[False, False]).reset_index(drop=True)


# =========================
# MATPLOTLIB CHARTS
# =========================

def plot_ytd_analogs(ticker_label, current_year, current, ytd_df, analog_df, n_days):
    palette = get_palette(len(analog_df))
    fig, ax = plt.subplots(figsize=(14.5, 7.5))
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
            lw=2.2,
            alpha=0.95,
            color=palette[idx],
            label=f"{yr} | score {row['Score']:.2f} | ρ {row['Correlation']:.2f}",
        )

    ax.plot(
        current.index,
        current.values,
        color="black",
        lw=3.4,
        label=f"{current_year} YTD",
        zorder=10,
    )

    ax.axvline(n_days, color="#6b7280", ls=":", lw=1.3, alpha=0.85)
    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)

    xmax = max(len(ytd_df[c].dropna()) for c in ytd_df.columns)
    ax.set_xlim(1, xmax)

    ax.set_title(
        f"{ticker_label} | Current YTD Path vs Composite-Ranked Historical Analogs",
        fontsize=15,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Trading Day of Year", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)

    set_percent_axis(ax, np.hstack(all_values))
    clean_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_setup_paths(ticker_label, current_path, close_px, selected_df, trailing_days):
    palette = get_palette(len(selected_df))
    fig, ax = plt.subplots(figsize=(14.5, 7.3))
    fig.patch.set_facecolor(CHART_FACE)

    all_values = [current_path.values]

    for idx, row in selected_df.reset_index(drop=True).iterrows():
        start_loc = int(row["_start_loc"])
        end_loc = int(row["_end_loc"])
        window = close_px.iloc[start_loc:end_loc]
        hist_path = normalized_price_path(window)
        if hist_path is None:
            continue

        all_values.append(hist_path.values)

        ax.plot(
            hist_path.index,
            hist_path.values,
            "--",
            lw=2.0,
            alpha=0.92,
            color=palette[idx],
            label=f"{int(row['Year'])} setup | score {row['Score']:.2f} | ρ {row['Correlation']:.2f}",
        )

    ax.plot(
        current_path.index,
        current_path.values,
        color="black",
        lw=3.4,
        label="Current trailing setup",
        zorder=10,
    )

    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)

    ax.set_title(
        f"{ticker_label} | Current {trailing_days}D Setup vs Prior Matched Setups",
        fontsize=15,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Trading Days in Setup Window", fontsize=12)
    ax.set_ylabel("Normalized Return", fontsize=12)
    ax.set_xlim(0, trailing_days - 1)

    set_percent_axis(ax, np.hstack(all_values))
    clean_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_forward_paths(ticker_label, close_px, selected_df, forward_matrix, trailing_days):
    palette = get_palette(len(selected_df))
    fig, ax = plt.subplots(figsize=(14.5, 7.6))
    fig.patch.set_facecolor(CHART_FACE)

    x_fwd = np.arange(0, trailing_days + 1)

    median_path = np.nanmedian(forward_matrix, axis=0)
    p25_path = np.nanpercentile(forward_matrix, 25, axis=0)
    p75_path = np.nanpercentile(forward_matrix, 75, axis=0)
    p10_path = np.nanpercentile(forward_matrix, 10, axis=0)
    p90_path = np.nanpercentile(forward_matrix, 90, axis=0)

    ax.fill_between(
        x_fwd,
        p10_path,
        p90_path,
        color="#d5dae3",
        alpha=0.45,
        label="10th to 90th %ile",
    )
    ax.fill_between(
        x_fwd,
        p25_path,
        p75_path,
        color="#aeb7c6",
        alpha=0.45,
        label="25th to 75th %ile",
    )
    ax.plot(
        x_fwd,
        median_path,
        color="black",
        lw=3.2,
        label="Median forward path",
        zorder=10,
    )

    all_values = [median_path, p10_path, p90_path]

    for idx, row in selected_df.reset_index(drop=True).iterrows():
        fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), trailing_days)
        if fwd_path is None:
            continue

        all_values.append(fwd_path.values)

        ax.plot(
            fwd_path.index,
            fwd_path.values,
            "--",
            lw=2.0,
            alpha=0.92,
            color=palette[idx],
            label=f"{int(row['Year'])} forward | score {row['Score']:.2f}",
        )

    ax.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)
    ax.axvline(0, color="#6b7280", ls=":", lw=1.0, alpha=0.85)

    ax.set_title(
        f"{ticker_label} | Forward Path After Similar {trailing_days}D Setups",
        fontsize=15,
        weight="bold",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Trading Days After Signal", fontsize=12)
    ax.set_ylabel("Forward Cumulative Return", fontsize=12)
    ax.set_xlim(0, trailing_days)

    set_percent_axis(ax, np.hstack(all_values))
    clean_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# =========================
# PAGE
# =========================

add_logo()
inject_app_css()
render_hero()

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
**Purpose:** Historical analog explorer comparing the current market path versus prior periods.

**What this tab shows**
- True YTD paths anchored to the prior year-end close.
- Composite-ranked calendar-year analogs using shape, magnitude, volatility, drawdown, and short-term slope.
- Rolling 252-day analog windows with cluster de-duplication.
- Forward return distributions after similar setups.
- Optional regime guardrails using trend, volatility, dollar, yield, and credit proxies.

**Data source**
- Yahoo Finance adjusted close history.
        """,
        unsafe_allow_html=False,
    )

    st.markdown("---")
    st.subheader("Analog Controls")

    top_n = st.slider("Top Analogs Shown", 1, 15, 5)
    min_corr = st.slider("Minimum Calendar-Year ρ", 0.00, 1.00, 0.00, 0.05, format="%.2f")
    rolling_min_corr = st.slider(
        "Minimum Rolling 252D ρ",
        0.00,
        1.00,
        ROLLING_MIN_CORR_DEFAULT,
        0.05,
        format="%.2f",
    )
    rolling_step = st.slider("Rolling Scan Step", 1, 10, ROLLING_STEP_DEFAULT)
    cluster_gap = st.slider("Minimum Spacing Between Rolling Matches", 21, 252, ROLLING_CLUSTER_GAP_DEFAULT, 21)
    max_one_per_year = st.checkbox("Keep max one rolling match per year", value=True)

    st.markdown("---")
    st.subheader("Filters")

    f_outliers = st.checkbox("Exclude analogs with extreme YTD returns", value=False)
    if f_outliers:
        lo, hi = st.slider("Allowed YTD Return Range (%)", -100, 1000, (-95, 300), 1)
    else:
        lo, hi = -100, 1000

    f_jumps = st.checkbox("Exclude analogs with large daily jumps", value=False)
    if f_jumps:
        max_jump = st.slider("Max Single-Day Move (%)", 5, 100, 25, 1)
    else:
        max_jump = 100

    use_regime_filter = st.checkbox("Apply optional regime guardrails", value=False)
    if use_regime_filter:
        min_regime_score = st.slider("Minimum Regime Match Score", 0.00, 1.00, 0.50, 0.10, format="%.2f")
    else:
        min_regime_score = 0.0


top_bar = st.container()
with top_bar:
    top_left, top_right = st.columns([1.2, 0.9], gap="large")
    with top_left:
        ticker_in = st.text_input("Ticker", "^SPX").strip().upper()
    metric_placeholder = top_right.empty()

ticker = TICKER_ALIASES.get(ticker_in, ticker_in)
ticker_label = ticker_in if ticker_in else ticker

ny_now = dt.datetime.now(NY_TZ)
this_year = ny_now.year

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
stale_bdays = business_days_behind(last_data_date, ny_now.date())

if stale_bdays > 1:
    st.warning(
        f"Latest close for {ticker_label} is {last_data_date:%Y-%m-%d}. "
        f"That is {stale_bdays} business days behind New York time."
    )

regime_data = {"asset": close_px}
missing_regime_symbols = []
current_regime_features = {}

if use_regime_filter:
    regime_data, missing_regime_symbols = load_optional_regime_data(close_px)
    current_regime_features = regime_features_for_date(regime_data, last_data_date)

    if missing_regime_symbols:
        st.caption(f"Some optional regime proxies failed to load: {', '.join(missing_regime_symbols)}")
    st.caption(f"Current regime guardrails: {current_regime_text(current_regime_features)}")

paths, date_maps = build_true_ytd_paths(raw, this_year)

if not paths:
    st.error("No usable yearly data found.")
    st.stop()

ytd_df = pd.DataFrame(paths)

if this_year not in ytd_df.columns:
    st.warning(f"No YTD data for {this_year}. Verify the ticker or the latest Yahoo close.")
    st.stop()

current = ytd_df[this_year].dropna()
n_days = len(current)

if n_days < MIN_DAYS_FOR_CORR:
    st.info(
        f"{this_year} has only {n_days} trading days so far. "
        f"Correlation-based analogs are unstable before {MIN_DAYS_FOR_CORR} trading days."
    )
    fig_early, ax_early = plt.subplots(figsize=(14.5, 7.2))
    fig_early.patch.set_facecolor(CHART_FACE)
    ax_early.plot(current.index, current.values, color="black", lw=3.4, label=f"{this_year} YTD")
    ax_early.axhline(0, color="#6b7280", ls="--", lw=1.0, alpha=0.85)
    ax_early.set_title(f"{ticker_label} | {this_year} YTD", fontsize=15, weight="bold", loc="left")
    ax_early.set_xlabel("Trading Day of Year")
    ax_early.set_ylabel("Cumulative Return")
    set_percent_axis(ax_early, current.values)
    clean_axes(ax_early)
    ax_early.legend(frameon=False)
    fig_early.tight_layout()
    st.pyplot(fig_early, clear_figure=True)
    plt.close(fig_early)
    st.caption("© 2026 AD Fund Management LP")
    st.stop()


# =========================
# CALENDAR-YEAR ANALOGS
# =========================

records = []

for yr in ytd_df.columns:
    yr = int(yr)
    if yr == this_year:
        continue

    ser = ytd_df[yr].dropna()
    if len(ser) < n_days:
        continue

    hist_slice = ser.iloc[:n_days]
    rho = safe_corr(current.values, hist_slice.values)
    if not np.isfinite(rho) or rho < min_corr:
        continue

    ret_n = float(hist_slice.iloc[-1])
    daily_ret = daily_returns_from_cum_path(hist_slice)
    max_d = float(daily_ret.abs().max()) if not daily_ret.empty else np.nan

    if f_outliers and not (lo / 100.0 <= ret_n <= hi / 100.0):
        continue
    if f_jumps and np.isfinite(max_d) and max_d > max_jump / 100.0:
        continue

    hist_date = pd.Timestamp(date_maps[yr].loc[n_days])
    regime_score = np.nan
    regime_detail = "N/A"

    if use_regime_filter:
        hist_regime_features = regime_features_for_date(regime_data, hist_date)
        regime_score, regime_detail = compare_regimes(current_regime_features, hist_regime_features)
        if np.isfinite(regime_score) and regime_score < min_regime_score:
            continue

    components = ytd_composite_score(
        current_path=current,
        hist_path=hist_slice,
        rho=rho,
        regime_score=regime_score if use_regime_filter else None,
    )

    full_year_return = float(ser.iloc[-1])
    forward_after_match = float((1.0 + full_year_return) / (1.0 + ret_n) - 1.0)

    records.append(
        {
            "Year": yr,
            "Match Date": hist_date,
            "Correlation": float(rho),
            "Score": components["Score"],
            "YTD at Match": ret_n,
            "Full-Year Return": full_year_return,
            "Return After Match Date": forward_after_match,
            "Realized Vol": annualized_vol_from_path(hist_slice),
            "Max Drawdown": max_drawdown_from_path(hist_slice),
            "Regime Score": regime_score,
            "Regime Detail": regime_detail,
            **components,
        }
    )

if not records:
    st.warning("No historical years meet the current filters.")
    st.stop()

calendar_df = pd.DataFrame(records).sort_values(["Score", "Correlation"], ascending=[False, False]).reset_index(drop=True)
top_calendar = calendar_df.head(top_n).copy()

median_after_match = float(np.nanmedian(top_calendar["Return After Match Date"])) if len(top_calendar) else np.nan

with metric_placeholder.container():
    render_inline_metric(
        label="Median Return After Match Date",
        value=fmt_pct(median_after_match),
    )

fig_ytd = plot_ytd_analogs(
    ticker_label=ticker_label,
    current_year=this_year,
    current=current,
    ytd_df=ytd_df,
    analog_df=top_calendar,
    n_days=n_days,
)
st.pyplot(fig_ytd, clear_figure=True)
plt.close(fig_ytd)

table_calendar = top_calendar.copy()
table_calendar["Match Date"] = pd.to_datetime(table_calendar["Match Date"]).dt.strftime("%Y-%m-%d")
for col in [
    "YTD at Match",
    "Full-Year Return",
    "Return After Match Date",
    "Realized Vol",
    "Max Drawdown",
]:
    table_calendar[col] = table_calendar[col].map(lambda x: fmt_pct(x))
for col in ["Correlation", "Score", "Endpoint Score", "Vol Score", "Drawdown Score", "Slope Score", "Regime Score"]:
    if col in table_calendar.columns:
        table_calendar[col] = table_calendar[col].map(lambda x: fmt_num(x))

display_cols = [
    "Year",
    "Match Date",
    "Score",
    "Correlation",
    "YTD at Match",
    "Full-Year Return",
    "Return After Match Date",
    "Realized Vol",
    "Max Drawdown",
]
if use_regime_filter:
    display_cols += ["Regime Score", "Regime Detail"]

st.markdown("**Composite-ranked calendar-year analogs**")
st.dataframe(table_calendar[display_cols], use_container_width=True, hide_index=True)


# =========================
# ROLLING TRAILING WINDOW ANALOGS
# =========================

st.markdown("<hr style='margin-top:18px; margin-bottom:12px;'>", unsafe_allow_html=True)
st.subheader(f"Rolling {TRAILING_DAYS}-Day Forward Analog Signal")

if len(close_px) < TRAILING_DAYS * 3:
    st.info(
        f"Need at least {TRAILING_DAYS * 3} trading days of history to run a clean "
        f"{TRAILING_DAYS}-day match plus a full {TRAILING_DAYS}-day forward window."
    )
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

current_start_loc = len(close_px) - TRAILING_DAYS
current_trailing = close_px.iloc[-TRAILING_DAYS:].copy()
current_trailing_path = normalized_price_path(current_trailing)

if current_trailing_path is None:
    st.warning("Current trailing window is not usable for the rolling analog signal.")
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

rolling_matches = []

for start_loc in range(0, len(close_px) - TRAILING_DAYS + 1, int(rolling_step)):
    end_loc = start_loc + TRAILING_DAYS
    signal_loc = end_loc - 1

    if end_loc + TRAILING_DAYS > current_start_loc:
        continue

    hist_window = close_px.iloc[start_loc:end_loc].copy()
    if len(hist_window) != TRAILING_DAYS or hist_window.isna().any():
        continue

    hist_path = normalized_price_path(hist_window)
    if hist_path is None or len(hist_path) != len(current_trailing_path):
        continue

    rho = safe_corr(current_trailing_path.values, hist_path.values)
    if not np.isfinite(rho) or rho < rolling_min_corr:
        continue

    signal_date = pd.Timestamp(hist_window.index[-1])

    regime_score = np.nan
    regime_detail = "N/A"

    if use_regime_filter:
        hist_regime_features = regime_features_for_date(regime_data, signal_date)
        regime_score, regime_detail = compare_regimes(current_regime_features, hist_regime_features)
        if np.isfinite(regime_score) and regime_score < min_regime_score:
            continue

    fwd_path = forward_path_from_signal(close_px, signal_loc, TRAILING_DAYS)
    if fwd_path is None:
        continue

    components = rolling_composite_score(
        current_path=current_trailing_path,
        hist_path=hist_path,
        rho=rho,
        regime_score=regime_score if use_regime_filter else None,
    )

    rolling_matches.append(
        {
            "Year": int(signal_date.year),
            "Match Start": pd.Timestamp(hist_window.index[0]),
            "Match End": pd.Timestamp(hist_window.index[-1]),
            "Signal Date": signal_date,
            "Correlation": float(rho),
            "Score": components["Score"],
            "Next 21D": float(fwd_path.iloc[min(21, len(fwd_path) - 1)]),
            "Next 63D": float(fwd_path.iloc[min(63, len(fwd_path) - 1)]),
            "Next 126D": float(fwd_path.iloc[min(126, len(fwd_path) - 1)]),
            "Next 252D": float(fwd_path.iloc[-1]),
            "Max DD Next 252D": max_drawdown_from_path(fwd_path),
            "Regime Score": regime_score,
            "Regime Detail": regime_detail,
            "_start_loc": int(start_loc),
            "_end_loc": int(end_loc),
            "_signal_loc": int(signal_loc),
            **components,
        }
    )

if not rolling_matches:
    st.warning(f"No rolling {TRAILING_DAYS}-day windows found with correlation above {rolling_min_corr:.2f}.")
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

rolling_df = pd.DataFrame(rolling_matches)
selected_rolling = select_clustered_matches(
    df=rolling_df,
    gap_days=int(cluster_gap),
    max_one_per_year=bool(max_one_per_year),
)

if selected_rolling.empty:
    st.warning("Rolling matches were found, but clustering removed all candidates. Lower the spacing filter.")
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

overlay_rolling = selected_rolling.head(top_n).copy()

forward_paths = []
for _, row in selected_rolling.iterrows():
    fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), TRAILING_DAYS)
    if fwd_path is not None and len(fwd_path) == TRAILING_DAYS + 1:
        forward_paths.append(fwd_path.values)

if not forward_paths:
    st.warning("No forward paths were available after rolling match filtering.")
    st.caption("© 2026 AD Fund Management LP")
    st.stop()

forward_matrix = np.vstack(forward_paths)

fig_forward = plot_forward_paths(
    ticker_label=ticker_label,
    close_px=close_px,
    selected_df=overlay_rolling,
    forward_matrix=forward_matrix,
    trailing_days=TRAILING_DAYS,
)
st.pyplot(fig_forward, clear_figure=True)
plt.close(fig_forward)

table_rolling = selected_rolling.copy()
for col in ["Match Start", "Match End", "Signal Date"]:
    table_rolling[col] = pd.to_datetime(table_rolling[col]).dt.strftime("%Y-%m-%d")

for col in [
    "Next 21D",
    "Next 63D",
    "Next 126D",
    "Next 252D",
    "Max DD Next 252D",
]:
    table_rolling[col] = table_rolling[col].map(lambda x: fmt_pct(x))

for col in ["Correlation", "Score", "Endpoint Score", "Vol Score", "Drawdown Score", "Regime Score"]:
    if col in table_rolling.columns:
        table_rolling[col] = table_rolling[col].map(lambda x: fmt_num(x))

rolling_cols = [
    "Year",
    "Match Start",
    "Match End",
    "Signal Date",
    "Score",
    "Correlation",
    "Cluster Windows",
    "Next 21D",
    "Next 63D",
    "Next 126D",
    "Next 252D",
    "Max DD Next 252D",
]
if use_regime_filter:
    rolling_cols += ["Regime Score", "Regime Detail"]

st.markdown(
    f"**Clustered rolling {TRAILING_DAYS}-day matches with correlation >= {rolling_min_corr:.2f}**"
)
st.dataframe(table_rolling[rolling_cols], use_container_width=True, hide_index=True)

st.caption("© 2026 AD Fund Management LP")
