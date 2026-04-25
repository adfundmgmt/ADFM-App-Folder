import datetime as dt
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf


# =========================
# CONFIG
# =========================

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

PLOT_TEMPLATE = "plotly_white"

st.set_page_config(page_title="Market Memory Explorer", layout="wide")


# =========================
# SMALL UI HELPERS
# =========================

def fmt_pct(x: float | int | None, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        if not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x):.{digits}%}"
    except Exception:
        return "N/A"


def fmt_num(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        if not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
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


def load_optional_regime_data(main_symbol: str, raw_close: pd.Series) -> tuple[dict[str, pd.Series], list[str]]:
    """
    Regime data is optional. The app should still work if one proxy fails.
    """
    data: dict[str, pd.Series] = {"asset": raw_close.dropna().copy()}
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


def build_true_ytd_paths(raw: pd.DataFrame, current_year: int) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """
    True YTD return path uses the prior year's final close as the base.
    This preserves the first trading day's gap instead of forcing day one to 0%.
    """
    close = raw["Close"].dropna().copy()
    paths: dict[int, pd.Series] = {}
    date_maps: dict[int, pd.Series] = {}

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


def normalized_price_path(series: pd.Series) -> pd.Series | None:
    series = series.dropna().astype(float)
    if len(series) < 2:
        return None
    first = float(series.iloc[0])
    if not np.isfinite(first) or first <= 0:
        return None
    out = series / first - 1.0
    out.index = np.arange(0, len(out))
    return out


def forward_path_from_signal(series: pd.Series, signal_loc: int, horizon: int) -> pd.Series | None:
    end_loc = signal_loc + horizon
    if signal_loc < 0 or end_loc >= len(series):
        return None

    window = series.iloc[signal_loc : end_loc + 1].copy()
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


def ytd_composite_score(
    current_path: pd.Series,
    hist_path: pd.Series,
    rho: float,
    regime_score: float | None = None,
) -> dict[str, float]:
    hist_path = hist_path.iloc[: len(current_path)].dropna().astype(float)
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

    regime_component = 0.50 if regime_score is None or not np.isfinite(regime_score) else float(regime_score)

    score = (
        0.50 * float(rho)
        + 0.20 * endpoint_score
        + 0.12 * vol_score
        + 0.10 * dd_score
        + 0.08 * slope_score
    )

    if regime_score is not None and np.isfinite(regime_score):
        score = 0.88 * score + 0.12 * regime_component

    return {
        "Score": float(score),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
        "Slope Score": float(slope_score),
    }


def rolling_composite_score(
    current_path: pd.Series,
    hist_path: pd.Series,
    rho: float,
    regime_score: float | None = None,
) -> dict[str, float]:
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

    regime_component = 0.50 if regime_score is None or not np.isfinite(regime_score) else float(regime_score)

    score = 0.58 * float(rho) + 0.18 * endpoint_score + 0.12 * vol_score + 0.12 * dd_score

    if regime_score is not None and np.isfinite(regime_score):
        score = 0.88 * score + 0.12 * regime_component

    return {
        "Score": float(score),
        "Endpoint Score": float(endpoint_score),
        "Vol Score": float(vol_score),
        "Drawdown Score": float(dd_score),
    }


# =========================
# REGIME FEATURES
# =========================

def value_on_or_before(series: pd.Series, date_value: pd.Timestamp) -> float | None:
    s = series.loc[: pd.Timestamp(date_value)].dropna()
    if s.empty:
        return None
    value = float(s.iloc[-1])
    return value if np.isfinite(value) else None


def window_on_or_before(series: pd.Series, date_value: pd.Timestamp, n: int) -> pd.Series:
    return series.loc[: pd.Timestamp(date_value)].dropna().tail(n)


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


def bucket_vix(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < 15:
        return "calm"
    if value < 25:
        return "normal"
    return "stressed"


def regime_features_for_date(regime_data: dict[str, pd.Series], date_value: pd.Timestamp) -> dict[str, str]:
    features: dict[str, str] = {}

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


def compare_regimes(current_features: dict[str, str], hist_features: dict[str, str]) -> tuple[float, str]:
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


def current_regime_text(features: dict[str, str]) -> str:
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
# CHART HELPERS
# =========================

def plot_ytd_analogs(
    ticker_label: str,
    current_year: int,
    current: pd.Series,
    ytd_df: pd.DataFrame,
    analog_df: pd.DataFrame,
    n_days: int,
) -> go.Figure:
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Plotly
    fig = go.Figure()

    for idx, row in analog_df.reset_index(drop=True).iterrows():
        yr = int(row["Year"])
        ser = ytd_df[yr].dropna()
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=ser.index,
                y=ser.values,
                mode="lines",
                name=f"{yr} | score {row['Score']:.2f} | ρ {row['Correlation']:.2f}",
                line=dict(width=2.0, dash="dash", color=color),
                opacity=0.95,
                hovertemplate=(
                    f"{yr}<br>"
                    "Trading Day=%{x}<br>"
                    "Cum Return=%{y:.2%}<br>"
                    f"Score={row['Score']:.3f}<br>"
                    f"ρ={row['Correlation']:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=current.index,
            y=current.values,
            mode="lines",
            name=f"{current_year} YTD",
            line=dict(width=3.6, color="black"),
            hovertemplate=(
                f"{current_year}<br>"
                "Trading Day=%{x}<br>"
                "Cum Return=%{y:.2%}<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=n_days, line_width=1.2, line_dash="dot", line_color="gray")
    fig.add_hline(y=0, line_width=1.0, line_dash="dash", line_color="gray")

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=650,
        title=f"{ticker_label} | Current YTD Path vs Composite-Ranked Historical Analogs",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=130),
        hovermode="x unified",
    )

    return fig


def plot_setup_paths(
    ticker_label: str,
    current_path: pd.Series,
    close_px: pd.Series,
    selected_df: pd.DataFrame,
    trailing_days: int,
) -> go.Figure:
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Plotly
    fig = go.Figure()

    for idx, row in selected_df.reset_index(drop=True).iterrows():
        start_loc = int(row["_start_loc"])
        end_loc = int(row["_end_loc"])
        window = close_px.iloc[start_loc:end_loc]
        hist_path = normalized_price_path(window)
        if hist_path is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=hist_path.index,
                y=hist_path.values,
                mode="lines",
                name=f"{int(row['Year'])} setup | score {row['Score']:.2f}",
                line=dict(width=1.9, dash="dash", color=colors[idx % len(colors)]),
                opacity=0.90,
                hovertemplate=(
                    f"{int(row['Year'])} setup<br>"
                    "Day=%{x}<br>"
                    "Return=%{y:.2%}<br>"
                    f"Signal={pd.Timestamp(row['Signal Date']).strftime('%Y-%m-%d')}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=current_path.index,
            y=current_path.values,
            mode="lines",
            name="Current trailing setup",
            line=dict(width=3.6, color="black"),
            hovertemplate="Current setup<br>Day=%{x}<br>Return=%{y:.2%}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1.0, line_dash="dash", line_color="gray")

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=590,
        title=f"{ticker_label} | Current {trailing_days}D Setup vs Prior Matched Setups",
        xaxis_title="Trading Days in Setup Window",
        yaxis_title="Normalized Return",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=130),
        hovermode="x unified",
    )

    return fig


def plot_forward_paths(
    ticker_label: str,
    close_px: pd.Series,
    selected_df: pd.DataFrame,
    forward_matrix: np.ndarray,
    trailing_days: int,
) -> go.Figure:
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Plotly
    x_fwd = np.arange(0, trailing_days + 1)

    median_path = np.nanmedian(forward_matrix, axis=0)
    p25_path = np.nanpercentile(forward_matrix, 25, axis=0)
    p75_path = np.nanpercentile(forward_matrix, 75, axis=0)
    p10_path = np.nanpercentile(forward_matrix, 10, axis=0)
    p90_path = np.nanpercentile(forward_matrix, 90, axis=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_fwd,
            y=p90_path,
            mode="lines",
            name="10th to 90th %ile",
            line=dict(width=0, color="rgba(150,150,150,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fwd,
            y=p10_path,
            mode="lines",
            name="10th to 90th %ile",
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.22)",
            line=dict(width=0, color="rgba(150,150,150,0.25)"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_fwd,
            y=p75_path,
            mode="lines",
            name="25th to 75th %ile",
            line=dict(width=0, color="rgba(120,120,120,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fwd,
            y=p25_path,
            mode="lines",
            name="25th to 75th %ile",
            fill="tonexty",
            fillcolor="rgba(120,120,120,0.28)",
            line=dict(width=0, color="rgba(120,120,120,0.25)"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_fwd,
            y=median_path,
            mode="lines",
            name="Median forward path",
            line=dict(width=3.3, color="black"),
            hovertemplate="Median<br>Day=%{x}<br>Return=%{y:.2%}<extra></extra>",
        )
    )

    for idx, row in selected_df.reset_index(drop=True).iterrows():
        fwd_path = forward_path_from_signal(close_px, int(row["_signal_loc"]), trailing_days)
        if fwd_path is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=fwd_path.index,
                y=fwd_path.values,
                mode="lines",
                name=f"{int(row['Year'])} forward | score {row['Score']:.2f}",
                line=dict(width=1.9, dash="dash", color=colors[idx % len(colors)]),
                opacity=0.90,
                hovertemplate=(
                    f"{int(row['Year'])} forward<br>"
                    "Day=%{x}<br>"
                    "Return=%{y:.2%}<br>"
                    f"Signal={pd.Timestamp(row['Signal Date']).strftime('%Y-%m-%d')}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_width=1.0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_width=1.0, line_dash="dot", line_color="gray")

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=650,
        title=f"{ticker_label} | Forward Path After Similar {trailing_days}D Setups",
        xaxis_title="Trading Days After Signal",
        yaxis_title="Forward Cumulative Return",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=140),
        hovermode="x unified",
    )

    return fig


# =========================
# PAGE
# =========================

add_logo()

st.title("Market Memory Explorer")
st.subheader("Compare the current market path with history, then measure what happened next")

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

    top_n = st.slider("Top Analogs Shown", 1, 15, 8)
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


ticker_col, note_col = st.columns([1, 2])
ticker_in = ticker_col.text_input("Ticker", "^SPX").strip().upper()
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

with note_col:
    if stale_bdays > 1:
        st.warning(
            f"Latest close for {ticker_label} is {last_data_date:%Y-%m-%d}. "
            f"That is {stale_bdays} business days behind New York time."
        )
    else:
        st.caption(f"Latest close: {last_data_date:%Y-%m-%d} | New York date: {ny_now:%Y-%m-%d}")

regime_data: dict[str, pd.Series] = {"asset": close_px}
missing_regime_symbols: list[str] = []
current_regime_features: dict[str, str] = {}

if use_regime_filter:
    regime_data, missing_regime_symbols = load_optional_regime_data(ticker, close_px)
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
    fig_early = go.Figure()
    fig_early.add_trace(
        go.Scatter(
            x=current.index,
            y=current.values,
            mode="lines",
            name=f"{this_year} YTD",
            line=dict(width=3.5, color="black"),
            hovertemplate="Trading Day=%{x}<br>Cum Return=%{y:.2%}<extra></extra>",
        )
    )
    fig_early.add_hline(y=0, line_width=1.0, line_dash="dash", line_color="gray")
    fig_early.update_layout(
        template=PLOT_TEMPLATE,
        height=580,
        title=f"{ticker_label} | {this_year} YTD",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        margin=dict(l=40, r=30, t=70, b=70),
    )
    st.plotly_chart(fig_early, use_container_width=True)
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

current_ret = float(current.iloc[-1])
median_final = float(np.nanmedian(top_calendar["Full-Year Return"])) if len(top_calendar) else np.nan
median_after_match = float(np.nanmedian(top_calendar["Return After Match Date"])) if len(top_calendar) else np.nan

m1, m2, m3 = st.columns(3)
m1.metric(f"{this_year} True YTD", fmt_pct(current_ret))
m2.metric("Median Full-Year Return of Shown Analogs", fmt_pct(median_final))
m3.metric("Median Return After Match Date", fmt_pct(median_after_match))

st.plotly_chart(
    plot_ytd_analogs(
        ticker_label=ticker_label,
        current_year=this_year,
        current=current,
        ytd_df=ytd_df,
        analog_df=top_calendar,
        n_days=n_days,
    ),
    use_container_width=True,
)

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
st.subheader(f"Rolling {TRAILING_DAYS}-Day Setup Match and Forward Signal")

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
current_signal_date = pd.Timestamp(close_px.index[-1])

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

s1, s2, s3 = st.columns(3)
s1.metric("Clustered Matches", f"{len(selected_rolling)}")
s2.metric("Median Next 252D", fmt_pct(float(np.nanmedian(selected_rolling["Next 252D"]))))
s3.metric("Positive 252D Hit Rate", fmt_pct(float(np.mean(selected_rolling["Next 252D"] > 0))))

st.plotly_chart(
    plot_setup_paths(
        ticker_label=ticker_label,
        current_path=current_trailing_path,
        close_px=close_px,
        selected_df=overlay_rolling,
        trailing_days=TRAILING_DAYS,
    ),
    use_container_width=True,
)

st.plotly_chart(
    plot_forward_paths(
        ticker_label=ticker_label,
        close_px=close_px,
        selected_df=overlay_rolling,
        forward_matrix=forward_matrix,
        trailing_days=TRAILING_DAYS,
    ),
    use_container_width=True,
)

dist_table = make_forward_distribution_table(selected_rolling)
if not dist_table.empty:
    st.markdown("**Forward return distribution after clustered rolling matches**")
    st.dataframe(dist_table, use_container_width=True, hide_index=True)

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

d1, d2, d3 = st.columns(3)
d1.metric("Median Max Drawdown", fmt_pct(float(np.nanmedian(selected_rolling["Max DD Next 252D"]))))
d2.metric("Median Next 63D", fmt_pct(float(np.nanmedian(selected_rolling["Next 63D"]))))
d3.metric("Median Next 126D", fmt_pct(float(np.nanmedian(selected_rolling["Next 126D"]))))

st.caption(
    f"Signal logic: compare the current trailing {TRAILING_DAYS}-day normalized price path with prior rolling "
    f"{TRAILING_DAYS}-day historical windows, score matches by correlation plus magnitude, volatility, drawdown, "
    f"and optional regime similarity, then plot the next {TRAILING_DAYS} trading days after each clustered signal."
)

st.caption("© 2026 AD Fund Management LP")
