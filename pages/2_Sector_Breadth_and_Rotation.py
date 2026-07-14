# sp_subsector_rotation_monitor_v12.py
# ADFM | S&P 500 Sector + Subsector Breadth & Rotation Monitor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Sector Breadth and Rotation",
    layout="wide",
)

st.title("Sector Breadth and Rotation")


# =============================================================================
# Constants
from adfm_sector_rotation_config import (
    BENCHMARKS,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_RETRIES,
    FORWARD_FILL_LIMIT,
    INTERVAL,
    LABEL_MODES,
    LOOKBACK_PERIOD,
    MAJOR_SECTORS,
    MAX_STALE_SESSIONS,
    MIN_SCORE_COMPONENTS,
    NEUTRAL_MAP_THRESHOLD,
    QUADRANT_LABELS,
    ROTATION_MODES,
    SECTOR_GROUP_COLORS,
    STATE_COLORS,
    SUBSECTOR_ROWS,
    TRAIL_OPTIONS,
    UNIVERSE_SCOPES,
    WINDOW_PRESETS,
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RotationConfig:
    short_window: int
    long_window: int
    short_label: str
    long_label: str


# =============================================================================
# Universe helpers
# =============================================================================

def major_sector_rows() -> List[Dict[str, str]]:
    return [
        {
            "Ticker": ticker,
            "Name": name,
            "Sector Group": name,
            "Tier": "Major Sector",
        }
        for ticker, name in MAJOR_SECTORS.items()
    ]


def validate_universe_definitions() -> List[str]:
    rows = major_sector_rows() + SUBSECTOR_ROWS
    df = pd.DataFrame(rows)
    errors: List[str] = []

    duplicate_tickers = sorted(
        df.loc[df["Ticker"].duplicated(keep=False), "Ticker"].unique().tolist()
    )
    if duplicate_tickers:
        errors.append(f"Duplicate ticker definitions: {', '.join(duplicate_tickers)}")

    unknown_groups = sorted(
        set(df["Sector Group"].dropna()) - set(SECTOR_GROUP_COLORS.keys())
    )
    if unknown_groups:
        errors.append(f"Sector groups without colors: {', '.join(unknown_groups)}")

    valid_tiers = {"Major Sector", "Core", "Thematic"}
    invalid_tiers = sorted(set(df["Tier"].dropna()) - valid_tiers)
    if invalid_tiers:
        errors.append(f"Invalid tier values: {', '.join(invalid_tiers)}")

    required_columns = {"Ticker", "Name", "Sector Group", "Tier"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        errors.append(f"Missing universe fields: {', '.join(sorted(missing_columns))}")

    return errors


def build_universe(scope: str) -> pd.DataFrame:
    major = major_sector_rows()
    core_subsectors = [row for row in SUBSECTOR_ROWS if row["Tier"] == "Core"]
    all_subsectors = SUBSECTOR_ROWS.copy()

    if scope == "Major sectors only":
        rows = major
    elif scope == "Core subsectors":
        rows = core_subsectors
    elif scope == "Core + thematic subsectors":
        rows = all_subsectors
    else:
        rows = core_subsectors

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    df["Sort Group"] = df["Sector Group"].map(
        {group: i for i, group in enumerate(SECTOR_GROUP_COLORS.keys())}
    )
    df["Sort Tier"] = df["Tier"].map(
        {"Major Sector": 0, "Core": 1, "Thematic": 2}
    ).fillna(9)
    df = df.sort_values(["Sort Group", "Sort Tier", "Ticker"]).drop(
        columns=["Sort Group", "Sort Tier"]
    )
    return df.reset_index(drop=True)


def filter_universe_by_groups(
    universe_df: pd.DataFrame,
    selected_groups: List[str],
) -> pd.DataFrame:
    if not selected_groups:
        return universe_df.iloc[0:0].copy()

    return universe_df[
        universe_df["Sector Group"].isin(selected_groups)
    ].reset_index(drop=True)


def filter_universe_by_data(
    universe_df: pd.DataFrame,
    raw_prices: pd.DataFrame,
    benchmark_ticker: str,
    min_valid_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = build_price_diagnostics(
        prices=raw_prices,
        tickers=universe_df["Ticker"].tolist(),
        min_valid_rows=min_valid_rows,
        benchmark_ticker=benchmark_ticker,
    )

    eligible = diagnostics.loc[
        diagnostics["Status"] == "OK",
        "Ticker",
    ].tolist()

    filtered = universe_df[
        universe_df["Ticker"].isin(eligible)
    ].reset_index(drop=True)
    return filtered, diagnostics


universe_definition_errors = validate_universe_definitions()
if universe_definition_errors:
    st.error("Universe-definition error: " + " | ".join(universe_definition_errors))
    st.stop()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("Settings")

    universe_scope = st.selectbox(
        "Universe",
        options=UNIVERSE_SCOPES,
        index=1,
        help="Use major sectors for top-down confirmation or subsectors for granular rotation work.",
    )

    base_universe = build_universe(universe_scope)
    group_options = base_universe["Sector Group"].drop_duplicates().tolist()

    selected_groups = st.multiselect(
        "Sector group filter",
        options=group_options,
        default=group_options,
    )

    selected_universe = filter_universe_by_groups(base_universe, selected_groups)

    benchmark = st.selectbox(
        "Benchmark",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{x} | {BENCHMARKS[x]}",
        index=0,
    )

    rotation_mode_label = st.radio(
        "Rotation mode",
        options=list(ROTATION_MODES.keys()),
        index=0,
        help="Benchmark-relative rotation uses sector/subsector performance versus the selected benchmark.",
    )
    rotation_mode = ROTATION_MODES[rotation_mode_label]

    window_choice = st.selectbox(
        "Rotation window",
        options=list(WINDOW_PRESETS.keys()),
        index=0,
    )

    trail_choice = st.selectbox(
        "Rotation trail",
        options=list(TRAIL_OPTIONS.keys()),
        index=1,
    )

    label_mode = st.selectbox(
        "Ticker labels on map",
        options=LABEL_MODES,
        index=0,
    )

    label_top_n = st.slider(
        "Label top N",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
        disabled=label_mode != "Top ranked only",
    )

    show_diagnostics = st.checkbox("Show data diagnostics", value=False)


# =============================================================================
# Helpers
# =============================================================================

def get_rotation_config(choice: str) -> RotationConfig:
    cfg = WINDOW_PRESETS[choice]
    return RotationConfig(
        short_window=cfg["short"],
        long_window=cfg["long"],
        short_label=cfg["label_short"],
        long_label=cfg["label_long"],
    )


def _safe_to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    try:
        if out.index.tz is not None:
            out.index = out.index.tz_convert(None)
    except Exception:
        pass

    return out


def _normalize_download(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Normalize yfinance output into a clean wide dataframe of unadjusted
    adjusted-close values when available, with close as the fallback.
    Handles both MultiIndex and single-index outputs.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    field_candidates = ["Adj Close", "Close"]
    out: Dict[str, pd.Series] = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            selected = None

            for field in field_candidates:
                if (field, ticker) in data.columns:
                    selected = data[(field, ticker)]
                    break

                if (ticker, field) in data.columns:
                    selected = data[(ticker, field)]
                    break

            if selected is not None:
                selected = pd.to_numeric(selected, errors="coerce")
                if selected.notna().any():
                    out[ticker] = selected

        return _safe_to_datetime_index(pd.DataFrame(out))

    if len(tickers) == 1:
        ticker = tickers[0]
        for field in field_candidates:
            if field in data.columns:
                s = pd.to_numeric(data[field], errors="coerce")
                if s.notna().any():
                    return _safe_to_datetime_index(pd.DataFrame({ticker: s}))

    return pd.DataFrame()


def _download_batch(
    tickers: List[str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    for attempt in range(DOWNLOAD_RETRIES):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                group_by="column",
                threads=True,
            )
            normalized = _normalize_download(raw, tickers)
            if not normalized.empty:
                return normalized
        except Exception:
            pass

        if attempt < DOWNLOAD_RETRIES - 1:
            time.sleep(0.75 * (attempt + 1))

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(
    tickers: List[str],
    period: str = LOOKBACK_PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Fetch raw prices without forward-filling. Downloads are chunked and any
    missing symbols receive an individual fallback request. Raw observations
    must remain untouched so staleness and missing-data checks stay valid.
    """
    unique_tickers = list(dict.fromkeys([t for t in tickers if t]))

    if not unique_tickers:
        return pd.DataFrame()

    pieces: List[pd.DataFrame] = []

    for start_idx in range(0, len(unique_tickers), DOWNLOAD_CHUNK_SIZE):
        chunk = unique_tickers[start_idx:start_idx + DOWNLOAD_CHUNK_SIZE]
        downloaded = _download_batch(chunk, period, interval)
        if not downloaded.empty:
            pieces.append(downloaded)

    if pieces:
        prices = pd.concat(pieces, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    else:
        prices = pd.DataFrame()

    missing_tickers = [
        ticker
        for ticker in unique_tickers
        if ticker not in prices.columns or not prices[ticker].notna().any()
    ]

    fallback_pieces: List[pd.DataFrame] = []
    for ticker in missing_tickers:
        downloaded = _download_batch([ticker], period, interval)
        if not downloaded.empty:
            fallback_pieces.append(downloaded)

    if fallback_pieces:
        prices = pd.concat([prices] + fallback_pieces, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    if prices.empty:
        return prices

    prices = _safe_to_datetime_index(prices)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def validate_benchmark(
    raw_prices: pd.DataFrame,
    benchmark_ticker: str,
    min_valid_rows: int,
) -> Tuple[bool, str]:
    if raw_prices.empty:
        return False, "No market data returned."

    if benchmark_ticker not in raw_prices.columns:
        return False, f"Benchmark data unavailable for {benchmark_ticker}."

    s = pd.to_numeric(
        raw_prices[benchmark_ticker],
        errors="coerce",
    ).dropna()

    if s.empty:
        return False, f"Benchmark data unavailable for {benchmark_ticker}."

    if len(s) < min_valid_rows:
        return (
            False,
            f"Benchmark has only {len(s)} valid rows; need at least {min_valid_rows}.",
        )

    return True, ""


def build_price_diagnostics(
    prices: pd.DataFrame,
    tickers: List[str],
    min_valid_rows: int = 100,
    benchmark_ticker: str = "",
) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Status",
        "First Date",
        "Latest Date",
        "Valid Rows",
        "Missing Values",
        "Last Price",
        "Stale Sessions",
        "Stale Days",
    ]

    if prices.empty:
        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Status": "Missing",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": "",
                    "Last Price": np.nan,
                    "Stale Sessions": np.nan,
                    "Stale Days": np.nan,
                }
                for ticker in tickers
            ],
            columns=columns,
        )

    if benchmark_ticker and benchmark_ticker in prices.columns:
        reference_series = pd.to_numeric(
            prices[benchmark_ticker],
            errors="coerce",
        ).dropna()
        reference_index = pd.DatetimeIndex(reference_series.index).sort_values().unique()
    else:
        reference_index = pd.DatetimeIndex(prices.index).sort_values().unique()

    if len(reference_index) == 0:
        reference_index = pd.DatetimeIndex(prices.index).sort_values().unique()

    reference_last = pd.to_datetime(reference_index.max())
    rows = []

    for ticker in tickers:
        if ticker not in prices.columns:
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": "Missing",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": "",
                    "Last Price": np.nan,
                    "Stale Sessions": np.nan,
                    "Stale Days": np.nan,
                }
            )
            continue

        raw_s = pd.to_numeric(prices[ticker], errors="coerce")
        s = raw_s.dropna()

        if s.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": "No valid data",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": int(raw_s.isna().sum()),
                    "Last Price": np.nan,
                    "Stale Sessions": np.nan,
                    "Stale Days": np.nan,
                }
            )
            continue

        first_ts = pd.to_datetime(s.index.min())
        last_ts = pd.to_datetime(s.index.max())
        active_reference_index = reference_index[reference_index >= first_ts]
        aligned_active = raw_s.reindex(active_reference_index)
        stale_sessions = int((reference_index > last_ts).sum())
        stale_days = int(max(0, (reference_last.date() - last_ts.date()).days))

        if len(s) < min_valid_rows:
            status = "Thin history"
        elif stale_sessions > MAX_STALE_SESSIONS:
            status = "Stale"
        else:
            status = "OK"

        rows.append(
            {
                "Ticker": ticker,
                "Status": status,
                "First Date": first_ts.date(),
                "Latest Date": last_ts.date(),
                "Valid Rows": int(len(s)),
                "Missing Values": int(aligned_active.isna().sum()),
                "Last Price": float(s.iloc[-1]),
                "Stale Sessions": stale_sessions,
                "Stale Days": stale_days,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def prepare_analysis_prices(
    raw_prices: pd.DataFrame,
    tickers: List[str],
    benchmark_ticker: str,
) -> pd.DataFrame:
    """
    Align all series to the selected benchmark's actual trading calendar and
    forward-fill only short market-calendar gaps. Eligibility and staleness are
    determined from raw_prices before this function is called.
    """
    if raw_prices.empty or benchmark_ticker not in raw_prices.columns:
        return pd.DataFrame()

    benchmark_series = pd.to_numeric(
        raw_prices[benchmark_ticker],
        errors="coerce",
    ).dropna()

    if benchmark_series.empty:
        return pd.DataFrame()

    benchmark_index = pd.DatetimeIndex(benchmark_series.index).sort_values().unique()
    available = [ticker for ticker in tickers if ticker in raw_prices.columns]

    if not available:
        return pd.DataFrame(index=benchmark_index)

    aligned = raw_prices[available].reindex(benchmark_index)
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    aligned = aligned.ffill(limit=FORWARD_FILL_LIMIT)
    aligned = aligned.sort_index()
    return aligned


def compute_relative_strength(
    prices: pd.DataFrame,
    item_tickers: List[str],
    benchmark_ticker: str,
) -> pd.DataFrame:
    rs = prices[item_tickers].div(prices[benchmark_ticker], axis=0)
    return rs.replace([np.inf, -np.inf], np.nan)


def compute_log_trend_zscore(
    frame: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    positive = frame.where(frame > 0)
    log_frame = np.log(positive)
    min_periods = max(20, window // 2)
    rolling_mean = log_frame.rolling(window, min_periods=min_periods).mean()
    rolling_std = log_frame.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    z = (log_frame - rolling_mean) / rolling_std
    return z.replace([np.inf, -np.inf], np.nan)


def pct_change_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    if len(frame) <= periods:
        return pd.Series(index=frame.columns, dtype=float)

    current = frame.iloc[-1]
    prior = frame.shift(periods).iloc[-1]
    out = current.div(prior) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def slope_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    return pct_change_last(frame, periods)


def acceleration_last(
    frame: pd.DataFrame,
    periods: int = 5,
) -> pd.Series:
    if len(frame) <= periods * 2:
        return pd.Series(index=frame.columns, dtype=float)

    current = frame.iloc[-1]
    prior = frame.shift(periods).iloc[-1]
    previous = frame.shift(periods * 2).iloc[-1]

    current_return = current.div(prior) - 1.0
    previous_return = prior.div(previous) - 1.0
    acceleration = current_return - previous_return
    return acceleration.replace([np.inf, -np.inf], np.nan)


def risk_adjusted_return_frame(
    frame: pd.DataFrame,
    periods: int,
) -> pd.DataFrame:
    raw_return = frame.div(frame.shift(periods)) - 1.0
    daily_return = frame.pct_change(fill_method=None)
    vol_window = min(252, max(63, periods))
    min_periods = max(20, min(63, vol_window // 2))
    daily_vol = daily_return.rolling(
        vol_window,
        min_periods=min_periods,
    ).std()
    expected_period_vol = daily_vol * np.sqrt(periods)
    standardized = raw_return.div(expected_period_vol)
    return standardized.replace([np.inf, -np.inf], np.nan)


def risk_adjusted_return_last(
    frame: pd.DataFrame,
    periods: int,
) -> pd.Series:
    standardized = risk_adjusted_return_frame(frame, periods)
    if standardized.empty:
        return pd.Series(index=frame.columns, dtype=float)
    return standardized.iloc[-1]


def classify_quadrant(
    long_value: float,
    short_value: float,
    neutral_threshold: float = NEUTRAL_MAP_THRESHOLD,
) -> str:
    if pd.isna(long_value) or pd.isna(short_value):
        return "Neutral"

    if abs(long_value) <= neutral_threshold or abs(short_value) <= neutral_threshold:
        return "Neutral"

    if long_value > 0 and short_value > 0:
        return "Q1"

    if long_value < 0 and short_value > 0:
        return "Q2"

    if long_value < 0 and short_value < 0:
        return "Q3"

    if long_value > 0 and short_value < 0:
        return "Q4"

    return "Neutral"


def pct_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    clean = pd.to_numeric(series, errors="coerce")
    return clean.rank(pct=True, method="average")


def weighted_percentile_score(
    components: Dict[str, pd.Series],
    weights: Dict[str, float],
    min_components: int = MIN_SCORE_COMPONENTS,
) -> Tuple[pd.Series, pd.Series]:
    if not components:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    component_frame = pd.DataFrame(components)
    rank_frame = component_frame.apply(pct_rank, axis=0)
    weight_series = pd.Series(weights, dtype=float).reindex(rank_frame.columns)

    available = rank_frame.notna()
    weighted_values = rank_frame.mul(weight_series, axis=1)
    available_weight = available.mul(weight_series, axis=1).sum(axis=1)
    component_count = available.sum(axis=1)

    score = weighted_values.sum(axis=1, min_count=1).div(
        available_weight.replace(0, np.nan)
    ) * 100.0
    score = score.where(component_count >= min_components)
    return score, component_count


def build_snapshot(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
) -> pd.DataFrame:
    item_tickers = universe_df["Ticker"].tolist()
    item_prices = prices[item_tickers].copy()
    rs = compute_relative_strength(prices, item_tickers, benchmark_ticker)

    if rotation_basis == "relative":
        rotation_frame = rs
        rotation_label = "Relative"
    else:
        rotation_frame = item_prices
        rotation_label = "Absolute"

    abs_short = pct_change_last(item_prices, cfg.short_window)
    abs_long = pct_change_last(item_prices, cfg.long_window)
    abs_slope_10d = slope_last(item_prices, 10)
    abs_slope_5d = slope_last(item_prices, 5)
    abs_accel = acceleration_last(item_prices, 5)

    rs_short = pct_change_last(rs, cfg.short_window)
    rs_long = pct_change_last(rs, cfg.long_window)
    rs_slope_10d = slope_last(rs, 10)
    rs_slope_5d = slope_last(rs, 5)
    rs_accel = acceleration_last(rs, 5)

    rotation_short = pct_change_last(rotation_frame, cfg.short_window)
    rotation_long = pct_change_last(rotation_frame, cfg.long_window)
    rotation_slope_10d = slope_last(rotation_frame, 10)
    rotation_accel = acceleration_last(rotation_frame, 5)

    map_y = rotation_short.copy()
    map_x = rotation_long.copy()

    rs_z_window = min(126, max(63, cfg.long_window))
    rs_z = compute_log_trend_zscore(rs, window=rs_z_window)
    rs_z_last = (
        rs_z.iloc[-1]
        if not rs_z.empty
        else pd.Series(index=item_tickers, dtype=float)
    )

    angle = np.degrees(np.arctan2(map_y, map_x))
    speed = np.sqrt(map_x.pow(2) + map_y.pow(2))

    if rotation_basis == "relative":
        score_components = {
            "rs_short": rs_short,
            "rs_long": rs_long,
            "rs_slope": rs_slope_10d,
            "rs_accel": rs_accel,
            "abs_short": abs_short,
            "abs_long": abs_long,
        }
        score_weights = {
            "rs_short": 0.25,
            "rs_long": 0.25,
            "rs_slope": 0.20,
            "rs_accel": 0.15,
            "abs_short": 0.10,
            "abs_long": 0.05,
        }
    else:
        score_components = {
            "abs_short": abs_short,
            "abs_long": abs_long,
            "abs_slope": abs_slope_10d,
            "abs_accel": abs_accel,
            "rs_short": rs_short,
            "rs_long": rs_long,
        }
        score_weights = {
            "abs_short": 0.25,
            "abs_long": 0.25,
            "abs_slope": 0.20,
            "abs_accel": 0.15,
            "rs_short": 0.10,
            "rs_long": 0.05,
        }

    composite, score_coverage = weighted_percentile_score(
        score_components,
        score_weights,
    )

    meta = universe_df.set_index("Ticker")

    snap = pd.DataFrame(
        {
            "Ticker": item_tickers,
            "Name": meta.reindex(item_tickers)["Name"].values,
            "Sector Group": meta.reindex(item_tickers)["Sector Group"].values,
            "Tier": meta.reindex(item_tickers)["Tier"].values,
            "Rotation Mode": rotation_label,
            f"Rotation {cfg.short_label}": rotation_short.reindex(item_tickers).values,
            f"Rotation {cfg.long_label}": rotation_long.reindex(item_tickers).values,
            f"Abs {cfg.short_label}": abs_short.reindex(item_tickers).values,
            f"Abs {cfg.long_label}": abs_long.reindex(item_tickers).values,
            f"RS {cfg.short_label}": rs_short.reindex(item_tickers).values,
            f"RS {cfg.long_label}": rs_long.reindex(item_tickers).values,
            "RS Slope 10D": rs_slope_10d.reindex(item_tickers).values,
            "RS Slope 5D": rs_slope_5d.reindex(item_tickers).values,
            "RS Accel": rs_accel.reindex(item_tickers).values,
            "Abs Slope 10D": abs_slope_10d.reindex(item_tickers).values,
            "Abs Slope 5D": abs_slope_5d.reindex(item_tickers).values,
            "Abs Accel": abs_accel.reindex(item_tickers).values,
            "Rotation Slope 10D": rotation_slope_10d.reindex(item_tickers).values,
            "Rotation Accel": rotation_accel.reindex(item_tickers).values,
            "RS Z": rs_z_last.reindex(item_tickers).values,
            "Map X": map_x.reindex(item_tickers).values,
            "Map Y": map_y.reindex(item_tickers).values,
            "Angle (deg)": angle.reindex(item_tickers).values,
            "Speed": speed.reindex(item_tickers).values,
            "Composite Score": composite.reindex(item_tickers).values,
            "Score Coverage": score_coverage.reindex(item_tickers).values,
        }
    )

    snap["QuadrantCode"] = [
        classify_quadrant(long_value, short_value)
        for long_value, short_value in zip(snap["Map X"], snap["Map Y"])
    ]
    snap["Quadrant"] = snap["QuadrantCode"].map(QUADRANT_LABELS)

    snap["State"] = np.select(
        [
            (snap["Quadrant"] == "Leading") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Leading") & (snap["Rotation Accel"] <= 0),
            (snap["Quadrant"] == "Improving") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Improving") & (snap["Rotation Accel"] <= 0),
            (snap["Quadrant"] == "Weakening") & (snap["Rotation Accel"] < 0),
            (snap["Quadrant"] == "Weakening") & (snap["Rotation Accel"] >= 0),
            (snap["Quadrant"] == "Lagging") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Lagging") & (snap["Rotation Accel"] <= 0),
        ],
        [
            "Leading and accelerating",
            "Leading but cooling",
            "Improving fast",
            "Improving but uneven",
            "Weakening fast",
            "Weakening but stabilizing",
            "Lagging but stabilizing",
            "Lagging and deteriorating",
        ],
        default=snap["Quadrant"],
    )

    snap["MarkerSize"] = 16 + (
        snap["Speed"].rank(pct=True).fillna(0.5) * 18
    )
    snap["Color"] = snap["Sector Group"].map(
        SECTOR_GROUP_COLORS
    ).fillna("#4b5563")

    snap = snap.sort_values(
        ["Composite Score", "Speed", "Ticker"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    snap["Rank"] = np.arange(1, len(snap) + 1)

    return snap


def compute_rank_delta(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
    compare_days: int = 5,
) -> pd.DataFrame:
    if len(prices) <= cfg.long_window + compare_days + 20:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Prior Rank",
                "Rank Δ",
                "Prior Quadrant",
                "Prior Composite Score",
            ]
        )

    past_prices = prices.iloc[:-compare_days].copy()

    past_snap = build_snapshot(
        past_prices,
        universe_df,
        benchmark_ticker,
        cfg,
        rotation_basis,
    )[["Ticker", "Rank", "Quadrant", "Composite Score"]].rename(
        columns={
            "Rank": "Prior Rank",
            "Quadrant": "Prior Quadrant",
            "Composite Score": "Prior Composite Score",
        }
    )

    current_snap = build_snapshot(
        prices,
        universe_df,
        benchmark_ticker,
        cfg,
        rotation_basis,
    )[["Ticker", "Rank"]]

    merged = current_snap.merge(past_snap, on="Ticker", how="left")
    merged["Rank Δ"] = merged["Prior Rank"] - merged["Rank"]

    return merged[
        [
            "Ticker",
            "Prior Rank",
            "Rank Δ",
            "Prior Quadrant",
            "Prior Composite Score",
        ]
    ]


def build_rotation_trails(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    trail_weeks: int,
    rotation_basis: str,
) -> Dict[str, pd.DataFrame]:
    """
    Build trail points from the same daily-window percentage-return
    methodology used for the current map point, then sample those coordinates
    at weekly intervals. The latest trail point therefore matches the marker.
    """
    if trail_weeks <= 0:
        return {}

    item_tickers = [
        ticker
        for ticker in universe_df["Ticker"].tolist()
        if ticker in prices.columns
    ]

    if not item_tickers:
        return {}

    if rotation_basis == "relative":
        if benchmark_ticker not in prices.columns:
            return {}
        rotation_frame = prices[item_tickers].div(
            prices[benchmark_ticker],
            axis=0,
        )
    else:
        rotation_frame = prices[item_tickers].copy()

    rotation_frame = rotation_frame.replace([np.inf, -np.inf], np.nan)
    map_x_history = rotation_frame.pct_change(cfg.long_window, fill_method=None)
    map_y_history = rotation_frame.pct_change(cfg.short_window, fill_method=None)

    trail_points: Dict[str, pd.DataFrame] = {}

    for ticker in item_tickers:
        coordinates = pd.DataFrame(
            {
                "x": map_x_history[ticker],
                "y": map_y_history[ticker],
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if coordinates.empty:
            continue

        weekly_coordinates = coordinates.resample("W-FRI").last().dropna()
        if weekly_coordinates.empty:
            continue

        trail_points[ticker] = weekly_coordinates.tail(trail_weeks)

    return trail_points


def make_rs_chart(
    rs_series: pd.Series,
    item_name: str,
    benchmark_ticker: str,
) -> go.Figure:
    rs_20 = rs_series.rolling(20).mean()
    rs_50 = rs_series.rolling(50).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rs_series.index,
            y=rs_series.values,
            mode="lines",
            name="RS Ratio",
            line=dict(width=2.4, color="#1f77b4"),
            hovertemplate="%{x|%Y-%m-%d}<br>RS: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rs_20.index,
            y=rs_20.values,
            mode="lines",
            name="20D MA",
            line=dict(width=1.4, dash="dot", color="#2ca02c"),
            hovertemplate="%{x|%Y-%m-%d}<br>20D MA: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rs_50.index,
            y=rs_50.values,
            mode="lines",
            name="50D MA",
            line=dict(width=1.4, dash="dash", color="#9467bd"),
            hovertemplate="%{x|%Y-%m-%d}<br>50D MA: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Relative Strength | {item_name} vs {benchmark_ticker}",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=20),
            pad=dict(b=12),
        ),
        height=440,
        margin=dict(l=10, r=10, t=95, b=10),
        xaxis_title="Date",
        yaxis_title="RS Ratio",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0.0,
            traceorder="normal",
            entrywidth=100,
            entrywidthmode="pixels",
            font=dict(size=13),
        ),
    )

    return fig


def make_rotation_scatter(
    snap: pd.DataFrame,
    cfg: RotationConfig,
    trail_points: Dict[str, pd.DataFrame],
    rotation_basis: str,
    benchmark_ticker: str,
    label_mode: str,
    label_top_n: int,
) -> go.Figure:
    x_col = "Map X"
    y_col = "Map Y"

    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_width=1, opacity=0.45)

    plot_df = snap.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid rotation data for the selected universe/window.",
            showarrow=False,
        )
        fig.update_layout(height=760)
        return fig

    label_set = set()
    if label_mode == "All tickers":
        label_set = set(plot_df["Ticker"])
    elif label_mode == "Top ranked only":
        label_set = set(plot_df.nsmallest(label_top_n, "Rank")["Ticker"])

    for ticker, trail_df in trail_points.items():
        if ticker not in plot_df["Ticker"].values or trail_df.empty:
            continue

        group = plot_df.loc[
            plot_df["Ticker"] == ticker,
            "Sector Group",
        ].iloc[0]
        color = SECTOR_GROUP_COLORS.get(group, "#4b5563")

        fig.add_trace(
            go.Scatter(
                x=trail_df["x"],
                y=trail_df["y"],
                mode="lines+markers",
                line=dict(color=color, width=1.2),
                marker=dict(size=4, color=color, opacity=0.5),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for _, row in plot_df.iterrows():
        ticker = row["Ticker"]
        border_color = STATE_COLORS.get(row["Quadrant"], "#111827")
        text = ticker if ticker in label_set else ""

        hovertemplate = (
            f"<b>{row['Name']} ({ticker})</b><br>"
            f"Group: {row['Sector Group']}<br>"
            f"Tier: {row['Tier']}<br>"
            f"Rotation mode: {row['Rotation Mode']}<br>"
            f"Rotation {cfg.short_label}: {row[f'Rotation {cfg.short_label}']:.2%}<br>"
            f"Rotation {cfg.long_label}: {row[f'Rotation {cfg.long_label}']:.2%}<br>"
            f"Map {cfg.short_label}: {row['Map Y']:.2%}<br>"
            f"Map {cfg.long_label}: {row['Map X']:.2%}<br>"
            f"Abs {cfg.short_label}: {row[f'Abs {cfg.short_label}']:.2%}<br>"
            f"Abs {cfg.long_label}: {row[f'Abs {cfg.long_label}']:.2%}<br>"
            f"RS {cfg.short_label}: {row[f'RS {cfg.short_label}']:.2%}<br>"
            f"RS {cfg.long_label}: {row[f'RS {cfg.long_label}']:.2%}<br>"
            f"RS Slope 10D: {row['RS Slope 10D']:.2%}<br>"
            f"RS Accel: {row['RS Accel']:.2%}<br>"
            f"Composite Score: {row['Composite Score']:.1f}<br>"
            f"Quadrant: {row['Quadrant']}<br>"
            f"State: {row['State']}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=[row[x_col]],
                y=[row[y_col]],
                mode="markers+text" if text else "markers",
                text=[text] if text else None,
                textposition="top center",
                marker=dict(
                    size=float(row["MarkerSize"]),
                    color=row["Color"],
                    line=dict(width=1.8, color=border_color),
                    opacity=0.90,
                ),
                name=ticker,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    x_min = float(plot_df[x_col].min())
    x_max = float(plot_df[x_col].max())
    y_min = float(plot_df[y_col].min())
    y_max = float(plot_df[y_col].max())

    x_pad = max(
        0.15,
        abs(x_max - x_min) * 0.20,
        float(plot_df[x_col].abs().max()) * 0.15,
    )
    y_pad = max(
        0.15,
        abs(y_max - y_min) * 0.20,
        float(plot_df[y_col].abs().max()) * 0.15,
    )

    if x_min == x_max:
        x_min -= x_pad
        x_max += x_pad

    if y_min == y_max:
        y_min -= y_pad
        y_max += y_pad

    fig.add_annotation(
        x=x_max + x_pad * 0.45,
        y=y_max + y_pad * 0.45,
        text="Leading",
        showarrow=False,
        font=dict(color=STATE_COLORS["Leading"], size=12),
    )

    fig.add_annotation(
        x=x_min - x_pad * 0.45,
        y=y_max + y_pad * 0.45,
        text="Improving",
        showarrow=False,
        font=dict(color=STATE_COLORS["Improving"], size=12),
    )

    fig.add_annotation(
        x=x_min - x_pad * 0.45,
        y=y_min - y_pad * 0.45,
        text="Lagging",
        showarrow=False,
        font=dict(color=STATE_COLORS["Lagging"], size=12),
    )

    fig.add_annotation(
        x=x_max + x_pad * 0.45,
        y=y_min - y_pad * 0.45,
        text="Weakening",
        showarrow=False,
        font=dict(color=STATE_COLORS["Weakening"], size=12),
    )

    mode_title = (
        f"Relative to {benchmark_ticker}"
        if rotation_basis == "relative"
        else "Absolute return"
    )

    fig.update_layout(
        title=f"Rotation Map | {mode_title} | {cfg.short_label} vs {cfg.long_label}",
        height=760,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(
            title=f"{cfg.long_label} rotation",
            tickformat=".1%",
            zeroline=False,
            range=[x_min - x_pad, x_max + x_pad],
        ),
        yaxis=dict(
            title=f"{cfg.short_label} rotation",
            tickformat=".1%",
            zeroline=False,
            range=[y_min - y_pad, y_max + y_pad],
        ),
    )

    return fig


def format_delta(val: float) -> str:
    if pd.isna(val):
        return ""

    sign = "+" if val > 0 else ""
    return f"{sign}{int(val)}"


def build_display_table(
    snap: pd.DataFrame,
    rank_delta: pd.DataFrame,
    cfg: RotationConfig,
) -> pd.DataFrame:
    table = snap.merge(rank_delta, on="Ticker", how="left")

    prior_quadrant = table["Prior Quadrant"].fillna("")
    current_quadrant = table["Quadrant"].fillna("")
    table["Quadrant Δ"] = np.where(
        (prior_quadrant == "") | (prior_quadrant == current_quadrant),
        "",
        prior_quadrant + " → " + current_quadrant,
    )

    table["Rank Δ"] = table["Rank Δ"].apply(format_delta)

    cols = [
        "Rank",
        "Ticker",
        "Name",
        "Sector Group",
        "Tier",
        "Quadrant",
        "State",
        "Composite Score",
        "Rank Δ",
        "Quadrant Δ",
        f"Rotation {cfg.short_label}",
        f"Rotation {cfg.long_label}",
        f"Abs {cfg.short_label}",
        f"Abs {cfg.long_label}",
        f"RS {cfg.short_label}",
        f"RS {cfg.long_label}",
        "RS Slope 10D",
        "RS Accel",
        "RS Z",
        "Angle (deg)",
        "Speed",
    ]

    return table[cols].copy()


def style_snapshot_table(df: pd.DataFrame, cfg: RotationConfig):
    fmt_map = {
        "Composite Score": "{:.1f}",
        f"Rotation {cfg.short_label}": "{:.2%}",
        f"Rotation {cfg.long_label}": "{:.2%}",
        f"Abs {cfg.short_label}": "{:.2%}",
        f"Abs {cfg.long_label}": "{:.2%}",
        f"RS {cfg.short_label}": "{:.2%}",
        f"RS {cfg.long_label}": "{:.2%}",
        "RS Slope 10D": "{:.2%}",
        "RS Accel": "{:.2%}",
        "RS Z": "{:.2f}",
        "Angle (deg)": "{:.1f}",
        "Speed": "{:.2f}",
    }

    fmt_map = {k: v for k, v in fmt_map.items() if k in df.columns}

    gradient_cols_primary = [
        c
        for c in [
            "Composite Score",
            f"Rotation {cfg.short_label}",
            f"Rotation {cfg.long_label}",
            f"RS {cfg.short_label}",
            f"RS {cfg.long_label}",
            "RS Slope 10D",
            "RS Accel",
        ]
        if c in df.columns
    ]

    gradient_cols_speed = [c for c in ["Speed"] if c in df.columns]

    styler = df.style.format(fmt_map)

    if gradient_cols_primary:
        styler = styler.background_gradient(
            subset=gradient_cols_primary,
            cmap="RdYlGn",
        )

    if gradient_cols_speed:
        styler = styler.background_gradient(
            subset=gradient_cols_speed,
            cmap="PuBuGn",
        )

    return styler


# =============================================================================
# Load data
# =============================================================================

cfg = get_rotation_config(window_choice)
min_required_rows = cfg.long_window + 30

if selected_universe.empty:
    st.warning("No sector groups selected.")
    st.stop()

all_tickers = list(
    dict.fromkeys(
        selected_universe["Ticker"].tolist()
        + list(BENCHMARKS.keys())
    )
)

with st.spinner("Loading market data..."):
    raw_prices = fetch_prices(all_tickers)

benchmark_ok, benchmark_error = validate_benchmark(
    raw_prices,
    benchmark,
    min_required_rows,
)

if not benchmark_ok:
    st.error(benchmark_error)
    st.stop()

eligible_universe, universe_diagnostics = filter_universe_by_data(
    universe_df=selected_universe,
    raw_prices=raw_prices,
    benchmark_ticker=benchmark,
    min_valid_rows=min_required_rows,
)

benchmark_diagnostics = build_price_diagnostics(
    prices=raw_prices,
    tickers=[benchmark],
    min_valid_rows=min_required_rows,
    benchmark_ticker=benchmark,
)

diagnostics_df = pd.concat(
    [universe_diagnostics, benchmark_diagnostics],
    ignore_index=True,
).drop_duplicates(subset=["Ticker"], keep="first")

if eligible_universe.empty:
    st.error(
        "No eligible sector/subsector ETFs have enough current history "
        "for the selected rotation window."
    )
    st.stop()

excluded_tickers = universe_diagnostics[
    universe_diagnostics["Status"] != "OK"
]

if not excluded_tickers.empty:
    with st.expander(
        f"Dropped {len(excluded_tickers)} ticker(s) with missing, thin, or stale data",
        expanded=False,
    ):
        st.dataframe(
            excluded_tickers,
            use_container_width=True,
            hide_index=True,
        )

item_tickers = eligible_universe["Ticker"].tolist()
analysis_tickers = list(dict.fromkeys(item_tickers + [benchmark]))
prices = prepare_analysis_prices(
    raw_prices=raw_prices,
    tickers=analysis_tickers,
    benchmark_ticker=benchmark,
)

if prices.empty:
    st.error("Unable to construct an analysis-ready price panel.")
    st.stop()

analysis_benchmark_ok, analysis_benchmark_error = validate_benchmark(
    prices,
    benchmark,
    min_required_rows,
)

if not analysis_benchmark_ok:
    st.error(analysis_benchmark_error)
    st.stop()


# =============================================================================
# Analytics
# =============================================================================

rs = compute_relative_strength(prices, item_tickers, benchmark)

snap = build_snapshot(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
)

rank_delta = compute_rank_delta(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
    compare_days=5,
)

trail_points = build_rotation_trails(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    trail_weeks=TRAIL_OPTIONS[trail_choice],
    rotation_basis=rotation_mode,
)

display_df = build_display_table(snap, rank_delta, cfg)


# =============================================================================
# Optional diagnostics
# =============================================================================

if show_diagnostics:
    with st.expander("Data diagnostics", expanded=True):
        st.dataframe(
            diagnostics_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Main charts
# =============================================================================

st.subheader("Rotation map")

rot_fig = make_rotation_scatter(
    snap=snap,
    cfg=cfg,
    trail_points=trail_points,
    rotation_basis=rotation_mode,
    benchmark_ticker=benchmark,
    label_mode=label_mode,
    label_top_n=label_top_n,
)

st.plotly_chart(rot_fig, use_container_width=True)

st.subheader("Relative strength")

rs_options = snap.sort_values("Rank")["Ticker"].tolist()
rs_default = rs_options.index("SMH") if "SMH" in rs_options else 0

universe_lookup = eligible_universe.set_index("Ticker")

rs_item = st.selectbox(
    "Ticker for RS chart",
    options=rs_options,
    format_func=lambda x: (
        f"{x} | "
        f"{universe_lookup.loc[x, 'Name']} | "
        f"{universe_lookup.loc[x, 'Sector Group']}"
    ),
    index=rs_default,
)

rs_meta = universe_lookup.loc[rs_item]

rs_fig = make_rs_chart(
    rs_series=rs[rs_item].dropna(),
    item_name=f"{rs_meta['Name']} ({rs_item})",
    benchmark_ticker=benchmark,
)

st.plotly_chart(rs_fig, use_container_width=True)


# =============================================================================
# Snapshot table
# =============================================================================

st.subheader("Rotation snapshot")

table_sorted = display_df.sort_values("Rank", ascending=True).reset_index(drop=True)

row_height = 34
table_height = min(780, max(180, row_height * (len(table_sorted) + 1)))

st.dataframe(
    style_snapshot_table(table_sorted, cfg),
    use_container_width=True,
    height=table_height,
    hide_index=True,
)

csv = table_sorted.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download snapshot CSV",
    data=csv,
    file_name="adfm_subsector_rotation_snapshot.csv",
    mime="text/csv",
)


# =============================================================================
# Footer
# =============================================================================

coverage = len(item_tickers)
benchmark_series_raw = pd.to_numeric(
    raw_prices[benchmark],
    errors="coerce",
).dropna()
latest_dt = pd.to_datetime(benchmark_series_raw.index.max()).date()

st.caption(
    f"Data through: {latest_dt} | Benchmark: {benchmark} | Rotation mode: {rotation_mode_label} | "
    f"Universe: {universe_scope} | Coverage: {coverage}/{len(selected_universe)} eligible tickers"
)

st.caption("© 2026 AD Fund Management LP")
