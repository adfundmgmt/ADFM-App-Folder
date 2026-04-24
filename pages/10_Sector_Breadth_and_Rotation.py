# sp_sector_rotation_monitor_v8.py
# ADFM | S&P 500 Sector Breadth & Rotation Monitor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
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
    page_title="S&P 500 Sector Breadth & Rotation Monitor",
    layout="wide",
)

st.title("S&P 500 Sector Breadth & Rotation Monitor")


# =============================================================================
# Constants
# =============================================================================

SECTORS: Dict[str, str] = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

BENCHMARKS: Dict[str, str] = {
    "SPY": "SPDR S&P 500",
    "RSP": "Invesco S&P 500 Equal Weight",
    "QQQ": "Invesco QQQ",
    "TLT": "iShares 20+ Year Treasury Bond",
}

MARKET_CONTEXT: Dict[str, str] = {
    "^VIX": "CBOE Volatility Index",
}

LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"

WINDOW_PRESETS = {
    "Fast (1M vs 3M)": {
        "short": 21,
        "long": 63,
        "label_short": "1M",
        "label_long": "3M",
    },
    "Intermediate (3M vs 6M)": {
        "short": 63,
        "long": 126,
        "label_short": "3M",
        "label_long": "6M",
    },
    "Trend (6M vs 12M)": {
        "short": 126,
        "long": 252,
        "label_short": "6M",
        "label_long": "12M",
    },
}

TRAIL_OPTIONS = {
    "None": 0,
    "4 weeks": 4,
    "8 weeks": 8,
    "12 weeks": 12,
}

ROTATION_MODES = {
    "Benchmark-relative rotation": "relative",
    "Absolute sector rotation": "absolute",
}

SECTOR_COLORS = {
    "XLC": "#4E79A7",
    "XLY": "#F28E2B",
    "XLP": "#59A14F",
    "XLE": "#E15759",
    "XLF": "#76B7B2",
    "XLV": "#EDC948",
    "XLI": "#B07AA1",
    "XLB": "#FF9DA7",
    "XLRE": "#9C755F",
    "XLK": "#2F5597",
    "XLU": "#BAB0AC",
}

QUADRANT_LABELS = {
    "Q1": "Leading",
    "Q2": "Improving",
    "Q3": "Lagging",
    "Q4": "Weakening",
    "Neutral": "Neutral",
}

STATE_COLORS = {
    "Leading": "#16a34a",
    "Improving": "#2563eb",
    "Lagging": "#dc2626",
    "Weakening": "#f59e0b",
    "Neutral": "#6b7280",
}

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"


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
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Sector breadth and rotation monitor for leadership, participation, and regime shifts.

        **Core read**
        - The default rotation map uses sector relative strength versus the selected benchmark.
        - Absolute sector returns are preserved in the table.
        - Composite rank combines RS momentum, RS trend, acceleration, and absolute participation.

        **Use case**
        - Identify whether leadership is broadening, narrowing, rotating, or breaking.
        - Separate genuine sector leadership from sectors that are only rising because the tape is strong.
        - Track whether rotation is confirmed by equal weight, growth, duration, and volatility signals.

        **Data source**
        - Yahoo Finance through `yfinance`.
        """
    )

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
        help="Benchmark-relative rotation is usually the cleaner default for leadership analysis.",
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

    rs_sector = st.selectbox(
        "Sector for RS chart",
        options=list(SECTORS.keys()),
        format_func=lambda x: f"{x} | {SECTORS[x]}",
        index=list(SECTORS.keys()).index("XLK"),
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
    Normalize yfinance output into a clean wide dataframe of adjusted closes.

    Handles:
    - MultiIndex columns with first level as field and second level as ticker.
    - MultiIndex columns with first level as ticker and second level as field.
    - Single ticker downloads with normal OHLC columns.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    field_candidates = ["Adj Close", "Close"]
    out = {}

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
                selected = pd.to_numeric(selected, errors="coerce").dropna()
                if not selected.empty:
                    out[ticker] = selected

        df = pd.DataFrame(out)
        return _safe_to_datetime_index(df)

    if len(tickers) == 1:
        ticker = tickers[0]
        for field in field_candidates:
            if field in data.columns:
                s = pd.to_numeric(data[field], errors="coerce").dropna()
                if not s.empty:
                    return _safe_to_datetime_index(pd.DataFrame({ticker: s}))

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(
    tickers: List[str],
    period: str = LOOKBACK_PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    unique_tickers = list(dict.fromkeys(tickers))

    try:
        raw = yf.download(
            tickers=unique_tickers,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    prices = _normalize_download(raw, unique_tickers)

    if prices.empty:
        return prices

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill()

    return prices


def validate_prices(prices: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    if prices.empty:
        return False, required

    missing = []
    for ticker in required:
        if ticker not in prices.columns:
            missing.append(ticker)
            continue

        if prices[ticker].dropna().empty:
            missing.append(ticker)

    return len(missing) == 0, missing


def build_price_diagnostics(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Status",
                "First Date",
                "Latest Date",
                "Valid Rows",
                "Missing Values",
                "Last Price",
                "Stale Days",
            ]
        )

    max_dt = pd.to_datetime(prices.index.max()).date()
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
                    "Stale Days": np.nan,
                }
            )
            continue

        s = pd.to_numeric(prices[ticker], errors="coerce").dropna()

        if s.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": "No valid data",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": int(prices[ticker].isna().sum()),
                    "Last Price": np.nan,
                    "Stale Days": np.nan,
                }
            )
            continue

        first_dt = pd.to_datetime(s.index.min()).date()
        last_dt = pd.to_datetime(s.index.max()).date()
        stale_days = (max_dt - last_dt).days

        if stale_days > 5:
            status = "Stale"
        elif len(s) < 100:
            status = "Thin history"
        else:
            status = "OK"

        rows.append(
            {
                "Ticker": ticker,
                "Status": status,
                "First Date": first_dt,
                "Latest Date": last_dt,
                "Valid Rows": int(len(s)),
                "Missing Values": int(prices[ticker].isna().sum()),
                "Last Price": float(s.iloc[-1]),
                "Stale Days": int(stale_days),
            }
        )

    return pd.DataFrame(rows)


def compute_relative_strength(
    prices: pd.DataFrame,
    sector_tickers: List[str],
    benchmark_ticker: str,
) -> pd.DataFrame:
    base = prices[sector_tickers].div(prices[benchmark_ticker], axis=0)
    base = base.replace([np.inf, -np.inf], np.nan)
    return base.dropna(how="all")


def compute_zscore(frame: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rolling_mean = frame.rolling(window).mean()
    rolling_std = frame.rolling(window).std().replace(0, np.nan)
    z = (frame - rolling_mean) / rolling_std
    return z.replace([np.inf, -np.inf], np.nan)


def pct_change_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    if len(frame) <= periods:
        return pd.Series(index=frame.columns, dtype=float)

    out = frame.pct_change(periods).iloc[-1]
    return out.replace([np.inf, -np.inf], np.nan)


def slope_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    if len(frame) <= periods:
        return pd.Series(index=frame.columns, dtype=float)

    out = frame.iloc[-1] / frame.shift(periods).iloc[-1] - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def classify_quadrant(long_ret: float, short_ret: float) -> str:
    if pd.isna(long_ret) or pd.isna(short_ret):
        return "Neutral"

    if long_ret > 0 and short_ret > 0:
        return "Q1"

    if long_ret < 0 and short_ret > 0:
        return "Q2"

    if long_ret < 0 and short_ret < 0:
        return "Q3"

    if long_ret > 0 and short_ret < 0:
        return "Q4"

    return "Neutral"


def pct_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    ranked = series.rank(pct=True)
    return ranked.fillna(0.5)


def build_snapshot(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
) -> pd.DataFrame:
    sector_tickers = list(sectors.keys())
    sector_prices = prices[sector_tickers].copy()

    rs = compute_relative_strength(prices, sector_tickers, benchmark_ticker)

    if rotation_basis == "relative":
        rotation_frame = rs
        rotation_label = "Relative"
    else:
        rotation_frame = sector_prices
        rotation_label = "Absolute"

    abs_short = pct_change_last(sector_prices, cfg.short_window)
    abs_long = pct_change_last(sector_prices, cfg.long_window)

    rs_short = pct_change_last(rs, cfg.short_window)
    rs_long = pct_change_last(rs, cfg.long_window)

    rotation_short = pct_change_last(rotation_frame, cfg.short_window)
    rotation_long = pct_change_last(rotation_frame, cfg.long_window)

    rs_slope_10d = slope_last(rs, 10)
    rs_slope_5d = slope_last(rs, 5)
    rs_accel = (rs_slope_5d - rs_slope_10d).replace([np.inf, -np.inf], np.nan)

    rotation_slope_10d = slope_last(rotation_frame, 10)
    rotation_slope_5d = slope_last(rotation_frame, 5)
    rotation_accel = (rotation_slope_5d - rotation_slope_10d).replace([np.inf, -np.inf], np.nan)

    rs_z_window = min(63, max(20, cfg.long_window))
    rs_z = compute_zscore(rs, window=rs_z_window)
    rs_z_last = rs_z.iloc[-1] if not rs_z.empty else pd.Series(index=sector_tickers, dtype=float)

    angle = np.degrees(np.arctan2(rotation_short, rotation_long))
    speed = np.sqrt(rotation_short.pow(2) + rotation_long.pow(2))

    composite = (
        0.25 * pct_rank(rs_short)
        + 0.25 * pct_rank(rs_long)
        + 0.20 * pct_rank(rs_slope_10d)
        + 0.15 * pct_rank(rs_accel)
        + 0.10 * pct_rank(abs_short)
        + 0.05 * pct_rank(abs_long)
    ) * 100.0

    snap = pd.DataFrame(
        {
            "Ticker": sector_tickers,
            "Sector": [sectors[t] for t in sector_tickers],
            "Rotation Mode": rotation_label,
            f"Rotation {cfg.short_label}": rotation_short.reindex(sector_tickers).values,
            f"Rotation {cfg.long_label}": rotation_long.reindex(sector_tickers).values,
            f"Abs {cfg.short_label}": abs_short.reindex(sector_tickers).values,
            f"Abs {cfg.long_label}": abs_long.reindex(sector_tickers).values,
            f"RS {cfg.short_label}": rs_short.reindex(sector_tickers).values,
            f"RS {cfg.long_label}": rs_long.reindex(sector_tickers).values,
            "RS Slope 10D": rs_slope_10d.reindex(sector_tickers).values,
            "RS Slope 5D": rs_slope_5d.reindex(sector_tickers).values,
            "RS Accel": rs_accel.reindex(sector_tickers).values,
            "Rotation Slope 10D": rotation_slope_10d.reindex(sector_tickers).values,
            "Rotation Accel": rotation_accel.reindex(sector_tickers).values,
            "RS Z": rs_z_last.reindex(sector_tickers).values,
            "Angle (deg)": angle.reindex(sector_tickers).values,
            "Speed": speed.reindex(sector_tickers).values,
            "Composite Score": composite.reindex(sector_tickers).values,
        }
    )

    snap["QuadrantCode"] = [
        classify_quadrant(lr, sr)
        for lr, sr in zip(
            snap[f"Rotation {cfg.long_label}"],
            snap[f"Rotation {cfg.short_label}"],
        )
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

    snap["MarkerSize"] = 18 + (snap["Speed"].rank(pct=True).fillna(0.5) * 20)
    snap["Color"] = snap["Ticker"].map(SECTOR_COLORS)

    snap = snap.sort_values("Composite Score", ascending=False).reset_index(drop=True)
    snap["Rank"] = np.arange(1, len(snap) + 1)

    return snap


def compute_rank_delta(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
    compare_days: int = 5,
) -> pd.DataFrame:
    if len(prices) <= cfg.long_window + compare_days + 15:
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
    current_prices = prices.copy()

    past_snap = build_snapshot(
        past_prices,
        sectors,
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

    curr_snap = build_snapshot(
        current_prices,
        sectors,
        benchmark_ticker,
        cfg,
        rotation_basis,
    )[["Ticker", "Rank", "Quadrant", "Composite Score"]]

    merged = curr_snap.merge(past_snap, on="Ticker", how="left")
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
    sectors: Dict[str, str],
    benchmark_ticker: str,
    cfg: RotationConfig,
    trail_weeks: int,
    rotation_basis: str,
) -> Dict[str, pd.DataFrame]:
    if trail_weeks <= 0:
        return {}

    sector_tickers = list(sectors.keys())
    needed = sector_tickers + [benchmark_ticker]
    available = [t for t in needed if t in prices.columns]

    weekly_prices = prices[available].resample("W-FRI").last().dropna(how="all")

    if weekly_prices.empty:
        return {}

    if rotation_basis == "relative":
        if benchmark_ticker not in weekly_prices.columns:
            return {}

        rotation_frame = weekly_prices[sector_tickers].div(
            weekly_prices[benchmark_ticker],
            axis=0,
        )
    else:
        rotation_frame = weekly_prices[sector_tickers]

    rotation_frame = rotation_frame.replace([np.inf, -np.inf], np.nan)

    short_weeks = max(1, round(cfg.short_window / 5))
    long_weeks = max(2, round(cfg.long_window / 5))

    trail_points = {}

    for ticker in sector_tickers:
        if ticker not in rotation_frame.columns:
            continue

        s = rotation_frame[ticker].dropna()
        if s.empty or len(s) <= long_weeks:
            continue

        short_ret = s.pct_change(short_weeks)
        long_ret = s.pct_change(long_weeks)

        df = pd.DataFrame(
            {
                "x": long_ret,
                "y": short_ret,
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            continue

        trail_points[ticker] = df.tail(trail_weeks)

    return trail_points


def make_summary_text(
    snap: pd.DataFrame,
    cfg: RotationConfig,
    benchmark_ticker: str,
    rotation_basis: str,
) -> str:
    if snap.empty:
        return "No valid sector data available."

    leaders = snap[snap["Quadrant"] == "Leading"].sort_values(
        "Composite Score",
        ascending=False,
    )
    improvers = snap[snap["Quadrant"] == "Improving"].sort_values(
        "Rotation Accel",
        ascending=False,
    )
    weakeners = snap[snap["Quadrant"] == "Weakening"].sort_values(
        "Rotation Accel",
        ascending=True,
    )
    laggards = snap[snap["Quadrant"] == "Lagging"].sort_values(
        "Composite Score",
        ascending=True,
    )

    leader_txt = ", ".join(leaders["Ticker"].head(3).tolist()) if not leaders.empty else "none"
    improver_txt = ", ".join(improvers["Ticker"].head(2).tolist()) if not improvers.empty else "none"
    weakener_txt = ", ".join(weakeners["Ticker"].head(2).tolist()) if not weakeners.empty else "none"
    laggard_txt = ", ".join(laggards["Ticker"].head(2).tolist()) if not laggards.empty else "none"

    positive_rs_short = int((snap[f"RS {cfg.short_label}"] > 0).sum())
    positive_rs_long = int((snap[f"RS {cfg.long_label}"] > 0).sum())
    positive_rs_slope = int((snap["RS Slope 10D"] > 0).sum())
    accelerating_rs = int((snap["RS Accel"] > 0).sum())

    mode_text = "benchmark-relative" if rotation_basis == "relative" else "absolute-return"

    text = (
        f"The map is currently reading {mode_text} rotation versus {benchmark_ticker}. "
        f"Leadership is concentrated in {leader_txt}. The main improvement bucket is {improver_txt}, "
        f"while deterioration is clearest in {weakener_txt}. Laggards remain {laggard_txt}. "
        f"Across the sector set, {positive_rs_short} of {len(snap)} sectors are outperforming over {cfg.short_label}, "
        f"{positive_rs_long} of {len(snap)} are outperforming over {cfg.long_label}, "
        f"{positive_rs_slope} of {len(snap)} have a positive 10-day RS slope, and "
        f"{accelerating_rs} of {len(snap)} are accelerating on a 5-day versus 10-day RS basis."
    )

    return text


def make_rs_chart(
    rs_series: pd.Series,
    sector_name: str,
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
        title=f"Relative Strength | {sector_name} vs {benchmark_ticker}",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Date",
        yaxis_title="RS Ratio",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
    )

    return fig


def make_rotation_scatter(
    snap: pd.DataFrame,
    cfg: RotationConfig,
    trail_points: Dict[str, pd.DataFrame],
    rotation_basis: str,
    benchmark_ticker: str,
) -> go.Figure:
    x_col = f"Rotation {cfg.long_label}"
    y_col = f"Rotation {cfg.short_label}"

    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_width=1, opacity=0.45)

    for ticker, trail_df in trail_points.items():
        if ticker not in snap["Ticker"].values or trail_df.empty:
            continue

        color = SECTOR_COLORS.get(ticker, "#4b5563")

        fig.add_trace(
            go.Scatter(
                x=trail_df["x"],
                y=trail_df["y"],
                mode="lines+markers",
                line=dict(color=color, width=1.4),
                marker=dict(size=5, color=color, opacity=0.6),
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for _, row in snap.iterrows():
        ticker = row["Ticker"]
        border_color = STATE_COLORS.get(row["Quadrant"], "#111827")

        hovertemplate = (
            f"<b>{row['Sector']} ({ticker})</b><br>"
            f"Rotation mode: {row['Rotation Mode']}<br>"
            f"Rotation {cfg.short_label}: {row[f'Rotation {cfg.short_label}']:.2%}<br>"
            f"Rotation {cfg.long_label}: {row[f'Rotation {cfg.long_label}']:.2%}<br>"
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
                mode="markers+text",
                text=[ticker],
                textposition="top center",
                marker=dict(
                    size=float(row["MarkerSize"]),
                    color=row["Color"],
                    line=dict(width=2.0, color=border_color),
                    opacity=0.92,
                ),
                name=ticker,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    x_min = float(snap[x_col].min())
    x_max = float(snap[x_col].max())
    y_min = float(snap[y_col].min())
    y_max = float(snap[y_col].max())

    x_pad = max(0.02, abs(x_max - x_min) * 0.20, snap[x_col].abs().max() * 0.15)
    y_pad = max(0.02, abs(y_max - y_min) * 0.20, snap[y_col].abs().max() * 0.15)

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
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(
            title=f"{cfg.long_label} rotation return",
            tickformat=".1%",
            zeroline=False,
            range=[x_min - x_pad, x_max + x_pad],
        ),
        yaxis=dict(
            title=f"{cfg.short_label} rotation return",
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

    table["Quadrant Δ"] = np.where(
        table["Prior Quadrant"].fillna("") == table["Quadrant"].fillna(""),
        "",
        table["Prior Quadrant"].fillna("") + " → " + table["Quadrant"].fillna(""),
    )

    table["Rank Δ"] = table["Rank Δ"].apply(format_delta)

    cols = [
        "Rank",
        "Ticker",
        "Sector",
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


def last_valid(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def pct_change_value(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-periods - 1] - 1.0)


def ratio_pct_change(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    periods: int,
) -> float:
    if numerator not in prices.columns or denominator not in prices.columns:
        return np.nan

    ratio = prices[numerator].div(prices[denominator]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) <= periods:
        return np.nan

    return float(ratio.iloc[-1] / ratio.iloc[-periods - 1] - 1.0)


def moving_average_stack(series: pd.Series, windows: List[int]) -> Tuple[int, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0, len(windows)

    last = float(s.iloc[-1])
    above = 0

    for window in windows:
        if len(s) < window:
            continue

        ma = float(s.rolling(window).mean().iloc[-1])
        if last > ma:
            above += 1

    return above, len(windows)


def make_market_context(prices: pd.DataFrame, benchmark_ticker: str) -> List[Dict[str, str]]:
    metrics = []

    if benchmark_ticker in prices.columns:
        s = prices[benchmark_ticker].dropna()
        last = last_valid(s)
        chg_5d = pct_change_value(s, 5)
        above, total = moving_average_stack(s, [21, 50, 100, 200])

        metrics.append(
            {
                "label": f"{benchmark_ticker} Trend",
                "value": f"{above}/{total} MAs",
                "delta": f"{chg_5d:+.2%} 5D" if not pd.isna(chg_5d) else "",
            }
        )

    if "RSP" in prices.columns and "SPY" in prices.columns:
        rsp_spy_20d = ratio_pct_change(prices, "RSP", "SPY", 20)
        metrics.append(
            {
                "label": "Equal Weight",
                "value": "RSP / SPY",
                "delta": f"{rsp_spy_20d:+.2%} 20D" if not pd.isna(rsp_spy_20d) else "",
            }
        )

    if "QQQ" in prices.columns and "SPY" in prices.columns:
        qqq_spy_20d = ratio_pct_change(prices, "QQQ", "SPY", 20)
        metrics.append(
            {
                "label": "Growth Tilt",
                "value": "QQQ / SPY",
                "delta": f"{qqq_spy_20d:+.2%} 20D" if not pd.isna(qqq_spy_20d) else "",
            }
        )

    if "TLT" in prices.columns and "SPY" in prices.columns:
        tlt_spy_20d = ratio_pct_change(prices, "TLT", "SPY", 20)
        metrics.append(
            {
                "label": "Duration Tilt",
                "value": "TLT / SPY",
                "delta": f"{tlt_spy_20d:+.2%} 20D" if not pd.isna(tlt_spy_20d) else "",
            }
        )

    if "^VIX" in prices.columns:
        vix = prices["^VIX"].dropna()
        vix_last = last_valid(vix)

        if len(vix) > 5:
            vix_5d_change = float(vix.iloc[-1] - vix.iloc[-6])
        else:
            vix_5d_change = np.nan

        metrics.append(
            {
                "label": "Volatility",
                "value": f"{vix_last:.1f}" if not pd.isna(vix_last) else "",
                "delta": f"{vix_5d_change:+.1f} pts 5D" if not pd.isna(vix_5d_change) else "",
            }
        )

    return metrics


def make_quadrant_counts(snap: pd.DataFrame) -> pd.DataFrame:
    order = ["Leading", "Improving", "Weakening", "Lagging", "Neutral"]
    counts = snap["Quadrant"].value_counts().reindex(order).fillna(0).astype(int)
    return counts.reset_index().rename(columns={"index": "Quadrant", "Quadrant": "Count"})


def make_score_bar_chart(snap: pd.DataFrame) -> go.Figure:
    df = snap.sort_values("Composite Score", ascending=True).copy()

    colors = []
    for value in df["Composite Score"]:
        if value >= 65:
            colors.append(PASTEL_GREEN)
        elif value <= 35:
            colors.append(PASTEL_RED)
        else:
            colors.append(PASTEL_GREY)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Composite Score"],
            y=df["Ticker"],
            orientation="h",
            marker=dict(color=colors),
            text=df["Composite Score"].round(1),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Composite Score: %{x:.1f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Composite Leadership Score",
        height=390,
        margin=dict(l=10, r=30, t=55, b=10),
        xaxis=dict(title="Score", range=[0, 105]),
        yaxis=dict(title=""),
        showlegend=False,
    )

    return fig


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
        "Speed": "{:.2%}",
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

all_tickers = list(
    dict.fromkeys(
        list(SECTORS.keys())
        + list(BENCHMARKS.keys())
        + list(MARKET_CONTEXT.keys())
    )
)

with st.spinner("Loading market data..."):
    prices = fetch_prices(all_tickers)

required_tickers = list(SECTORS.keys()) + [benchmark]
ok, missing = validate_prices(prices, required_tickers)

if not ok:
    st.error(f"Data unavailable for: {', '.join(missing)}")
    st.stop()

prices = prices.sort_index().ffill()

min_required_rows = cfg.long_window + 20
if len(prices) < min_required_rows:
    st.error(f"Not enough data for the selected window. Need at least {min_required_rows} rows.")
    st.stop()


# =============================================================================
# Analytics
# =============================================================================

sector_tickers = list(SECTORS.keys())

rs = compute_relative_strength(prices, sector_tickers, benchmark)

snap = build_snapshot(
    prices=prices,
    sectors=SECTORS,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
)

rank_delta = compute_rank_delta(
    prices=prices,
    sectors=SECTORS,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
    compare_days=5,
)

trail_points = build_rotation_trails(
    prices=prices,
    sectors=SECTORS,
    benchmark_ticker=benchmark,
    cfg=cfg,
    trail_weeks=TRAIL_OPTIONS[trail_choice],
    rotation_basis=rotation_mode,
)

summary_text = make_summary_text(
    snap=snap,
    cfg=cfg,
    benchmark_ticker=benchmark,
    rotation_basis=rotation_mode,
)

display_df = build_display_table(snap, rank_delta, cfg)
diagnostics_df = build_price_diagnostics(prices, all_tickers)
context_metrics = make_market_context(prices, benchmark)


# =============================================================================
# Market context
# =============================================================================

st.subheader("Market structure")

if context_metrics:
    context_cols = st.columns(len(context_metrics))

    for col, metric in zip(context_cols, context_metrics):
        col.metric(
            metric["label"],
            metric["value"],
            metric["delta"],
        )

if show_diagnostics:
    with st.expander("Data diagnostics", expanded=True):
        st.dataframe(
            diagnostics_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Top dashboard
# =============================================================================

leader_row = snap.sort_values("Composite Score", ascending=False).iloc[0]
worst_row = snap.sort_values("Composite Score", ascending=True).iloc[0]
best_rs_row = snap.sort_values("RS Slope 10D", ascending=False).iloc[0]
worst_rs_row = snap.sort_values("RS Slope 10D", ascending=True).iloc[0]
leading_count = int((snap["Quadrant"] == "Leading").sum())
accelerating_count = int((snap["RS Accel"] > 0).sum())

st.subheader("Sector tape")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Top Composite",
    f"{leader_row['Ticker']}",
    f"{leader_row['Composite Score']:.1f}",
)

c2.metric(
    "Weakest Composite",
    f"{worst_row['Ticker']}",
    f"{worst_row['Composite Score']:.1f}",
)

c3.metric(
    "Best RS Slope",
    f"{best_rs_row['Ticker']}",
    f"{best_rs_row['RS Slope 10D']:.2%}",
)

c4.metric(
    "Weakest RS Slope",
    f"{worst_rs_row['Ticker']}",
    f"{worst_rs_row['RS Slope 10D']:.2%}",
)

c5.metric(
    "Breadth / Accel",
    f"{leading_count} leading",
    f"{accelerating_count} accelerating",
)

st.markdown(
    f"""
    <div style="
        padding: 0.75rem 0.95rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 0.65rem;
        margin-bottom: 0.85rem;
        line-height: 1.45;
    ">
        <b>Regime summary:</b> {summary_text}
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Main charts
# =============================================================================

left, right = st.columns([1.03, 1.17])

with left:
    st.subheader("Relative strength")

    rs_fig = make_rs_chart(
        rs_series=rs[rs_sector].dropna(),
        sector_name=f"{SECTORS[rs_sector]} ({rs_sector})",
        benchmark_ticker=benchmark,
    )

    st.plotly_chart(rs_fig, use_container_width=True)

    score_fig = make_score_bar_chart(snap)
    st.plotly_chart(score_fig, use_container_width=True)

with right:
    st.subheader("Rotation map")

    rot_fig = make_rotation_scatter(
        snap=snap,
        cfg=cfg,
        trail_points=trail_points,
        rotation_basis=rotation_mode,
        benchmark_ticker=benchmark,
    )

    st.plotly_chart(rot_fig, use_container_width=True)


# =============================================================================
# Snapshot table
# =============================================================================

st.subheader("Rotation snapshot")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.1, 1.4, 1.1, 1.0])

with filter_col1:
    sort_col = st.selectbox(
        "Sort by",
        options=[
            "Rank",
            "Ticker",
            "Sector",
            "Quadrant",
            "State",
            "Composite Score",
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
        ],
        index=5,
    )

with filter_col2:
    quadrant_options = [
        q
        for q in ["Leading", "Improving", "Weakening", "Lagging", "Neutral"]
        if q in display_df["Quadrant"].unique()
    ]

    selected_quadrants = st.multiselect(
        "Quadrant filter",
        options=quadrant_options,
        default=quadrant_options,
    )

with filter_col3:
    state_options = sorted(display_df["State"].dropna().unique().tolist())

    selected_states = st.multiselect(
        "State filter",
        options=state_options,
        default=state_options,
    )

with filter_col4:
    ascending = st.toggle("Ascending", value=False)
    only_accelerating = st.toggle("RS accel only", value=False)

table_filtered = display_df.copy()

if selected_quadrants:
    table_filtered = table_filtered[table_filtered["Quadrant"].isin(selected_quadrants)]

if selected_states:
    table_filtered = table_filtered[table_filtered["State"].isin(selected_states)]

if only_accelerating:
    table_filtered = table_filtered[table_filtered["RS Accel"] > 0]

if table_filtered.empty:
    st.info("No sectors match the current filter selection.")
else:
    table_sorted = table_filtered.sort_values(
        sort_col,
        ascending=ascending,
    ).reset_index(drop=True)

    row_height = 35
    table_height = min(720, max(160, row_height * (len(table_sorted) + 1)))

    st.dataframe(
        style_snapshot_table(table_sorted, cfg),
        use_container_width=True,
        height=table_height,
        hide_index=True,
    )


# =============================================================================
# Quadrant counts and notes
# =============================================================================

q_left, q_right = st.columns([0.85, 1.15])

with q_left:
    st.subheader("Quadrant counts")
    quadrant_counts = make_quadrant_counts(snap)
    st.dataframe(
        quadrant_counts,
        use_container_width=True,
        hide_index=True,
    )

with q_right:
    st.subheader("Interpretation guide")
    st.markdown(
        f"""
        - **Leading:** positive {cfg.short_label} and {cfg.long_label} rotation.
        - **Improving:** negative {cfg.long_label} rotation, positive {cfg.short_label} rotation.
        - **Weakening:** positive {cfg.long_label} rotation, negative {cfg.short_label} rotation.
        - **Lagging:** negative {cfg.short_label} and {cfg.long_label} rotation.
        - **Composite Score:** weighted score using RS {cfg.short_label}, RS {cfg.long_label}, 10-day RS slope, RS acceleration, and absolute participation.
        """
    )


# =============================================================================
# Footer
# =============================================================================

coverage = len([c for c in sector_tickers if c in prices.columns and not prices[c].dropna().empty])
latest_dt = pd.to_datetime(prices.index.max()).date()

st.caption(
    f"Data through: {latest_dt} | Benchmark: {benchmark} | Rotation mode: {rotation_mode_label} | "
    f"Coverage: {coverage}/{len(sector_tickers)} sector ETFs"
)

st.caption("© 2026 AD Fund Management LP")
