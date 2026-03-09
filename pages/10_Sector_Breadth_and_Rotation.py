# sp_sector_rotation_monitor_v7.py
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

# ------------------------------- Page config -------------------------------

st.set_page_config(
    page_title="S&P 500 Sector Breadth & Rotation Monitor",
    layout="wide",
)

st.title("S&P 500 Sector Breadth & Rotation Monitor")

# ------------------------------- Constants ---------------------------------

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

LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"

WINDOW_PRESETS = {
    "Fast (1M vs 3M)": {"short": 21, "long": 63, "label_short": "1M", "label_long": "3M"},
    "Intermediate (3M vs 6M)": {"short": 63, "long": 126, "label_short": "3M", "label_long": "6M"},
    "Trend (6M vs 12M)": {"short": 126, "long": 252, "label_short": "6M", "label_long": "12M"},
}

TRAIL_OPTIONS = {
    "None": 0,
    "4 weeks": 4,
    "8 weeks": 8,
    "12 weeks": 12,
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

# ------------------------------- Sidebar -----------------------------------

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Purpose: Sector breadth and rotation monitor for leadership, participation, and regime shifts.

        What it covers
        • Core signals and summary outputs for this dashboard
        • Key context needed to interpret current regime or setup
        • Practical view designed for quick internal decision support

        Data source
        • Public market and macro data feeds used throughout the app
        """
    )

    benchmark = st.selectbox(
        "Benchmark",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{x} | {BENCHMARKS[x]}",
        index=0,
    )

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

# ------------------------------- Data classes ------------------------------

@dataclass
class RotationConfig:
    short_window: int
    long_window: int
    short_label: str
    long_label: str


# ------------------------------- Helpers -----------------------------------

def _normalize_download(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Normalize yfinance download output into a clean wide dataframe of adjusted closes.
    Handles both single-index and MultiIndex outputs.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        candidates = ["Adj Close", "Close"]
        out = {}
        for t in tickers:
            for field in candidates:
                try:
                    s = data[(field, t)].dropna()
                    if not s.empty:
                        out[t] = s
                        break
                except Exception:
                    continue
        return pd.DataFrame(out)

    if len(tickers) == 1:
        for field in ["Adj Close", "Close"]:
            if field in data.columns:
                return pd.DataFrame({tickers[0]: data[field].dropna()})

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_prices(tickers: List[str], period: str = LOOKBACK_PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    prices = _normalize_download(raw, tickers)

    if prices.empty:
        return prices

    prices = prices.sort_index().ffill()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def validate_prices(prices: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    missing = [t for t in required if t not in prices.columns]
    if prices.empty:
        return False, required
    return len(missing) == 0, missing


def get_rotation_config(choice: str) -> RotationConfig:
    cfg = WINDOW_PRESETS[choice]
    return RotationConfig(
        short_window=cfg["short"],
        long_window=cfg["long"],
        short_label=cfg["label_short"],
        long_label=cfg["label_long"],
    )


def compute_relative_strength(prices: pd.DataFrame, sector_tickers: List[str], benchmark_ticker: str) -> pd.DataFrame:
    base = prices[sector_tickers].div(prices[benchmark_ticker], axis=0)
    return base.dropna(how="all")


def compute_rs_zscore(rs: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rolling_mean = rs.rolling(window).mean()
    rolling_std = rs.rolling(window).std().replace(0, np.nan)
    z = (rs - rolling_mean) / rolling_std
    return z


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


def build_snapshot(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    benchmark: str,
    cfg: RotationConfig,
) -> pd.DataFrame:
    sector_tickers = list(sectors.keys())
    rs = compute_relative_strength(prices, sector_tickers, benchmark)
    rs_z = compute_rs_zscore(rs, window=min(63, max(20, cfg.long_window)))

    short_ret = prices[sector_tickers].pct_change(cfg.short_window).iloc[-1]
    long_ret = prices[sector_tickers].pct_change(cfg.long_window).iloc[-1]

    rs_short = rs.pct_change(cfg.short_window).iloc[-1]
    rs_long = rs.pct_change(cfg.long_window).iloc[-1]

    rs_slope_10d = (rs.iloc[-1] / rs.shift(10).iloc[-1] - 1.0).replace([np.inf, -np.inf], np.nan)
    rs_slope_5d = (rs.iloc[-1] / rs.shift(5).iloc[-1] - 1.0).replace([np.inf, -np.inf], np.nan)
    accel = (rs_slope_5d - rs_slope_10d).replace([np.inf, -np.inf], np.nan)

    angle = np.degrees(np.arctan2(short_ret, long_ret))
    speed = np.sqrt(short_ret.pow(2) + long_ret.pow(2))
    rs_z_last = rs_z.iloc[-1]

    snap = pd.DataFrame({
        "Ticker": sector_tickers,
        "Sector": [sectors[t] for t in sector_tickers],
        cfg.short_label: short_ret.values,
        cfg.long_label: long_ret.values,
        "RS " + cfg.short_label: rs_short.values,
        "RS " + cfg.long_label: rs_long.values,
        "RS Slope 10D": rs_slope_10d.values,
        "RS Slope 5D": rs_slope_5d.values,
        "Accel": accel.values,
        "Angle (deg)": angle.values,
        "Speed": speed.values,
        "RS Z": rs_z_last.values,
    })

    snap["QuadrantCode"] = [
        classify_quadrant(lr, sr)
        for lr, sr in zip(snap[cfg.long_label], snap[cfg.short_label])
    ]
    snap["Quadrant"] = snap["QuadrantCode"].map(QUADRANT_LABELS)

    snap["State"] = np.select(
        [
            (snap["Quadrant"] == "Leading") & (snap["Accel"] > 0),
            (snap["Quadrant"] == "Leading") & (snap["Accel"] <= 0),
            (snap["Quadrant"] == "Improving") & (snap["Accel"] > 0),
            (snap["Quadrant"] == "Weakening") & (snap["Accel"] < 0),
            (snap["Quadrant"] == "Lagging") & (snap["Accel"] > 0),
        ],
        [
            "Leading and accelerating",
            "Leading but cooling",
            "Improving fast",
            "Weakening fast",
            "Lagging but stabilizing",
        ],
        default=snap["Quadrant"],
    )

    snap["MarkerSize"] = 18 + (snap["Speed"].rank(pct=True).fillna(0.5) * 20)
    snap["Color"] = snap["Ticker"].map(SECTOR_COLORS)

    snap = snap.sort_values("Speed", ascending=False).reset_index(drop=True)
    snap["Rank"] = np.arange(1, len(snap) + 1)

    return snap


def compute_rank_delta(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    benchmark: str,
    cfg: RotationConfig,
    compare_days: int = 5,
) -> pd.DataFrame:
    if len(prices) <= cfg.long_window + compare_days + 5:
        return pd.DataFrame(columns=["Ticker", "Prior Rank", "Rank Δ", "Prior Quadrant"])

    past_prices = prices.iloc[:-compare_days].copy()
    current_prices = prices.copy()

    past_snap = build_snapshot(past_prices, sectors, benchmark, cfg)[["Ticker", "Rank", "Quadrant"]].rename(
        columns={"Rank": "Prior Rank", "Quadrant": "Prior Quadrant"}
    )
    curr_snap = build_snapshot(current_prices, sectors, benchmark, cfg)[["Ticker", "Rank", "Quadrant"]]

    merged = curr_snap.merge(past_snap, on="Ticker", how="left")
    merged["Rank Δ"] = merged["Prior Rank"] - merged["Rank"]
    return merged[["Ticker", "Prior Rank", "Rank Δ", "Prior Quadrant"]]


def build_rotation_trails(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    cfg: RotationConfig,
    trail_weeks: int,
) -> Dict[str, pd.DataFrame]:
    if trail_weeks <= 0:
        return {}

    sector_tickers = list(sectors.keys())
    trail_points = {}
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")

    for t in sector_tickers:
        if t not in weekly_prices.columns:
            continue
        short_ret = weekly_prices[t].pct_change(max(1, round(cfg.short_window / 5)))
        long_ret = weekly_prices[t].pct_change(max(2, round(cfg.long_window / 5)))

        df = pd.DataFrame({
            "x": long_ret,
            "y": short_ret,
        }).dropna()

        if df.empty:
            continue

        trail_points[t] = df.tail(trail_weeks)

    return trail_points


def make_summary_text(snap: pd.DataFrame, cfg: RotationConfig, benchmark: str) -> str:
    if snap.empty:
        return "No valid sector data available."

    leaders = snap[snap["Quadrant"] == "Leading"].sort_values("Speed", ascending=False)
    improvers = snap[snap["Quadrant"] == "Improving"].sort_values("Accel", ascending=False)
    weakeners = snap[snap["Quadrant"] == "Weakening"].sort_values("Accel")
    laggards = snap[snap["Quadrant"] == "Lagging"].sort_values("Speed")

    leader_txt = ", ".join(leaders["Ticker"].head(3).tolist()) if not leaders.empty else "none"
    improver_txt = ", ".join(improvers["Ticker"].head(2).tolist()) if not improvers.empty else "none"
    weakener_txt = ", ".join(weakeners["Ticker"].head(2).tolist()) if not weakeners.empty else "none"
    laggard_txt = ", ".join(laggards["Ticker"].head(2).tolist()) if not laggards.empty else "none"

    breadth_count = int((snap["RS Slope 10D"] > 0).sum())
    accel_count = int((snap["Accel"] > 0).sum())

    text = (
        f"Leadership versus {benchmark} is concentrated in {leader_txt}. "
        f"The main improvement bucket is {improver_txt}, while near-term deterioration is showing up most clearly in {weakener_txt}. "
        f"Laggards remain {laggard_txt}. Across the full sector set, {breadth_count} of {len(snap)} sectors have a positive 10-day RS slope "
        f"and {accel_count} of {len(snap)} are accelerating on a 5-day versus 10-day basis. "
        f"Read together, this gives you both the current state and whether rotation is broadening or narrowing beneath the surface."
    )
    return text


def make_rs_chart(rs_series: pd.Series, sector_name: str, benchmark: str) -> go.Figure:
    rs_20 = rs_series.rolling(20).mean()
    rs_50 = rs_series.rolling(50).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rs_series.index,
        y=rs_series.values,
        mode="lines",
        name="RS Ratio",
        line=dict(width=2.4, color="#1f77b4"),
        hovertemplate="%{x|%Y-%m-%d}<br>RS: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=rs_20.index,
        y=rs_20.values,
        mode="lines",
        name="20D MA",
        line=dict(width=1.4, dash="dot", color="#2ca02c"),
        hovertemplate="%{x|%Y-%m-%d}<br>20D MA: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=rs_50.index,
        y=rs_50.values,
        mode="lines",
        name="50D MA",
        line=dict(width=1.4, dash="dash", color="#9467bd"),
        hovertemplate="%{x|%Y-%m-%d}<br>50D MA: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Relative Strength | {sector_name} vs {benchmark}",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Date",
        yaxis_title="RS Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    return fig


def make_rotation_scatter(
    snap: pd.DataFrame,
    cfg: RotationConfig,
    trail_points: Dict[str, pd.DataFrame],
) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_width=1, opacity=0.45)

    for t, trail_df in trail_points.items():
        if t not in snap["Ticker"].values or trail_df.empty:
            continue

        color = SECTOR_COLORS.get(t, "#4b5563")
        fig.add_trace(go.Scatter(
            x=trail_df["x"],
            y=trail_df["y"],
            mode="lines+markers",
            line=dict(color=color, width=1.4),
            marker=dict(size=5, color=color, opacity=0.6),
            opacity=0.45,
            hoverinfo="skip",
            showlegend=False,
        ))

    for _, r in snap.iterrows():
        border_color = STATE_COLORS.get(r["Quadrant"], "#111827")

        fig.add_trace(go.Scatter(
            x=[r[cfg.long_label]],
            y=[r[cfg.short_label]],
            mode="markers+text",
            text=[r["Ticker"]],
            textposition="top center",
            marker=dict(
                size=float(r["MarkerSize"]),
                color=r["Color"],
                line=dict(width=2.0, color=border_color),
                opacity=0.9,
            ),
            name=r["Ticker"],
            hovertemplate=(
                f"<b>{r['Sector']} ({r['Ticker']})</b><br>"
                f"{cfg.short_label}: {r[cfg.short_label]:.2%}<br>"
                f"{cfg.long_label}: {r[cfg.long_label]:.2%}<br>"
                f"RS {cfg.short_label}: {r['RS ' + cfg.short_label]:.2%}<br>"
                f"RS {cfg.long_label}: {r['RS ' + cfg.long_label]:.2%}<br>"
                f"RS Slope 10D: {r['RS Slope 10D']:.2%}<br>"
                f"Accel: {r['Accel']:.2%}<br>"
                f"Speed: {r['Speed']:.2%}<br>"
                f"Quadrant: {r['Quadrant']}<br>"
                f"State: {r['State']}<extra></extra>"
            ),
            showlegend=False,
        ))

    x_pad = max(0.02, snap[cfg.long_label].abs().max() * 0.20)
    y_pad = max(0.02, snap[cfg.short_label].abs().max() * 0.20)

    fig.add_annotation(
        x=snap[cfg.long_label].max() + x_pad * 0.45,
        y=snap[cfg.short_label].max() + y_pad * 0.45,
        text="Leading",
        showarrow=False,
        font=dict(color=STATE_COLORS["Leading"], size=12),
    )
    fig.add_annotation(
        x=snap[cfg.long_label].min() - x_pad * 0.45,
        y=snap[cfg.short_label].max() + y_pad * 0.45,
        text="Improving",
        showarrow=False,
        font=dict(color=STATE_COLORS["Improving"], size=12),
    )
    fig.add_annotation(
        x=snap[cfg.long_label].min() - x_pad * 0.45,
        y=snap[cfg.short_label].min() - y_pad * 0.45,
        text="Lagging",
        showarrow=False,
        font=dict(color=STATE_COLORS["Lagging"], size=12),
    )
    fig.add_annotation(
        x=snap[cfg.long_label].max() + x_pad * 0.45,
        y=snap[cfg.short_label].min() - y_pad * 0.45,
        text="Weakening",
        showarrow=False,
        font=dict(color=STATE_COLORS["Weakening"], size=12),
    )

    fig.update_layout(
        title=f"Rotation Map | {cfg.short_label} vs {cfg.long_label}",
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(
            title=f"{cfg.long_label} total return",
            tickformat=".1%",
            zeroline=False,
            range=[snap[cfg.long_label].min() - x_pad, snap[cfg.long_label].max() + x_pad],
        ),
        yaxis=dict(
            title=f"{cfg.short_label} total return",
            tickformat=".1%",
            zeroline=False,
            range=[snap[cfg.short_label].min() - y_pad, snap[cfg.short_label].max() + y_pad],
        ),
    )

    return fig


def format_delta(val: float) -> str:
    if pd.isna(val):
        return ""
    sign = "+" if val > 0 else ""
    return f"{sign}{int(val)}"


def build_display_table(snap: pd.DataFrame, rank_delta: pd.DataFrame, cfg: RotationConfig) -> pd.DataFrame:
    table = snap.merge(rank_delta, on="Ticker", how="left")

    table["Quadrant Δ"] = np.where(
        table["Prior Quadrant"].fillna("") == table["Quadrant"].fillna(""),
        "",
        table["Prior Quadrant"].fillna("") + " → " + table["Quadrant"].fillna("")
    )

    table["Rank Δ"] = table["Rank Δ"].apply(format_delta)

    cols = [
        "Rank", "Ticker", "Sector", "Quadrant", "State", "Rank Δ", "Quadrant Δ",
        cfg.short_label, cfg.long_label,
        "RS " + cfg.short_label, "RS " + cfg.long_label,
        "RS Slope 10D", "Accel", "RS Z", "Angle (deg)", "Speed"
    ]
    out = table[cols].copy()
    return out


# ------------------------------- Load data ---------------------------------

all_tickers = list(SECTORS.keys()) + list(BENCHMARKS.keys())
prices = fetch_prices(all_tickers)

ok, missing = validate_prices(prices, list(SECTORS.keys()) + [benchmark])
if not ok:
    st.error(f"Data unavailable for: {', '.join(missing)}")
    st.stop()

# Keep only columns we need, but retain all benchmarks already downloaded for flexibility.
prices = prices.sort_index().ffill()

cfg = get_rotation_config(window_choice)
min_required_rows = cfg.long_window + 15
if len(prices) < min_required_rows:
    st.error(f"Not enough data for the selected window. Need at least {min_required_rows} rows.")
    st.stop()

# ------------------------------- Analytics ---------------------------------

sector_tickers = list(SECTORS.keys())
rs = compute_relative_strength(prices, sector_tickers, benchmark)
snap = build_snapshot(prices, SECTORS, benchmark, cfg)
rank_delta = compute_rank_delta(prices, SECTORS, benchmark, cfg, compare_days=5)
trail_points = build_rotation_trails(
    prices=prices[list(SECTORS.keys())],
    sectors=SECTORS,
    cfg=cfg,
    trail_weeks=TRAIL_OPTIONS[trail_choice],
)
summary_text = make_summary_text(snap, cfg, benchmark)
display_df = build_display_table(snap, rank_delta, cfg)

# ------------------------------- Top dashboard -----------------------------

leader_row = snap.sort_values("Speed", ascending=False).iloc[0]
worst_row = snap.sort_values(cfg.short_label, ascending=True).iloc[0]
best_rs_row = snap.sort_values("RS Slope 10D", ascending=False).iloc[0]
worst_rs_row = snap.sort_values("RS Slope 10D", ascending=True).iloc[0]
breadth_thrust = int((snap["Quadrant"] == "Leading").sum())
accelerating = int((snap["Accel"] > 0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Top Sector", f"{leader_row['Ticker']}", f"{leader_row['Speed']:.2%} speed")
c2.metric("Worst Short-Term", f"{worst_row['Ticker']}", f"{worst_row[cfg.short_label]:.2%}")
c3.metric("Best RS Slope", f"{best_rs_row['Ticker']}", f"{best_rs_row['RS Slope 10D']:.2%}")
c4.metric("Weakest RS Slope", f"{worst_rs_row['Ticker']}", f"{worst_rs_row['RS Slope 10D']:.2%}")
c5.metric("Breadth / Accel", f"{breadth_thrust} leading", f"{accelerating} accelerating")

st.markdown(
    f"""
    <div style="padding: 0.65rem 0.9rem; border: 1px solid rgba(128,128,128,0.25); border-radius: 0.6rem; margin-bottom: 0.75rem;">
        <b>Regime summary:</b> {summary_text}
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Main charts -------------------------------

left, right = st.columns([1.02, 1.18])

with left:
    st.subheader("Relative strength")
    rs_fig = make_rs_chart(
        rs_series=rs[rs_sector].dropna(),
        sector_name=f"{SECTORS[rs_sector]} ({rs_sector})",
        benchmark=benchmark,
    )
    st.plotly_chart(rs_fig, use_container_width=True)

with right:
    st.subheader("Rotation map")
    rot_fig = make_rotation_scatter(snap=snap, cfg=cfg, trail_points=trail_points)
    st.plotly_chart(rot_fig, use_container_width=True)

# ------------------------------- Snapshot table ----------------------------

st.subheader("Rotation snapshot")

sort_col = st.selectbox(
    "Sort by",
    options=[
        "Rank", "Ticker", "Sector", "Quadrant", "State",
        cfg.short_label, cfg.long_label,
        "RS " + cfg.short_label, "RS " + cfg.long_label,
        "RS Slope 10D", "Accel", "RS Z", "Angle (deg)", "Speed"
    ],
    index=13,
)

ascending = st.toggle("Ascending", value=False)

table_sorted = display_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

fmt_map = {
    cfg.short_label: "{:.2%}",
    cfg.long_label: "{:.2%}",
    "RS " + cfg.short_label: "{:.2%}",
    "RS " + cfg.long_label: "{:.2%}",
    "RS Slope 10D": "{:.2%}",
    "Accel": "{:.2%}",
    "RS Z": "{:.2f}",
    "Angle (deg)": "{:.1f}",
    "Speed": "{:.2%}",
}

styler = (
    table_sorted.style
    .format(fmt_map)
    .background_gradient(subset=[cfg.short_label, cfg.long_label], cmap="RdYlGn")
    .background_gradient(subset=["RS Slope 10D", "Accel"], cmap="RdYlGn")
    .background_gradient(subset=["Speed"], cmap="PuBuGn")
)

row_height = 35
table_height = row_height * (len(table_sorted) + 1)

st.dataframe(
    styler,
    use_container_width=True,
    height=table_height,
    hide_index=True,
)

# ------------------------------- Footer ------------------------------------

coverage = len([c for c in sector_tickers if c in prices.columns])
latest_dt = pd.to_datetime(prices.index.max()).date()

st.caption(
    f"Data through: {latest_dt} | Benchmark: {benchmark} | Coverage: {coverage}/{len(sector_tickers)} sector ETFs"
)
st.caption("© 2026 AD Fund Management LP")
