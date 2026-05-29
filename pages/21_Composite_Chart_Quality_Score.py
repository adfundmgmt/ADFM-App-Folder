from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# Page config and styling
# ============================================================
st.set_page_config(page_title="Composite Chart Quality Score", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
    .meta {color: #6b7280; font-size: 0.90rem; margin-top: -0.35rem; margin-bottom: 1rem;}
    .section-rule {border: 0; border-top: 1px solid #e5e7eb; margin: 1.25rem 0 1.1rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TITLE = "Composite Chart Quality Score"
SUBTITLE = "Daily technical quality screen powered by the CCQS production cache."

CCQS_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "adfundmgmt/Composite-Chart-Quality-Score/main/data/cache/dashboard"
)

TTL = 60 * 30

STATE_LABELS = {
    "TRENDING": "Trending",
    "PULLBACK": "Pullback",
    "CONSOLIDATING": "Consolidating",
    "EXHAUSTION": "Parabolic",
    "DETERIORATING": "Breaking Down",
    "INDETERMINATE": "No Edge",
    "UNCLASSIFIED": "No Edge",
}

TIER_LABELS = {
    "ELITE_LEADER": "Elite Leader",
    "STRONG_LEADER": "Strong Leader",
    "ESTABLISHED_LEADER": "Established Leader",
    "EMERGING_LEADER": "Emerging Leader",
    "STRONG_PERFORMER": "Steady",
    "NEUTRAL": "Neutral",
    "WEAK_PERFORMER": "Weak Performer",
    "DETERIORATING": "Fading Leader",
    "WEAK_LAGGARD": "Weak Laggard",
    "UNCLASSIFIED": "No RS Signal",
}

COMPONENT_DISPLAY_NAMES = {
    "s_rs": "Relative Strength",
    "s_rs_leadership": "Relative Strength Leadership",
    "s_residual_momentum": "Residual Momentum",
    "s_rsl": "Relative Strength Line",
    "s_trend_slope": "Trend Slope",
    "s_structure": "Structure",
    "s_mtf": "Multi-Timeframe Alignment",
    "s_extension": "Extension",
    "s_momentum": "Momentum",
    "s_volume": "Volume Pattern",
}

PERIOD_DAYS = {
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 1095,
    "5Y": 1825,
    "INCEPTION": None,
}


# ============================================================
# Cached data loaders
# ============================================================
@st.cache_data(ttl=TTL, show_spinner=False)
def read_remote_parquet(name: str) -> pd.DataFrame:
    url = f"{CCQS_RAW_BASE}/{name}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_parquet(BytesIO(response.content))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def read_remote_json(name: str) -> dict:
    url = f"{CCQS_RAW_BASE}/{name}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def _date_level(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        return "date"
    return None


def _ticker_level(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex) and "ticker" in df.index.names:
        return "ticker"
    return None


def latest_snapshot(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    level = _date_level(df)
    if df.empty or level is None:
        return pd.DataFrame(), None

    latest = pd.Timestamp(df.index.get_level_values(level).max())
    latest_df = df.xs(latest, level=level).copy()
    return latest_df, latest


def series_delta_by_trading_days(ccqs: pd.DataFrame, latest: pd.Timestamp, days: int) -> pd.Series:
    if ccqs.empty or "ccqs" not in ccqs.columns:
        return pd.Series(dtype=float)

    level = _date_level(ccqs)
    if level is None:
        return pd.Series(dtype=float)

    dates = pd.Index(ccqs.index.get_level_values(level).unique()).sort_values()
    if latest not in dates or len(dates) <= days:
        return pd.Series(dtype=float)

    prior = dates[dates.get_loc(latest) - days]
    current_series = ccqs.xs(latest, level=level)["ccqs"].astype(float)
    prior_series = ccqs.xs(prior, level=level)["ccqs"].astype(float)
    return current_series - prior_series


@st.cache_data(ttl=TTL, show_spinner=False)
def load_dashboard_snapshot() -> Tuple[pd.DataFrame, str]:
    ccqs = read_remote_parquet("ccqs.parquet")
    if ccqs.empty:
        return pd.DataFrame(), ""

    states = read_remote_parquet("state.parquet")
    leadership = read_remote_parquet("leadership.parquet")
    setups = read_remote_parquet("setups.parquet")
    features = read_remote_parquet("features.parquet")

    ccqs_latest, latest = latest_snapshot(ccqs)
    if ccqs_latest.empty or latest is None:
        return pd.DataFrame(), ""

    state_latest, _ = latest_snapshot(states)
    leadership_latest, _ = latest_snapshot(leadership)
    setup_latest, _ = latest_snapshot(setups)
    feature_latest, _ = latest_snapshot(features)

    change_1d = series_delta_by_trading_days(ccqs, latest, 1)
    change_5d = series_delta_by_trading_days(ccqs, latest, 5)
    change_21d = series_delta_by_trading_days(ccqs, latest, 21)

    out = pd.DataFrame(index=ccqs_latest.index)
    out["ccqs"] = pd.to_numeric(ccqs_latest.get("ccqs"), errors="coerce")
    out["grade"] = ccqs_latest.get("grade", pd.Series("", index=ccqs_latest.index)).astype(str)

    if not leadership_latest.empty and "leadership_tier" in leadership_latest.columns:
        out["leadership_tier"] = leadership_latest["leadership_tier"].reindex(out.index).astype(str)
    else:
        out["leadership_tier"] = "UNCLASSIFIED"

    if not state_latest.empty and "primary_state" in state_latest.columns:
        out["primary_state"] = state_latest["primary_state"].reindex(out.index).astype(str)
    else:
        out["primary_state"] = "INDETERMINATE"

    if not setup_latest.empty:
        if "setup_label" in setup_latest.columns:
            out["setup"] = setup_latest["setup_label"].reindex(out.index).astype(str)
        elif "setup" in setup_latest.columns:
            out["setup"] = setup_latest["setup"].reindex(out.index).astype(str)
        else:
            out["setup"] = ""
    else:
        out["setup"] = ""

    if not feature_latest.empty:
        for col in ["rs_rating_spy", "information_ratio_252d", "pct_ma_50", "pct_ma_200", "pct_from_52w_high"]:
            if col in feature_latest.columns:
                out[col] = pd.to_numeric(feature_latest[col].reindex(out.index), errors="coerce")

    out["ccqs_change_1d"] = change_1d.reindex(out.index)
    out["ccqs_change_5d"] = change_5d.reindex(out.index)
    out["ccqs_change_21d"] = change_21d.reindex(out.index)

    out = out.dropna(subset=["ccqs"])
    out.index = out.index.astype(str)
    out.index.name = "ticker"

    latest_str = latest.strftime("%Y-%m-%d")
    return out.sort_values("ccqs", ascending=False), latest_str


@st.cache_data(ttl=TTL, show_spinner=False)
def load_themes_snapshot() -> Tuple[pd.DataFrame, str]:
    themes = read_remote_parquet("theme_aggregates.parquet")
    if themes.empty:
        return pd.DataFrame(), ""

    themes_latest, latest = latest_snapshot(themes)
    if themes_latest.empty or latest is None:
        return pd.DataFrame(), ""

    out = themes_latest.copy()
    out = out[out.get("theme_ccqs", pd.Series(index=out.index)).notna()]
    if "n_constituents" in out.columns:
        out = out[out["n_constituents"] >= 3]

    out = out.reset_index().rename(columns={"basket": "theme"})
    if "theme" not in out.columns:
        out = out.rename(columns={out.columns[0]: "theme"})

    keep = [
        "theme",
        "theme_ccqs",
        "theme_class",
        "momentum_class",
        "pct_above_50dma",
        "n_constituents",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()
    out = out.sort_values("theme_ccqs", ascending=False)

    return out, latest.strftime("%Y-%m-%d")


@st.cache_data(ttl=TTL, show_spinner=False)
def load_ticker_history(ticker: str, period: str) -> pd.DataFrame:
    ccqs = read_remote_parquet("ccqs.parquet")
    if ccqs.empty:
        return pd.DataFrame()

    date_level = _date_level(ccqs)
    ticker_level = _ticker_level(ccqs)
    if date_level is None or ticker_level is None:
        return pd.DataFrame()

    ticker = str(ticker).upper().strip()
    mask = ccqs.index.get_level_values(ticker_level).astype(str) == ticker
    sub = ccqs.loc[mask, ["ccqs", "grade"]].copy()

    if sub.empty:
        return pd.DataFrame()

    if period != "INCEPTION":
        latest_date = pd.Timestamp(sub.index.get_level_values(date_level).max())
        cutoff = latest_date - pd.Timedelta(days=PERIOD_DAYS[period])
        sub = sub[sub.index.get_level_values(date_level) >= cutoff]

    out = sub.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    return out[["date", "ccqs", "grade"]]


@st.cache_data(ttl=TTL, show_spinner=False)
def load_ticker_components(ticker: str) -> pd.DataFrame:
    components = read_remote_parquet("components.parquet")
    if components.empty:
        return pd.DataFrame(columns=["Component", "Z-Score"])

    date_level = _date_level(components)
    ticker_level = _ticker_level(components)
    if date_level is None or ticker_level is None:
        return pd.DataFrame(columns=["Component", "Z-Score"])

    ticker = str(ticker).upper().strip()
    latest = components.index.get_level_values(date_level).max()
    mask = (
        (components.index.get_level_values(ticker_level).astype(str) == ticker)
        & (components.index.get_level_values(date_level) == latest)
    )
    sub = components.loc[mask]
    if sub.empty:
        return pd.DataFrame(columns=["Component", "Z-Score"])

    row = sub.iloc[-1]
    rows = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        display = COMPONENT_DISPLAY_NAMES.get(str(key), str(key).removeprefix("s_").replace("_", " ").title())
        rows.append({"Component": display, "Z-Score": float(value)})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Component", "Z-Score"])

    return out.sort_values("Z-Score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_ticker_key_metrics(ticker: str) -> pd.DataFrame:
    features = read_remote_parquet("features.parquet")
    if features.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    date_level = _date_level(features)
    ticker_level = _ticker_level(features)
    if date_level is None or ticker_level is None:
        return pd.DataFrame(columns=["Metric", "Value"])

    ticker = str(ticker).upper().strip()
    latest = features.index.get_level_values(date_level).max()
    mask = (
        (features.index.get_level_values(ticker_level).astype(str) == ticker)
        & (features.index.get_level_values(date_level) == latest)
    )
    sub = features.loc[mask]
    if sub.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    row = sub.iloc[-1]

    def pct(key: str) -> str:
        value = row.get(key, np.nan)
        return f"{float(value):+.1f}%" if pd.notna(value) else "-"

    def num(key: str, digits: int = 1) -> str:
        value = row.get(key, np.nan)
        return f"{float(value):,.{digits}f}" if pd.notna(value) else "-"

    def integer(key: str) -> str:
        value = row.get(key, np.nan)
        return f"{int(value):,}" if pd.notna(value) else "-"

    rows = [
        ("% from 50-day Moving Average", pct("pct_ma_50")),
        ("% from 200-day Moving Average", pct("pct_ma_200")),
        ("% from 52-week High", pct("pct_from_52w_high")),
        ("RS Rating vs SPY", num("rs_rating_spy", 1)),
        ("Information Ratio 252D", num("information_ratio_252d", 2)),
        ("Relative Strength Index 14D", num("rsi_14", 1)),
        ("Realized Volatility 60D", num("realized_vol_60", 1)),
        ("Distribution Days 25D", integer("distribution_days_25")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ============================================================
# Formatting and display helpers
# ============================================================
def display_state(value: str) -> str:
    return STATE_LABELS.get(str(value), str(value).replace("_", " ").title())


def display_tier(value: str) -> str:
    return TIER_LABELS.get(str(value), str(value).replace("_", " ").title())


def fmt_signed(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):+.{digits}f}"


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}%"


def score_band(score: float) -> str:
    if pd.isna(score):
        return ""
    if score >= 80:
        return "S / A quality"
    if score >= 55:
        return "B quality"
    if score >= 30:
        return "C quality"
    return "D quality"


def style_ccqs(value):
    if pd.isna(value):
        return ""
    v = float(value)
    if v >= 80:
        return "background-color: #d8f3dc"
    if v >= 55:
        return "background-color: #edf6f9"
    if v >= 30:
        return "background-color: #fff3cd"
    return "background-color: #fde2e2"


def style_signed(value):
    if pd.isna(value):
        return ""
    return "background-color: #d8f3dc" if float(value) >= 0 else "background-color: #fde2e2"


def prepare_top_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.head(n).copy()
    d = d.reset_index().rename(columns={"ticker": "Ticker"})
    d["Leadership Tier"] = d["leadership_tier"].map(display_tier)
    d["State"] = d["primary_state"].map(display_state)
    d["Chart Quality"] = d["ccqs"].map(score_band)

    cols = [
        "Ticker",
        "ccqs",
        "grade",
        "Chart Quality",
        "Leadership Tier",
        "State",
        "setup",
        "ccqs_change_1d",
        "ccqs_change_5d",
        "ccqs_change_21d",
        "rs_rating_spy",
        "information_ratio_252d",
    ]
    cols = [c for c in cols if c in d.columns]
    d = d[cols]

    return d.rename(columns={
        "ccqs": "CCQS",
        "grade": "Grade",
        "setup": "Setup",
        "ccqs_change_1d": "Δ 1D",
        "ccqs_change_5d": "Δ 5D",
        "ccqs_change_21d": "Δ 21D",
        "rs_rating_spy": "RS vs SPY",
        "information_ratio_252d": "IR 252D",
    })


def prepare_themes_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.head(n).copy()
    d = d.rename(columns={
        "theme": "Theme",
        "theme_ccqs": "Theme CCQS",
        "theme_class": "Theme Class",
        "momentum_class": "Momentum",
        "pct_above_50dma": "% > 50D MA",
        "n_constituents": "Members",
    })
    return d


def plot_ccqs_history(history: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    if history.empty:
        fig.update_layout(height=360)
        return fig

    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["ccqs"],
        mode="lines",
        name="CCQS",
        line=dict(width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>CCQS: %{y:.1f}<extra></extra>",
    ))

    for level, label in [(30, "D/C"), (55, "C/B"), (80, "B/A"), (92, "A/S")]:
        fig.add_hline(
            y=level,
            line_width=1,
            line_dash="dot",
            annotation_text=label,
            annotation_position="right",
        )

    fig.update_layout(
        title=dict(text=f"{ticker} | CCQS Trajectory", x=0, xanchor="left"),
        height=390,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="CCQS", range=[0, 100], showgrid=True),
        xaxis=dict(showgrid=True),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


# ============================================================
# Sidebar
# ============================================================
st.title(TITLE)
st.caption(SUBTITLE)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Display the latest CCQS technical-quality output inside the ADFM app.

        **Source:** The page reads the CCQS production cache from the Composite-Chart-Quality-Score repository.

        **Method:** Streamlit is display-only here. The scoring pipeline remains in the CCQS repo.
        """
    )
    st.divider()
    st.markdown("### Controls")

    rows_to_show = st.selectbox("Rows", options=[25, 50, 100, 200], index=1)
    min_score = st.slider("Minimum CCQS", min_value=0, max_value=100, value=0, step=5)
    selected_grades = st.multiselect("Grades", options=["S", "A", "B", "C", "D"], default=[])
    show_themes = st.checkbox("Show theme scores", value=True)
    show_movers = st.checkbox("Show daily movers", value=True)
    show_detail = st.checkbox("Show ticker detail", value=True)


# ============================================================
# Main data load
# ============================================================
with st.spinner("Loading CCQS cache..."):
    df, snapshot_date = load_dashboard_snapshot()
    themes_df, themes_date = load_themes_snapshot()

if df.empty:
    st.error("No CCQS data found. Confirm the CCQS dashboard cache is available in the source repository.")
    st.stop()

filtered = df.copy()
filtered = filtered[filtered["ccqs"] >= float(min_score)]
if selected_grades:
    filtered = filtered[filtered["grade"].isin(selected_grades)]

st.markdown(
    f"<div class='meta'>{snapshot_date} · {len(df):,} names scored · {len(filtered):,} after filters</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Top stocks
# ============================================================
st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)
st.markdown("## Top Stocks")

top_table = prepare_top_table(filtered.sort_values("ccqs", ascending=False), int(rows_to_show))

styled_top = top_table.style.map(style_ccqs, subset=["CCQS"])
for col in ["Δ 1D", "Δ 5D", "Δ 21D"]:
    if col in top_table.columns:
        styled_top = styled_top.map(style_signed, subset=[col])

st.dataframe(
    styled_top.format({
        "CCQS": "{:.1f}",
        "Δ 1D": "{:+.2f}",
        "Δ 5D": "{:+.2f}",
        "Δ 21D": "{:+.2f}",
        "RS vs SPY": "{:.1f}",
        "IR 252D": "{:.2f}",
    }, na_rep="-"),
    use_container_width=True,
    hide_index=True,
    height=min(900, 38 + 35 * max(3, len(top_table))),
)

# ============================================================
# Themes
# ============================================================
if show_themes:
    st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)
    st.markdown("## Themes")

    if themes_df.empty:
        st.info("Theme-level CCQS cache is unavailable.")
    else:
        theme_table = prepare_themes_table(themes_df, int(rows_to_show))
        styled_theme = theme_table.style.map(style_ccqs, subset=["Theme CCQS"])
        st.dataframe(
            styled_theme.format({
                "Theme CCQS": "{:.1f}",
                "% > 50D MA": "{:.0f}%",
            }, na_rep="-"),
            use_container_width=True,
            hide_index=True,
            height=min(850, 38 + 35 * max(3, len(theme_table))),
        )

# ============================================================
# What changed today
# ============================================================
if show_movers:
    st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)
    st.markdown("## What Changed Today")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### Strongest Risers")
        risers = df.dropna(subset=["ccqs_change_1d"]).sort_values("ccqs_change_1d", ascending=False).head(10)
        risers_table = prepare_top_table(risers, 10)[["Ticker", "CCQS", "Leadership Tier", "State", "Δ 1D"]]
        st.dataframe(
            risers_table.style.map(style_signed, subset=["Δ 1D"]).format({"CCQS": "{:.1f}", "Δ 1D": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.markdown("### Largest Decliners")
        decliners = df.dropna(subset=["ccqs_change_1d"]).sort_values("ccqs_change_1d", ascending=True).head(10)
        decliners_table = prepare_top_table(decliners, 10)[["Ticker", "CCQS", "Leadership Tier", "State", "Δ 1D"]]
        st.dataframe(
            decliners_table.style.map(style_signed, subset=["Δ 1D"]).format({"CCQS": "{:.1f}", "Δ 1D": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_c:
        st.markdown("### Highest Quality Pullbacks")
        pullbacks = df[
            (df["ccqs"] >= 70)
            & (df["primary_state"].isin(["PULLBACK", "CONSOLIDATING"]))
        ].sort_values("ccqs", ascending=False).head(10)
        pullbacks_table = prepare_top_table(pullbacks, 10)[["Ticker", "CCQS", "Leadership Tier", "State", "Setup"]]
        st.dataframe(
            pullbacks_table.style.map(style_ccqs, subset=["CCQS"]).format({"CCQS": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# Ticker detail
# ============================================================
if show_detail:
    st.markdown("<hr class='section-rule'/>", unsafe_allow_html=True)
    st.markdown("## Ticker Detail")

    tickers = sorted(filtered.index.astype(str).tolist())
    default_idx = tickers.index("NVDA") if "NVDA" in tickers else 0
    selected_ticker = st.selectbox("Ticker", options=tickers, index=default_idx)

    row = filtered.loc[selected_ticker]
    st.markdown(
        f"**{selected_ticker}** · CCQS **{row['ccqs']:.1f}** ({row['grade']}) · "
        f"{display_tier(row['leadership_tier'])} · {display_state(row['primary_state'])} · "
        f"{str(row.get('setup', ''))}"
    )

    period = st.selectbox(
        "CCQS history period",
        options=list(PERIOD_DAYS.keys()),
        index=list(PERIOD_DAYS.keys()).index("6M"),
    )

    history = load_ticker_history(selected_ticker, period)
    st.plotly_chart(plot_ccqs_history(history, selected_ticker), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Component Z-Scores")
        components = load_ticker_components(selected_ticker)
        st.dataframe(
            components.style.map(style_signed, subset=["Z-Score"]).format({"Z-Score": "{:+.2f}"}, na_rep="-"),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with right:
        st.markdown("### Key Metrics")
        key_metrics = load_ticker_key_metrics(selected_ticker)
        st.dataframe(
            key_metrics,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

# ============================================================
# Methodology note
# ============================================================
with st.expander("Methodology Note", expanded=False):
    st.markdown(
        """
        CCQS is a 0-100 technical quality score built from standardized relative strength,
        leadership, trend, structure, multi-timeframe alignment, extension, momentum,
        residual momentum, and volume-pattern inputs. This page consumes the latest CCQS
        dashboard cache and does not recompute the production score inside the ADFM app.
        """
    )
