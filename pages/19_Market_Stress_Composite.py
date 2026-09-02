from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from adfm_core.palette import PASTEL
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_sidebar_about,
)


TITLE = "Market Stress Composite"
SPX = "^GSPC"
IXIC = "^IXIC"

FOREIGN_EQUITIES: Dict[str, str] = {
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^HSI": "Hang Seng",
    "^AXJO": "ASX 200",
}

CARRY_FX: Dict[str, str] = {
    "AUDJPY=X": "AUD/JPY",
    "NZDJPY=X": "NZD/JPY",
}

HAVEN_FX: Dict[str, str] = {
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF",
}

FOREIGN_BONDS: Dict[str, str] = {
    "IGLT.L": "UK Gilts ETF",
    "IEGA.AS": "Euro Govt Bonds ETF",
    "2510.T": "Japan Govt Bonds ETF",
    "IGB.AX": "Australia Govt Bonds ETF",
}

ALL_TICKERS = sorted(
    set(
        [SPX, IXIC]
        + list(FOREIGN_EQUITIES)
        + list(CARRY_FX)
        + list(HAVEN_FX)
        + list(FOREIGN_BONDS)
    )
)

DEFAULT_Z_YEARS = 3
DEFAULT_SMOOTH_DAYS = 10
LEAD_HORIZON = 21
EARLY_STAGE_DD63 = -0.07

WATCH_RISK = 0.90
WATCH_DISLOCATION = 1.10
HEDGE_RISK = 1.25
HEDGE_DISLOCATION = 1.50
FRACTURE_LEVEL = 1.50

INDEX_COLOR = "#202124"
RISK_COLOR = PASTEL["rose"]
DISLOCATION_COLOR = PASTEL["lavender"]
WATCH_COLOR = PASTEL["amber"]
THRESHOLD_COLOR = "#B7B7B7"


st.set_page_config(page_title=TITLE, layout="wide")
inject_explorer_style()
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
render_page_header(
    PageHeader(
        title=TITLE,
        description="Foreign rates, FX and equity stress designed to warn before the U.S. tape breaks.",
        eyebrow="ADFM Global Fracture Monitor",
    )
)


# ---------------- Helpers ----------------
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: List[str], start: date) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
            elif (ticker, "Adj Close") in raw.columns:
                out[ticker] = pd.to_numeric(raw[(ticker, "Adj Close")], errors="coerce")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if col and tickers:
            out[tickers[0]] = pd.to_numeric(raw[col], errors="coerce")

    df = pd.DataFrame(out)
    if df.empty:
        return df

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx.normalize()
    return df.sort_index().groupby(level=0).last()


def robust_z(s: pd.Series, window: int, min_periods: int = 126) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    minp = min(min_periods, max(40, window // 4))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    return ((s - mean) / std.replace(0, np.nan)).clip(-5, 5)


def pct_return(px: pd.Series, days: int) -> pd.Series:
    return px.pct_change(days, fill_method=None)


def safe_mean(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return frame.mean(axis=1, skipna=True)


def available_cols(px: pd.DataFrame, universe: Dict[str, str]) -> List[str]:
    return [t for t in universe if t in px.columns and px[t].notna().sum() >= 80]


def latest_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def fmt_score(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:+.2f}"


def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:+.2f}%"


def future_min(px: pd.Series, horizon: int) -> pd.Series:
    forward = pd.concat([px.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return forward.min(axis=1, skipna=False)


def classify_market(direction_z: float, shock_z: float) -> str:
    direction = 0.0 if pd.isna(direction_z) else direction_z
    shock = 0.0 if pd.isna(shock_z) else shock_z
    if direction >= 1.50 or shock >= 2.00:
        return "High stress"
    if direction >= 0.90 or shock >= 1.20:
        return "Watch"
    return "Normal"


def regime_label(risk: float, dislocation: float) -> str:
    if pd.isna(risk) or pd.isna(dislocation):
        return "Insufficient data"
    if risk >= FRACTURE_LEVEL and dislocation >= FRACTURE_LEVEL:
        return "Global fracture"
    if risk >= HEDGE_RISK or dislocation >= HEDGE_DISLOCATION:
        return "Hedge pressure"
    if risk >= WATCH_RISK or dislocation >= WATCH_DISLOCATION:
        return "Watch"
    return "Neutral"


def action_label(risk: float, dislocation: float, us_dd63: float) -> str:
    if pd.isna(risk) or pd.isna(dislocation) or pd.isna(us_dd63):
        return "Insufficient data"
    if us_dd63 <= -0.10:
        return "Late: stress already in U.S. tape"
    if risk >= FRACTURE_LEVEL and dislocation >= FRACTURE_LEVEL:
        return "High alert: hedge"
    if (
        risk >= HEDGE_RISK
        or dislocation >= HEDGE_DISLOCATION
        or (risk >= WATCH_RISK and dislocation >= WATCH_DISLOCATION)
    ):
        return "Add protection"
    if risk >= WATCH_RISK or dislocation >= WATCH_DISLOCATION:
        return "Watch / prepare"
    return "No hedge signal"


# ---------------- Sidebar ----------------
with st.sidebar:
    render_sidebar_about("19_Market_Stress_Composite.py")
    st.header("Settings")

    lookback_years = st.selectbox(
        "Chart lookback",
        [1, 2, 3, 5, 10, 25, 50],
        index=3,
        format_func=lambda x: f"{x} year" if x == 1 else f"{x} years",
    )

    target_mode = st.radio(
        "U.S. overlay",
        ["Auto", "S&P 500", "Nasdaq Composite"],
        index=0,
    )

    z_window_years = st.slider(
        "Normalization window",
        1,
        5,
        DEFAULT_Z_YEARS,
        1,
    )

    smoothing_mode = st.selectbox(
        "Signal speed",
        ["Fast - 3D", "Base - 5D", "Slow - 10D", "21D", "63D"],
        index=2,
    )
    smooth_days = {
        "Fast - 3D": 3,
        "Base - 5D": 5,
        "Slow - 10D": 10,
        "21D": 21,
        "63D": 63,
    }[smoothing_mode]


# ---------------- Data ----------------
# Pull enough prehistory to normalize the signal at the left edge of long charts.
history_years = max(12, lookback_years + z_window_years + 2)
history_start = date.today() - timedelta(days=int(history_years * 365.25))
px = load_prices(ALL_TICKERS, history_start)

if px.empty or SPX not in px.columns:
    st.error("Yahoo Finance did not return enough market data to build the Global Fracture Monitor.")
    st.stop()

calendar = px[SPX].dropna().index
px = px.reindex(calendar).ffill(limit=2)

eq_cols = available_cols(px, FOREIGN_EQUITIES)
carry_cols = available_cols(px, CARRY_FX)
haven_cols = available_cols(px, HAVEN_FX)
bond_cols = available_cols(px, FOREIGN_BONDS)

if len(eq_cols) < 3 or len(carry_cols) < 1:
    st.warning("Some foreign-market series are unavailable today. Scores are reweighted across available inputs.")

z_window = int(252 * z_window_years)


# ---------------- Risk-Off score ----------------
eq_r21 = pd.DataFrame({t: pct_return(px[t], 21) for t in eq_cols})
eq_r63 = pd.DataFrame({t: pct_return(px[t], 63) for t in eq_cols})

eq_weak_21 = safe_mean(pd.DataFrame({t: -robust_z(eq_r21[t], z_window) for t in eq_cols}))
eq_weak_63 = safe_mean(pd.DataFrame({t: -robust_z(eq_r63[t], z_window) for t in eq_cols}))

breadth_neg21 = (eq_r21 < 0).mean(axis=1) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
below_ma = pd.DataFrame(
    {t: (px[t] < px[t].rolling(100, min_periods=60).mean()).astype(float) for t in eq_cols}
)
breadth_ma = below_ma.mean(axis=1) if not below_ma.empty else pd.Series(index=calendar, dtype=float)
breadth_z = robust_z(0.50 * breadth_neg21 + 0.50 * breadth_ma, z_window)

spx_r21 = pct_return(px[SPX], 21)
relative = pd.DataFrame({t: eq_r21[t] - spx_r21 for t in eq_cols})
relative_weak = safe_mean(pd.DataFrame({t: -robust_z(relative[t], z_window) for t in eq_cols}))

carry_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in carry_cols})
carry_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in carry_cols})
carry_stress = 0.35 * safe_mean(carry_5) + 0.65 * safe_mean(carry_21)

haven_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in haven_cols})
haven_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in haven_cols})
haven_stress = 0.35 * safe_mean(haven_5) + 0.65 * safe_mean(haven_21)

risk_components = pd.DataFrame(
    {
        "Foreign equity weakness": 0.25 * eq_weak_21 + 0.10 * eq_weak_63,
        "Foreign breadth": 0.20 * breadth_z,
        "Foreign vs U.S.": 0.15 * relative_weak,
        "Carry unwind": 0.20 * carry_stress,
        "Haven FX": 0.10 * haven_stress,
    }
)
risk_raw = risk_components.sum(axis=1, min_count=2)
risk_score = robust_z(risk_raw, z_window).ewm(
    span=smooth_days,
    adjust=False,
    min_periods=1,
).mean()


# ---------------- Dislocation score ----------------
eq_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in eq_cols})
)

fx_all = carry_cols + haven_cols
fx_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in fx_all})
)

bond_shock = safe_mean(
    pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in bond_cols})
)

eq_dispersion = eq_r21.std(axis=1, skipna=True) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
dispersion_z = robust_z(eq_dispersion, z_window)

dislocation_components = pd.DataFrame(
    {
        "Foreign bond shock": 0.40 * bond_shock,
        "FX shock": 0.25 * fx_shock,
        "Foreign equity shock": 0.20 * eq_shock,
        "Cross-country dispersion": 0.15 * dispersion_z,
    }
)
dislocation_raw = dislocation_components.sum(axis=1, min_count=2)
dislocation_score = robust_z(dislocation_raw, z_window).ewm(
    span=smooth_days,
    adjust=False,
    min_periods=1,
).mean()


# ---------------- U.S. overlay ----------------
def corr_with_future_drawdown(signal: pd.Series, target: pd.Series) -> float:
    fwd_dd = future_min(target, LEAD_HORIZON) / target - 1.0
    aligned = pd.concat([signal, fwd_dd], axis=1).dropna()
    if len(aligned) < 100:
        return np.nan
    return float(aligned.iloc[:, 0].corr(-aligned.iloc[:, 1]))


def choose_target() -> tuple[str, str]:
    if target_mode == "S&P 500":
        return SPX, "S&P 500"
    if target_mode == "Nasdaq Composite":
        return (IXIC, "Nasdaq Composite") if IXIC in px.columns else (SPX, "S&P 500")

    combined = pd.concat(
        [risk_score.rename("Risk-Off"), dislocation_score.rename("Dislocation")],
        axis=1,
    ).mean(axis=1)

    candidates = [(SPX, "S&P 500")]
    if IXIC in px.columns:
        candidates.append((IXIC, "Nasdaq Composite"))

    best = candidates[0]
    best_metric = -np.inf
    for ticker, label in candidates:
        relationship = corr_with_future_drawdown(combined, px[ticker])
        move_21 = abs(latest_valid(pct_return(px[ticker], 21)))
        metric = 0.85 * (0.0 if pd.isna(relationship) else relationship) + 0.15 * (
            0.0 if pd.isna(move_21) else move_21
        )
        if metric > best_metric:
            best_metric = metric
            best = (ticker, label)
    return best


target_ticker, target_label = choose_target()
target_px = px[target_ticker].dropna()

risk_now = latest_valid(risk_score)
dislocation_now = latest_valid(dislocation_score)
regime = regime_label(risk_now, dislocation_now)

us_high63 = target_px.rolling(63, min_periods=20).max()
us_dd63 = target_px / us_high63 - 1.0
us_dd63_now = latest_valid(us_dd63)
action = action_label(risk_now, dislocation_now, us_dd63_now)

watch_signal = (
    ((risk_score >= WATCH_RISK) | (dislocation_score >= WATCH_DISLOCATION))
    & (us_dd63 > EARLY_STAGE_DD63)
).fillna(False)
watch_onset = watch_signal & ~watch_signal.shift(1, fill_value=False)
onset_dates = watch_onset[watch_onset].index

signal_age = "No active watch"
if watch_signal.iloc[-1] and len(onset_dates):
    onset_pos = int(calendar.get_indexer([onset_dates[-1]])[0])
    current_pos = int(calendar.get_indexer([calendar[-1]])[0])
    signal_age = f"{current_pos - onset_pos} sessions"

with st.sidebar:
    st.markdown("---")
    st.subheader("Current read")
    st.caption(f"Risk-Off {fmt_score(risk_now)}  |  Dislocation {fmt_score(dislocation_now)}")
    st.caption(f"{regime}  |  {target_label}")
    st.caption(f"{action}  |  Watch age: {signal_age}")

    with st.expander("Signal construction", expanded=False):
        st.markdown(
            """
            **Risk-Off:** foreign equity weakness and breadth, foreign underperformance versus the U.S., carry unwind, and haven-FX strength.

            **Dislocation:** abnormal foreign bond, FX and equity moves plus cross-country dispersion.

            Fresh watch signals are suppressed once the selected U.S. index is more than 7% below its 63-session high.
            """
        )

    with st.expander("Data health", expanded=False):
        health_rows = []
        for ticker in ALL_TICKERS:
            s = px[ticker].dropna() if ticker in px.columns else pd.Series(dtype=float)
            health_rows.append(
                {
                    "Ticker": ticker,
                    "Obs": len(s),
                    "Last": s.index.max().date().isoformat() if len(s) else "",
                }
            )
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True, height=260)


# ---------------- Two-panel interactive chart ----------------
cutoff = pd.Timestamp(date.today() - timedelta(days=int(lookback_years * 365.25)))
plot_idx = target_px.index[target_px.index >= cutoff]

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.62, 0.38],
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=target_px.reindex(plot_idx),
        name=target_label,
        mode="lines",
        line=dict(width=2.4, color=INDEX_COLOR),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            + target_label
            + ": %{y:,.2f}<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)

visible_onsets = onset_dates[onset_dates >= cutoff]
if len(visible_onsets):
    marker_risk = risk_score.reindex(visible_onsets)
    marker_dis = dislocation_score.reindex(visible_onsets)
    custom = np.column_stack(
        [
            [f"{x:+.2f}" if pd.notna(x) else "NA" for x in marker_risk],
            [f"{x:+.2f}" if pd.notna(x) else "NA" for x in marker_dis],
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=visible_onsets,
            y=target_px.reindex(visible_onsets).values,
            name="Watch onset",
            mode="markers",
            marker=dict(
                size=8,
                symbol="diamond",
                color=WATCH_COLOR,
                line=dict(color=INDEX_COLOR, width=0.8),
            ),
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Watch onset"
                "<br>Risk-Off: %{customdata[0]}"
                "<br>Dislocation: %{customdata[1]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=risk_score.reindex(plot_idx),
        name="Global Risk-Off",
        mode="lines",
        line=dict(width=2.0, color=RISK_COLOR),
        hovertemplate="%{x|%Y-%m-%d}<br>Risk-Off: %{y:+.2f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=plot_idx,
        y=dislocation_score.reindex(plot_idx),
        name="Global Dislocation",
        mode="lines",
        line=dict(width=1.9, dash="dot", color=DISLOCATION_COLOR),
        hovertemplate="%{x|%Y-%m-%d}<br>Dislocation: %{y:+.2f}<extra></extra>",
    ),
    row=2,
    col=1,
)

for level in [WATCH_RISK, HEDGE_RISK, FRACTURE_LEVEL]:
    fig.add_hline(
        y=level,
        row=2,
        col=1,
        line_dash="dash",
        line_width=1,
        line_color=THRESHOLD_COLOR,
        opacity=0.35,
    )

fig.add_hline(
    y=0.0,
    row=2,
    col=1,
    line_width=1,
    line_color=THRESHOLD_COLOR,
    opacity=0.30,
)

fig.update_layout(
    height=700,
    margin=dict(l=25, r=25, t=45, b=25),
    hovermode="x",
    legend=dict(orientation="h", y=1.04, x=0),
)
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
fig.update_yaxes(title_text=target_label, tickformat=",.2f", row=1, col=1)
fig.update_yaxes(title_text="Global stress score", tickformat=".2f", row=2, col=1)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    },
)

st.markdown(
    f"**Actionable read:** {action}  |  "
    f"Risk-Off {fmt_score(risk_now)}  |  "
    f"Dislocation {fmt_score(dislocation_now)}  |  "
    f"{target_label} vs 63D high {fmt_pct(us_dd63_now)}  |  "
    f"Current watch age: {signal_age}"
)


# ---------------- Main table ----------------
st.subheader("Global market moves")
rows = []

for ticker in eq_cols:
    direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
    shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
    rows.append(
        {
            "Bucket": "Foreign equities",
            "Market": FOREIGN_EQUITIES[ticker],
            "Ticker": ticker,
            "5D": latest_valid(pct_return(px[ticker], 5)),
            "21D": latest_valid(pct_return(px[ticker], 21)),
            "63D": latest_valid(pct_return(px[ticker], 63)),
            "Risk-Off Z": direction_z,
            "Shock Z": shock_z,
            "Status": classify_market(direction_z, shock_z),
        }
    )

for ticker in carry_cols:
    direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
    shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
    rows.append(
        {
            "Bucket": "Carry FX",
            "Market": CARRY_FX[ticker],
            "Ticker": ticker,
            "5D": latest_valid(pct_return(px[ticker], 5)),
            "21D": latest_valid(pct_return(px[ticker], 21)),
            "63D": latest_valid(pct_return(px[ticker], 63)),
            "Risk-Off Z": direction_z,
            "Shock Z": shock_z,
            "Status": classify_market(direction_z, shock_z),
        }
    )

for ticker in haven_cols:
    direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
    shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
    rows.append(
        {
            "Bucket": "Haven FX",
            "Market": HAVEN_FX[ticker],
            "Ticker": ticker,
            "5D": latest_valid(pct_return(px[ticker], 5)),
            "21D": latest_valid(pct_return(px[ticker], 21)),
            "63D": latest_valid(pct_return(px[ticker], 63)),
            "Risk-Off Z": direction_z,
            "Shock Z": shock_z,
            "Status": classify_market(direction_z, shock_z),
        }
    )

for ticker in bond_cols:
    shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
    rows.append(
        {
            "Bucket": "Foreign bonds",
            "Market": FOREIGN_BONDS[ticker],
            "Ticker": ticker,
            "5D": latest_valid(pct_return(px[ticker], 5)),
            "21D": latest_valid(pct_return(px[ticker], 21)),
            "63D": latest_valid(pct_return(px[ticker], 63)),
            "Risk-Off Z": np.nan,
            "Shock Z": shock_z,
            "Status": classify_market(np.nan, shock_z),
        }
    )

moves = pd.DataFrame(rows)
if not moves.empty:
    status_rank = {"High stress": 2, "Watch": 1, "Normal": 0}
    moves["_rank"] = moves["Status"].map(status_rank).fillna(0)
    moves["_stress"] = moves[["Risk-Off Z", "Shock Z"]].max(axis=1, skipna=True)
    moves = moves.sort_values(["_rank", "_stress"], ascending=[False, False]).drop(
        columns=["_rank", "_stress"]
    )

    styled_moves = moves.style.format(
        {
            "5D": "{:+.2%}",
            "21D": "{:+.2%}",
            "63D": "{:+.2%}",
            "Risk-Off Z": "{:+.2f}",
            "Shock Z": "{:+.2f}",
        },
        na_rep="",
    )
    st.dataframe(styled_moves, use_container_width=True, hide_index=True)

render_footer(
    data_note="Primary inputs: Yahoo Finance market history; local last-good cache on provider failure."
)
