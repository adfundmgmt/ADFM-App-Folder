from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

TITLE = "Rates Regime Monitor"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (3, 10)

SERIES: Dict[str, str] = {
    "DGS3MO": "3M",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
    "DFII10": "10Y Real",
    "T10YIE": "10Y Breakeven",
}

TENORS: Dict[str, Tuple[str, float]] = {
    "DGS3MO": ("3M", 0.25),
    "DGS2": ("2Y", 2.0),
    "DGS5": ("5Y", 5.0),
    "DGS10": ("10Y", 10.0),
    "DGS30": ("30Y", 30.0),
}
TENOR_ORDER = list(TENORS.keys())

PERIODS = {
    "1D": {"kind": "row", "rows": 1, "threshold": 4},
    "1W": {"kind": "calendar", "days": 7, "threshold": 8},
    "1M": {"kind": "calendar", "months": 1, "threshold": 15},
    "3M": {"kind": "calendar", "months": 3, "threshold": 30},
    "YTD": {"kind": "ytd", "threshold": 35},
}

COLORS = {
    "ink": "#111827",
    "muted": "#6b7280",
    "border": "#e5e7eb",
    "panel": "#ffffff",
    "soft": "#f8fafc",
    "blue": "#2563eb",
    "purple": "#7c3aed",
    "green": "#059669",
    "red": "#dc2626",
    "amber": "#d97706",
    "grey": "#9ca3af",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1540px;}
        .adfm-title {font-size: 1.75rem; line-height: 1.15; font-weight: 760; margin-bottom: 0.2rem; color: #111827;}
        .adfm-subtitle {font-size: 0.94rem; color: #6b7280; margin-bottom: 1.05rem;}
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 13px; padding: 13px 15px 11px 15px; min-height: 96px; box-shadow: 0 1px 5px rgba(15,23,42,0.045);}
        .metric-label {font-size: 0.71rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.055em; margin-bottom: 0.4rem;}
        .metric-value {font-size: 1.26rem; font-weight: 760; color: #111827; line-height: 1.18;}
        .metric-footnote {font-size: 0.75rem; color: #6b7280; margin-top: 0.42rem; line-height: 1.34;}
        .section-title {font-size: 1.02rem; font-weight: 760; color: #111827; margin-top: 0.95rem; margin-bottom: 0.45rem;}
        .small-note {font-size: 0.80rem; color: #6b7280; margin-top: -0.25rem; margin-bottom: 0.45rem;}
        .note-box {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 11px 13px; color: #475569; font-size: 0.84rem; line-height: 1.42;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.65rem 0.75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_float(x: object) -> float:
    try:
        value = float(x)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return safe_float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    clean = df.dropna(how="all")
    if clean.empty:
        return None
    return pd.Timestamp(clean.index[-1])


def value_on_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index <= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[-1])


def first_value_on_or_after(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index >= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[0])


def change_bp(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan

    last_value = safe_float(clean.iloc[-1])
    last_idx = pd.Timestamp(clean.index[-1])
    spec = PERIODS[period]

    if spec["kind"] == "row":
        rows = int(spec.get("rows", 1))
        if len(clean) <= rows:
            return np.nan
        anchor = safe_float(clean.iloc[-rows - 1])
    elif spec["kind"] == "calendar":
        target = last_idx - pd.DateOffset(days=int(spec.get("days", 0)))
        if "months" in spec:
            target = last_idx - pd.DateOffset(months=int(spec["months"]))
        anchor = value_on_or_before(clean, target)
    elif spec["kind"] == "ytd":
        jan_first = pd.Timestamp(date(last_idx.year, 1, 1))
        anchor = first_value_on_or_after(clean, jan_first)
    else:
        return np.nan

    if not np.isfinite(last_value) or not np.isfinite(anchor):
        return np.nan
    return float((last_value - anchor) * 100.0)


def fmt_pct(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.2f}%"


def fmt_bp(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:+.0f} bps"


def fmt_num(x: float, decimals: int = 1) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:,.{decimals}f}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_ids: Tuple[str, ...], start_date: date) -> pd.DataFrame:
    frames: List[pd.Series] = []
    for series_id in series_ids:
        try:
            response = requests.get(
                FRED_URL,
                params={"id": series_id, "cosd": start_date.isoformat()},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
            if raw.empty or len(raw.columns) < 2:
                continue
            raw.columns = ["date", series_id]
            values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")
            frame = pd.Series(values.to_numpy(), index=pd.to_datetime(raw["date"]), name=series_id)
            frames.append(frame)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.ffill().dropna(how="all")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"DGS10", "DGS2"}.issubset(out.columns):
        out["2s10s"] = out["DGS10"] - out["DGS2"]
    if {"DGS30", "DGS5"}.issubset(out.columns):
        out["5s30s"] = out["DGS30"] - out["DGS5"]
    if {"DGS10", "DGS3MO"}.issubset(out.columns):
        out["3m10y"] = out["DGS10"] - out["DGS3MO"]
    if {"DGS10", "T10YIE"}.issubset(out.columns):
        out["REAL10_CALC"] = out["DGS10"] - out["T10YIE"]
    return out


def available_curve_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ["2s10s", "3m10y", "5s30s"] if c in df.columns and df[c].dropna().any()]


def classify_regime(df: pd.DataFrame, period: str, curve_col: str) -> Tuple[str, str, str]:
    ten = change_bp(df["DGS10"], period) if "DGS10" in df else np.nan
    curve = change_bp(df[curve_col], period) if curve_col in df else np.nan
    threshold = float(PERIODS[period]["threshold"])

    if not np.isfinite(ten):
        return "Insufficient Data", "Need a valid 10Y series.", COLORS["amber"]

    if not np.isfinite(curve):
        if ten > threshold:
            return "Rates Rising", f"10Y yield up {fmt_bp(ten)} over {period}; curve unavailable.", COLORS["red"]
        if ten < -threshold:
            return "Rates Falling", f"10Y yield down {fmt_bp(ten)} over {period}; curve unavailable.", COLORS["green"]
        return "Range / Mixed", f"10Y move inside the {threshold:.0f} bp signal band.", COLORS["amber"]

    if abs(ten) < threshold and abs(curve) < threshold:
        return "Range / Mixed", f"10Y and {curve_col} are inside the {threshold:.0f} bp signal band.", COLORS["amber"]
    if ten > threshold and curve > threshold:
        return "Bear Steepener", f"10Y up {fmt_bp(ten)}; {curve_col} steepened {fmt_bp(curve)} over {period}.", COLORS["red"]
    if ten > threshold and curve < -threshold:
        return "Bear Flattener", f"10Y up {fmt_bp(ten)}; {curve_col} flattened {fmt_bp(curve)} over {period}.", COLORS["red"]
    if ten < -threshold and curve > threshold:
        return "Bull Steepener", f"10Y down {fmt_bp(ten)}; {curve_col} steepened {fmt_bp(curve)} over {period}.", COLORS["green"]
    if ten < -threshold and curve < -threshold:
        return "Bull Flattener", f"10Y down {fmt_bp(ten)}; {curve_col} flattened {fmt_bp(curve)} over {period}.", COLORS["green"]
    if ten > threshold:
        return "Bearish Rates Impulse", f"10Y up {fmt_bp(ten)}; curve signal is mixed.", COLORS["red"]
    if ten < -threshold:
        return "Bullish Rates Impulse", f"10Y down {fmt_bp(ten)}; curve signal is mixed.", COLORS["green"]
    return "Curve Signal", f"10Y quiet, but {curve_col} moved {fmt_bp(curve)} over {period}.", COLORS["amber"]


def regime_read(regime: str) -> str:
    reads = {
        "Bear Steepener": "Higher long-end yields with a steeper curve. Usually the most hostile mix for long-duration equity, levered growth, and bond proxies unless it is driven by better nominal growth.",
        "Bear Flattener": "Front-end or intermediate rates are tightening the curve while yields rise. This is usually the cleanest policy-pressure regime and tends to pressure cyclicals, small caps, and credit appetite.",
        "Bull Steepener": "Yields are falling while the curve steepens. This often reflects growth deterioration, Fed-cut pricing, or a flight-to-duration bid. Good for duration, less clean for equities unless liquidity is improving.",
        "Bull Flattener": "Yields are falling while the curve flattens. Usually a long-end duration bid without a strong reflation impulse. Quality and defensive duration often screen better than cyclicals.",
        "Bearish Rates Impulse": "The level move matters more than the curve. Duration is the pressure point. Equity duration and crowded quality growth should be watched first.",
        "Bullish Rates Impulse": "The level move is supportive for duration. The equity read depends on whether falling yields reflect easing liquidity or weakening growth.",
        "Curve Signal": "The curve is moving more than the level of 10Y yields. The trade is about policy path, term premium, and growth expectations rather than outright duration alone.",
        "Range / Mixed": "No clean rates regime. Treat the curve as background information and avoid over-reading small basis-point changes.",
    }
    return reads.get(regime, "Signal quality is low. Check data freshness and the underlying series before acting on the classification.")


def period_matrix(df: pd.DataFrame, rows: List[str]) -> pd.DataFrame:
    out = []
    for col in rows:
        if col not in df.columns:
            continue
        label = SERIES.get(col, col)
        if col in ["2s10s", "5s30s", "3m10y"]:
            label = col
        out.append(
            {
                "Series": label,
                "Latest": latest(df[col]),
                **{period: change_bp(df[col], period) for period in PERIODS},
            }
        )
    return pd.DataFrame(out)


def build_shock_table(df: pd.DataFrame, curve_col: str, lookback_days: int, top_n: int) -> pd.DataFrame:
    cols = [c for c in ["DGS10", curve_col, "DFII10", "T10YIE"] if c in df.columns]
    if "DGS10" not in cols or len(df) < 5:
        return pd.DataFrame()

    recent = df[cols].dropna(how="all").tail(lookback_days).copy()
    daily_bp = recent.diff() * 100.0
    if daily_bp.empty:
        return pd.DataFrame()

    weights = {"DGS10": 1.00, curve_col: 0.70, "DFII10": 0.60, "T10YIE": 0.45}
    score = pd.Series(0.0, index=daily_bp.index)
    for col in daily_bp.columns:
        score = score.add(daily_bp[col].abs().fillna(0.0) * weights.get(col, 0.5), fill_value=0.0)

    shock = daily_bp.copy()
    shock["Shock Score"] = score
    shock = shock.dropna(how="all")
    if shock.empty:
        return pd.DataFrame()

    component_cols = [c for c in ["DGS10", curve_col, "DFII10", "T10YIE"] if c in shock.columns]
    shock["Driver"] = shock[component_cols].abs().idxmax(axis=1).replace(
        {"DGS10": "10Y", curve_col: curve_col, "DFII10": "Real", "T10YIE": "Breakeven"}
    )
    shock = shock.sort_values("Shock Score", ascending=False).head(top_n).sort_index()
    shock = shock.rename(columns={"DGS10": "10Y bp", curve_col: f"{curve_col} bp", "DFII10": "Real bp", "T10YIE": "BE bp"})
    shock.insert(0, "Date", shock.index.date)
    return shock.reset_index(drop=True)


def metric_card(label: str, value: str, footnote: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_plot_layout(fig: go.Figure, height: int, y_title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=18, b=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False, title_text=y_title)
    return fig


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        FRED-only Treasury monitor for curve shape, real yields, breakevens, and regime pressure. 
        Built to answer one question quickly: is the rates move helping or hurting risk?
        """
    )

    st.divider()
    st.header("Controls")
    lookback_years = st.selectbox("History", [1, 2, 3, 5, 10, 20], index=3)
    regime_period = st.radio("Regime window", list(PERIODS.keys()), index=2, horizontal=True)
    curve_choices_default = ["2s10s", "3m10y", "5s30s"]
    selected_curve = st.selectbox("Curve gauge", curve_choices_default, index=0)
    curve_compare = st.radio("Curve comparison", ["1W", "1M", "3M", "YTD"], index=1, horizontal=True)
    show_full_history = st.checkbox("Show full history chart", value=True)
    show_table = st.checkbox("Show data table", value=False)

    st.divider()
    st.caption("No Yahoo fallback. If data is missing, the page shows the gap instead of mixing sources.")

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Curve shape, level shock, real-rate pressure, breakevens, and outlier days. FRED-only to avoid proxy drift.</div>",
    unsafe_allow_html=True,
)

start = date.today() - timedelta(days=int(lookback_years * 365.25) + 15)
rates = add_derived(fetch_fred(tuple(SERIES.keys()), start))

if rates.empty or "DGS10" not in rates.columns or rates["DGS10"].dropna().empty:
    st.error("No usable FRED Treasury data loaded. Check connection or FRED availability.")
    st.stop()

curve_cols = available_curve_columns(rates)
if not curve_cols:
    st.error("No usable curve spreads can be calculated from the loaded FRED data.")
    st.stop()
if selected_curve not in curve_cols:
    selected_curve = curve_cols[0]

last_obs = latest_date(rates)
if last_obs is not None:
    age_days = (pd.Timestamp(date.today()) - last_obs.normalize()).days
    if age_days > 4:
        st.warning(f"Last FRED observation is {last_obs.date()}. Data may be stale.")

regime, regime_note, regime_color = classify_regime(rates, regime_period, selected_curve)
real_col = "DFII10" if "DFII10" in rates.columns and rates["DFII10"].dropna().any() else "REAL10_CALC"

cards = [
    ("Regime", regime, regime_note, regime_color),
    ("10Y Treasury", fmt_pct(latest(rates["DGS10"])), f"{regime_period} {fmt_bp(change_bp(rates['DGS10'], regime_period))}", COLORS["blue"]),
    (selected_curve, fmt_bp(latest(rates[selected_curve]) * 100.0), f"{regime_period} {fmt_bp(change_bp(rates[selected_curve], regime_period))}", COLORS["purple"]),
    ("10Y Real", fmt_pct(latest(rates[real_col])) if real_col in rates else "N/A", f"{regime_period} {fmt_bp(change_bp(rates[real_col], regime_period))}" if real_col in rates else "Unavailable", COLORS["green"]),
    ("10Y Breakeven", fmt_pct(latest(rates["T10YIE"])) if "T10YIE" in rates else "N/A", f"{regime_period} {fmt_bp(change_bp(rates['T10YIE'], regime_period))}" if "T10YIE" in rates else "Unavailable", COLORS["amber"]),
]

for col, card in zip(st.columns(5), cards):
    with col:
        metric_card(*card)

st.markdown("<div class='section-title'>Read-through</div>", unsafe_allow_html=True)
st.markdown(f"<div class='note-box'>{regime_read(regime)}</div>", unsafe_allow_html=True)

available_tenors = [c for c in TENOR_ORDER if c in rates.columns and rates[c].dropna().any()]
curve_data = rates[available_tenors].dropna(how="all")

left, right = st.columns([1.05, 0.95])

with left:
    st.markdown("<div class='section-title'>Yield Curve Snapshot</div>", unsafe_allow_html=True)
    if len(available_tenors) < 2 or curve_data.empty:
        st.info("At least two tenors are needed for the curve snapshot.")
    else:
        latest_curve = curve_data.iloc[-1]
        last_idx = pd.Timestamp(curve_data.index[-1])
        compare_target = last_idx - pd.DateOffset(days=7)
        if curve_compare == "1M":
            compare_target = last_idx - pd.DateOffset(months=1)
        elif curve_compare == "3M":
            compare_target = last_idx - pd.DateOffset(months=3)
        elif curve_compare == "YTD":
            compare_target = pd.Timestamp(date(last_idx.year, 1, 1))

        comparison_curve = []
        for tenor in available_tenors:
            if curve_compare == "YTD":
                comparison_curve.append(first_value_on_or_after(curve_data[tenor], compare_target))
            else:
                comparison_curve.append(value_on_or_before(curve_data[tenor], compare_target))

        x_vals = [TENORS[c][1] for c in available_tenors]
        x_labels = [TENORS[c][0] for c in available_tenors]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=latest_curve.values, mode="lines+markers", name="Latest", line=dict(color=COLORS["blue"], width=3)))
        fig.add_trace(go.Scatter(x=x_vals, y=comparison_curve, mode="lines+markers", name=f"{curve_compare} ago", line=dict(color=COLORS["grey"], width=2, dash="dash")))
        fig.update_layout(xaxis=dict(title="Tenor", tickvals=x_vals, ticktext=x_labels), yaxis=dict(title="Yield (%)"))
        st.plotly_chart(clean_plot_layout(fig, 390), use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Pressure Matrix</div>", unsafe_allow_html=True)
    matrix_rows = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", selected_curve, real_col, "T10YIE"]
    matrix = period_matrix(rates, matrix_rows)
    if matrix.empty:
        st.info("No period matrix available.")
    else:
        heat_cols = list(PERIODS.keys())
        z = matrix[heat_cols].to_numpy(dtype=float)
        text = np.full(z.shape, "", dtype=object)
        finite_mask = np.isfinite(z)
        text[finite_mask] = np.round(z[finite_mask], 0).astype(int).astype(str)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=heat_cols,
                y=matrix["Series"],
                colorscale="RdYlGn_r",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="bps"),
            )
        )
        st.plotly_chart(clean_plot_layout(fig, 390), use_container_width=True)

st.markdown("<div class='section-title'>Outlier Rate Days</div>", unsafe_allow_html=True)
st.markdown("<div class='small-note'>Weighted daily shock score from 10Y level, selected curve, real yields, and breakevens. The goal is to flag days worth reviewing, not to predict the next move.</div>", unsafe_allow_html=True)
shock = build_shock_table(rates, selected_curve, lookback_days=min(len(rates), 300), top_n=10)
if shock.empty:
    st.info("Not enough history to calculate outlier days.")
else:
    chart = shock.copy()
    chart["Date Label"] = chart["Date"].astype(str)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart["Date Label"],
            y=chart["Shock Score"],
            text=chart["Driver"],
            textposition="outside",
            name="Shock score",
            marker=dict(color=COLORS["blue"]),
        )
    )
    fig.update_layout(xaxis=dict(title="Date"), yaxis=dict(title="Weighted bp shock"), showlegend=False)
    st.plotly_chart(clean_plot_layout(fig, 330), use_container_width=True)

    display_shock = shock.copy()
    for col in display_shock.columns:
        if col.endswith("bp") or col == "Shock Score":
            display_shock[col] = display_shock[col].map(lambda x: fmt_num(x, 0))
    st.dataframe(display_shock, use_container_width=True, hide_index=True)

if show_full_history:
    st.markdown("<div class='section-title'>Level + Curve History</div>", unsafe_allow_html=True)
    hist = make_subplots(specs=[[{"secondary_y": True}]])
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS10"], name="10Y", line=dict(color=COLORS["blue"], width=2)), secondary_y=False)
    if "DGS2" in rates.columns:
        hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS2"], name="2Y", line=dict(color=COLORS["purple"], width=1.6)), secondary_y=False)
    if real_col in rates.columns:
        hist.add_trace(go.Scatter(x=rates.index, y=rates[real_col], name="10Y Real", line=dict(color=COLORS["green"], width=1.4)), secondary_y=False)
    hist.add_trace(go.Scatter(x=rates.index, y=rates[selected_curve] * 100.0, name=f"{selected_curve} bp", line=dict(color=COLORS["amber"], width=2)), secondary_y=True)
    hist.update_yaxes(title_text="Yield (%)", secondary_y=False)
    hist.update_yaxes(title_text="Curve bps", secondary_y=True)
    st.plotly_chart(clean_plot_layout(hist, 430), use_container_width=True)

if show_table:
    st.markdown("<div class='section-title'>Raw Rates Table</div>", unsafe_allow_html=True)
    table_cols = [c for c in ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "DFII10", "T10YIE", "2s10s", "3m10y", "5s30s"] if c in rates.columns]
    table = rates[table_cols].tail(260).copy()
    rename = {**SERIES, "2s10s": "2s10s", "3m10y": "3m10y", "5s30s": "5s30s"}
    table = table.rename(columns=rename)
    st.dataframe(table, use_container_width=True)
