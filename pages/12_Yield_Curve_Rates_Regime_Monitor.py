from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

TITLE = "Yield Curve + Rates Regime Monitor"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (4, 12)

SERIES = {
    "DGS3MO": "3M",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
    "DFII10": "10Y Real",
    "T10YIE": "10Y Breakeven",
    "FEDFUNDS": "Fed Funds",
}

TENOR_ORDER = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]
TENOR_LABELS = ["3M", "2Y", "5Y", "10Y", "30Y"]
TENOR_YEARS = [0.25, 2, 5, 10, 30]

COLORS = {
    "green": "#52b788",
    "red": "#e85d5d",
    "amber": "#f59e0b",
    "blue": "#4c78a8",
    "purple": "#8b5cf6",
    "grey": "#8b949e",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1550px;}
        .adfm-title {font-size: 1.85rem; font-weight: 750; margin-bottom: 0.1rem; color: #111827;}
        .adfm-subtitle {font-size: 0.96rem; color: #6b7280; margin-bottom: 1.1rem;}
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 14px; padding: 13px 15px 10px 15px; min-height: 92px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
        .metric-label {font-size: 0.74rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.045em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.32rem; font-weight: 750; color: #111827; line-height: 1.12;}
        .metric-footnote {font-size: 0.76rem; color: #9ca3af; margin-top: 0.42rem;}
        .section-title {font-size: 1.03rem; font-weight: 750; color: #111827; margin-top: 0.5rem; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def change_bp(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float((clean.iloc[-1] - clean.iloc[-periods - 1]) * 100)


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.2f}%"


def fmt_bp(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:+.0f} bps"


def regime_color(regime: str) -> str:
    if "Bull" in regime:
        return COLORS["green"]
    if "Bear" in regime:
        return COLORS["red"]
    return COLORS["amber"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_map: Dict[str, str], start: date) -> pd.DataFrame:
    frames: List[pd.Series] = []
    for series_id in series_map:
        try:
            response = requests.get(FRED_URL, params={"id": series_id, "cosd": start.isoformat()}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text))
            if raw.empty or len(raw.columns) < 2:
                continue
            raw.columns = ["date", series_id]
            values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")
            frames.append(pd.Series(values.values, index=pd.to_datetime(raw["date"]), name=series_id))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().ffill()


def classify_rates(df: pd.DataFrame, window: int) -> tuple[str, str]:
    ten = change_bp(df["DGS10"], window) if "DGS10" in df else np.nan
    curve = change_bp(df["2s10s"], window) if "2s10s" in df else np.nan
    if not np.isfinite(ten) or not np.isfinite(curve):
        return "Insufficient Data", "Need more yield history to classify the rates move."
    if ten > 5 and curve > 5:
        return "Bear Steepener", "Long rates rising faster than front-end rates; inflation, supply, or term-premium pressure."
    if ten > 5 and curve < -5:
        return "Bear Flattener", "Rates rising while the curve flattens; policy-tightening pressure."
    if ten < -5 and curve > 5:
        return "Bull Steepener", "Rates falling while the curve steepens; growth scare or easing expectation behavior."
    if ten < -5 and curve < -5:
        return "Bull Flattener", "Rates falling while the curve flattens; duration bid with slower front-end repricing."
    return "Range / Mixed", "Rates and curve changes are not large enough to define a clean regime."


with st.sidebar:
    st.header("Settings")
    lookback_years = st.selectbox("History", [1, 2, 3, 5, 10, 20], index=3)
    regime_window = st.selectbox("Regime window", [5, 21, 63, 126], index=1, format_func=lambda x: f"{x} trading days")
    comparison_window = st.selectbox("Curve comparison", [21, 63, 126, 252], index=1, format_func=lambda x: f"{x} trading days ago")
    show_table = st.checkbox("Show rates table", value=True)

    st.divider()
    st.header("About This Tool")
    st.markdown(
        """
        Tracks the U.S. Treasury curve, real yields, breakevens, and front-end policy rate.

        The regime label is based on the joint move in 10Y yields and the 2s10s curve
        over the selected window.
        """
    )

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Treasury curve shape, real yields, breakevens, and steepener/flattener regimes.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 120)
rates = fetch_fred(SERIES, start_date)
if rates.empty:
    st.error("No FRED rates data loaded. Check data-source connectivity.")
    st.stop()

if "DGS10" in rates and "DGS2" in rates:
    rates["2s10s"] = rates["DGS10"] - rates["DGS2"]
if "DGS30" in rates and "DGS5" in rates:
    rates["5s30s"] = rates["DGS30"] - rates["DGS5"]
if "DGS10" in rates and "DGS3MO" in rates:
    rates["3m10y"] = rates["DGS10"] - rates["DGS3MO"]
if "DGS10" in rates and "T10YIE" in rates:
    rates["Market Real 10Y"] = rates["DGS10"] - rates["T10YIE"]

regime, note = classify_rates(rates, regime_window)

cards = [
    ("Rates Regime", regime, note, regime_color(regime)),
    ("10Y Treasury", fmt_pct(latest(rates["DGS10"])) if "DGS10" in rates else "N/A", f"{regime_window}D {fmt_bp(change_bp(rates['DGS10'], regime_window))}" if "DGS10" in rates else "N/A", COLORS["blue"]),
    ("2s10s Curve", fmt_bp(latest(rates["2s10s"]) * 100) if "2s10s" in rates else "N/A", f"{regime_window}D {fmt_bp(change_bp(rates['2s10s'], regime_window))}" if "2s10s" in rates else "N/A", COLORS["purple"]),
    ("10Y Breakeven", fmt_pct(latest(rates["T10YIE"])) if "T10YIE" in rates else "N/A", f"{regime_window}D {fmt_bp(change_bp(rates['T10YIE'], regime_window))}" if "T10YIE" in rates else "N/A", COLORS["amber"]),
    ("10Y Real Yield", fmt_pct(latest(rates["DFII10"])) if "DFII10" in rates else "N/A", f"{regime_window}D {fmt_bp(change_bp(rates['DFII10'], regime_window))}" if "DFII10" in rates else "N/A", COLORS["green"]),
]

for col, (label, value, footnote, color) in zip(st.columns(5), cards):
    with col:
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

left, right = st.columns([1.05, 0.95])
with left:
    st.markdown("<div class='section-title'>Yield Curve Snapshot</div>", unsafe_allow_html=True)
    curve_data = rates[TENOR_ORDER].dropna(how="all")
    latest_curve = curve_data.iloc[-1]
    comparison_curve = curve_data.iloc[max(0, len(curve_data) - comparison_window - 1)]
    one_year_curve = curve_data.iloc[max(0, len(curve_data) - 252 - 1)]
    curve_fig = go.Figure()
    curve_fig.add_trace(go.Scatter(x=TENOR_YEARS, y=latest_curve.values, mode="lines+markers", name="Latest", line=dict(color=COLORS["blue"], width=3)))
    curve_fig.add_trace(go.Scatter(x=TENOR_YEARS, y=comparison_curve.values, mode="lines+markers", name=f"{comparison_window}D ago", line=dict(color=COLORS["grey"], dash="dash")))
    curve_fig.add_trace(go.Scatter(x=TENOR_YEARS, y=one_year_curve.values, mode="lines+markers", name="1Y ago", line=dict(color=COLORS["amber"], dash="dot")))
    curve_fig.update_layout(height=410, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(title="Tenor", tickvals=TENOR_YEARS, ticktext=TENOR_LABELS), yaxis=dict(title="Yield (%)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(curve_fig, use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Curve Change Heatmap</div>", unsafe_allow_html=True)
    rows = []
    for series_id, label in zip(TENOR_ORDER, TENOR_LABELS):
        if series_id in rates:
            rows.append({"Tenor": label, "Latest": latest(rates[series_id]), "5D bp": change_bp(rates[series_id], 5), "21D bp": change_bp(rates[series_id], 21), "63D bp": change_bp(rates[series_id], 63), "126D bp": change_bp(rates[series_id], 126)})
    heat = pd.DataFrame(rows).set_index("Tenor")
    fig = go.Figure(data=go.Heatmap(z=heat[["5D bp", "21D bp", "63D bp", "126D bp"]].values, x=["5D", "21D", "63D", "126D"], y=heat.index, colorscale="RdYlGn_r", zmid=0, text=np.round(heat[["5D bp", "21D bp", "63D bp", "126D bp"]].values, 0), texttemplate="%{text:.0f}", colorbar=dict(title="bps")))
    fig.update_layout(height=410, margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-title'>Rates Regime History</div>", unsafe_allow_html=True)
hist = make_subplots(specs=[[{"secondary_y": True}]])
if "DGS10" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS10"], name="10Y", line=dict(color=COLORS["blue"])), secondary_y=False)
if "DGS2" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS2"], name="2Y", line=dict(color=COLORS["purple"])), secondary_y=False)
if "2s10s" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["2s10s"] * 100, name="2s10s bps", line=dict(color=COLORS["amber"])), secondary_y=True)
if "DFII10" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DFII10"], name="10Y Real", line=dict(color=COLORS["green"])), secondary_y=False)
hist.update_layout(height=460, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
hist.update_yaxes(title_text="Yield (%)", secondary_y=False)
hist.update_yaxes(title_text="Curve bps", secondary_y=True)
st.plotly_chart(hist, use_container_width=True)

if show_table:
    st.markdown("<div class='section-title'>Rates Table</div>", unsafe_allow_html=True)
    display = pd.DataFrame(rows)
    if not display.empty:
        display["Latest"] = display["Latest"].map(lambda x: f"{x:.2f}%")
        for col in ["5D bp", "21D bp", "63D bp", "126D bp"]:
            display[col] = display[col].map(lambda x: f"{x:+.0f}" if np.isfinite(x) else "N/A")
        st.dataframe(display, use_container_width=True, hide_index=True)
