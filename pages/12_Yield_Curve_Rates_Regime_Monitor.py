from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

TITLE = "Yield Curve + Rates Regime Monitor"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (3, 8)

SERIES = {
    "DGS3MO": "3M",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
    "DFII10": "10Y Real",
    "T10YIE": "10Y Breakeven",
}

YAHOO_YIELD_FALLBACKS = {
    "DGS3MO": "^IRX",
    "DGS5": "^FVX",
    "DGS10": "^TNX",
    "DGS30": "^TYX",
}

TENOR_META = {
    "DGS3MO": ("3M", 0.25),
    "DGS2": ("2Y", 2),
    "DGS5": ("5Y", 5),
    "DGS10": ("10Y", 10),
    "DGS30": ("30Y", 30),
}
TENOR_ORDER = list(TENOR_META.keys())

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
        .block-container {padding-top: 3.0rem; padding-bottom: 2rem; max-width: 1550px;}
        .adfm-title {font-size: 1.9rem; line-height: 1.18; font-weight: 760; margin: 0 0 0.25rem 0; color: #111827;}
        .adfm-subtitle {font-size: 0.96rem; color: #6b7280; margin-bottom: 1.15rem;}
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 12px; padding: 13px 15px 10px 15px; min-height: 92px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
        .metric-label {font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.045em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.24rem; font-weight: 750; color: #111827; line-height: 1.18;}
        .metric-footnote {font-size: 0.75rem; color: #9ca3af; margin-top: 0.42rem; line-height: 1.35;}
        .section-title {font-size: 1.03rem; font-weight: 750; color: #111827; margin-top: 0.9rem; margin-bottom: 0.45rem;}
        .note-box {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; color: #475569; font-size: 0.86rem; line-height: 1.45;}
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


def card_color(regime: str) -> str:
    if "Bull" in regime or "Falling" in regime:
        return COLORS["green"]
    if "Bear" in regime or "Rising" in regime:
        return COLORS["red"]
    return COLORS["amber"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_map: Dict[str, str], start: date) -> pd.DataFrame:
    frames: List[pd.Series] = []
    for series_id in series_map:
        try:
            response = requests.get(
                FRED_URL,
                params={"id": series_id, "cosd": start.isoformat()},
                timeout=REQUEST_TIMEOUT,
            )
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_yields(period: str) -> pd.DataFrame:
    tickers = list(YAHOO_YIELD_FALLBACKS.values())
    try:
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
    out = pd.DataFrame(index=close.index)
    reverse = {v: k for k, v in YAHOO_YIELD_FALLBACKS.items()}
    for ticker, series_id in reverse.items():
        if ticker in close:
            out[series_id] = pd.to_numeric(close[ticker], errors="coerce") / 10.0
    return out.dropna(how="all").ffill()


def add_curves(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DGS10" in out and "DGS2" in out:
        out["2s10s"] = out["DGS10"] - out["DGS2"]
    if "DGS30" in out and "DGS5" in out:
        out["5s30s"] = out["DGS30"] - out["DGS5"]
    if "DGS10" in out and "DGS3MO" in out:
        out["3m10y"] = out["DGS10"] - out["DGS3MO"]
    if "DGS10" in out and "T10YIE" in out:
        out["Market Real 10Y"] = out["DGS10"] - out["T10YIE"]
    return out


def classify_rates(df: pd.DataFrame, window: int) -> tuple[str, str, str | None]:
    ten = change_bp(df["DGS10"], window) if "DGS10" in df else np.nan
    curve_col = next((c for c in ["2s10s", "3m10y", "5s30s"] if c in df), None)
    curve = change_bp(df[curve_col], window) if curve_col else np.nan

    if not np.isfinite(ten):
        return "Insufficient Data", "Need a 10Y yield series to classify the rates move.", curve_col
    if not np.isfinite(curve):
        if ten > 5:
            return "Rates Rising", "10Y yields are rising; curve data is incomplete in this session.", curve_col
        if ten < -5:
            return "Rates Falling", "10Y yields are falling; curve data is incomplete in this session.", curve_col
        return "Range / Mixed", "10Y yield change is not large enough to define a clean regime.", curve_col

    if ten > 5 and curve > 5:
        return "Bear Steepener", f"10Y yields rising and {curve_col} steepening.", curve_col
    if ten > 5 and curve < -5:
        return "Bear Flattener", f"10Y yields rising and {curve_col} flattening.", curve_col
    if ten < -5 and curve > 5:
        return "Bull Steepener", f"10Y yields falling and {curve_col} steepening.", curve_col
    if ten < -5 and curve < -5:
        return "Bull Flattener", f"10Y yields falling and {curve_col} flattening.", curve_col
    return "Range / Mixed", "Rates and curve changes are not large enough to define a clean regime.", curve_col


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
        Tracks the U.S. Treasury curve, real yields, breakevens, and rates regimes.
        FRED is used where available; Yahoo yield indices fill common Treasury gaps.
        """
    )

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Treasury curve shape, real yields, breakevens, and steepener/flattener regimes.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 120)
rates = fetch_fred(SERIES, start_date)
yahoo_rates = fetch_yahoo_yields(f"{lookback_years}y")

if rates.empty:
    rates = yahoo_rates.copy()
elif not yahoo_rates.empty:
    rates = rates.combine_first(yahoo_rates).ffill()
rates = add_curves(rates)

available_tenors = [c for c in TENOR_ORDER if c in rates and rates[c].dropna().any()]
if rates.empty or "DGS10" not in rates:
    st.error("No usable rates data loaded from FRED or Yahoo yield proxies.")
    st.stop()

regime, note, curve_col = classify_rates(rates, regime_window)

cards = [
    ("Rates Regime", regime, note, card_color(regime)),
    ("10Y Treasury", fmt_pct(latest(rates["DGS10"])), f"{regime_window}D {fmt_bp(change_bp(rates['DGS10'], regime_window))}", COLORS["blue"]),
    ("Curve Gauge", fmt_bp(latest(rates[curve_col]) * 100) if curve_col else "N/A", curve_col or "Curve unavailable", COLORS["purple"]),
    ("10Y Breakeven", fmt_pct(latest(rates["T10YIE"])) if "T10YIE" in rates else "N/A", f"{regime_window}D {fmt_bp(change_bp(rates['T10YIE'], regime_window))}" if "T10YIE" in rates else "FRED only", COLORS["amber"]),
    ("10Y Real Yield", fmt_pct(latest(rates["DFII10"])) if "DFII10" in rates else (fmt_pct(latest(rates["Market Real 10Y"])) if "Market Real 10Y" in rates else "N/A"), "FRED real or nominal less breakeven", COLORS["green"]),
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
    if len(available_tenors) < 2:
        st.info("At least two tenor series are needed to draw the curve snapshot.")
    else:
        curve_data = rates[available_tenors].dropna(how="all")
        latest_curve = curve_data.iloc[-1]
        comparison_curve = curve_data.iloc[max(0, len(curve_data) - comparison_window - 1)]
        one_year_curve = curve_data.iloc[max(0, len(curve_data) - 252 - 1)]
        x_vals = [TENOR_META[c][1] for c in available_tenors]
        x_labels = [TENOR_META[c][0] for c in available_tenors]
        curve_fig = go.Figure()
        curve_fig.add_trace(go.Scatter(x=x_vals, y=latest_curve.values, mode="lines+markers", name="Latest", line=dict(color=COLORS["blue"], width=3)))
        curve_fig.add_trace(go.Scatter(x=x_vals, y=comparison_curve.values, mode="lines+markers", name=f"{comparison_window}D ago", line=dict(color=COLORS["grey"], dash="dash")))
        curve_fig.add_trace(go.Scatter(x=x_vals, y=one_year_curve.values, mode="lines+markers", name="1Y ago", line=dict(color=COLORS["amber"], dash="dot")))
        curve_fig.update_layout(height=385, margin=dict(l=12, r=12, t=15, b=15), xaxis=dict(title="Tenor", tickvals=x_vals, ticktext=x_labels), yaxis=dict(title="Yield (%)"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(curve_fig, use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Rate Change Heatmap</div>", unsafe_allow_html=True)
    rows = []
    for series_id in available_tenors:
        label = TENOR_META[series_id][0]
        rows.append({"Tenor": label, "Latest": latest(rates[series_id]), "5D bp": change_bp(rates[series_id], 5), "21D bp": change_bp(rates[series_id], 21), "63D bp": change_bp(rates[series_id], 63), "126D bp": change_bp(rates[series_id], 126)})
    heat = pd.DataFrame(rows).set_index("Tenor") if rows else pd.DataFrame()
    if heat.empty:
        st.info("No tenor history available for the heatmap.")
    else:
        heat_cols = ["5D bp", "21D bp", "63D bp", "126D bp"]
        fig = go.Figure(data=go.Heatmap(z=heat[heat_cols].values, x=["5D", "21D", "63D", "126D"], y=heat.index, colorscale="RdYlGn_r", zmid=0, text=np.round(heat[heat_cols].values, 0), texttemplate="%{text:.0f}", colorbar=dict(title="bps")))
        fig.update_layout(height=385, margin=dict(l=12, r=12, t=15, b=15), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-title'>Rates Regime History</div>", unsafe_allow_html=True)
hist = make_subplots(specs=[[{"secondary_y": True}]])
if "DGS10" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS10"], name="10Y", line=dict(color=COLORS["blue"])), secondary_y=False)
if "DGS2" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS2"], name="2Y", line=dict(color=COLORS["purple"])), secondary_y=False)
if "DGS3MO" in rates:
    hist.add_trace(go.Scatter(x=rates.index, y=rates["DGS3MO"], name="3M", line=dict(color=COLORS["grey"])), secondary_y=False)
if curve_col:
    hist.add_trace(go.Scatter(x=rates.index, y=rates[curve_col] * 100, name=f"{curve_col} bps", line=dict(color=COLORS["amber"])), secondary_y=True)
hist.update_layout(height=420, margin=dict(l=12, r=12, t=20, b=15), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
hist.update_yaxes(title_text="Yield (%)", secondary_y=False)
hist.update_yaxes(title_text="Curve bps", secondary_y=True)
st.plotly_chart(hist, use_container_width=True)

if show_table and rows:
    st.markdown("<div class='section-title'>Rates Table</div>", unsafe_allow_html=True)
    display = pd.DataFrame(rows)
    display["Latest"] = display["Latest"].map(lambda x: f"{x:.2f}%" if np.isfinite(x) else "N/A")
    for col in ["5D bp", "21D bp", "63D bp", "126D bp"]:
        display[col] = display[col].map(lambda x: f"{x:+.0f}" if np.isfinite(x) else "N/A")
    st.dataframe(display, use_container_width=True, hide_index=True)
