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

TITLE = "Global Macro Regime Dashboard"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (4, 12)

FRED_SERIES = {
    "NAPM": "ISM Manufacturing PMI",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Claims",
    "CPIAUCSL": "Headline CPI",
    "CPILFESL": "Core CPI",
    "T10YIE": "10Y Breakeven",
    "DGS10": "10Y Treasury",
    "FEDFUNDS": "Fed Funds",
    "NFCI": "Chicago Fed NFCI",
}

MARKET_TICKERS = ["SPY", "HYG", "LQD", "UUP", "GLD", "CPER", "TLT", "^VIX"]

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
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 14px; padding: 13px 15px 10px 15px; min-height: 96px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
        .metric-label {font-size: 0.74rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.045em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.28rem; font-weight: 750; color: #111827; line-height: 1.12;}
        .metric-footnote {font-size: 0.76rem; color: #9ca3af; margin-top: 0.42rem;}
        .section-title {font-size: 1.03rem; font-weight: 750; color: #111827; margin-top: 0.5rem; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def pct_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-periods - 1] - 1.0)


def percentile_score(series: pd.Series, invert: bool = False) -> float:
    clean = series.dropna()
    if len(clean) < 12:
        return 0.0
    score = (float(clean.rank(pct=True).iloc[-1]) - 0.5) * 2.0
    return -score if invert else score


def score_color(score: float) -> str:
    if score >= 0.35:
        return COLORS["green"]
    if score <= -0.35:
        return COLORS["red"]
    return COLORS["amber"]


def fmt_pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"].ffill()
    return data[["Close"]].rename(columns={"Close": tickers[0]}).ffill()


def classify(growth: float, inflation: float, policy: float, risk: float) -> tuple[str, str]:
    if growth >= 0.25 and inflation <= 0.15 and risk >= 0:
        return "Goldilocks / Expansion", "Growth firm, inflation manageable, risk appetite constructive."
    if growth >= 0.15 and inflation > 0.15:
        return "Reflation / Policy Watch", "Growth is holding up, but inflation keeps rates-sensitive assets exposed."
    if growth < -0.15 and inflation > 0.15:
        return "Stagflation Pressure", "Growth is fading while inflation remains sticky."
    if growth < -0.15 and risk < -0.15:
        return "Slowdown / Risk-Off", "Growth and market signals are both deteriorating."
    if policy < -0.45 and risk < 0:
        return "Tight Financial Conditions", "Policy and financial conditions are leaning restrictive."
    return "Transition / Mixed Macro", "Signals are cross-current; watch breadth, credit, and rates confirmation."


with st.sidebar:
    st.header("Settings")
    lookback_years = st.selectbox("Macro history", [3, 5, 10, 20], index=1)
    market_period = st.selectbox("Market history", ["1y", "2y", "3y", "5y"], index=2)
    smoothing = st.slider("Monthly smoothing", 1, 6, 3)
    show_raw = st.checkbox("Show raw data", value=False)

    st.divider()
    st.header("About This Tool")
    st.markdown(
        """
        Broad macro regime dashboard built from public growth, inflation,
        policy, financial-conditions, and market proxies.

        Scores are directional and percentile-based. They are intended to
        organize the macro backdrop, not replace judgment around individual releases.
        """
    )

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Growth, inflation, policy, financial conditions, and risk appetite in one macro map.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 120)
fred = fetch_fred(FRED_SERIES, start_date)
market = fetch_market(MARKET_TICKERS, market_period)

if fred.empty and market.empty:
    st.error("No macro or market data loaded. Check data-source connectivity.")
    st.stop()

monthly = fred.resample("M").last().ffill() if not fred.empty else pd.DataFrame()
if smoothing > 1 and not monthly.empty:
    monthly = monthly.rolling(smoothing, min_periods=1).mean()

rows = []
growth_parts = []
inflation_parts = []
policy_parts = []
risk_parts = []

if "NAPM" in monthly:
    growth_parts.append(percentile_score(monthly["NAPM"]))
    rows.append(["ISM PMI", f"{latest(monthly['NAPM']):.1f}", f"3M change {change(monthly['NAPM'], 3):+.1f}"])
if "UNRATE" in monthly:
    growth_parts.append(np.clip(-change(monthly["UNRATE"], 6), -1, 1))
    rows.append(["Unemployment", fmt_pct(latest(monthly["UNRATE"])), f"6M change {change(monthly['UNRATE'], 6):+.1f} pts"])
if "ICSA" in monthly:
    claims = monthly["ICSA"].rolling(3, min_periods=1).mean()
    growth_parts.append(np.clip(-pct_change(claims, 6) * 4, -1, 1))
    rows.append(["Initial Claims", f"{latest(claims) / 1000:.0f}k", f"6M change {pct_change(claims, 6):+.1%}"])

if "CPIAUCSL" in monthly:
    cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
    inflation_parts.append(percentile_score(cpi_yoy))
    rows.append(["Headline CPI YoY", fmt_pct(latest(cpi_yoy)), f"3M change {change(cpi_yoy, 3):+.1f} pts"])
if "CPILFESL" in monthly:
    core_yoy = monthly["CPILFESL"].pct_change(12) * 100
    inflation_parts.append(percentile_score(core_yoy))
    rows.append(["Core CPI YoY", fmt_pct(latest(core_yoy)), f"3M change {change(core_yoy, 3):+.1f} pts"])
if "T10YIE" in monthly:
    inflation_parts.append(percentile_score(monthly["T10YIE"]))
    rows.append(["10Y Breakeven", fmt_pct(latest(monthly["T10YIE"])), f"6M change {change(monthly['T10YIE'], 6) * 100:+.0f} bps"])

if "DGS10" in monthly and "CPIAUCSL" in monthly:
    real_rate = monthly["DGS10"] - monthly["CPIAUCSL"].pct_change(12) * 100
    policy_parts.append(percentile_score(real_rate, invert=True))
    rows.append(["Ex-post Real 10Y", fmt_pct(latest(real_rate)), f"6M change {change(real_rate, 6):+.1f} pts"])
if "NFCI" in monthly:
    policy_parts.append(percentile_score(monthly["NFCI"], invert=True))
    rows.append(["NFCI", f"{latest(monthly['NFCI']):.2f}", f"6M change {change(monthly['NFCI'], 6):+.2f}"])

if not market.empty:
    if "SPY" in market:
        risk_parts.append(np.clip(pct_change(market["SPY"], 63) * 5, -1, 1))
        rows.append(["SPY 3M", fmt_pct(pct_change(market["SPY"], 63) * 100), "Risk appetite"])
    if "HYG" in market and "LQD" in market:
        ratio = market["HYG"] / market["LQD"]
        risk_parts.append(np.clip(pct_change(ratio, 63) * 8, -1, 1))
        rows.append(["HYG/LQD 3M", fmt_pct(pct_change(ratio, 63) * 100), "Credit risk appetite"])
    if "^VIX" in market:
        risk_parts.append(percentile_score(market["^VIX"], invert=True))
        rows.append(["VIX", f"{latest(market['^VIX']):.1f}", "Lower is easier"])
    if "UUP" in market:
        risk_parts.append(np.clip(-pct_change(market["UUP"], 63) * 8, -1, 1))
        rows.append(["Dollar 3M", fmt_pct(pct_change(market["UUP"], 63) * 100), "Higher can tighten global liquidity"])

growth_score = float(np.nanmean(growth_parts)) if growth_parts else 0.0
inflation_score = float(np.nanmean(inflation_parts)) if inflation_parts else 0.0
policy_score = float(np.nanmean(policy_parts)) if policy_parts else 0.0
risk_score = float(np.nanmean(risk_parts)) if risk_parts else 0.0
composite = float(np.nanmean([growth_score, -inflation_score, policy_score, risk_score]))
regime, regime_note = classify(growth_score, inflation_score, policy_score, risk_score)

cards = [
    ("Regime", regime, regime_note, score_color(composite)),
    ("Growth", f"{growth_score:+.2f}", "Positive means improving/firm", score_color(growth_score)),
    ("Inflation", f"{inflation_score:+.2f}", "Positive means more pressure", score_color(-inflation_score)),
    ("Policy / Conditions", f"{policy_score:+.2f}", "Positive means easier", score_color(policy_score)),
    ("Risk Appetite", f"{risk_score:+.2f}", "Positive means constructive tape", score_color(risk_score)),
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

st.markdown("<div class='section-title'>Macro Scoreboard</div>", unsafe_allow_html=True)
score_fig = go.Figure(
    go.Bar(
        x=["Growth", "Inflation Pressure", "Policy / Conditions", "Risk Appetite", "Composite"],
        y=[growth_score, inflation_score, policy_score, risk_score, composite],
        marker_color=[score_color(growth_score), score_color(-inflation_score), score_color(policy_score), score_color(risk_score), score_color(composite)],
    )
)
score_fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
score_fig.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[-1, 1], title="Score"), plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(score_fig, use_container_width=True)

left, right = st.columns([1.05, 0.95])
with left:
    st.markdown("<div class='section-title'>Key Macro Inputs</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows, columns=["Input", "Latest", "Read-through"]), use_container_width=True, hide_index=True)

with right:
    st.markdown("<div class='section-title'>Market Context</div>", unsafe_allow_html=True)
    if market.empty:
        st.info("Market data unavailable.")
    else:
        rebased = market.dropna(how="all").copy()
        rebased = rebased / rebased.ffill().bfill().iloc[0] * 100
        fig = go.Figure()
        for ticker in ["SPY", "HYG", "LQD", "UUP", "GLD", "TLT"]:
            if ticker in rebased:
                fig.add_trace(go.Scatter(x=rebased.index, y=rebased[ticker], mode="lines", name=ticker))
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Rebased to 100", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

if not monthly.empty:
    st.markdown("<div class='section-title'>Growth, Inflation, Policy History</div>", unsafe_allow_html=True)
    history_fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "NAPM" in monthly:
        history_fig.add_trace(go.Scatter(x=monthly.index, y=monthly["NAPM"], name="ISM PMI", line=dict(color=COLORS["blue"])), secondary_y=False)
    if "CPIAUCSL" in monthly:
        cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
        history_fig.add_trace(go.Scatter(x=cpi_yoy.index, y=cpi_yoy, name="CPI YoY", line=dict(color=COLORS["red"])), secondary_y=True)
    if "DGS10" in monthly:
        history_fig.add_trace(go.Scatter(x=monthly.index, y=monthly["DGS10"], name="10Y Yield", line=dict(color=COLORS["purple"])), secondary_y=True)
    history_fig.update_layout(height=430, margin=dict(l=20, r=20, t=25, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    history_fig.update_yaxes(title_text="ISM", secondary_y=False)
    history_fig.update_yaxes(title_text="Percent", secondary_y=True)
    st.plotly_chart(history_fig, use_container_width=True)

if show_raw:
    st.markdown("<div class='section-title'>Raw FRED Data</div>", unsafe_allow_html=True)
    st.dataframe(fred.tail(120), use_container_width=True)
