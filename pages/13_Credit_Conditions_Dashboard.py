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

TITLE = "Credit Conditions Dashboard"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (3, 8)

FRED_SERIES = {
    "BAMLH0A0HYM2": "HY OAS",
    "BAMLC0A0CM": "IG OAS",
    "BAMLH0A3HYC": "CCC OAS",
    "NFCI": "Chicago Fed NFCI",
}

MARKET_TICKERS = ["HYG", "JNK", "LQD", "BKLN", "EMB", "KRE", "SPY", "TLT", "IEF", "SHY"]

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
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%); border: 1px solid #e5e7eb; border-radius: 12px; padding: 13px 15px 10px 15px; min-height: 96px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
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


def drawdown(series: pd.Series, window: int = 126) -> float:
    clean = series.dropna().tail(window)
    if clean.empty:
        return np.nan
    return float(clean.iloc[-1] / clean.cummax().iloc[-1] - 1.0)


def percentile_score(series: pd.Series, invert: bool = False) -> float:
    clean = series.dropna()
    if len(clean) < 30:
        return np.nan
    score = (float(clean.rank(pct=True).iloc[-1]) - 0.5) * 2.0
    return -score if invert else score


def mean_score(parts: List[float]) -> float:
    clean = [x for x in parts if np.isfinite(x)]
    return float(np.mean(clean)) if clean else 0.0


def fmt_pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def fmt_bps(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.0f} bps"


def signal_color(score: float) -> str:
    if score <= -0.35:
        return COLORS["green"]
    if score >= 0.35:
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
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    try:
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all").ffill()


def classify_credit(score: float) -> str:
    if score <= -0.55:
        return "Easy / Risk-On"
    if score <= -0.15:
        return "Constructive"
    if score < 0.35:
        return "Neutral / Watch"
    if score < 0.65:
        return "Tightening"
    return "Stress / De-risk"


with st.sidebar:
    st.header("Settings")
    fred_years = st.selectbox("Spread history", [3, 5, 10, 20], index=2)
    market_period = st.selectbox("Market proxy history", ["1y", "2y", "3y", "5y"], index=2)
    signal_window = st.selectbox("Signal window", [21, 63, 126], index=1, format_func=lambda x: f"{x} trading days")
    show_raw = st.checkbox("Show raw data", value=False)

    st.divider()
    st.header("About This Tool")
    st.markdown(
        """
        Tracks credit spreads when FRED is available and always falls back to
        liquid credit-market proxies: HY vs IG, banks vs equities, loans, EM debt,
        and duration-sensitive credit pressure.
        """
    )

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Credit spreads, credit ETF ratios, bank stress proxies, and early warning conditions.</div>",
    unsafe_allow_html=True,
)

start = date.today() - timedelta(days=int(fred_years * 365.25) + 120)
fred = fetch_fred(FRED_SERIES, start)
market = fetch_market(MARKET_TICKERS, market_period)

if fred.empty and market.empty:
    st.error("No credit or market data loaded. Check data-source connectivity.")
    st.stop()

score_parts = []
rows = []
proxy = pd.DataFrame(index=market.index) if not market.empty else pd.DataFrame()

if "BAMLH0A0HYM2" in fred:
    hy = fred["BAMLH0A0HYM2"]
    score_parts.append(percentile_score(hy))
    rows.append(["HY OAS", fmt_bps(latest(hy) * 100), f"{signal_window}D {change(hy, signal_window) * 100:+.0f} bps", "Wider is tighter"])
if "BAMLC0A0CM" in fred:
    ig = fred["BAMLC0A0CM"]
    score_parts.append(percentile_score(ig))
    rows.append(["IG OAS", fmt_bps(latest(ig) * 100), f"{signal_window}D {change(ig, signal_window) * 100:+.0f} bps", "Wider is tighter"])
if "BAMLH0A3HYC" in fred:
    ccc = fred["BAMLH0A3HYC"]
    score_parts.append(percentile_score(ccc))
    rows.append(["CCC OAS", fmt_bps(latest(ccc) * 100), f"{signal_window}D {change(ccc, signal_window) * 100:+.0f} bps", "Lower-quality stress"])
if "NFCI" in fred:
    nfci = fred["NFCI"]
    score_parts.append(percentile_score(nfci))
    rows.append(["NFCI", f"{latest(nfci):.2f}", f"{signal_window}D {change(nfci, signal_window):+.2f}", "Higher is tighter"])

if not market.empty:
    if "HYG" in market and "LQD" in market:
        proxy["HYG/LQD"] = market["HYG"] / market["LQD"]
        score_parts.append(np.clip(-pct_change(proxy["HYG/LQD"], signal_window) * 8, -1, 1))
        rows.append(["HYG/LQD", fmt_pct(pct_change(proxy["HYG/LQD"], signal_window) * 100), f"{signal_window}D relative return", "HY beta vs IG"])
    if "JNK" in market and "LQD" in market:
        proxy["JNK/LQD"] = market["JNK"] / market["LQD"]
        score_parts.append(np.clip(-pct_change(proxy["JNK/LQD"], signal_window) * 8, -1, 1))
    if "KRE" in market and "SPY" in market:
        proxy["KRE/SPY"] = market["KRE"] / market["SPY"]
        score_parts.append(np.clip(-pct_change(proxy["KRE/SPY"], signal_window) * 6, -1, 1))
        rows.append(["KRE/SPY", fmt_pct(pct_change(proxy["KRE/SPY"], signal_window) * 100), f"{signal_window}D relative return", "Bank stress proxy"])
    if "BKLN" in market and "LQD" in market:
        proxy["BKLN/LQD"] = market["BKLN"] / market["LQD"]
        rows.append(["BKLN/LQD", fmt_pct(pct_change(proxy["BKLN/LQD"], signal_window) * 100), f"{signal_window}D relative return", "Loan risk appetite"])
    if "EMB" in market and "LQD" in market:
        proxy["EMB/LQD"] = market["EMB"] / market["LQD"]
        score_parts.append(np.clip(-pct_change(proxy["EMB/LQD"], signal_window) * 7, -1, 1))
    if "HYG" in market:
        score_parts.append(np.clip(-drawdown(market["HYG"], 126) * 5, -1, 1))
        rows.append(["HYG drawdown", fmt_pct(drawdown(market["HYG"], 126) * 100), "6M drawdown from high", "Absolute credit stress"])

credit_score = mean_score(score_parts)
credit_regime = classify_credit(credit_score)
hy_oas = latest(fred["BAMLH0A0HYM2"]) if "BAMLH0A0HYM2" in fred else np.nan
ig_oas = latest(fred["BAMLC0A0CM"]) if "BAMLC0A0CM" in fred else np.nan
ccc_oas = latest(fred["BAMLH0A3HYC"]) if "BAMLH0A3HYC" in fred else np.nan
nfci_latest = latest(fred["NFCI"]) if "NFCI" in fred else np.nan

primary_proxy = pct_change(proxy["HYG/LQD"], signal_window) * 100 if "HYG/LQD" in proxy else np.nan
bank_proxy = pct_change(proxy["KRE/SPY"], signal_window) * 100 if "KRE/SPY" in proxy else np.nan
hyg_dd = drawdown(market["HYG"], 126) * 100 if not market.empty and "HYG" in market else np.nan

cards = [
    ("Credit Regime", credit_regime, f"Composite score {credit_score:+.2f}", signal_color(credit_score)),
    ("HY OAS", fmt_bps(hy_oas * 100) if np.isfinite(hy_oas) else "FRED N/A", "Spread source", COLORS["blue"]),
    ("HYG/LQD", fmt_pct(primary_proxy), f"{signal_window}D relative", COLORS["green"] if np.isfinite(primary_proxy) and primary_proxy > 0 else COLORS["red"]),
    ("KRE/SPY", fmt_pct(bank_proxy), f"{signal_window}D relative", COLORS["green"] if np.isfinite(bank_proxy) and bank_proxy > 0 else COLORS["red"]),
    ("HYG Drawdown", fmt_pct(hyg_dd), "6M drawdown", COLORS["red"] if np.isfinite(hyg_dd) and hyg_dd < -3 else COLORS["amber"]),
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
    if any(c in fred for c in ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC"]):
        st.markdown("<div class='section-title'>Spread Dashboard</div>", unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "BAMLH0A0HYM2" in fred:
            fig.add_trace(go.Scatter(x=fred.index, y=fred["BAMLH0A0HYM2"] * 100, name="HY OAS", line=dict(color=COLORS["red"])), secondary_y=False)
        if "BAMLC0A0CM" in fred:
            fig.add_trace(go.Scatter(x=fred.index, y=fred["BAMLC0A0CM"] * 100, name="IG OAS", line=dict(color=COLORS["blue"])), secondary_y=False)
        if "BAMLH0A3HYC" in fred:
            fig.add_trace(go.Scatter(x=fred.index, y=fred["BAMLH0A3HYC"] * 100, name="CCC OAS", line=dict(color=COLORS["purple"])), secondary_y=True)
        fig.update_yaxes(title_text="HY / IG OAS bps", secondary_y=False)
        fig.update_yaxes(title_text="CCC OAS bps", secondary_y=True)
    else:
        st.markdown("<div class='section-title'>Credit Proxy Dashboard</div>", unsafe_allow_html=True)
        if proxy.empty:
            st.info("No market proxy data loaded.")
            fig = None
        else:
            rebased = proxy.dropna(how="all")
            rebased = rebased / rebased.ffill().bfill().iloc[0] * 100
            fig = go.Figure()
            for col_name in rebased:
                fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col_name], mode="lines", name=col_name))
            fig.update_yaxes(title_text="Rebased to 100")
    if 'fig' in locals() and fig is not None:
        fig.update_layout(height=390, margin=dict(l=12, r=12, t=15, b=15), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Credit Signal Table</div>", unsafe_allow_html=True)
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Signal", "Latest / Change", "Window", "Read-through"]), use_container_width=True, hide_index=True, height=390)
    else:
        st.info("No credit signals loaded yet.")

st.markdown("<div class='section-title'>Market Credit Proxies</div>", unsafe_allow_html=True)
if proxy.empty:
    st.info("Market proxy data unavailable.")
else:
    rebased = proxy.dropna(how="all")
    rebased = rebased / rebased.ffill().bfill().iloc[0] * 100
    proxy_fig = go.Figure()
    for col_name in rebased:
        proxy_fig.add_trace(go.Scatter(x=rebased.index, y=rebased[col_name], mode="lines", name=col_name))
    proxy_fig.update_layout(height=390, margin=dict(l=12, r=12, t=15, b=15), yaxis_title="Rebased to 100", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(proxy_fig, use_container_width=True)

if fred.empty:
    st.markdown(
        "<div class='note-box'>FRED credit-spread series did not load in this session, so the dashboard is using liquid ETF proxies only.</div>",
        unsafe_allow_html=True,
    )

if show_raw and not fred.empty:
    st.markdown("<div class='section-title'>Raw Credit Data</div>", unsafe_allow_html=True)
    st.dataframe(fred.tail(160), use_container_width=True)
