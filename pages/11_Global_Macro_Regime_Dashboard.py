from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

TITLE = "Global Macro Regime"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (3, 8)

# Keep the data set deliberately small. These are enough to cover growth,
# inflation, rates, financial conditions, credit, dollar, commodities, and risk.
FRED_SERIES: Dict[str, str] = {
    "NAPM": "ISM Manufacturing PMI",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Claims",
    "CPIAUCSL": "Headline CPI",
    "CPILFESL": "Core CPI",
    "T10YIE": "10Y Breakeven",
    "DGS10": "10Y Treasury",
    "NFCI": "Chicago Fed NFCI",
}

MARKET_TICKERS = ["SPY", "HYG", "LQD", "UUP", "GLD", "CPER", "TLT", "^VIX"]
DISPLAY_TICKERS = ["SPY", "HYG/LQD", "TLT", "UUP", "GLD", "CPER", "^VIX"]

COLORS = {
    "green": "#2f9e44",
    "red": "#d9480f",
    "amber": "#f59f00",
    "blue": "#1971c2",
    "purple": "#7048e8",
    "grey": "#868e96",
    "dark": "#111827",
    "muted": "#6b7280",
    "border": "#e5e7eb",
    "bg": "#f8fafc",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.4rem; padding-bottom: 2.0rem; max-width: 1550px;}
        .adfm-title {font-size: 1.85rem; line-height: 1.15; font-weight: 780; margin: 0 0 0.20rem 0; color: #111827;}
        .adfm-subtitle {font-size: 0.94rem; color: #6b7280; margin-bottom: 1.00rem;}
        .metric-card {background: linear-gradient(180deg, #ffffff 0%, #fbfbfb 100%); border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 15px 12px 15px; min-height: 106px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);}
        .metric-label {font-size: 0.71rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.055em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.18rem; font-weight: 760; color: #111827; line-height: 1.18;}
        .metric-footnote {font-size: 0.76rem; color: #6b7280; margin-top: 0.48rem; line-height: 1.34;}
        .section-title {font-size: 1.02rem; font-weight: 760; color: #111827; margin-top: 0.82rem; margin-bottom: 0.44rem;}
        .note-box {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 11px 13px; color: #475569; font-size: 0.85rem; line-height: 1.42;}
        .pill {display:inline-block; border: 1px solid #e5e7eb; border-radius: 999px; padding: 2px 8px; font-size: 0.74rem; color:#6b7280; background:#ffffff; margin-right: 5px;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; padding: 10px 12px; border-radius: 12px;}
        hr {margin-top: 0.7rem; margin-bottom: 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    idx = df.dropna(how="all").index
    if len(idx) == 0:
        return "N/A"
    return pd.Timestamp(idx[-1]).strftime("%b %d, %Y")


def delta(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def ret(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    base = float(clean.iloc[-periods - 1])
    if base == 0 or not np.isfinite(base):
        return np.nan
    return float(clean.iloc[-1] / base - 1.0)


def ytd_ret(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    latest_idx = pd.Timestamp(clean.index[-1])
    start = clean[clean.index >= pd.Timestamp(date(latest_idx.year, 1, 1))]
    if len(start) < 2:
        return np.nan
    base = float(start.iloc[0])
    if base == 0 or not np.isfinite(base):
        return np.nan
    return float(clean.iloc[-1] / base - 1.0)


def percentile_score(series: pd.Series, invert: bool = False, min_obs: int = 18) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_obs:
        return np.nan
    score = (float(clean.rank(pct=True).iloc[-1]) - 0.5) * 2.0
    return -score if invert else score


def clip_score(value: float, scale: float, invert: bool = False) -> float:
    if not np.isfinite(value) or scale == 0:
        return np.nan
    score = float(np.clip(value / scale, -1.0, 1.0))
    return -score if invert else score


def mean_score(parts: Iterable[float]) -> float:
    clean = [float(x) for x in parts if np.isfinite(x)]
    return float(np.mean(clean)) if clean else 0.0


def score_color(score: float) -> str:
    if score >= 0.35:
        return COLORS["green"]
    if score <= -0.35:
        return COLORS["red"]
    return COLORS["amber"]


def fmt_num(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def fmt_ret(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x * 100:+.{digits}f}%"


def fmt_bps(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x * 100:+.0f} bps"


def score_label(score: float) -> str:
    if score >= 0.55:
        return "Strong"
    if score >= 0.20:
        return "Firm"
    if score <= -0.55:
        return "Weak"
    if score <= -0.20:
        return "Soft"
    return "Mixed"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
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
            index = pd.to_datetime(raw["date"], errors="coerce")
            frames.append(pd.Series(values.values, index=index, name=series_id).dropna(how="all"))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().ffill()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    try:
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.columns = [str(c) for c in close.columns]
    return close.dropna(how="all").ffill()


def build_monthly(fred: pd.DataFrame, smoothing: int) -> pd.DataFrame:
    if fred.empty:
        return pd.DataFrame()
    monthly = fred.resample("ME").last().ffill()
    if smoothing > 1:
        monthly = monthly.rolling(smoothing, min_periods=1).mean()
    return monthly


def market_series(market: pd.DataFrame, ticker: str) -> pd.Series:
    if market.empty:
        return pd.Series(dtype=float)
    if ticker == "HYG/LQD" and {"HYG", "LQD"}.issubset(market.columns):
        return market["HYG"] / market["LQD"]
    if ticker in market.columns:
        return market[ticker]
    return pd.Series(dtype=float)


def classify_regime(growth: float, inflation: float, policy: float, risk: float) -> Tuple[str, str]:
    if growth >= 0.25 and inflation <= 0.10 and policy >= -0.10 and risk >= 0.10:
        return "Goldilocks / Expansion", "Growth is firm, inflation pressure is contained, and the tape confirms easier conditions."
    if growth >= 0.20 and inflation > 0.20:
        return "Reflation / Policy Watch", "Nominal growth is holding, but inflation pressure leaves duration and multiples exposed."
    if growth < -0.20 and inflation > 0.20:
        return "Stagflation Pressure", "The bad mix is weaker growth with sticky prices. Credit and cyclicals need confirmation before adding risk."
    if growth < -0.20 and inflation <= 0.00 and risk < 0.00:
        return "Disinflationary Slowdown", "Growth is fading and inflation pressure is cooling. Duration can work if credit does not break."
    if policy < -0.35 and risk < -0.05:
        return "Liquidity Squeeze", "Policy and financial conditions are restrictive and market confirmation is weak."
    if risk >= 0.35 and policy >= 0.00 and growth >= -0.10:
        return "Risk-On Transition", "The tape is improving before the macro data has fully confirmed it."
    return "Mixed / Transition", "The regime is cross-current. Positioning should wait for confirmation from credit, rates, and the dollar."


def dominant_pressure(scores: Dict[str, float]) -> Tuple[str, float, str]:
    ordered = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    label, value = ordered[0]
    if label == "Inflation":
        text = "Inflation is the main pressure point" if value > 0 else "Disinflation is the main pressure point"
    elif label == "Policy / Liquidity":
        text = "Liquidity is easing" if value > 0 else "Policy/liquidity is the main tightening pressure"
    elif label == "Risk Appetite":
        text = "The tape is confirming risk appetite" if value > 0 else "Risk appetite is the main pressure point"
    else:
        text = "Growth is confirming" if value > 0 else "Growth is the main pressure point"
    return label, value, text


def asset_readthrough(growth: float, inflation: float, policy: float, risk: float, composite: float) -> str:
    if growth < -0.20 and risk < -0.20:
        return "Defensive bias. Prefer quality, duration hedges, and lower cyclicality until credit stabilizes."
    if growth < -0.20 and inflation <= 0.00:
        return "Slowdown bias. Duration improves, cyclicals need discipline, and credit is the confirmation variable."
    if growth >= 0.20 and inflation > 0.20:
        return "Nominal growth bias. Equities can hold up, but duration and long-multiple risk remain exposed."
    if policy < -0.35:
        return "Liquidity-sensitive assets need confirmation. Strong dollar or higher real yields can pressure breadth."
    if composite >= 0.25 and risk >= 0.15:
        return "Constructive tape. Risk can be owned, but the invalidation is a turn lower in credit or breadth."
    return "Keep sizing moderate. The data does not yet justify a one-way macro bet."


def market_confirmation(composite: float, risk: float) -> str:
    spread = risk - composite
    if risk >= 0.25 and composite >= 0.10:
        return "Macro and tape agree."
    if risk <= -0.25 and composite <= -0.10:
        return "Macro and tape are deteriorating together."
    if spread >= 0.35:
        return "Tape is stronger than macro."
    if spread <= -0.35:
        return "Macro is stronger than tape."
    return "No clean confirmation yet."


def add_signal(rows: List[Dict[str, object]], group: str, signal: str, latest_value: str, move: str, score: float, readthrough: str) -> None:
    rows.append(
        {
            "Group": group,
            "Signal": signal,
            "Latest": latest_value,
            "Move": move,
            "Score": score,
            "Read-through": readthrough,
        }
    )


def build_scores(monthly: pd.DataFrame, market: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    growth_parts: List[float] = []
    inflation_parts: List[float] = []
    policy_parts: List[float] = []
    risk_parts: List[float] = []

    if "NAPM" in monthly:
        s = percentile_score(monthly["NAPM"])
        growth_parts.append(s)
        add_signal(rows, "Growth", "ISM Manufacturing PMI", fmt_num(latest(monthly["NAPM"]), 1), f"3M {delta(monthly['NAPM'], 3):+.1f}", s, "Above-trend PMI supports cyclicals; deterioration pressures earnings breadth.")

    if "UNRATE" in monthly:
        s = clip_score(delta(monthly["UNRATE"], 6), 0.75, invert=True)
        growth_parts.append(s)
        add_signal(rows, "Growth", "Unemployment Rate", fmt_pct(latest(monthly["UNRATE"]), 1), f"6M {delta(monthly['UNRATE'], 6):+.1f} pts", s, "Rising unemployment weakens the soft-landing case.")

    if "ICSA" in monthly:
        claims = monthly["ICSA"].rolling(3, min_periods=1).mean()
        s = clip_score(ret(claims, 6), 0.20, invert=True)
        growth_parts.append(s)
        add_signal(rows, "Growth", "Initial Claims", f"{latest(claims) / 1000:.0f}k", f"6M {ret(claims, 6):+.1%}", s, "Claims are the labor-market early warning signal.")

    if "CPIAUCSL" in monthly:
        cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
        s = percentile_score(cpi_yoy)
        inflation_parts.append(s)
        add_signal(rows, "Inflation", "Headline CPI YoY", fmt_pct(latest(cpi_yoy), 1), f"3M {delta(cpi_yoy, 3):+.1f} pts", s, "Higher inflation pressure keeps the policy reaction function restrictive.")

    if "CPILFESL" in monthly:
        core_yoy = monthly["CPILFESL"].pct_change(12) * 100
        s = percentile_score(core_yoy)
        inflation_parts.append(s)
        add_signal(rows, "Inflation", "Core CPI YoY", fmt_pct(latest(core_yoy), 1), f"3M {delta(core_yoy, 3):+.1f} pts", s, "Core inflation is the cleaner policy signal.")

    if "T10YIE" in monthly:
        s = percentile_score(monthly["T10YIE"])
        inflation_parts.append(s)
        add_signal(rows, "Inflation", "10Y Breakeven", fmt_pct(latest(monthly["T10YIE"]), 2), f"6M {fmt_bps(delta(monthly['T10YIE'], 6))}", s, "Breakevens capture market-implied inflation risk.")

    if "DGS10" in monthly and "CPIAUCSL" in monthly:
        cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
        real_10y = monthly["DGS10"] - cpi_yoy
        s = percentile_score(real_10y, invert=True)
        policy_parts.append(s)
        add_signal(rows, "Policy / Liquidity", "Ex-post Real 10Y", fmt_pct(latest(real_10y), 1), f"6M {delta(real_10y, 6):+.1f} pts", s, "Higher real rates tighten multiples, credit, and housing.")

    if "NFCI" in monthly:
        s = percentile_score(monthly["NFCI"], invert=True)
        policy_parts.append(s)
        add_signal(rows, "Policy / Liquidity", "Chicago Fed NFCI", fmt_num(latest(monthly["NFCI"]), 2), f"6M {delta(monthly['NFCI'], 6):+.2f}", s, "Lower NFCI means easier financial conditions.")

    if not market.empty:
        spy = market_series(market, "SPY")
        if not spy.empty:
            s_3m = clip_score(ret(spy, 63), 0.10)
            s_6m = clip_score(ret(spy, 126), 0.18)
            risk_parts.append(s_3m)
            growth_parts.append(s_6m)
            add_signal(rows, "Risk Appetite", "SPY", fmt_ret(ret(spy, 1), 1), f"3M {fmt_ret(ret(spy, 63), 1)}", s_3m, "Equity tape confirms or rejects the macro regime.")

        hyg_lqd = market_series(market, "HYG/LQD")
        if not hyg_lqd.empty:
            s = clip_score(ret(hyg_lqd, 63), 0.035)
            risk_parts.append(s)
            policy_parts.append(clip_score(ret(hyg_lqd, 126), 0.055))
            add_signal(rows, "Risk Appetite", "HYG/LQD", fmt_ret(ret(hyg_lqd, 5), 1), f"3M {fmt_ret(ret(hyg_lqd, 63), 1)}", s, "Credit leadership is the key risk confirmation signal.")

        vix = market_series(market, "^VIX")
        if not vix.empty:
            s = percentile_score(vix, invert=True)
            risk_parts.append(s)
            add_signal(rows, "Risk Appetite", "VIX", fmt_num(latest(vix), 1), f"1W {delta(vix, 5):+.1f} pts", s, "Falling vol eases financial conditions; rising vol tightens them.")

        uup = market_series(market, "UUP")
        if not uup.empty:
            s = clip_score(ret(uup, 63), 0.055, invert=True)
            risk_parts.append(s)
            policy_parts.append(clip_score(ret(uup, 126), 0.08, invert=True))
            add_signal(rows, "Policy / Liquidity", "Dollar ETF", fmt_ret(ret(uup, 1), 1), f"3M {fmt_ret(ret(uup, 63), 1)}", s, "A stronger dollar usually tightens global liquidity.")

        copper = market_series(market, "CPER")
        if not copper.empty:
            s = clip_score(ret(copper, 126), 0.18)
            growth_parts.append(s)
            add_signal(rows, "Growth", "Copper ETF", fmt_ret(ret(copper, 1), 1), f"6M {fmt_ret(ret(copper, 126), 1)}", s, "Copper is a market-based cyclical growth proxy.")

        tlt = market_series(market, "TLT")
        if not tlt.empty:
            s = clip_score(ret(tlt, 63), 0.10)
            policy_parts.append(s)
            add_signal(rows, "Policy / Liquidity", "TLT", fmt_ret(ret(tlt, 1), 1), f"3M {fmt_ret(ret(tlt, 63), 1)}", s, "Long-duration strength can signal easier rates pressure or rising slowdown risk.")

    scores = {
        "Growth": mean_score(growth_parts),
        "Inflation": mean_score(inflation_parts),
        "Policy / Liquidity": mean_score(policy_parts),
        "Risk Appetite": mean_score(risk_parts),
    }
    scores["Composite"] = mean_score([scores["Growth"], -scores["Inflation"], scores["Policy / Liquidity"], scores["Risk Appetite"]])
    return scores, pd.DataFrame(rows)


def create_score_bar(scores: Dict[str, float]) -> go.Figure:
    labels = ["Growth", "Inflation Pressure", "Policy / Liquidity", "Risk Appetite", "Composite"]
    values = [scores["Growth"], scores["Inflation"], scores["Policy / Liquidity"], scores["Risk Appetite"], scores["Composite"]]
    colors = [score_color(scores["Growth"]), score_color(-scores["Inflation"]), score_color(scores["Policy / Liquidity"]), score_color(scores["Risk Appetite"]), score_color(scores["Composite"])]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[f"{v:+.2f}" for v in values], textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)
    fig.update_layout(height=292, margin=dict(l=8, r=45, t=14, b=18), xaxis=dict(range=[-1, 1], title="Score"), yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def create_regime_map(growth: float, inflation: float) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=-1, y1=0, fillcolor="rgba(47,158,68,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, fillcolor="rgba(245,159,0,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=-1, x1=0, y0=0, y1=1, fillcolor="rgba(217,72,15,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=-1, x1=0, y0=-1, y1=0, fillcolor="rgba(25,113,194,0.08)", line_width=0)
    fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
    fig.add_vline(x=0, line_color="#9ca3af", line_width=1)
    fig.add_trace(go.Scatter(x=[growth], y=[inflation], mode="markers+text", marker=dict(size=22, color=score_color(growth - inflation), line=dict(color="#111827", width=1)), text=["Now"], textposition="top center", name="Current"))
    fig.add_annotation(x=0.55, y=-0.72, text="Goldilocks", showarrow=False, font=dict(size=12, color="#2f9e44"))
    fig.add_annotation(x=0.55, y=0.72, text="Reflation", showarrow=False, font=dict(size=12, color="#b7791f"))
    fig.add_annotation(x=-0.55, y=0.72, text="Stagflation", showarrow=False, font=dict(size=12, color="#d9480f"))
    fig.add_annotation(x=-0.55, y=-0.72, text="Slowdown", showarrow=False, font=dict(size=12, color="#1971c2"))
    fig.update_layout(height=292, margin=dict(l=12, r=12, t=14, b=18), xaxis=dict(title="Growth", range=[-1, 1], zeroline=False), yaxis=dict(title="Inflation Pressure", range=[-1, 1], zeroline=False), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def create_market_heatmap(market: pd.DataFrame, tickers: List[str]) -> go.Figure:
    horizons = [("Today", 1), ("1W", 5), ("1M", 21), ("3M", 63), ("YTD", None)]
    z: List[List[float]] = []
    text: List[List[str]] = []
    rows: List[str] = []
    for ticker in tickers:
        s = market_series(market, ticker)
        if s.empty:
            continue
        vals = []
        labels = []
        for _, window in horizons:
            value = ytd_ret(s) if window is None else ret(s, window)
            vals.append(value * 100 if np.isfinite(value) else np.nan)
            labels.append(fmt_ret(value, 1))
        z.append(vals)
        text.append(labels)
        rows.append(ticker)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[h[0] for h in horizons],
            y=rows,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#d9480f"], [0.5, "#ffffff"], [1, "#2f9e44"]],
            zmid=0,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(height=330, margin=dict(l=8, r=8, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def create_rebased_chart(market: pd.DataFrame, tickers: List[str]) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        s = market_series(market, ticker).dropna()
        if s.empty:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, mode="lines", name=ticker, line=dict(width=2)))
    fig.update_layout(height=335, margin=dict(l=10, r=10, t=12, b=12), yaxis_title="Rebased to 100", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def create_history_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "NAPM" in monthly:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["NAPM"], name="ISM PMI", yaxis="y1", line=dict(width=2)))
    if "CPIAUCSL" in monthly:
        cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
        fig.add_trace(go.Scatter(x=cpi_yoy.index, y=cpi_yoy, name="Headline CPI YoY", yaxis="y2", line=dict(width=2)))
    if "DGS10" in monthly:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["DGS10"], name="10Y Treasury", yaxis="y2", line=dict(width=2)))
    fig.update_layout(
        height=365,
        margin=dict(l=10, r=10, t=12, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="ISM", side="left"),
        yaxis2=dict(title="Percent", overlaying="y", side="right"),
    )
    return fig


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Lightweight macro regime map built from FRED macro series and liquid market proxies.
        The first screen is designed to answer three questions: what regime are we in,
        what is the pressure point, and whether the tape confirms it.
        """
    )

    st.divider()
    st.header("Controls")
    lookback_years = st.selectbox("Macro lookback", [3, 5, 10, 20], index=1)
    market_period = st.selectbox("Market history", ["6mo", "1y", "2y", "3y", "5y"], index=3)
    smoothing = st.slider("Monthly smoothing", 1, 6, 3)
    show_inputs = st.checkbox("Show input table", value=True)
    show_history = st.checkbox("Show macro history", value=True)
    show_raw = st.checkbox("Show raw FRED data", value=False)

st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Regime, pressure point, liquidity signal, and market confirmation without turning the page into a terminal.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 395)
fred = fetch_fred(FRED_SERIES, start_date)
market = fetch_market(MARKET_TICKERS, market_period)

if fred.empty and market.empty:
    st.error("No macro or market data loaded. Check FRED/yfinance connectivity.")
    st.stop()

monthly = build_monthly(fred, smoothing)
scores, signal_table = build_scores(monthly, market)
regime, regime_note = classify_regime(scores["Growth"], scores["Inflation"], scores["Policy / Liquidity"], scores["Risk Appetite"])
pressure_label, pressure_value, pressure_text = dominant_pressure({k: scores[k] for k in ["Growth", "Inflation", "Policy / Liquidity", "Risk Appetite"]})
confirmation = market_confirmation(scores["Composite"], scores["Risk Appetite"])
readthrough = asset_readthrough(scores["Growth"], scores["Inflation"], scores["Policy / Liquidity"], scores["Risk Appetite"], scores["Composite"])

fred_loaded = len([c for c in FRED_SERIES if c in fred.columns]) if not fred.empty else 0
market_loaded = len([c for c in MARKET_TICKERS if c in market.columns]) if not market.empty else 0

st.markdown(
    f"<span class='pill'>FRED: {fred_loaded}/{len(FRED_SERIES)} loaded, latest {latest_date(fred)}</span>"
    f"<span class='pill'>Market: {market_loaded}/{len(MARKET_TICKERS)} loaded, latest {latest_date(market)}</span>",
    unsafe_allow_html=True,
)

cards = [
    ("Regime", regime, regime_note, score_color(scores["Composite"])),
    ("Pressure Point", f"{pressure_label} {pressure_value:+.2f}", pressure_text, score_color(-pressure_value if pressure_label == "Inflation" else pressure_value)),
    ("Market Confirmation", confirmation, f"Risk score {scores['Risk Appetite']:+.2f}; composite {scores['Composite']:+.2f}.", score_color(scores["Risk Appetite"])),
    ("Asset Read-Through", score_label(scores["Composite"]), readthrough, score_color(scores["Composite"])),
]

for col, (label, value, footnote, color) in zip(st.columns(4), cards):
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

left, right = st.columns([1.0, 1.0])
with left:
    st.markdown("<div class='section-title'>Regime Map</div>", unsafe_allow_html=True)
    st.plotly_chart(create_regime_map(scores["Growth"], scores["Inflation"]), use_container_width=True)
with right:
    st.markdown("<div class='section-title'>Macro Scoreboard</div>", unsafe_allow_html=True)
    st.plotly_chart(create_score_bar(scores), use_container_width=True)

if not market.empty:
    st.markdown("<div class='section-title'>Market Confirmation Heatmap</div>", unsafe_allow_html=True)
    st.plotly_chart(create_market_heatmap(market, DISPLAY_TICKERS), use_container_width=True)

left, right = st.columns([1.0, 1.0])
with left:
    st.markdown("<div class='section-title'>Market Context</div>", unsafe_allow_html=True)
    if market.empty:
        st.info("Market data unavailable.")
    else:
        st.plotly_chart(create_rebased_chart(market, ["SPY", "HYG/LQD", "TLT", "UUP", "GLD", "CPER"]), use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Decision Notes</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='note-box'>
            <b>Regime:</b> {regime}. {regime_note}<br><br>
            <b>Pressure:</b> {pressure_text}.<br><br>
            <b>Read-through:</b> {readthrough}<br><br>
            <b>Invalidation:</b> watch the HYG/LQD ratio, VIX, UUP, and 10Y real-rate signal. A regime call without credit or dollar confirmation should be sized smaller.
        </div>
        """,
        unsafe_allow_html=True,
    )

if show_inputs:
    st.markdown("<div class='section-title'>Key Inputs</div>", unsafe_allow_html=True)
    if signal_table.empty:
        st.info("Input table unavailable. The page is running from partial market or macro data.")
    else:
        view = signal_table.copy()
        view["Score"] = view["Score"].map(lambda x: f"{x:+.2f}" if np.isfinite(x) else "N/A")
        st.dataframe(view, use_container_width=True, hide_index=True, height=360)

if show_history:
    if not monthly.empty:
        st.markdown("<div class='section-title'>Macro History</div>", unsafe_allow_html=True)
        st.plotly_chart(create_history_chart(monthly), use_container_width=True)
    else:
        st.markdown(
            "<div class='note-box'>FRED macro series did not load, so the regime is based on market proxies only.</div>",
            unsafe_allow_html=True,
        )

if show_raw and not fred.empty:
    st.markdown("<div class='section-title'>Raw FRED Data</div>", unsafe_allow_html=True)
    st.dataframe(fred.tail(180), use_container_width=True)
