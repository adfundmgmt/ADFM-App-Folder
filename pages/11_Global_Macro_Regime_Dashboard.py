from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Tuple

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
    "FEDFUNDS": "Fed Funds",
    "DGS10": "10Y Treasury",
    "NFCI": "Chicago Fed NFCI",
    "USREC": "NBER Recession Indicator",
}

MARKET_TICKERS = {
    "SPY": "S&P 500",
    "RSP": "Equal Weight S&P 500",
    "HYG": "High Yield Credit",
    "LQD": "Investment Grade Credit",
    "UUP": "U.S. Dollar",
    "GLD": "Gold",
    "CPER": "Copper",
    "TLT": "Long Duration",
    "^VIX": "VIX",
}

PALETTE = {
    "green": "#52b788",
    "red": "#e85d5d",
    "amber": "#f59e0b",
    "blue": "#4c78a8",
    "purple": "#8b5cf6",
    "grey": "#8b949e",
    "grid": "#e5e7eb",
    "text": "#111827",
}


st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1550px;}
        .adfm-title {font-size: 1.85rem; font-weight: 750; margin-bottom: 0.1rem; color: #111827;}
        .adfm-subtitle {font-size: 0.96rem; color: #6b7280; margin-bottom: 1.1rem;}
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
            border: 1px solid #e5e7eb; border-radius: 14px;
            padding: 13px 15px 10px 15px; min-height: 96px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }
        .metric-label {font-size: 0.74rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.045em; margin-bottom: 0.42rem;}
        .metric-value {font-size: 1.3rem; font-weight: 750; color: #111827; line-height: 1.12;}
        .metric-footnote {font-size: 0.76rem; color: #9ca3af; margin-top: 0.42rem;}
        .section-title {font-size: 1.03rem; font-weight: 750; color: #111827; margin-top: 0.5rem; margin-bottom: 0.5rem;}
        .section-subtitle {font-size: 0.88rem; color: #6b7280; margin-bottom: 0.9rem;}
        .note-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.85rem; margin-bottom: 1rem;}
        .decision-note {background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; min-height: 118px;}
        .decision-label {color: #8b5f39; font-size: 0.71rem; font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem;}
        .decision-copy {color: #334155; font-size: 0.88rem; line-height: 1.48;}
        .context-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem; margin-bottom: 0.85rem;}
        .context-card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 11px 13px;}
        .context-label {color: #6b7280; font-size: 0.69rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase;}
        .context-value {color: #111827; font-size: 1.15rem; font-weight: 750; margin: 0.2rem 0;}
        .context-copy {color: #64748b; font-size: 0.76rem; line-height: 1.4;}
        @media (max-width: 1000px) {
            .note-grid, .context-grid {grid-template-columns: 1fr;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct_fmt(x: float | None, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def bp_fmt(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.0f} bps"


def score_color(score: float) -> str:
    if score >= 0.45:
        return PALETTE["green"]
    if score <= -0.45:
        return PALETTE["red"]
    return PALETTE["amber"]


def latest_valid(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def safe_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def safe_pct_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-periods - 1] - 1.0)


def percentile_score(series: pd.Series, invert: bool = False) -> float:
    clean = series.dropna()
    if len(clean) < 12:
        return 0.0
    rank = clean.rank(pct=True).iloc[-1]
    score = (float(rank) - 0.5) * 2.0
    return -score if invert else score


def historical_change_score(series: pd.Series, periods: int) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    changes = clean.pct_change(periods).dropna()
    if len(changes) < 20:
        return np.nan
    rank = float(changes.rank(pct=True).iloc[-1])
    return float(np.clip((rank - 0.5) * 2.0, -1.0, 1.0))


def historical_trend_score(series: pd.Series, window: int = 200) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    trend = clean / clean.rolling(window, min_periods=min(60, window)).mean() - 1.0
    trend = trend.dropna()
    if len(trend) < 20:
        return np.nan
    rank = float(trend.rank(pct=True).iloc[-1])
    return float(np.clip((rank - 0.5) * 2.0, -1.0, 1.0))


def market_confirmation_frame(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()

    definitions: list[tuple[str, str, pd.Series | None]] = [
        ("Equity trend", "SPY: broad risk appetite and earnings confidence", market["SPY"] if "SPY" in market else None),
        (
            "Equity breadth",
            "RSP/SPY: equal-weight participation versus mega-cap concentration",
            market["RSP"] / market["SPY"] if {"RSP", "SPY"}.issubset(market.columns) else None,
        ),
        (
            "Credit appetite",
            "HYG/LQD: lower-quality credit versus investment grade",
            market["HYG"] / market["LQD"] if {"HYG", "LQD"}.issubset(market.columns) else None,
        ),
        (
            "Cyclical growth",
            "CPER/GLD: industrial demand versus defensive scarcity",
            market["CPER"] / market["GLD"] if {"CPER", "GLD"}.issubset(market.columns) else None,
        ),
        (
            "Dollar liquidity",
            "Inverse UUP: a softer dollar generally eases global conditions",
            1.0 / market["UUP"] if "UUP" in market else None,
        ),
        (
            "Volatility",
            "Inverse VIX: lower implied volatility confirms risk tolerance",
            1.0 / market["^VIX"] if "^VIX" in market else None,
        ),
    ]

    rows: list[dict[str, object]] = []
    horizons = ["1M", "3M", "6M", "Trend"]
    horizon_weights = [0.20, 0.35, 0.30, 0.15]
    for signal, rationale, series in definitions:
        if series is None:
            continue
        scores = {
            "1M": historical_change_score(series, 21),
            "3M": historical_change_score(series, 63),
            "6M": historical_change_score(series, 126),
            "Trend": historical_trend_score(series),
        }
        available = [(scores[key], weight) for key, weight in zip(horizons, horizon_weights) if np.isfinite(scores[key])]
        if not available:
            continue
        aggregate = float(np.average([item[0] for item in available], weights=[item[1] for item in available]))
        rows.append({"Signal": signal, "Why it matters": rationale, **scores, "Composite": aggregate})
    return pd.DataFrame(rows)


def score_phrase(value: float) -> str:
    if value >= 0.5:
        return "strongly constructive"
    if value >= 0.15:
        return "constructive"
    if value <= -0.5:
        return "strongly defensive"
    if value <= -0.15:
        return "defensive"
    return "neutral"


def build_decision_notes(
    regime: str,
    growth: float,
    inflation: float,
    policy: float,
    risk: float,
    composite: float,
    confirmations: pd.DataFrame,
    macro_inputs_available: int,
) -> list[tuple[str, str]]:
    dimensions = {
        "growth": growth,
        "inflation relief": -inflation,
        "policy/conditions": policy,
        "risk appetite": risk,
    }
    ordered = sorted(dimensions.items(), key=lambda item: item[1], reverse=True)
    strongest_name, strongest_value = ordered[0]
    weakest_name, weakest_value = ordered[-1]
    market_scores = confirmations["Composite"].dropna() if not confirmations.empty else pd.Series(dtype=float)
    positive = int((market_scores > 0.15).sum())
    negative = int((market_scores < -0.15).sum())
    total = len(market_scores)
    breadth = positive / total if total else np.nan
    market_average = float(market_scores.mean()) if total else np.nan

    base_case = (
        f"{regime}. The composite is {score_phrase(composite)} ({composite:+.2f}). "
        f"The strongest pillar is {strongest_name} ({strongest_value:+.2f}); "
        f"the main drag is {weakest_name} ({weakest_value:+.2f})."
    )
    if total:
        leaders = confirmations.nlargest(min(2, total), "Composite")["Signal"].tolist()
        laggards = confirmations.nsmallest(min(2, total), "Composite")["Signal"].tolist()
        confirmation = (
            f"{positive} of {total} market lenses are constructive and {negative} {'is' if negative == 1 else 'are'} defensive "
            f"(average {market_average:+.2f}). Strongest: {', '.join(leaders)}. "
            f"Weakest: {', '.join(laggards)}."
        )
    else:
        confirmation = "Market confirmation is unavailable, so the regime read should be treated as macro-only."

    macro_direction = np.sign(composite) if abs(composite) >= 0.15 else 0
    market_direction = np.sign(market_average) if np.isfinite(market_average) and abs(market_average) >= 0.15 else 0
    if macro_direction and market_direction and macro_direction != market_direction:
        tension = (
            "Macro and markets disagree. Keep position sizes below normal until credit, breadth, and volatility "
            "move back into line with the macro composite."
        )
    elif strongest_value - weakest_value >= 0.75:
        tension = (
            f"Internal dispersion is high: {strongest_name} and {weakest_name} send materially different signals. "
            "Favor relative-value expressions over a large one-way macro bet."
        )
    else:
        tension = "The main pillars are reasonably aligned; the current regime has no large internal contradiction."

    if composite >= 0.35 and (not np.isfinite(breadth) or breadth >= 0.5):
        stance = "Risk budget can remain constructive, emphasizing exposures confirmed by breadth and credit."
    elif composite <= -0.35 or (np.isfinite(breadth) and breadth < 0.34):
        stance = "Keep risk below normal, prioritize liquidity and quality, and require broader confirmation before adding beta."
    else:
        stance = "Run a balanced book, keep optionality, and add risk only where macro and market confirmation agree."

    coverage = min(macro_inputs_available / 8.0, 1.0)
    if total:
        coverage = 0.65 * coverage + 0.35 * min(total / 6.0, 1.0)
    confidence = "high" if coverage >= 0.85 else "moderate" if coverage >= 0.6 else "low"
    invalidation = (
        f"Confidence is {confidence}. Reassess if growth falls below -0.15, inflation pressure rises above +0.15, "
        f"policy/conditions falls below -0.35, or fewer than 2 of {max(total, 6)} market lenses remain constructive."
    )
    return [
        ("Base case", base_case),
        ("Market confirmation", confirmation),
        ("Tensions and sizing", tension),
        ("Decision and invalidation", f"{stance} {invalidation}"),
    ]


def history_context_items(monthly: pd.DataFrame) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []

    def add_item(label: str, series: pd.Series, value_format: str, threshold_copy: str) -> None:
        clean = series.dropna()
        if clean.empty:
            return
        value = float(clean.iloc[-1])
        percentile = float(clean.rank(pct=True).iloc[-1] * 100)
        six_month_change = safe_change(clean, 6)
        change_text = f"{six_month_change:+.2f}" if np.isfinite(six_month_change) else "N/A"
        items.append(
            (
                label,
                value_format.format(value),
                f"{percentile:.0f}th percentile in view; 6M change {change_text}. {threshold_copy}",
            )
        )

    if "NAPM" in monthly:
        ism = monthly["NAPM"]
        add_item("Growth pulse", ism, "{:.1f}", "Above 50 signals expansion." if latest_valid(ism) >= 50 else "Below 50 signals contraction.")
    if "CPILFESL" in monthly:
        core = monthly["CPILFESL"].pct_change(12) * 100
        add_item(
            "Core inflation",
            core,
            "{:.1f}%",
            "Above the Fed's 2% objective." if latest_valid(core) > 2 else "At or below the Fed's 2% objective.",
        )
    if "DGS10" in monthly:
        add_item("10Y Treasury", monthly["DGS10"], "{:.2f}%", "Higher levels tighten discount rates and financing costs.")
    if "NFCI" in monthly:
        nfci = monthly["NFCI"]
        add_item(
            "Financial conditions",
            nfci,
            "{:.2f}",
            "Above zero is tighter than average." if latest_valid(nfci) > 0 else "Below zero is easier than average.",
        )
    return items


def recession_windows(monthly: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if "USREC" not in monthly:
        return []
    recession = monthly["USREC"].fillna(0).ge(0.5)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    for timestamp, active in recession.items():
        if active and start is None:
            start = pd.Timestamp(timestamp)
        elif not active and start is not None:
            windows.append((start, pd.Timestamp(timestamp)))
            start = None
    if start is not None:
        windows.append((start, pd.Timestamp(monthly.index[-1])))
    return windows


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
    return pd.concat(frames, axis=1).sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.ffill()


def classify_regime(growth: float, inflation: float, policy: float, risk: float) -> Tuple[str, str]:
    if growth >= 0.25 and inflation <= 0.15 and risk >= 0:
        return "Goldilocks / Expansion", "Growth firm, inflation manageable, risk appetite constructive."
    if growth >= 0.15 and inflation > 0.15:
        return "Reflation / Policy Watch", "Growth is holding up, but inflation pressure keeps rates-sensitive assets exposed."
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
    smooth = st.slider("Score smoothing", 1, 6, 3)
    show_raw = st.checkbox("Show raw inputs", value=False)

    st.divider()
    st.header("About This Tool")
    st.markdown(
        """
        Broad macro regime dashboard built from public growth, inflation, policy,
        financial conditions, and market proxies.

        Scores are directional and percentile-based. They are meant to organize
        the macro backdrop, not replace judgment around individual releases.
        """
    )


st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Growth, inflation, policy, financial conditions, and risk appetite in one macro map.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 120)

fred = fetch_fred(FRED_SERIES, start_date)
market = fetch_market(list(MARKET_TICKERS.keys()), market_period)

if fred.empty and market.empty:
    st.error("No macro or market data loaded. Check data-source connectivity.")
    st.stop()

history_monthly = fred.resample("ME").last().ffill() if not fred.empty else pd.DataFrame()
monthly = history_monthly.copy()
if smooth > 1 and not monthly.empty:
    monthly = monthly.rolling(smooth, min_periods=1).mean()
    if "USREC" in history_monthly:
        monthly["USREC"] = history_monthly["USREC"]

metric_rows = []

growth_parts = []
if "NAPM" in monthly:
    growth_parts.append(percentile_score(monthly["NAPM"]))
    metric_rows.append(("ISM PMI", f"{latest_valid(monthly['NAPM']):.1f}", f"3M change {safe_change(monthly['NAPM'], 3):+.1f}"))
if "UNRATE" in monthly:
    growth_parts.append(np.clip(-safe_change(monthly["UNRATE"], 6), -1, 1))
    metric_rows.append(("Unemployment", pct_fmt(latest_valid(monthly["UNRATE"])), f"6M change {safe_change(monthly['UNRATE'], 6):+.1f} pts"))
if "ICSA" in monthly:
    claims_13w = monthly["ICSA"].rolling(3, min_periods=1).mean()
    growth_parts.append(np.clip(-safe_pct_change(claims_13w, 6) * 4, -1, 1))
    metric_rows.append(("Initial Claims", f"{latest_valid(claims_13w) / 1000:.0f}k", f"6M change {safe_pct_change(claims_13w, 6):+.1%}"))

inflation_parts = []
if "CPIAUCSL" in monthly:
    cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
    inflation_parts.append(percentile_score(cpi_yoy))
    metric_rows.append(("Headline CPI YoY", pct_fmt(latest_valid(cpi_yoy)), f"3M change {safe_change(cpi_yoy, 3):+.1f} pts"))
if "CPILFESL" in monthly:
    core_yoy = monthly["CPILFESL"].pct_change(12) * 100
    inflation_parts.append(percentile_score(core_yoy))
    metric_rows.append(("Core CPI YoY", pct_fmt(latest_valid(core_yoy)), f"3M change {safe_change(core_yoy, 3):+.1f} pts"))
if "T10YIE" in monthly:
    inflation_parts.append(percentile_score(monthly["T10YIE"]))
    metric_rows.append(("10Y Breakeven", pct_fmt(latest_valid(monthly["T10YIE"])), f"6M change {safe_change(monthly['T10YIE'], 6) * 100:+.0f} bps"))

policy_parts = []
if "FEDFUNDS" in monthly and "DGS10" in monthly and "CPIAUCSL" in monthly:
    real_rate = monthly["DGS10"] - monthly["CPIAUCSL"].pct_change(12) * 100
    policy_parts.append(percentile_score(real_rate, invert=True))
    metric_rows.append(("Ex-post Real 10Y", pct_fmt(latest_valid(real_rate)), f"6M change {safe_change(real_rate, 6):+.1f} pts"))
if "NFCI" in monthly:
    policy_parts.append(percentile_score(monthly["NFCI"], invert=True))
    metric_rows.append(("NFCI", f"{latest_valid(monthly['NFCI']):.2f}", f"6M change {safe_change(monthly['NFCI'], 6):+.2f}"))

risk_parts = []
if not market.empty:
    returns = market.pct_change()
    if "SPY" in market:
        risk_parts.append(np.clip(safe_pct_change(market["SPY"], 63) * 5, -1, 1))
        metric_rows.append(("SPY 3M", pct_fmt(safe_pct_change(market["SPY"], 63) * 100), "Risk appetite"))
    if "HYG" in market and "LQD" in market:
        ratio = market["HYG"] / market["LQD"]
        risk_parts.append(np.clip(safe_pct_change(ratio, 63) * 8, -1, 1))
        metric_rows.append(("HYG/LQD 3M", pct_fmt(safe_pct_change(ratio, 63) * 100), "Credit risk appetite"))
    if "^VIX" in market:
        risk_parts.append(percentile_score(market["^VIX"], invert=True))
        metric_rows.append(("VIX", f"{latest_valid(market['^VIX']):.1f}", "Lower is easier"))
    if "UUP" in market:
        risk_parts.append(np.clip(-safe_pct_change(market["UUP"], 63) * 8, -1, 1))
        metric_rows.append(("Dollar 3M", pct_fmt(safe_pct_change(market["UUP"], 63) * 100), "Higher can tighten global liquidity"))

growth_score = float(np.nanmean(growth_parts)) if growth_parts else 0.0
inflation_score = float(np.nanmean(inflation_parts)) if inflation_parts else 0.0
policy_score = float(np.nanmean(policy_parts)) if policy_parts else 0.0
risk_score = float(np.nanmean(risk_parts)) if risk_parts else 0.0
composite = float(np.nanmean([growth_score, -inflation_score, policy_score, risk_score]))
regime, regime_note = classify_regime(growth_score, inflation_score, policy_score, risk_score)
confirmations = market_confirmation_frame(market)
decision_notes = build_decision_notes(
    regime,
    growth_score,
    inflation_score,
    policy_score,
    risk_score,
    composite,
    confirmations,
    len(growth_parts) + len(inflation_parts) + len(policy_parts) + len(risk_parts),
)

cols = st.columns(5)
cards = [
    ("Regime", regime, regime_note, score_color(composite)),
    ("Growth", f"{growth_score:+.2f}", "Positive means improving/firm", score_color(growth_score)),
    ("Inflation", f"{inflation_score:+.2f}", "Positive means more pressure", score_color(-inflation_score)),
    ("Policy / Conditions", f"{policy_score:+.2f}", "Positive means easier", score_color(policy_score)),
    ("Risk Appetite", f"{risk_score:+.2f}", "Positive means constructive tape", score_color(risk_score)),
]

for col, (label, value, footnote, color) in zip(cols, cards):
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
        marker_color=[
            score_color(growth_score),
            score_color(-inflation_score),
            score_color(policy_score),
            score_color(risk_score),
            score_color(composite),
        ],
    )
)
score_fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
score_fig.update_layout(
    height=330,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis=dict(range=[-1, 1], title="Score"),
    xaxis=dict(title=""),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(score_fig, use_container_width=True)

st.markdown("<div class='section-title'>Market Confirmation Heatmap</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='section-subtitle'>
        A fixed cross-asset confirmation set, chosen for economic meaning rather than recent performance.
        Each cell ranks the latest move against that signal's own history; positive values confirm easier,
        broader risk-taking conditions and negative values flag defensive pressure.
    </div>
    """,
    unsafe_allow_html=True,
)
if confirmations.empty:
    st.info("Not enough market history is available to calculate confirmation percentiles.")
else:
    heatmap_columns = ["1M", "3M", "6M", "Trend", "Composite"]
    heatmap_values = confirmations[heatmap_columns].to_numpy(dtype=float)
    heatmap_text = np.where(np.isfinite(heatmap_values), np.vectorize(lambda value: f"{value:+.2f}")(heatmap_values), "N/A")
    rationale = np.repeat(confirmations["Why it matters"].to_numpy()[:, None], len(heatmap_columns), axis=1)
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=heatmap_values,
            x=heatmap_columns,
            y=confirmations["Signal"],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, "#9f3d3d"],
                [0.35, "#e8b4a2"],
                [0.5, "#f3efe7"],
                [0.65, "#a8d5c8"],
                [1.0, "#1f6f5f"],
            ],
            text=heatmap_text,
            texttemplate="%{text}",
            customdata=rationale,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f}<br>%{customdata}<extra></extra>",
            colorbar=dict(title="Confirmation", tickvals=[-1, -0.5, 0, 0.5, 1]),
        )
    )
    heatmap_fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(side="top", title=""),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    with st.expander("Why these signals?"):
        st.dataframe(
            confirmations[["Signal", "Why it matters", "Composite"]].style.format({"Composite": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("<div class='section-title'>Decision Notes</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>A decision framework that separates the base case, confirmation, contradictions, sizing, and invalidation.</div>",
    unsafe_allow_html=True,
)
notes_html = "".join(
    f'<div class="decision-note"><div class="decision-label">{label}</div>'
    f'<div class="decision-copy">{copy}</div></div>'
    for label, copy in decision_notes
)
st.markdown(f"<div class='note-grid'>{notes_html}</div>", unsafe_allow_html=True)

chart_cols = st.columns([1.05, 0.95])

with chart_cols[0]:
    st.markdown("<div class='section-title'>Key Macro Inputs</div>", unsafe_allow_html=True)
    table = pd.DataFrame(metric_rows, columns=["Input", "Latest", "Read-through"])
    st.dataframe(table, use_container_width=True, hide_index=True)

with chart_cols[1]:
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
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Rebased to 100",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

if not monthly.empty:
    st.markdown("<div class='section-title'>Macro History</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='section-subtitle'>
            {lookback_years} years of macro context with {smooth}-month smoothing. Dashed lines mark economic
            thresholds; shaded bands mark NBER recessions. Percentiles below are calculated within the selected view,
            so today's level is compared with a relevant cycle rather than shown in isolation.
        </div>
        """,
        unsafe_allow_html=True,
    )
    context_html = "".join(
        f'<div class="context-card"><div class="context-label">{label}</div>'
        f'<div class="context-value">{value}</div><div class="context-copy">{copy}</div></div>'
        for label, value, copy in history_context_items(monthly)
    )
    st.markdown(f"<div class='context-grid'>{context_html}</div>", unsafe_allow_html=True)

    history_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("Growth and labor", "Inflation", "Rates and financial conditions"),
    )
    if "NAPM" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["NAPM"], name="ISM PMI", line=dict(color=PALETTE["blue"])),
            row=1,
            col=1,
            secondary_y=False,
        )
        history_fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8", row=1, col=1)
    if "UNRATE" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["UNRATE"], name="Unemployment", line=dict(color=PALETTE["grey"])),
            row=1,
            col=1,
            secondary_y=True,
        )
    if "CPIAUCSL" in monthly:
        cpi_yoy = monthly["CPIAUCSL"].pct_change(12) * 100
        history_fig.add_trace(
            go.Scatter(x=cpi_yoy.index, y=cpi_yoy, name="Headline CPI YoY", line=dict(color=PALETTE["red"])),
            row=2,
            col=1,
            secondary_y=False,
        )
    if "CPILFESL" in monthly:
        core_yoy = monthly["CPILFESL"].pct_change(12) * 100
        history_fig.add_trace(
            go.Scatter(x=core_yoy.index, y=core_yoy, name="Core CPI YoY", line=dict(color="#b45309")),
            row=2,
            col=1,
            secondary_y=False,
        )
    if "T10YIE" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["T10YIE"], name="10Y Breakeven", line=dict(color="#d97706", dash="dot")),
            row=2,
            col=1,
            secondary_y=True,
        )
    history_fig.add_hline(y=2, line_dash="dash", line_color="#94a3b8", row=2, col=1)
    if "DGS10" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["DGS10"], name="10Y Yield", line=dict(color=PALETTE["purple"])),
            row=3,
            col=1,
            secondary_y=False,
        )
    if "FEDFUNDS" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["FEDFUNDS"], name="Fed Funds", line=dict(color="#475569", dash="dot")),
            row=3,
            col=1,
            secondary_y=False,
        )
    if "NFCI" in monthly:
        history_fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly["NFCI"], name="NFCI", line=dict(color="#9f3d3d")),
            row=3,
            col=1,
            secondary_y=True,
        )
        history_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=3, col=1, secondary_y=True)

    for recession_start, recession_end in recession_windows(history_monthly):
        for row in [1, 2, 3]:
            history_fig.add_vrect(
                x0=recession_start,
                x1=recession_end,
                fillcolor="#94a3b8",
                opacity=0.13,
                line_width=0,
                row=row,
                col=1,
            )

    history_fig.update_layout(
        height=760,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    history_fig.update_yaxes(title_text="ISM", row=1, col=1, secondary_y=False)
    history_fig.update_yaxes(title_text="Unemployment %", row=1, col=1, secondary_y=True)
    history_fig.update_yaxes(title_text="CPI %", row=2, col=1, secondary_y=False)
    history_fig.update_yaxes(title_text="Breakeven %", row=2, col=1, secondary_y=True)
    history_fig.update_yaxes(title_text="Policy / yield %", row=3, col=1, secondary_y=False)
    history_fig.update_yaxes(title_text="NFCI", row=3, col=1, secondary_y=True)
    st.plotly_chart(history_fig, use_container_width=True)

if show_raw:
    st.markdown("<div class='section-title'>Raw FRED Data</div>", unsafe_allow_html=True)
    st.dataframe(fred.tail(120), use_container_width=True)
