from __future__ import annotations

from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# =============================================================================
# PAGE CONFIG
# =============================================================================

TITLE = "Global Macro Regime Monitor"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = (4, 12)

FRED_SERIES = {
    "NAPM": "ISM Manufacturing PMI",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Jobless Claims",
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
    "bg": "#f7f8fa",
    "card": "#ffffff",
    "border": "#d9dee7",
    "text": "#111827",
    "muted": "#6b7280",
    "faint": "#9ca3af",
    "grid": "#e5e7eb",
    "green": "#1f7a5a",
    "red": "#a63d40",
    "amber": "#b7791f",
    "blue": "#315f95",
    "purple": "#6d4aff",
    "slate": "#334155",
}

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f8fa;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1550px;
        }

        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .page-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            font-size: 0.92rem;
            color: #6b7280;
            margin-bottom: 1.0rem;
        }

        .section-title {
            font-size: 1.0rem;
            font-weight: 800;
            color: #111827;
            margin-top: 0.9rem;
            margin-bottom: 0.35rem;
        }

        .section-subtitle {
            font-size: 0.84rem;
            color: #6b7280;
            margin-bottom: 0.7rem;
            line-height: 1.45;
        }

        .brief-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr 1fr;
            gap: 0.75rem;
            margin-bottom: 0.9rem;
        }

        .brief-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 12px;
            padding: 13px 15px;
            min-height: 116px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .brief-label {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.45rem;
        }

        .brief-value {
            font-size: 1.2rem;
            font-weight: 800;
            color: #111827;
            line-height: 1.2;
            margin-bottom: 0.45rem;
        }

        .brief-copy {
            font-size: 0.80rem;
            line-height: 1.42;
            color: #475569;
        }

        .pillar-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.8rem;
        }

        .pillar-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 12px;
            padding: 12px 13px;
            min-height: 92px;
        }

        .pillar-label {
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.35rem;
        }

        .pillar-value {
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }

        .pillar-note {
            font-size: 0.76rem;
            color: #64748b;
            line-height: 1.35;
        }

        .note-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 12px;
            padding: 13px 15px;
            min-height: 112px;
        }

        .note-label {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #7c5a2b;
            margin-bottom: 0.45rem;
        }

        .note-copy {
            font-size: 0.84rem;
            color: #334155;
            line-height: 1.48;
        }

        .data-status {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 10px;
            padding: 9px 12px;
            color: #475569;
            font-size: 0.78rem;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 1150px) {
            .brief-grid, .pillar-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def is_valid(x: float | None) -> bool:
    return x is not None and np.isfinite(x)


def pct_fmt(x: float | None, digits: int = 1) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def num_fmt(x: float | None, digits: int = 2) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{x:.{digits}f}"


def signed_fmt(x: float | None, digits: int = 2) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{x:+.{digits}f}"


def score_color(score: float) -> str:
    if score >= 0.30:
        return PALETTE["green"]
    if score <= -0.30:
        return PALETTE["red"]
    return PALETTE["amber"]


def score_label(score: float) -> str:
    if score >= 0.55:
        return "Strong positive"
    if score >= 0.25:
        return "Positive"
    if score <= -0.55:
        return "Strong negative"
    if score <= -0.25:
        return "Negative"
    return "Mixed"


def latest_valid(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(series: pd.Series) -> str:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return "N/A"
    return pd.Timestamp(clean.index[-1]).strftime("%b %d, %Y")


def safe_change(series: pd.Series, periods: int) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def safe_pct_change(series: pd.Series, periods: int) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= periods:
        return np.nan
    base = clean.iloc[-periods - 1]
    if base == 0 or not np.isfinite(base):
        return np.nan
    return float(clean.iloc[-1] / base - 1.0)


def percentile_score(series: pd.Series, invert: bool = False) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 12:
        return 0.0
    rank = clean.rank(pct=True).iloc[-1]
    score = (float(rank) - 0.5) * 2.0
    return -score if invert else score


def zscore(series: pd.Series, window: int = 36, invert: bool = False) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    mean = clean.rolling(window, min_periods=max(12, window // 3)).mean()
    std = clean.rolling(window, min_periods=max(12, window // 3)).std()
    z = (clean - mean) / std
    z = z.clip(-3, 3)
    return -z if invert else z


def normalize_score(value: float) -> float:
    if not is_valid(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def score_phrase(value: float) -> str:
    if value >= 0.55:
        return "strongly constructive"
    if value >= 0.20:
        return "constructive"
    if value <= -0.55:
        return "strongly defensive"
    if value <= -0.20:
        return "defensive"
    return "mixed"


def regime_color(regime: str) -> str:
    if "Goldilocks" in regime or "Expansion" in regime:
        return PALETTE["green"]
    if "Slowdown" in regime or "Stagflation" in regime or "Tight" in regime:
        return PALETTE["red"]
    return PALETTE["amber"]


# =============================================================================
# DATA
# =============================================================================

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
            frames.append(
                pd.Series(
                    values.values,
                    index=pd.to_datetime(raw["date"], errors="coerce"),
                    name=series_id,
                )
            )
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: List[str], period: str) -> pd.DataFrame:
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()
            elif "Adj Close" in data.columns.get_level_values(0):
                close = data["Adj Close"].copy()
            else:
                return pd.DataFrame()
        else:
            close = data[["Close"]].rename(columns={"Close": tickers[0]})

        close = close.reindex(columns=[ticker for ticker in tickers if ticker in close.columns])
        return close.ffill().dropna(how="all")

    except Exception:
        return pd.DataFrame()


# =============================================================================
# SCORING
# =============================================================================

def classify_regime(growth: float, inflation: float, policy: float, risk: float) -> Tuple[str, str]:
    inflation_relief = -inflation

    if growth >= 0.25 and inflation <= 0.10 and risk >= 0.00:
        return "Goldilocks / Expansion", "Growth is firm, inflation pressure is contained, and the tape is not rejecting risk."

    if growth >= 0.20 and inflation > 0.10:
        return "Reflation / Policy Watch", "Growth is holding up, but inflation pressure can keep the front end and duration exposed."

    if growth < -0.20 and inflation > 0.10:
        return "Stagflation Pressure", "Growth is fading while inflation remains sticky, the worst mix for policy flexibility."

    if growth < -0.20 and risk < -0.20:
        return "Slowdown / Risk-Off", "Growth and market confirmation are deteriorating together."

    if policy < -0.40 and risk < 0.00:
        return "Tight Financial Conditions", "Policy and financial conditions are restrictive enough to pressure risk assets."

    if inflation_relief >= 0.25 and policy >= 0.15 and growth >= -0.10:
        return "Disinflationary Easing", "Inflation pressure is fading and policy conditions are becoming less restrictive."

    return "Transition / Mixed Macro", "The macro impulse is not one-way. Positioning should respect cross-currents across growth, inflation, policy, and tape."


def market_confirmation_frame(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()

    definitions: list[tuple[str, str, pd.Series | None]] = [
        (
            "Equity trend",
            "SPY confirms broad risk appetite and earnings confidence.",
            market["SPY"] if "SPY" in market else None,
        ),
        (
            "Breadth",
            "RSP/SPY shows equal-weight participation versus mega-cap concentration.",
            market["RSP"] / market["SPY"] if {"RSP", "SPY"}.issubset(market.columns) else None,
        ),
        (
            "Credit",
            "HYG/LQD captures lower-quality credit appetite versus investment grade.",
            market["HYG"] / market["LQD"] if {"HYG", "LQD"}.issubset(market.columns) else None,
        ),
        (
            "Cyclicals",
            "Copper/gold captures cyclical demand versus defensive scarcity.",
            market["CPER"] / market["GLD"] if {"CPER", "GLD"}.issubset(market.columns) else None,
        ),
        (
            "Dollar liquidity",
            "A softer dollar generally eases global financial conditions.",
            1.0 / market["UUP"] if "UUP" in market else None,
        ),
        (
            "Volatility",
            "Lower VIX confirms risk tolerance and easier hedging conditions.",
            1.0 / market["^VIX"] if "^VIX" in market else None,
        ),
    ]

    rows: list[dict[str, object]] = []
    horizons = {
        "1W": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
    }

    for signal, rationale, series in definitions:
        if series is None:
            continue

        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 150:
            continue

        scores = {}
        for label, days in horizons.items():
            changes = clean.pct_change(days).dropna()
            if len(changes) < 50:
                scores[label] = np.nan
            else:
                rank = float(changes.rank(pct=True).iloc[-1])
                scores[label] = normalize_score((rank - 0.5) * 2.0)

        trend = clean / clean.rolling(200, min_periods=80).mean() - 1.0
        trend = trend.dropna()
        if len(trend) >= 50:
            trend_rank = float(trend.rank(pct=True).iloc[-1])
            scores["Trend"] = normalize_score((trend_rank - 0.5) * 2.0)
        else:
            scores["Trend"] = np.nan

        available = [v for v in scores.values() if np.isfinite(v)]
        if not available:
            continue

        composite = np.nanmean(
            [
                scores.get("1W", np.nan) * 0.10,
                scores.get("1M", np.nan) * 0.25,
                scores.get("3M", np.nan) * 0.35,
                scores.get("6M", np.nan) * 0.20,
                scores.get("Trend", np.nan) * 0.10,
            ]
        )

        rows.append(
            {
                "Signal": signal,
                "Why it matters": rationale,
                **scores,
                "Composite": float(composite),
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.sort_values("Composite", ascending=False).reset_index(drop=True)


def build_scores(monthly: pd.DataFrame, market: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    metric_rows = []

    growth_parts = []
    if "NAPM" in monthly:
        ism = monthly["NAPM"]
        growth_parts.append(percentile_score(ism))
        metric_rows.append(
            {
                "Group": "Growth",
                "Input": "ISM Manufacturing PMI",
                "Latest": f"{latest_valid(ism):.1f}",
                "Latest Date": latest_date(ism),
                "1M": signed_fmt(safe_change(ism, 1), 1),
                "3M": signed_fmt(safe_change(ism, 3), 1),
                "Read-through": "Above 50 is expansionary.",
            }
        )

    if "UNRATE" in monthly:
        unrate = monthly["UNRATE"]
        growth_parts.append(normalize_score(-safe_change(unrate, 6)))
        metric_rows.append(
            {
                "Group": "Growth",
                "Input": "Unemployment Rate",
                "Latest": pct_fmt(latest_valid(unrate), 1),
                "Latest Date": latest_date(unrate),
                "1M": signed_fmt(safe_change(unrate, 1), 1),
                "3M": signed_fmt(safe_change(unrate, 3), 1),
                "Read-through": "Higher unemployment weakens the labor impulse.",
            }
        )

    if "ICSA" in monthly:
        claims = monthly["ICSA"].rolling(3, min_periods=1).mean()
        growth_parts.append(normalize_score(-safe_pct_change(claims, 6) * 4))
        metric_rows.append(
            {
                "Group": "Growth",
                "Input": "Initial Claims, 3M Avg",
                "Latest": f"{latest_valid(claims) / 1000:.0f}k",
                "Latest Date": latest_date(claims),
                "1M": pct_fmt(safe_pct_change(claims, 1) * 100, 1),
                "3M": pct_fmt(safe_pct_change(claims, 3) * 100, 1),
                "Read-through": "Rising claims pressure the growth score.",
            }
        )

    inflation_parts = []
    if "CPIAUCSL" in monthly:
        headline = monthly["CPIAUCSL"].pct_change(12) * 100
        inflation_parts.append(percentile_score(headline))
        metric_rows.append(
            {
                "Group": "Inflation",
                "Input": "Headline CPI YoY",
                "Latest": pct_fmt(latest_valid(headline), 1),
                "Latest Date": latest_date(headline),
                "1M": signed_fmt(safe_change(headline, 1), 1),
                "3M": signed_fmt(safe_change(headline, 3), 1),
                "Read-through": "Higher inflation pressure is negative for duration and policy flexibility.",
            }
        )

    if "CPILFESL" in monthly:
        core = monthly["CPILFESL"].pct_change(12) * 100
        inflation_parts.append(percentile_score(core))
        metric_rows.append(
            {
                "Group": "Inflation",
                "Input": "Core CPI YoY",
                "Latest": pct_fmt(latest_valid(core), 1),
                "Latest Date": latest_date(core),
                "1M": signed_fmt(safe_change(core, 1), 1),
                "3M": signed_fmt(safe_change(core, 3), 1),
                "Read-through": "Sticky core inflation limits easing optionality.",
            }
        )

    if "T10YIE" in monthly:
        breakeven = monthly["T10YIE"]
        inflation_parts.append(percentile_score(breakeven))
        metric_rows.append(
            {
                "Group": "Inflation",
                "Input": "10Y Breakeven",
                "Latest": pct_fmt(latest_valid(breakeven), 2),
                "Latest Date": latest_date(breakeven),
                "1M": f"{safe_change(breakeven, 1) * 100:+.0f} bps" if is_valid(safe_change(breakeven, 1)) else "N/A",
                "3M": f"{safe_change(breakeven, 3) * 100:+.0f} bps" if is_valid(safe_change(breakeven, 3)) else "N/A",
                "Read-through": "Breakevens show market inflation compensation.",
            }
        )

    policy_parts = []
    if {"DGS10", "CPIAUCSL"}.issubset(monthly.columns):
        real_10y = monthly["DGS10"] - monthly["CPIAUCSL"].pct_change(12) * 100
        policy_parts.append(percentile_score(real_10y, invert=True))
        metric_rows.append(
            {
                "Group": "Policy",
                "Input": "Ex-post Real 10Y",
                "Latest": pct_fmt(latest_valid(real_10y), 1),
                "Latest Date": latest_date(real_10y),
                "1M": signed_fmt(safe_change(real_10y, 1), 1),
                "3M": signed_fmt(safe_change(real_10y, 3), 1),
                "Read-through": "Higher real yields tighten the valuation and credit backdrop.",
            }
        )

    if "NFCI" in monthly:
        nfci = monthly["NFCI"]
        policy_parts.append(percentile_score(nfci, invert=True))
        metric_rows.append(
            {
                "Group": "Policy",
                "Input": "Chicago Fed NFCI",
                "Latest": num_fmt(latest_valid(nfci), 2),
                "Latest Date": latest_date(nfci),
                "1M": signed_fmt(safe_change(nfci, 1), 2),
                "3M": signed_fmt(safe_change(nfci, 3), 2),
                "Read-through": "Below zero is easier than average financial conditions.",
            }
        )

    risk_parts = []
    if not market.empty:
        if "SPY" in market:
            spy_3m = safe_pct_change(market["SPY"], 63)
            risk_parts.append(normalize_score(spy_3m * 5))
            metric_rows.append(
                {
                    "Group": "Market",
                    "Input": "S&P 500, 3M",
                    "Latest": pct_fmt(spy_3m * 100, 1),
                    "Latest Date": latest_date(market["SPY"]),
                    "1M": pct_fmt(safe_pct_change(market["SPY"], 21) * 100, 1),
                    "3M": pct_fmt(spy_3m * 100, 1),
                    "Read-through": "Higher equities confirm risk appetite.",
                }
            )

        if {"HYG", "LQD"}.issubset(market.columns):
            credit_ratio = market["HYG"] / market["LQD"]
            credit_3m = safe_pct_change(credit_ratio, 63)
            risk_parts.append(normalize_score(credit_3m * 8))
            metric_rows.append(
                {
                    "Group": "Market",
                    "Input": "HYG / LQD, 3M",
                    "Latest": pct_fmt(credit_3m * 100, 1),
                    "Latest Date": latest_date(credit_ratio),
                    "1M": pct_fmt(safe_pct_change(credit_ratio, 21) * 100, 1),
                    "3M": pct_fmt(credit_3m * 100, 1),
                    "Read-through": "Higher ratio confirms credit risk appetite.",
                }
            )

        if "^VIX" in market:
            vix = market["^VIX"]
            risk_parts.append(percentile_score(vix, invert=True))
            metric_rows.append(
                {
                    "Group": "Market",
                    "Input": "VIX",
                    "Latest": num_fmt(latest_valid(vix), 1),
                    "Latest Date": latest_date(vix),
                    "1M": signed_fmt(safe_change(vix, 21), 1),
                    "3M": signed_fmt(safe_change(vix, 63), 1),
                    "Read-through": "Lower volatility confirms easier market conditions.",
                }
            )

        if "UUP" in market:
            dollar = market["UUP"]
            dollar_3m = safe_pct_change(dollar, 63)
            risk_parts.append(normalize_score(-dollar_3m * 8))
            metric_rows.append(
                {
                    "Group": "Market",
                    "Input": "U.S. Dollar, 3M",
                    "Latest": pct_fmt(dollar_3m * 100, 1),
                    "Latest Date": latest_date(dollar),
                    "1M": pct_fmt(safe_pct_change(dollar, 21) * 100, 1),
                    "3M": pct_fmt(dollar_3m * 100, 1),
                    "Read-through": "A stronger dollar can tighten global liquidity.",
                }
            )

    growth_score = float(np.nanmean(growth_parts)) if growth_parts else 0.0
    inflation_score = float(np.nanmean(inflation_parts)) if inflation_parts else 0.0
    policy_score = float(np.nanmean(policy_parts)) if policy_parts else 0.0
    risk_score = float(np.nanmean(risk_parts)) if risk_parts else 0.0
    composite = float(np.nanmean([growth_score, -inflation_score, policy_score, risk_score]))

    scores = {
        "Growth": growth_score,
        "Inflation Pressure": inflation_score,
        "Policy / Conditions": policy_score,
        "Risk Appetite": risk_score,
        "Composite": composite,
        "Inputs Available": len(growth_parts) + len(inflation_parts) + len(policy_parts) + len(risk_parts),
    }

    return scores, pd.DataFrame(metric_rows)


def build_decision_notes(
    regime: str,
    scores: dict,
    confirmations: pd.DataFrame,
) -> dict:
    growth = scores["Growth"]
    inflation = scores["Inflation Pressure"]
    policy = scores["Policy / Conditions"]
    risk = scores["Risk Appetite"]
    composite = scores["Composite"]

    dimensions = {
        "growth": growth,
        "inflation relief": -inflation,
        "policy/conditions": policy,
        "risk appetite": risk,
    }

    strongest_name, strongest_value = max(dimensions.items(), key=lambda item: item[1])
    weakest_name, weakest_value = min(dimensions.items(), key=lambda item: item[1])

    market_scores = confirmations["Composite"].dropna() if not confirmations.empty else pd.Series(dtype=float)
    positive = int((market_scores > 0.20).sum())
    negative = int((market_scores < -0.20).sum())
    total = len(market_scores)
    market_average = float(market_scores.mean()) if total else np.nan

    if composite >= 0.35 and (not total or positive >= max(2, total // 2)):
        stance = "Constructive but selective"
        action = "Risk can stay on, but size should be concentrated in assets confirmed by credit and breadth."
    elif composite <= -0.35 or (total and positive <= 1):
        stance = "Defensive"
        action = "Keep gross liquidity high, reduce weak-beta exposure, and require credit or breadth repair before adding risk."
    else:
        stance = "Balanced / tactical"
        action = "Avoid a large one-way book. Favor relative-value trades and liquid expressions until the market confirms the macro signal."

    if total and np.sign(composite) != 0 and np.sign(market_average) != 0 and np.sign(composite) != np.sign(market_average):
        pressure = "Macro and market confirmation disagree."
    elif strongest_value - weakest_value > 0.75:
        pressure = f"Internal dispersion is high: {strongest_name} is leading while {weakest_name} is the drag."
    else:
        pressure = f"The main pressure point is {weakest_name}; the strongest offset is {strongest_name}."

    if total:
        leaders = confirmations.nlargest(min(2, total), "Composite")["Signal"].tolist()
        laggards = confirmations.nsmallest(min(2, total), "Composite")["Signal"].tolist()
        confirmation = (
            f"{positive} of {total} market signals are constructive; {negative} are defensive. "
            f"Leadership: {', '.join(leaders)}. Weakness: {', '.join(laggards)}."
        )
    else:
        confirmation = "Market confirmation is unavailable. Treat the read as macro-only."

    invalidation = (
        "Reassess if growth turns negative, inflation pressure rises, policy/conditions falls below -0.35, "
        "or fewer than two market signals remain constructive."
    )

    return {
        "stance": stance,
        "action": action,
        "pressure": pressure,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "base": (
            f"{regime}. Composite is {score_phrase(composite)} at {composite:+.2f}. "
            f"Growth {growth:+.2f}, inflation pressure {inflation:+.2f}, "
            f"policy/conditions {policy:+.2f}, risk appetite {risk:+.2f}."
        ),
    }


# =============================================================================
# CHARTS
# =============================================================================

def apply_layout(fig: go.Figure, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=32, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        showlegend=showlegend,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#e5e7eb",
        tickfont=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        zeroline=False,
        linecolor="#e5e7eb",
        tickfont=dict(color="#64748b"),
    )
    return fig


def score_bar_chart(scores: dict) -> go.Figure:
    rows = ["Growth", "Inflation Relief", "Policy / Conditions", "Risk Appetite", "Composite"]
    values = [
        scores["Growth"],
        -scores["Inflation Pressure"],
        scores["Policy / Conditions"],
        scores["Risk Appetite"],
        scores["Composite"],
    ]

    colors = [score_color(v) for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=rows,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=45, t=10, b=20),
        xaxis=dict(range=[-1, 1], title="", tickvals=[-1, -0.5, 0, 0.5, 1]),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def confirmation_heatmap(confirmations: pd.DataFrame) -> go.Figure:
    columns = ["1W", "1M", "3M", "6M", "Trend", "Composite"]
    values = confirmations[columns].to_numpy(dtype=float)
    text = np.where(np.isfinite(values), np.vectorize(lambda v: f"{v:+.2f}")(values), "N/A")

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=columns,
            y=confirmations["Signal"],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.00, "#8f2d35"],
                [0.35, "#d8a092"],
                [0.50, "#f2f0ea"],
                [0.65, "#93c7b4"],
                [1.00, "#1f7a5a"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f}<extra></extra>",
            colorbar=dict(
                title="",
                thickness=10,
                len=0.80,
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(side="top", title=""),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def chart_growth(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Manufacturing cycle", "Labor pressure"),
    )

    if "NAPM" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["NAPM"],
                name="ISM Manufacturing PMI",
                line=dict(color=PALETTE["blue"], width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8", row=1, col=1)

    if "UNRATE" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["UNRATE"],
                name="Unemployment Rate",
                line=dict(color="#475569", width=2),
            ),
            row=2,
            col=1,
        )

    if "ICSA" in monthly:
        claims = monthly["ICSA"].rolling(3, min_periods=1).mean() / 1000
        fig.add_trace(
            go.Scatter(
                x=claims.index,
                y=claims,
                name="Initial Claims, 3M Avg (k)",
                line=dict(color=PALETTE["red"], width=2),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="PMI", row=1, col=1)
    fig.update_yaxes(title_text="% / k", row=2, col=1)
    return apply_layout(fig, height=430)


def chart_inflation(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "CPIAUCSL" in monthly:
        headline = monthly["CPIAUCSL"].pct_change(12) * 100
        fig.add_trace(
            go.Scatter(
                x=headline.index,
                y=headline,
                name="Headline CPI YoY",
                line=dict(color=PALETTE["red"], width=2),
            )
        )

    if "CPILFESL" in monthly:
        core = monthly["CPILFESL"].pct_change(12) * 100
        fig.add_trace(
            go.Scatter(
                x=core.index,
                y=core,
                name="Core CPI YoY",
                line=dict(color="#9a4f13", width=2),
            )
        )

    if "T10YIE" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["T10YIE"],
                name="10Y Breakeven",
                line=dict(color=PALETTE["blue"], width=2, dash="dot"),
            )
        )

    fig.add_hline(y=2, line_dash="dash", line_color="#94a3b8")
    fig.update_yaxes(title_text="%")
    return apply_layout(fig, height=390)


def chart_policy(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Rates", "Financial conditions"),
    )

    if "DGS10" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["DGS10"],
                name="10Y Treasury",
                line=dict(color=PALETTE["purple"], width=2),
            ),
            row=1,
            col=1,
        )

    if "FEDFUNDS" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["FEDFUNDS"],
                name="Fed Funds",
                line=dict(color="#475569", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

    if {"DGS10", "CPIAUCSL"}.issubset(monthly.columns):
        real_10y = monthly["DGS10"] - monthly["CPIAUCSL"].pct_change(12) * 100
        fig.add_trace(
            go.Scatter(
                x=real_10y.index,
                y=real_10y,
                name="Ex-post Real 10Y",
                line=dict(color=PALETTE["green"], width=2),
            ),
            row=1,
            col=1,
        )

    if "NFCI" in monthly:
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly["NFCI"],
                name="Chicago Fed NFCI",
                line=dict(color=PALETTE["red"], width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=1)

    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)
    return apply_layout(fig, height=430)


def chart_market(market: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()

    if market.empty:
        return apply_layout(fig, height=390)

    chosen = [t for t in tickers if t in market.columns]
    if not chosen:
        return apply_layout(fig, height=390)

    rebased = market[chosen].dropna(how="all").copy()
    rebased = rebased.ffill().dropna(how="all")

    if rebased.empty:
        return apply_layout(fig, height=390)

    rebased = rebased / rebased.iloc[0] * 100

    for ticker in chosen:
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased[ticker],
                mode="lines",
                name=MARKET_TICKERS.get(ticker, ticker),
                line=dict(width=2),
            )
        )

    fig.update_yaxes(title_text="Rebased to 100")
    return apply_layout(fig, height=390)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Controls")

    lookback_years = st.selectbox(
        "Macro history",
        [3, 5, 10, 20],
        index=1,
        help="Controls the FRED history window used for charts and percentile scores.",
    )

    market_period = st.selectbox(
        "Market history",
        ["1y", "2y", "3y", "5y"],
        index=2,
        help="Controls the yfinance market history window.",
    )

    smooth = st.slider(
        "Macro smoothing",
        1,
        6,
        3,
        help="Applies rolling monthly smoothing to macro inputs. Recession indicator is not smoothed.",
    )

    market_chart_selection = st.multiselect(
        "Market chart",
        options=list(MARKET_TICKERS.keys()),
        default=["SPY", "RSP", "HYG", "LQD", "UUP", "TLT", "GLD"],
        format_func=lambda x: MARKET_TICKERS.get(x, x),
    )

    show_raw = st.checkbox("Show raw data", value=False)

    st.divider()
    st.header("About")
    st.markdown(
        """
        Regime monitor built from public macro data and liquid market proxies.

        Scores are percentile-based directional signals. The dashboard is designed to organize the backdrop, identify contradictions, and frame risk sizing.
        """
    )


# =============================================================================
# LOAD DATA
# =============================================================================

st.markdown(f"<div class='page-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-subtitle'>Growth, inflation, policy, financial conditions, and cross-asset confirmation in one decision page.</div>",
    unsafe_allow_html=True,
)

start_date = date.today() - timedelta(days=int(lookback_years * 365.25) + 420)

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

scores, input_table = build_scores(monthly, market)
confirmations = market_confirmation_frame(market)

regime, regime_note = classify_regime(
    scores["Growth"],
    scores["Inflation Pressure"],
    scores["Policy / Conditions"],
    scores["Risk Appetite"],
)

notes = build_decision_notes(regime, scores, confirmations)

fred_latest = max([pd.Timestamp(fred[col].dropna().index[-1]) for col in fred.columns if not fred[col].dropna().empty], default=None)
market_latest = max([pd.Timestamp(market[col].dropna().index[-1]) for col in market.columns if not market[col].dropna().empty], default=None)

fred_text = fred_latest.strftime("%b %d, %Y") if fred_latest is not None else "N/A"
market_text = market_latest.strftime("%b %d, %Y") if market_latest is not None else "N/A"

st.markdown(
    f"""
    <div class="data-status">
        <b>Data status:</b> latest FRED observation {fred_text}; latest market close {market_text}; 
        macro smoothing {smooth}M; macro window {lookback_years}Y; market window {market_period}.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TOP DECISION BRIEF
# =============================================================================

brief_html = f"""
<div class="brief-grid">
    <div class="brief-card">
        <div class="brief-label">Current regime</div>
        <div class="brief-value" style="color:{regime_color(regime)};">{regime}</div>
        <div class="brief-copy">{regime_note}</div>
    </div>
    <div class="brief-card">
        <div class="brief-label">Portfolio stance</div>
        <div class="brief-value">{notes["stance"]}</div>
        <div class="brief-copy">{notes["action"]}</div>
    </div>
    <div class="brief-card">
        <div class="brief-label">Pressure point</div>
        <div class="brief-value">What matters now</div>
        <div class="brief-copy">{notes["pressure"]}</div>
    </div>
    <div class="brief-card">
        <div class="brief-label">Invalidation</div>
        <div class="brief-value">When to change</div>
        <div class="brief-copy">{notes["invalidation"]}</div>
    </div>
</div>
"""
st.markdown(brief_html, unsafe_allow_html=True)


# =============================================================================
# PILLAR CARDS
# =============================================================================

pillar_cards = [
    ("Growth", scores["Growth"], "Positive means firming growth impulse."),
    ("Inflation", scores["Inflation Pressure"], "Positive means more inflation pressure."),
    ("Policy / Conditions", scores["Policy / Conditions"], "Positive means easier policy/conditions."),
    ("Risk Appetite", scores["Risk Appetite"], "Positive means constructive market tape."),
    ("Composite", scores["Composite"], "Net macro and market regime score."),
]

pillar_html = "<div class='pillar-grid'>"
for label, value, note in pillar_cards:
    pillar_html += f"""
    <div class="pillar-card">
        <div class="pillar-label">{label}</div>
        <div class="pillar-value" style="color:{score_color(value if label != "Inflation" else -value)};">{value:+.2f}</div>
        <div class="pillar-note">{score_label(value)}. {note}</div>
    </div>
    """
pillar_html += "</div>"
st.markdown(pillar_html, unsafe_allow_html=True)


# =============================================================================
# REGIME MAP + CONFIRMATION
# =============================================================================

left, right = st.columns([0.9, 1.1])

with left:
    st.markdown("<div class='section-title'>Regime Scores</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Inflation is shown separately in the cards, but inflation relief is used in the composite.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(score_bar_chart(scores), use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Cross-Asset Confirmation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Sorted by composite signal. Positive values confirm easier risk conditions; negative values flag defensive pressure.</div>",
        unsafe_allow_html=True,
    )
    if confirmations.empty:
        st.info("Not enough market history is available to calculate confirmation percentiles.")
    else:
        st.plotly_chart(confirmation_heatmap(confirmations), use_container_width=True)


# =============================================================================
# DECISION NOTES
# =============================================================================

st.markdown("<div class='section-title'>Decision Notes</div>", unsafe_allow_html=True)

note_cols = st.columns(3)

note_blocks = [
    ("Base case", notes["base"]),
    ("Market confirmation", notes["confirmation"]),
    ("Risk sizing", notes["action"]),
]

for col, (label, copy) in zip(note_cols, note_blocks):
    with col:
        st.markdown(
            f"""
            <div class="note-card">
                <div class="note-label">{label}</div>
                <div class="note-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# WHAT CHANGED
# =============================================================================

st.markdown("<div class='section-title'>What Changed</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Levels and changes by macro group. This table is the fastest way to see whether the regime is moving or just sitting at an extreme.</div>",
    unsafe_allow_html=True,
)

if input_table.empty:
    st.info("No input table available.")
else:
    st.dataframe(
        input_table[
            ["Group", "Input", "Latest", "Latest Date", "1M", "3M", "Read-through"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# CHARTS
# =============================================================================

st.markdown("<div class='section-title'>Macro and Market Charts</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Charts are separated by economic function to avoid mixed-axis distortion.</div>",
    unsafe_allow_html=True,
)

tab_growth, tab_inflation, tab_policy, tab_market = st.tabs(
    ["Growth & Labor", "Inflation", "Policy & Conditions", "Market Tape"]
)

with tab_growth:
    if monthly.empty:
        st.info("Macro data unavailable.")
    else:
        st.plotly_chart(chart_growth(monthly), use_container_width=True)

with tab_inflation:
    if monthly.empty:
        st.info("Macro data unavailable.")
    else:
        st.plotly_chart(chart_inflation(monthly), use_container_width=True)

with tab_policy:
    if monthly.empty:
        st.info("Macro data unavailable.")
    else:
        st.plotly_chart(chart_policy(monthly), use_container_width=True)

with tab_market:
    if market.empty:
        st.info("Market data unavailable.")
    else:
        st.plotly_chart(chart_market(market, market_chart_selection), use_container_width=True)


# =============================================================================
# EXPLAINERS
# =============================================================================

with st.expander("Signal definitions"):
    if confirmations.empty:
        st.info("Market confirmation table unavailable.")
    else:
        st.dataframe(
            confirmations[["Signal", "Why it matters", "Composite"]].style.format(
                {"Composite": "{:+.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Scoring methodology"):
    st.markdown(
        """
        Scores are normalized to a -1 to +1 range.

        Growth uses ISM, unemployment, and claims. Inflation pressure uses headline CPI, core CPI, and 10Y breakevens. Policy and conditions use real yields and the Chicago Fed NFCI. Risk appetite uses equities, credit, volatility, and dollar liquidity.

        Composite score uses growth, inflation relief, policy/conditions, and risk appetite. Market confirmation is calculated separately so the tool can show when the macro read and the tape disagree.
        """
    )


# =============================================================================
# RAW DATA
# =============================================================================

if show_raw:
    st.markdown("<div class='section-title'>Raw Data</div>", unsafe_allow_html=True)

    raw_tabs = st.tabs(["FRED", "Market"])

    with raw_tabs[0]:
        if fred.empty:
            st.info("No FRED data loaded.")
        else:
            st.dataframe(fred.tail(160), use_container_width=True)

    with raw_tabs[1]:
        if market.empty:
            st.info("No market data loaded.")
        else:
            st.dataframe(market.tail(160), use_container_width=True)
