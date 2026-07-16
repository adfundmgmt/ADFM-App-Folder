from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.market_data import configure_yfinance_cache
from adfm_core.regime_math import rolling_percentile_previous
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
    render_selection_note,
)

configure_yfinance_cache()


# =============================================================================
# PAGE CONFIG
# =============================================================================

TITLE = "Global Macro Regime Dashboard"
SUBTITLE = "Market-implied regime monitor built from prices, ETFs, futures, FX, yield indexes, credit proxies, commodities, and volatility."

TICKERS: Dict[str, str] = {
    # Equity / risk
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "RSP": "Equal Weight S&P 500",
    "IWM": "Russell 2000",
    "DIA": "Dow Industrials",
    "ACWI": "Global Equities",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "SMH": "Semiconductors",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    # Credit
    "HYG": "High Yield Credit",
    "JNK": "High Yield Credit 2",
    "LQD": "Investment Grade Credit",
    "BKLN": "Senior Loans",
    # Rates / duration
    "SHY": "1-3Y Treasuries",
    "IEF": "7-10Y Treasuries",
    "TLT": "20Y+ Treasuries",
    "TIP": "TIPS",
    "^IRX": "13W Bill Yield",
    "^FVX": "5Y Treasury Yield",
    "^TNX": "10Y Treasury Yield",
    "^TYX": "30Y Treasury Yield",
    # Dollar / FX
    "DX-Y.NYB": "U.S. Dollar Index",
    "UUP": "Dollar ETF",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "EURGBP=X": "EUR/GBP",
    # Commodities
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "GC=F": "Gold",
    "HG=F": "Copper",
    "NG=F": "Natural Gas",
    "GLD": "Gold ETF",
    "CPER": "Copper ETF",
    # Volatility
    "^VIX": "VIX",
    "^VVIX": "VVIX",
}

CORE_DISPLAY = [
    "SPY",
    "QQQ",
    "RSP",
    "IWM",
    "HYG",
    "LQD",
    "TLT",
    "IEF",
    "^TNX",
    "DX-Y.NYB",
    "UUP",
    "CL=F",
    "BZ=F",
    "GC=F",
    "HG=F",
    "^VIX",
]

HORIZONS = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
}

SCORE_COLUMNS = ["1D", "1W", "1M", "3M", "YTD", "Trend", "Composite"]

PALETTE = {
    "bg": "#ffffff",
    "card": "#ffffff",
    "border": "#d9dee7",
    "text": "#111827",
    "muted": "#6b7280",
    "faint": "#94a3b8",
    "grid": "#e5e7eb",
    "green": "#1f7a5a",
    "red": "#a63d40",
    "amber": "#b7791f",
    "blue": "#315f95",
    "purple": "#6d4aff",
    "slate": "#334155",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>
        html, body, .stApp, main, [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
        }
        .block-container {
            padding-top: 3.25rem;
            padding-bottom: 2.5rem;
            max-width: 1560px;
        }
        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.96) !important;
            border-bottom: 1px solid #f1f5f9;
        }
        .page-title {
            padding-top: 0.25rem;
        }
        div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
            color: #111827;
            letter-spacing: -0.01em;
        }
        .page-title {
            font-size: 1.62rem;
            font-weight: 850;
            color: #111827;
            letter-spacing: -0.025em;
            margin-bottom: 0.12rem;
        }
        .page-subtitle {
            font-size: 0.91rem;
            color: #64748b;
            margin-bottom: 0.85rem;
            line-height: 1.38;
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
        .brief-grid {
            display: grid;
            grid-template-columns: 1.25fr 1fr 1fr 1fr;
            gap: 0.72rem;
            margin-bottom: 0.82rem;
        }
        .brief-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 13px;
            padding: 13px 15px;
            min-height: 118px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
        }
        .brief-label, .pillar-label, .note-label {
            font-size: 0.67rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.42rem;
        }
        .brief-value {
            font-size: 1.13rem;
            font-weight: 850;
            color: #111827;
            line-height: 1.18;
            margin-bottom: 0.44rem;
        }
        .brief-copy {
            font-size: 0.80rem;
            color: #475569;
            line-height: 1.42;
        }
        .pillar-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 0.62rem;
            margin-bottom: 0.82rem;
        }
        .pillar-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 12px;
            padding: 11px 12px;
            min-height: 91px;
        }
        .pillar-value {
            font-size: 1.18rem;
            font-weight: 850;
            line-height: 1.08;
            margin-bottom: 0.30rem;
        }
        .pillar-note {
            font-size: 0.745rem;
            color: #64748b;
            line-height: 1.32;
        }
        .section-title {
            font-size: 0.98rem;
            font-weight: 850;
            color: #111827;
            margin-top: 0.78rem;
            margin-bottom: 0.30rem;
            letter-spacing: -0.01em;
        }
        .section-subtitle {
            font-size: 0.80rem;
            color: #64748b;
            margin-bottom: 0.62rem;
            line-height: 1.42;
        }
        .note-card {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 12px;
            padding: 12px 14px;
            min-height: 104px;
            margin-bottom: 0.75rem;
        }
        .note-label { color: #7c5a2b; }
        .note-copy {
            font-size: 0.82rem;
            color: #334155;
            line-height: 1.45;
        }
        .mini-caption {
            color: #64748b;
            font-size: 0.76rem;
            line-height: 1.38;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d9dee7;
            border-radius: 11px;
            overflow: hidden;
        }
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.35rem;
            padding: 0 0.9rem;
        }
        @media (max-width: 1250px) {
            .brief-grid { grid-template-columns: 1fr 1fr; }
            .pillar-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 780px) {
            .brief-grid, .pillar-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_explorer_style()


# =============================================================================
# HELPERS
# =============================================================================


def is_valid(x: float | int | None) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def clean_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def latest_valid(series: pd.Series) -> float:
    clean = clean_series(series)
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(series: pd.Series) -> str:
    clean = clean_series(series)
    if clean.empty:
        return "N/A"
    return pd.Timestamp(clean.index[-1]).strftime("%b %d, %Y")


def pct_fmt(x: float | None, digits: int = 1) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{float(x):.{digits}f}%"


def num_fmt(x: float | None, digits: int = 2) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def signed_num_fmt(x: float | None, digits: int = 2) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{float(x):+.{digits}f}"


def signed_pct_fmt(x: float | None, digits: int = 1) -> str:
    if not is_valid(x):
        return "N/A"
    return f"{float(x):+.{digits}f}%"


def yield_level_fmt(ticker: str, value: float | None) -> str:
    if not is_valid(value):
        return "N/A"
    if ticker in {"^IRX", "^FVX", "^TNX", "^TYX"}:
        return f"{float(value) / 10:.2f}%"
    return num_fmt(value, 2)


def level_fmt(ticker: str, value: float | None) -> str:
    if not is_valid(value):
        return "N/A"
    if ticker in {"^VIX", "^VVIX"}:
        return num_fmt(value, 1)
    if ticker in {"^IRX", "^FVX", "^TNX", "^TYX"}:
        return yield_level_fmt(ticker, value)
    if ticker.endswith("=X"):
        return num_fmt(value, 4)
    return num_fmt(value, 2)


def safe_pct_change(series: pd.Series, periods: int) -> float:
    clean = clean_series(series)
    if len(clean) <= periods:
        return np.nan
    base = clean.iloc[-periods - 1]
    last = clean.iloc[-1]
    if not is_valid(base) or base == 0:
        return np.nan
    return float(last / base - 1.0)


def ytd_return(series: pd.Series) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    last_date = pd.Timestamp(clean.index[-1])
    same_year = clean[clean.index >= pd.Timestamp(date(last_date.year, 1, 1))]
    if len(same_year) < 2:
        return np.nan
    base = same_year.iloc[0]
    if not is_valid(base) or base == 0:
        return np.nan
    return float(same_year.iloc[-1] / base - 1.0)


def weighted_average(values: Dict[str, float], weights: Dict[str, float]) -> float:
    used = [
        (values[k], weights[k]) for k in weights if k in values and is_valid(values[k])
    ]
    if not used:
        return 0.0
    total_weight = sum(w for _, w in used)
    if total_weight == 0:
        return 0.0
    return float(sum(v * w for v, w in used) / total_weight)


def normalize_score(value: float | None) -> float:
    if not is_valid(value):
        return 0.0
    return float(np.clip(float(value), -1.0, 1.0))


def percentile_score_from_change(series: pd.Series, periods: int) -> float:
    clean = clean_series(series)
    if len(clean) <= max(periods + 5, 35):
        return np.nan
    changes = clean.pct_change(periods).replace([np.inf, -np.inf], np.nan).dropna()
    if len(changes) < 30:
        return np.nan
    rank_series = rolling_percentile_previous(
        changes,
        window=min(756, max(30, len(changes) - 1)),
        min_periods=min(30, max(1, len(changes) - 1)),
    )
    rank = latest_valid(rank_series)
    if not is_valid(rank):
        return np.nan
    return normalize_score((rank - 0.5) * 2.0)


def trend_score(series: pd.Series) -> float:
    clean = clean_series(series)
    if len(clean) < 60:
        return np.nan
    window = min(200, max(50, len(clean) // 2))
    ma = clean.rolling(window, min_periods=max(30, window // 2)).mean()
    spread = (clean / ma - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if len(spread) < 25:
        return np.nan
    rank_series = rolling_percentile_previous(
        spread,
        window=min(756, max(30, len(spread) - 1)),
        min_periods=min(30, max(1, len(spread) - 1)),
    )
    rank = latest_valid(rank_series)
    if not is_valid(rank):
        return np.nan
    return normalize_score((rank - 0.5) * 2.0)


def ytd_percentile_score(series: pd.Series) -> float:
    """Compare the current YTD move with prior moves of the same trading-day length."""
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    last_date = pd.Timestamp(clean.index[-1])
    year_slice = clean.loc[clean.index >= pd.Timestamp(date(last_date.year, 1, 1))]
    periods = max(1, len(year_slice) - 1)
    return percentile_score_from_change(clean, periods)


def score_color(score: float | None) -> str:
    if not is_valid(score):
        return PALETTE["amber"]
    score = float(score)
    if score >= 0.25:
        return PALETTE["green"]
    if score <= -0.25:
        return PALETTE["red"]
    return PALETTE["amber"]


def score_label(score: float | None) -> str:
    if not is_valid(score):
        return "N/A"
    score = float(score)
    if score >= 0.55:
        return "Strong positive"
    if score >= 0.25:
        return "Positive"
    if score <= -0.55:
        return "Strong negative"
    if score <= -0.25:
        return "Negative"
    return "Mixed"


def score_phrase(score: float | None) -> str:
    if not is_valid(score):
        return "unavailable"
    score = float(score)
    if score >= 0.55:
        return "strongly constructive"
    if score >= 0.25:
        return "constructive"
    if score <= -0.55:
        return "strongly defensive"
    if score <= -0.25:
        return "defensive"
    return "mixed"


# =============================================================================
# DATA
# =============================================================================


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_prices(
    tickers: Tuple[str, ...], period: str
) -> Tuple[pd.DataFrame, List[str]]:
    if not tickers:
        return pd.DataFrame(), []

    try:
        data = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame(), list(tickers)

    if data is None or data.empty:
        return pd.DataFrame(), list(tickers)

    close = pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = list(data.columns.get_level_values(0).unique())
            level1 = list(data.columns.get_level_values(1).unique())

            if "Close" in level0:
                close = data["Close"].copy()
            elif "Close" in level1:
                close = data.xs("Close", level=1, axis=1).copy()
        else:
            if "Close" in data.columns and len(tickers) == 1:
                close = data[["Close"]].rename(columns={"Close": tickers[0]}).copy()
            elif "Adj Close" in data.columns and len(tickers) == 1:
                close = (
                    data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]}).copy()
                )
    except Exception:
        close = pd.DataFrame()

    if close.empty:
        return pd.DataFrame(), list(tickers)

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close = close.loc[:, ~close.columns.duplicated()].copy()

    ordered_cols = [ticker for ticker in tickers if ticker in close.columns]
    close = close.reindex(columns=ordered_cols)
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.ffill().dropna(axis=1, how="all").dropna(how="all")

    failed = [
        ticker
        for ticker in tickers
        if ticker not in close.columns or close[ticker].dropna().empty
    ]
    return close, failed


def pick_series(
    prices: pd.DataFrame, primary: str, fallback: str | None = None
) -> Tuple[pd.Series | None, str | None]:
    if primary in prices.columns and not prices[primary].dropna().empty:
        return prices[primary], primary
    if fallback and fallback in prices.columns and not prices[fallback].dropna().empty:
        return prices[fallback], fallback
    return None, None


def safe_ratio(
    prices: pd.DataFrame, numerator: str, denominator: str
) -> pd.Series | None:
    if numerator not in prices.columns or denominator not in prices.columns:
        return None
    denom = prices[denominator].replace(0, np.nan)
    ratio = prices[numerator] / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return ratio if not ratio.empty else None


# =============================================================================
# SIGNALS
# =============================================================================


def build_signal_definitions(prices: pd.DataFrame) -> List[dict]:
    signals: List[dict] = []

    def add_signal(
        group: str,
        signal: str,
        proxy: str,
        series: pd.Series | None,
        why: str,
    ) -> None:
        if series is None:
            return
        clean = clean_series(series)
        if len(clean) < 45:
            return
        signals.append(
            {
                "Group": group,
                "Signal": signal,
                "Proxy": proxy,
                "Series": clean,
                "Why it matters": why,
            }
        )

    add_signal(
        "Equity Tape",
        "Equity trend",
        "SPY",
        prices["SPY"] if "SPY" in prices else None,
        "The cleanest liquid proxy for broad U.S. risk appetite.",
    )
    add_signal(
        "Equity Tape",
        "Breadth",
        "RSP / SPY",
        safe_ratio(prices, "RSP", "SPY"),
        "Equal weight versus cap weight shows whether the tape is broadening or narrowing.",
    )
    add_signal(
        "Equity Tape",
        "Small-cap liquidity",
        "IWM / SPY",
        safe_ratio(prices, "IWM", "SPY"),
        "Small-cap leadership usually requires easier liquidity and stronger domestic cyclicals.",
    )
    add_signal(
        "Equity Tape",
        "Nasdaq leadership",
        "QQQ / SPY",
        safe_ratio(prices, "QQQ", "SPY"),
        "Growth and mega-cap leadership are a useful check on duration-sensitive equity risk.",
    )
    add_signal(
        "Equity Tape",
        "Semiconductor leadership",
        "SMH / SPY",
        safe_ratio(prices, "SMH", "SPY"),
        "Semis capture the highest-beta AI and capex leadership pocket.",
    )

    add_signal(
        "Credit",
        "Credit risk appetite",
        "HYG / LQD",
        safe_ratio(prices, "HYG", "LQD"),
        "High yield versus investment grade is the fastest liquid proxy for credit risk appetite.",
    )
    add_signal(
        "Credit",
        "Loan risk appetite",
        "BKLN / SHY",
        safe_ratio(prices, "BKLN", "SHY"),
        "Senior loans versus short Treasuries help flag credit demand outside plain high yield.",
    )

    add_signal(
        "Rates",
        "Duration bid",
        "TLT / SHY",
        safe_ratio(prices, "TLT", "SHY"),
        "Long duration outperforming bills points to lower-rate pressure or a growth scare hedge bid.",
    )

    if "^TNX" in prices:
        add_signal(
            "Rates",
            "Lower 10Y yield impulse",
            "1 / ^TNX",
            1.0 / prices["^TNX"].replace(0, np.nan),
            "Lower 10-year yields ease valuation pressure and usually help duration-sensitive assets.",
        )

    if {"^TYX", "^IRX"}.issubset(prices.columns):
        curve = prices["^TYX"] - prices["^IRX"]
        add_signal(
            "Rates",
            "Curve steepening impulse",
            "^TYX - ^IRX",
            curve,
            "A steeper curve can mark easing expectations, term-premium pressure, or cyclical reflation depending on the rest of the tape.",
        )

    dollar_series, dollar_proxy = pick_series(prices, "DX-Y.NYB", "UUP")
    if dollar_series is not None and dollar_proxy is not None:
        add_signal(
            "Dollar Liquidity",
            "Softer dollar liquidity",
            f"1 / {dollar_proxy}",
            1.0 / dollar_series.replace(0, np.nan),
            "A softer dollar generally loosens global financial conditions and helps non-U.S. risk.",
        )

    add_signal(
        "Dollar Liquidity",
        "EUR/USD pressure",
        "EURUSD=X",
        prices["EURUSD=X"] if "EURUSD=X" in prices else None,
        "EUR/USD strength is a simple check on dollar weakness and global liquidity relief.",
    )

    copper_gold = safe_ratio(prices, "HG=F", "GC=F")
    copper_gold_proxy = "Copper / Gold futures"
    if copper_gold is None:
        copper_gold = safe_ratio(prices, "CPER", "GLD")
        copper_gold_proxy = "CPER / GLD"
    add_signal(
        "Commodities",
        "Copper/gold cyclicality",
        copper_gold_proxy,
        copper_gold,
        "Copper versus gold tracks cyclical demand against defensive scarcity demand.",
    )

    crude_series, crude_proxy = pick_series(prices, "CL=F", "BZ=F")
    if crude_series is not None and crude_proxy is not None:
        add_signal(
            "Commodities",
            "Crude disinflation relief",
            f"1 / {crude_proxy}",
            1.0 / crude_series.replace(0, np.nan),
            "Lower crude reduces inflation pressure and can improve the policy optionality mix.",
        )

    if "^VIX" in prices:
        add_signal(
            "Volatility",
            "Volatility relief",
            "1 / ^VIX",
            1.0 / prices["^VIX"].replace(0, np.nan),
            "Lower VIX confirms risk tolerance and easier hedging conditions.",
        )

    if "^VVIX" in prices:
        add_signal(
            "Volatility",
            "Vol-of-vol relief",
            "1 / ^VVIX",
            1.0 / prices["^VVIX"].replace(0, np.nan),
            "VVIX is a useful stress proxy when index volatility is calm but option demand is unstable.",
        )

    return signals


def score_signal(series: pd.Series) -> Dict[str, float]:
    scores = {
        label: percentile_score_from_change(series, days)
        for label, days in HORIZONS.items()
    }
    scores["YTD"] = ytd_percentile_score(series)
    scores["Trend"] = trend_score(series)
    scores["Composite"] = weighted_average(
        scores,
        {
            "1D": 0.00,
            "1W": 0.10,
            "1M": 0.30,
            "3M": 0.35,
            "YTD": 0.10,
            "Trend": 0.15,
        },
    )
    return scores


def build_signal_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for item in build_signal_definitions(prices):
        series = item["Series"]
        scores = score_signal(series)
        row = {
            "Group": item["Group"],
            "Signal": item["Signal"],
            "Proxy": item["Proxy"],
            "Why it matters": item["Why it matters"],
            "Latest Date": latest_date(series),
            **scores,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("Composite", ascending=False).reset_index(drop=True)
    return frame


def group_scores(signals: pd.DataFrame) -> Dict[str, float]:
    if signals.empty:
        return {}

    groups = signals.groupby("Group")["Composite"].mean().to_dict()

    equity = groups.get("Equity Tape", 0.0)
    credit = groups.get("Credit", 0.0)
    rates = groups.get("Rates", 0.0)
    dollar = groups.get("Dollar Liquidity", 0.0)
    commodities = groups.get("Commodities", 0.0)
    vol = groups.get("Volatility", 0.0)

    composite = weighted_average(
        {
            "Equity Tape": equity,
            "Credit": credit,
            "Rates": rates,
            "Dollar Liquidity": dollar,
            "Commodities": commodities,
            "Volatility": vol,
        },
        {
            "Equity Tape": 0.26,
            "Credit": 0.20,
            "Rates": 0.16,
            "Dollar Liquidity": 0.14,
            "Commodities": 0.12,
            "Volatility": 0.12,
        },
    )

    groups["Composite"] = composite
    return groups


def classify_regime(scores: Dict[str, float]) -> Tuple[str, str]:
    equity = scores.get("Equity Tape", 0.0)
    credit = scores.get("Credit", 0.0)
    rates = scores.get("Rates", 0.0)
    dollar = scores.get("Dollar Liquidity", 0.0)
    commodities = scores.get("Commodities", 0.0)
    vol = scores.get("Volatility", 0.0)
    composite = scores.get("Composite", 0.0)

    if composite >= 0.35 and equity >= 0.15 and credit >= 0.10 and vol >= 0.00:
        return (
            "Risk-on / liquidity easing",
            "Equity tape, credit, and volatility are confirming easier conditions. The burden of proof shifts to breadth and credit staying firm.",
        )

    if equity >= 0.20 and credit >= 0.10 and rates <= -0.15:
        return (
            "Reflation / rate pressure",
            "Risk assets are holding up while the rates impulse is tightening. This works until yields begin to pressure multiples or credit.",
        )

    if rates >= 0.25 and dollar >= 0.10 and commodities >= 0.00 and equity > -0.15:
        return (
            "Disinflationary easing",
            "Duration and dollar liquidity are improving without a decisive break in risk assets. That is the cleanest easing mix.",
        )

    if equity <= -0.25 and credit <= -0.20:
        return (
            "Risk-off / credit stress",
            "Equity weakness is being confirmed by credit. Liquidity matters more than valuation until credit stabilizes.",
        )

    if rates <= -0.25 and dollar <= -0.20:
        return (
            "Tightening impulse",
            "Rates and dollar liquidity are moving in the wrong direction together. That usually raises the hurdle for long-duration risk.",
        )

    if commodities <= -0.30 and equity <= 0.00 and credit <= 0.00:
        return (
            "Growth scare / defensive bid",
            "Commodity cyclicality and risk appetite are weak together. The market is pricing weaker nominal growth rather than clean disinflation.",
        )

    return (
        "Mixed / tactical",
        "The tape is not one-way. This is a relative-value backdrop until credit, breadth, or rates resolve the contradiction.",
    )


def build_decision_notes(
    regime: str, regime_note: str, scores: Dict[str, float], signals: pd.DataFrame
) -> Dict[str, str]:
    composite = scores.get("Composite", 0.0)
    dimensions = {k: v for k, v in scores.items() if k != "Composite"}
    strongest_name, strongest_value = (
        max(dimensions.items(), key=lambda item: item[1])
        if dimensions
        else ("N/A", np.nan)
    )
    weakest_name, weakest_value = (
        min(dimensions.items(), key=lambda item: item[1])
        if dimensions
        else ("N/A", np.nan)
    )

    if signals.empty:
        positive = negative = total = 0
        leaders: List[str] = []
        laggards: List[str] = []
    else:
        market_scores = signals["Composite"].dropna()
        positive = int((market_scores > 0.20).sum())
        negative = int((market_scores < -0.20).sum())
        total = int(len(market_scores))
        leaders = signals.nlargest(min(2, len(signals)), "Composite")["Signal"].tolist()
        laggards = signals.nsmallest(min(2, len(signals)), "Composite")[
            "Signal"
        ].tolist()

    equity = scores.get("Equity Tape", 0.0)
    credit = scores.get("Credit", 0.0)

    if composite >= 0.30 and credit >= 0.00:
        stance = "Constructive but selective"
        action = "Risk can stay on, but sizing should be concentrated where credit, breadth, and trend confirm together."
        invalidation = "Cut risk if HYG/LQD turns negative, breadth rolls over, or VIX begins rising while SPY is flat."
    elif composite <= -0.30 or credit <= -0.30:
        stance = "Defensive"
        action = "Keep gross liquidity high, avoid weak-beta exposure, and require credit repair before adding cyclical risk."
        invalidation = "Reassess if credit turns positive, breadth stops deteriorating, and volatility relief broadens beyond one session."
    else:
        stance = "Balanced / tactical"
        action = "Avoid a large one-way book. Favor liquid relative-value trades until the tape confirms a cleaner regime."
        invalidation = "Change posture when the composite clears ±0.35 with credit and breadth moving in the same direction."

    if (
        is_valid(strongest_value)
        and is_valid(weakest_value)
        and strongest_value - weakest_value > 0.70
    ):
        pressure = f"Internal dispersion is high: {strongest_name} is leading while {weakest_name} is the drag."
    else:
        pressure = f"The main pressure point is {weakest_name}; the strongest offset is {strongest_name}."

    confirmation = (
        f"{positive} of {total} signals are constructive; {negative} are defensive. "
        f"Leadership: {', '.join(leaders) if leaders else 'N/A'}. "
        f"Weakness: {', '.join(laggards) if laggards else 'N/A'}."
    )

    base = (
        f"{regime}. Composite is {score_phrase(composite)} at {composite:+.2f}. "
        f"Equity tape {scores.get('Equity Tape', 0.0):+.2f}, credit {scores.get('Credit', 0.0):+.2f}, "
        f"rates {scores.get('Rates', 0.0):+.2f}, dollar liquidity {scores.get('Dollar Liquidity', 0.0):+.2f}, "
        f"commodities {scores.get('Commodities', 0.0):+.2f}, volatility {scores.get('Volatility', 0.0):+.2f}."
    )

    return {
        "stance": stance,
        "action": action,
        "pressure": pressure,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "base": base,
        "regime_note": regime_note,
        "equity_credit": f"Equity tape {equity:+.2f}; credit {credit:+.2f}. Credit remains the higher-quality confirmation signal.",
    }


# =============================================================================
# TABLES
# =============================================================================


def asset_return_rows(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for ticker in tickers:
        if ticker not in prices.columns or prices[ticker].dropna().empty:
            continue
        series = prices[ticker]
        row = {
            "Ticker": ticker,
            "Asset": TICKERS.get(ticker, ticker),
            "Latest": level_fmt(ticker, latest_valid(series)),
            "Latest Date": latest_date(series),
        }
        for label, periods in HORIZONS.items():
            row[label] = safe_pct_change(series, periods) * 100
        row["YTD"] = ytd_return(series) * 100
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    sort_col = "1M" if "1M" in frame else "YTD"
    frame = frame.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return frame


def display_asset_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for col in ["1D", "1W", "1M", "3M", "YTD"]:
        if col in out:
            out[col] = out[col].map(lambda x: signed_pct_fmt(x, 1))
    return out


def display_signal_table(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals[
        [
            "Group",
            "Signal",
            "Proxy",
            "Latest Date",
            "1D",
            "1W",
            "1M",
            "3M",
            "YTD",
            "Trend",
            "Composite",
            "Why it matters",
        ]
    ].copy()
    for col in ["1D", "1W", "1M", "3M", "YTD", "Trend", "Composite"]:
        out[col] = out[col].map(lambda x: signed_num_fmt(x, 2))
    return out


# =============================================================================
# CHARTS
# =============================================================================


def apply_layout(
    fig: go.Figure, height: int = 360, showlegend: bool = True
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=24, t=30, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
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


def score_bar_chart(scores: Dict[str, float]) -> go.Figure:
    labels = [
        "Equity Tape",
        "Credit",
        "Rates",
        "Dollar Liquidity",
        "Commodities",
        "Volatility",
        "Composite",
    ]
    values = [scores.get(label, np.nan) for label in labels]
    colors = [score_color(value) for value in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[signed_num_fmt(v, 2) for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        height=318,
        margin=dict(l=10, r=48, t=12, b=20),
        xaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def signal_heatmap(signals: pd.DataFrame) -> go.Figure:
    if signals.empty:
        return apply_layout(go.Figure(), height=330, showlegend=False)

    values = signals[SCORE_COLUMNS].to_numpy(dtype=float)
    text = np.where(
        np.isfinite(values), np.vectorize(lambda v: f"{v:+.2f}")(values), "N/A"
    )

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=SCORE_COLUMNS,
            y=signals["Signal"],
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
            colorbar=dict(thickness=10, len=0.80, tickvals=[-1, -0.5, 0, 0.5, 1]),
        )
    )
    fig.update_layout(
        height=max(320, 28 * len(signals) + 90),
        margin=dict(l=10, r=10, t=14, b=20),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=11),
    )
    return fig


def asset_performance_heatmap(asset_returns: pd.DataFrame) -> go.Figure:
    if asset_returns.empty:
        return apply_layout(go.Figure(), height=360, showlegend=False)

    cols = ["1D", "1W", "1M", "3M", "YTD"]
    frame = asset_returns.dropna(
        subset=[c for c in cols if c in asset_returns], how="all"
    ).copy()
    frame = frame.head(22)
    values = frame[cols].to_numpy(dtype=float)
    text = np.where(
        np.isfinite(values), np.vectorize(lambda v: f"{v:+.1f}%")(values), "N/A"
    )
    labels = frame["Ticker"] + "  " + frame["Asset"]

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=cols,
            y=labels,
            zmid=0,
            colorscale=[
                [0.00, "#8f2d35"],
                [0.42, "#d8a092"],
                [0.50, "#f2f0ea"],
                [0.58, "#93c7b4"],
                [1.00, "#1f7a5a"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f}%<extra></extra>",
            colorbar=dict(title="%", thickness=10, len=0.78),
        )
    )
    fig.update_layout(
        height=max(360, 24 * len(frame) + 86),
        margin=dict(l=10, r=10, t=14, b=20),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=11),
    )
    return fig


def horizon_bar_chart(asset_returns: pd.DataFrame, horizon: str) -> go.Figure:
    if asset_returns.empty or horizon not in asset_returns.columns:
        return apply_layout(go.Figure(), height=320, showlegend=False)

    frame = asset_returns.dropna(subset=[horizon]).copy()
    frame = frame.sort_values(horizon, ascending=True).tail(22)
    colors = [score_color(v / 8.0) for v in frame[horizon]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame[horizon],
            y=frame["Ticker"] + "  " + frame["Asset"],
            orientation="h",
            marker_color=colors,
            text=[signed_pct_fmt(v, 1) for v in frame[horizon]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Return: %{x:+.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        height=max(330, 24 * len(frame) + 70),
        margin=dict(l=10, r=55, t=14, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(color="#334155", size=11),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False, title=f"{horizon} return")
    fig.update_yaxes(showgrid=False)
    return fig


def rebased_price_chart(prices: pd.DataFrame, tickers: List[str]) -> go.Figure:
    fig = go.Figure()
    chosen = [
        ticker
        for ticker in tickers
        if ticker in prices.columns and not prices[ticker].dropna().empty
    ]
    if not chosen:
        return apply_layout(fig, height=390)

    frame = prices[chosen].ffill().dropna(how="all")
    frame = frame.dropna(how="all")

    for ticker in chosen:
        series = clean_series(frame[ticker])
        if len(series) < 2:
            continue
        rebased = series / series.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased,
                mode="lines",
                name=f"{ticker} {TICKERS.get(ticker, ticker)}",
                line=dict(width=2),
            )
        )

    fig.update_yaxes(title_text="Rebased to 100")
    return apply_layout(fig, height=395)


def signal_line_chart(prices: pd.DataFrame, signal_names: List[str]) -> go.Figure:
    fig = go.Figure()
    definitions = build_signal_definitions(prices)
    lookup = {item["Signal"]: item for item in definitions}

    for name in signal_names:
        item = lookup.get(name)
        if not item:
            continue
        series = clean_series(item["Series"])
        if len(series) < 2:
            continue
        rebased = series / series.iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased,
                mode="lines",
                name=name,
                line=dict(width=2),
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_yaxes(title_text="Constructive proxy rebased to 100")
    return apply_layout(fig, height=395)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Cross-asset regime monitor built from liquid market proxies rather than economic-release endpoints. The output is designed to be fast, stable, and comparable across ADFM tabs.

        It should be read as a tape-and-liquidity dashboard, not as an official CPI, ISM, claims, or Fed-release page.
        """
    )

    st.divider()
    st.header("Controls")

    period = st.selectbox(
        "Market history",
        ["6mo", "1y", "2y", "3y", "5y"],
        index=4,
        help="Longer windows improve percentile scoring. Shorter windows load faster but reduce signal depth.",
    )

    primary_horizon = st.selectbox(
        "Primary move horizon",
        ["1D", "1W", "1M", "3M", "YTD"],
        index=2,
        help="Controls the main ranked performance chart.",
    )

    default_assets = [
        "SPY",
        "QQQ",
        "RSP",
        "IWM",
        "HYG",
        "LQD",
        "TLT",
        "^TNX",
        "DX-Y.NYB",
        "CL=F",
        "GC=F",
        "HG=F",
        "^VIX",
    ]
    selected_assets = st.multiselect(
        "Asset chart",
        options=list(TICKERS.keys()),
        default=[ticker for ticker in default_assets if ticker in TICKERS],
        format_func=lambda ticker: f"{ticker} · {TICKERS.get(ticker, ticker)}",
    )

    default_signal_names = [
        "Equity trend",
        "Breadth",
        "Credit risk appetite",
        "Duration bid",
        "Softer dollar liquidity",
        "Copper/gold cyclicality",
        "Volatility relief",
    ]
    selected_signal_names = st.multiselect(
        "Signal chart",
        options=[
            "Equity trend",
            "Breadth",
            "Small-cap liquidity",
            "Nasdaq leadership",
            "Semiconductor leadership",
            "Credit risk appetite",
            "Loan risk appetite",
            "Duration bid",
            "Lower 10Y yield impulse",
            "Curve steepening impulse",
            "Softer dollar liquidity",
            "EUR/USD pressure",
            "Copper/gold cyclicality",
            "Crude disinflation relief",
            "Volatility relief",
            "Vol-of-vol relief",
        ],
        default=default_signal_names,
    )

    show_raw = st.checkbox("Show raw data", value=False)


# =============================================================================
# LOAD DATA
# =============================================================================

render_page_header(
    PageHeader(
        title=TITLE,
        description=SUBTITLE,
        eyebrow="ADFM Cross-Asset Regimes",
    )
)

all_tickers = tuple(TICKERS.keys())
prices, failed = fetch_market_prices(all_tickers, period)

if prices.empty:
    st.error("No market data loaded. Check connectivity or reduce the ticker list.")
    st.stop()

signals = build_signal_table(prices)
scores = group_scores(signals)
regime, regime_note = classify_regime(scores)
notes = build_decision_notes(regime, regime_note, scores, signals)
asset_returns = asset_return_rows(
    prices, [ticker for ticker in CORE_DISPLAY if ticker in TICKERS]
)

latest_market_dt = max(
    [
        pd.Timestamp(prices[col].dropna().index[-1])
        for col in prices.columns
        if not prices[col].dropna().empty
    ],
    default=None,
)
latest_market_text = (
    latest_market_dt.strftime("%b %d, %Y") if latest_market_dt is not None else "N/A"
)
loaded_count = len([col for col in prices.columns if not prices[col].dropna().empty])
failed_text = (
    ", ".join(failed[:8]) + ("..." if len(failed) > 8 else "") if failed else "None"
)
signal_values = (
    signals["Composite"].dropna() if not signals.empty else pd.Series(dtype=float)
)
constructive_share = (
    float((signal_values > 0.20).mean() * 100.0) if not signal_values.empty else np.nan
)
defensive_share = (
    float((signal_values < -0.20).mean() * 100.0) if not signal_values.empty else np.nan
)
constructive_text = (
    f"{constructive_share:.0f}%" if is_valid(constructive_share) else "N/A"
)
defensive_text = f"{defensive_share:.0f}%" if is_valid(defensive_share) else "N/A"

st.markdown(
    f"""
    <div class="data-status">
        <b>Data status:</b> latest market observation {latest_market_text}; loaded {loaded_count}/{len(TICKERS)} tickers; history {period}; unavailable tickers: {failed_text}.
    </div>
    """,
    unsafe_allow_html=True,
)

render_selection_note(
    "Active regime read",
    f"{regime}. {regime_note} Signal breadth is {constructive_text} constructive and "
    f"{defensive_text} defensive across {len(signal_values)} available signals. "
    f"Scores use causal percentiles against the selected {period} history.",
)


# =============================================================================
# TOP DECISION BRIEF
# =============================================================================

brief_html = f"""
<div class="brief-grid">
    <div class="brief-card">
        <div class="brief-label">Current regime</div>
        <div class="brief-value" style="color:{score_color(scores.get("Composite", 0.0))};">{regime}</div>
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

pillar_items = [
    (
        "Equity Tape",
        scores.get("Equity Tape", np.nan),
        "Trend, breadth, small caps, Nasdaq, semis.",
    ),
    (
        "Credit",
        scores.get("Credit", np.nan),
        "High yield and loans versus safer bonds.",
    ),
    ("Rates", scores.get("Rates", np.nan), "Duration bid, yields, and curve impulse."),
    (
        "Dollar Liquidity",
        scores.get("Dollar Liquidity", np.nan),
        "DXY/UUP and EUR/USD pressure.",
    ),
    ("Commodities", scores.get("Commodities", np.nan), "Copper/gold and crude relief."),
    ("Volatility", scores.get("Volatility", np.nan), "VIX and vol-of-vol relief."),
    ("Composite", scores.get("Composite", np.nan), "Weighted regime score."),
]

pillar_cols = st.columns(len(pillar_items), gap="small")
for col, (label, value, note) in zip(pillar_cols, pillar_items):
    with col:
        st.markdown(
            f"""
            <div class="pillar-card">
                <div class="pillar-label">{label}</div>
                <div class="pillar-value" style="color:{score_color(value)};">{signed_num_fmt(value, 2)}</div>
                <div class="pillar-note">{score_label(value)}. {note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# VISUALS FIRST
# =============================================================================

left, right = st.columns([0.85, 1.15])

with left:
    st.markdown(
        "<div class='section-title'>Regime Scores</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-subtitle'>Positive scores mean easier risk conditions; negative scores flag tightening or defensive pressure.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(score_bar_chart(scores), width="stretch")

with right:
    st.markdown(
        "<div class='section-title'>Signal Confirmation Matrix</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Percentile-based scores across short-term moves, YTD impulse, trend, and composite confirmation.</div>",
        unsafe_allow_html=True,
    )
    if signals.empty:
        st.info("Not enough market data to calculate signal scores.")
    else:
        st.plotly_chart(signal_heatmap(signals), width="stretch")

st.markdown(
    "<div class='section-title'>Cross-Asset Performance</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='section-subtitle'>Price moves across the liquid proxy set. Futures, FX, and yield indexes are included when usable data is available.</div>",
    unsafe_allow_html=True,
)

perf_left, perf_right = st.columns([1.05, 0.95])
with perf_left:
    st.plotly_chart(asset_performance_heatmap(asset_returns), width="stretch")
with perf_right:
    st.plotly_chart(horizon_bar_chart(asset_returns, primary_horizon), width="stretch")


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
# CHART TABS
# =============================================================================

st.markdown(
    "<div class='section-title'>Market Structure Charts</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='section-subtitle'>The signal chart uses constructive proxies, so higher usually means easier conditions or better risk confirmation.</div>",
    unsafe_allow_html=True,
)

tab_assets, tab_signals, tab_tables = st.tabs(
    ["Asset Tape", "Constructive Signals", "Tables"]
)

with tab_assets:
    st.plotly_chart(rebased_price_chart(prices, selected_assets), width="stretch")

with tab_signals:
    st.plotly_chart(signal_line_chart(prices, selected_signal_names), width="stretch")

with tab_tables:
    st.markdown("<div class='section-title'>Asset Moves</div>", unsafe_allow_html=True)
    if asset_returns.empty:
        st.info("No asset return table available.")
    else:
        st.dataframe(
            display_asset_table(asset_returns),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        "<div class='section-title'>Signal Definitions and Scores</div>",
        unsafe_allow_html=True,
    )
    if signals.empty:
        st.info("No signal table available.")
    else:
        st.dataframe(display_signal_table(signals), width="stretch", hide_index=True)


# =============================================================================
# METHODOLOGY / RAW DATA
# =============================================================================

with st.expander("Scoring methodology"):
    st.markdown(
        """
        This page uses liquid market proxies rather than macro-release endpoints.

        Each signal is transformed so that a higher series is generally more constructive for liquidity or risk appetite. Examples: HYG/LQD higher is constructive for credit; 1/VIX higher is volatility relief; 1/DXY or 1/UUP higher is softer-dollar liquidity; 1/crude higher is crude disinflation relief. The tool scores 1D, 1W, 1M, and 3M moves against prior observations only, compares the current YTD move with historical moves of the same trading-day length, adds a trend score, and rolls those into a -1 to +1 composite. The 1D score is diagnostic but intentionally receives no composite weight.

        This is a market-implied regime tool. It is designed to answer what the tape is confirming now, not what CPI, payrolls, ISM, claims, or Fed funds officially printed.
        """
    )

if show_raw:
    st.markdown(
        "<div class='section-title'>Raw Market Data</div>", unsafe_allow_html=True
    )
    st.dataframe(prices.tail(220), width="stretch")


render_footer()
