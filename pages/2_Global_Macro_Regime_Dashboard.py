from __future__ import annotations

from datetime import timedelta
from html import escape
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.market_data import (
    close_panel,
    configure_yfinance_cache,
    fetch_daily_ohlcv,
)
from adfm_core.primary_data import fetch_fred_series
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_page_header,
)

configure_yfinance_cache()


# =============================================================================
# PAGE CONFIG
# =============================================================================

TITLE = "Global Macro Regime Dashboard"
SUBTITLE = (
    "A transparent cross-asset read of growth, inflation, rates, liquidity and risk appetite. "
    "Every regime call is tied to observable market moves, yield changes or primary-source data."
)

TICKERS: Dict[str, str] = {
    # Equity / risk
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "RSP": "Equal Weight S&P 500",
    "IWM": "Russell 2000",
    "EFA": "Developed ex-US",
    "EEM": "Emerging Markets",
    "FXI": "China Large Caps",
    "SMH": "Semiconductors",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    # Credit / rates
    "HYG": "High Yield Credit",
    "LQD": "Investment Grade Credit",
    "BKLN": "Senior Loans",
    "SHY": "1-3Y Treasuries",
    "IEF": "7-10Y Treasuries",
    "TLT": "20Y+ Treasuries",
    "TIP": "TIPS",
    # FX
    "DX-Y.NYB": "U.S. Dollar Index",
    "UUP": "Dollar ETF",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    # Commodities
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "GC=F": "Gold",
    "HG=F": "Copper",
    # Volatility
    "^VIX": "VIX",
}

PERFORMANCE_TICKERS = [
    "SPY",
    "QQQ",
    "RSP",
    "IWM",
    "EFA",
    "EEM",
    "FXI",
    "SMH",
    "XLF",
    "XLI",
    "XLE",
    "HYG",
    "LQD",
    "IEF",
    "TLT",
    "UUP",
    "GC=F",
    "CL=F",
    "HG=F",
]

PALETTE = {
    "text": "#111827",
    "muted": "#64748b",
    "faint": "#94a3b8",
    "border": "#dfe3e8",
    "grid": "#e5e7eb",
    "green": "#6f9653",
    "red": "#b75b54",
    "amber": "#d9a525",
    "blue": "#4569ad",
    "slate": "#475569",
    "green_bg": "#f3f8ef",
    "red_bg": "#fbf3f2",
    "amber_bg": "#fff9e9",
    "blue_bg": "#f3f6fb",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
inject_explorer_style()


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
            max-width: 1580px;
        }
        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.96) !important;
            border-bottom: 1px solid #f1f5f9;
        }
        .data-status {
            color: #64748b;
            font-size: 0.73rem;
            line-height: 1.4;
            margin: -0.15rem 0 0.62rem;
        }
        .regime-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.72fr);
            gap: 1.15rem;
            align-items: stretch;
            border: 1px solid rgba(100, 116, 139, 0.20);
            border-left: 4px solid var(--hero-accent, #4569ad);
            border-radius: 10px;
            background: #ffffff;
            padding: 15px 17px;
            margin: 0.35rem 0 0.78rem;
        }
        .hero-kicker {
            font-size: 0.63rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--hero-accent, #4569ad);
            margin-bottom: 0.34rem;
        }
        .hero-title {
            font-size: 1.48rem;
            line-height: 1.12;
            font-weight: 850;
            color: #111827;
            letter-spacing: -0.025em;
            margin-bottom: 0.34rem;
        }
        .hero-copy {
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.46;
        }
        .hero-side {
            border-left: 1px solid rgba(100, 116, 139, 0.18);
            padding-left: 1.05rem;
        }
        .hero-side-label {
            font-size: 0.61rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.31rem;
        }
        .hero-side-value {
            font-size: 1.00rem;
            font-weight: 820;
            color: #111827;
            line-height: 1.25;
            margin-bottom: 0.27rem;
        }
        .state-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            border: 1px solid rgba(100, 116, 139, 0.18);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 0.92rem;
        }
        .state-card {
            min-height: 118px;
            padding: 12px 13px;
            border-right: 1px solid rgba(100, 116, 139, 0.15);
            background: #ffffff;
        }
        .state-card:last-child { border-right: 0; }
        .state-label {
            font-size: 0.61rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.38rem;
        }
        .state-value {
            font-size: 1.03rem;
            font-weight: 850;
            line-height: 1.16;
            margin-bottom: 0.35rem;
        }
        .state-evidence {
            font-size: 0.70rem;
            color: #64748b;
            line-height: 1.36;
        }
        .section-title {
            font-size: 1.02rem;
            font-weight: 850;
            color: #111827;
            margin-top: 0.95rem;
            margin-bottom: 0.26rem;
            letter-spacing: -0.01em;
        }
        .section-subtitle {
            font-size: 0.79rem;
            color: #64748b;
            margin-bottom: 0.56rem;
            line-height: 1.42;
        }
        .transition-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-bottom: 0.75rem;
        }
        .transition-card {
            border: 1px solid #dfe3e8;
            border-radius: 9px;
            padding: 11px 13px;
            background: #ffffff;
            min-height: 102px;
        }
        .transition-date {
            color: #64748b;
            font-size: 0.66rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .transition-regime {
            color: #111827;
            font-size: 0.93rem;
            font-weight: 820;
            margin-bottom: 0.28rem;
        }
        .transition-detail {
            color: #64748b;
            font-size: 0.71rem;
            line-height: 1.36;
        }
        .change-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.64rem;
            margin-bottom: 0.86rem;
        }
        .change-card {
            border: 1px solid #dfe3e8;
            border-radius: 9px;
            padding: 10px 12px;
            background: #ffffff;
            min-height: 104px;
        }
        .change-name {
            font-size: 0.72rem;
            font-weight: 820;
            color: #334155;
            margin-bottom: 0.24rem;
        }
        .change-move {
            font-size: 1.00rem;
            font-weight: 850;
            line-height: 1.1;
            margin-bottom: 0.28rem;
        }
        .change-read {
            font-size: 0.70rem;
            line-height: 1.36;
            color: #64748b;
        }
        .tension-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.66rem;
            margin-bottom: 0.84rem;
        }
        .tension-card {
            border: 1px solid #dfe3e8;
            border-radius: 9px;
            padding: 11px 13px;
            background: #ffffff;
            min-height: 108px;
        }
        .tension-status {
            font-size: 0.61rem;
            font-weight: 850;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.26rem;
        }
        .tension-title {
            color: #111827;
            font-size: 0.88rem;
            font-weight: 820;
            margin-bottom: 0.28rem;
        }
        .tension-copy {
            color: #64748b;
            font-size: 0.72rem;
            line-height: 1.40;
        }
        .macro-table-wrap {
            border: 1px solid #dfe3e8;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 0.7rem;
        }
        table.macro-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            font-size: 0.74rem;
        }
        .macro-table thead th {
            text-align: right;
            padding: 9px 10px;
            color: #64748b;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid #dfe3e8;
            background: #fbfcfd;
            white-space: nowrap;
        }
        .macro-table thead th:first-child,
        .macro-table thead th:nth-child(2),
        .macro-table thead th:last-child {
            text-align: left;
        }
        .macro-table tbody td {
            padding: 8px 10px;
            border-bottom: 1px solid #edf0f3;
            color: #334155;
            text-align: right;
            vertical-align: middle;
            white-space: nowrap;
        }
        .macro-table tbody tr:last-child td { border-bottom: 0; }
        .macro-table tbody td:first-child,
        .macro-table tbody td:nth-child(2),
        .macro-table tbody td:last-child {
            text-align: left;
        }
        .macro-group {
            color: #64748b !important;
            font-size: 0.66rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .macro-name {
            font-weight: 780;
            color: #111827 !important;
        }
        .read-chip {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 999px;
            font-size: 0.65rem;
            font-weight: 760;
            white-space: nowrap;
        }
        .read-pos { background: #eef6e9; color: #4f733a; }
        .read-neg { background: #faeeee; color: #9d4c47; }
        .read-mix { background: #fff6dd; color: #876517; }
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
        }
        @media (max-width: 1250px) {
            .state-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .state-card:nth-child(3) { border-right: 0; }
            .state-card:nth-child(-n+3) { border-bottom: 1px solid rgba(100, 116, 139, 0.15); }
        }
        @media (max-width: 850px) {
            .regime-hero { grid-template-columns: 1fr; }
            .hero-side { border-left: 0; border-top: 1px solid rgba(100, 116, 139, 0.18); padding: 0.8rem 0 0; }
            .state-grid, .transition-grid, .change-grid, .tension-grid { grid-template-columns: 1fr; }
            .state-card { border-right: 0; border-bottom: 1px solid rgba(100, 116, 139, 0.15); }
            .state-card:last-child { border-bottom: 0; }
            .macro-table-wrap { overflow-x: auto; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def is_valid(value: object) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_index()


def value_asof(series: pd.Series | None, asof: pd.Timestamp) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    eligible = clean.loc[clean.index <= pd.Timestamp(asof)]
    return float(eligible.iloc[-1]) if not eligible.empty else np.nan


def pct_change_days(series: pd.Series | None, asof: pd.Timestamp, days: int) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    end = value_asof(clean, asof)
    start = value_asof(clean, pd.Timestamp(asof) - pd.Timedelta(days=days))
    if not is_valid(end) or not is_valid(start) or float(start) == 0:
        return np.nan
    return float((float(end) / float(start) - 1.0) * 100.0)


def abs_change_days(series: pd.Series | None, asof: pd.Timestamp, days: int) -> float:
    end = value_asof(series, asof)
    start = value_asof(series, pd.Timestamp(asof) - pd.Timedelta(days=days))
    if not is_valid(end) or not is_valid(start):
        return np.nan
    return float(end) - float(start)


def ytd_change_pct(series: pd.Series | None, asof: pd.Timestamp) -> float:
    clean = clean_series(series)
    if clean.empty:
        return np.nan
    end = value_asof(clean, asof)
    start = value_asof(clean, pd.Timestamp(year=pd.Timestamp(asof).year, month=1, day=1))
    if not is_valid(end) or not is_valid(start) or float(start) == 0:
        return np.nan
    return float((float(end) / float(start) - 1.0) * 100.0)


def safe_ratio(prices: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    if numerator not in prices.columns or denominator not in prices.columns:
        return pd.Series(dtype=float)
    denom = pd.to_numeric(prices[denominator], errors="coerce").replace(0, np.nan)
    numer = pd.to_numeric(prices[numerator], errors="coerce")
    return clean_series(numer / denom)


def fmt_pct(value: float | None) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):+.2f}%"


def fmt_bp(value: float | None) -> str:
    return "N/A" if not is_valid(value) else f"{float(value):+.0f} bp"


def fmt_bn(value: float | None) -> str:
    return "N/A" if not is_valid(value) else f"${float(value):+,.0f}bn"


def fmt_level(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if not is_valid(value):
        return "N/A"
    return f"{float(value):,.{digits}f}{suffix}"


def score_color(read: str) -> str:
    positive_words = {
        "Improving",
        "Falling",
        "Easing",
        "Broad risk-on",
        "Constructive",
        "Tighter",
        "Lower",
        "Weaker",
    }
    negative_words = {
        "Weakening",
        "Rising",
        "Tightening",
        "Risk-off",
        "Defensive",
        "Wider",
        "Higher",
        "Stronger",
    }
    if read in positive_words:
        return PALETTE["green"]
    if read in negative_words:
        return PALETTE["red"]
    return PALETTE["amber"]


def state_background(read: str) -> str:
    color = score_color(read)
    if color == PALETTE["green"]:
        return PALETTE["green_bg"]
    if color == PALETTE["red"]:
        return PALETTE["red_bg"]
    return PALETTE["amber_bg"]


def signal_vote(move: float, threshold: float, orientation: int = 1) -> int:
    if not is_valid(move):
        return 0
    adjusted = float(move) * int(orientation)
    if adjusted >= threshold:
        return 1
    if adjusted <= -threshold:
        return -1
    return 0


def classify_votes(
    indicators: List[dict],
    positive_label: str,
    negative_label: str,
    mixed_label: str = "Mixed",
) -> Tuple[str, str]:
    votes = [int(item.get("vote", 0)) for item in indicators if item.get("available", True)]
    positives = sum(v > 0 for v in votes)
    negatives = sum(v < 0 for v in votes)
    neutrals = sum(v == 0 for v in votes)
    if positives >= max(2, negatives + 1):
        state = positive_label
    elif negatives >= max(2, positives + 1):
        state = negative_label
    else:
        state = mixed_label
    detail = f"{positives} positive · {negatives} negative · {neutrals} neutral"
    return state, detail


def pct_indicator(
    name: str,
    series: pd.Series,
    asof: pd.Timestamp,
    threshold: float,
    orientation: int = 1,
    horizon_days: int = 30,
) -> dict:
    move = pct_change_days(series, asof, horizon_days)
    return {
        "name": name,
        "move": move,
        "move_text": fmt_pct(move),
        "vote": signal_vote(move, threshold, orientation),
        "available": is_valid(move),
    }


def bp_indicator(
    name: str,
    series: pd.Series,
    asof: pd.Timestamp,
    threshold: float,
    orientation: int = 1,
    horizon_days: int = 30,
) -> dict:
    move = abs_change_days(series, asof, horizon_days) * 100.0
    return {
        "name": name,
        "move": move,
        "move_text": fmt_bp(move),
        "vote": signal_vote(move, threshold, orientation),
        "available": is_valid(move),
    }


def bn_indicator(
    name: str,
    series: pd.Series,
    asof: pd.Timestamp,
    threshold: float,
    orientation: int = 1,
    horizon_days: int = 30,
) -> dict:
    move = abs_change_days(series, asof, horizon_days) / 1_000.0
    return {
        "name": name,
        "move": move,
        "move_text": fmt_bn(move),
        "vote": signal_vote(move, threshold, orientation),
        "available": is_valid(move),
    }


def evidence_text(indicators: List[dict], limit: int = 3) -> str:
    usable = [item for item in indicators if item.get("available", False)]
    if not usable:
        return "Underlying observations unavailable."
    parts = [
        f"{item['name']} {item['move_text']}"
        for item in usable[:limit]
    ]
    return " · ".join(parts)


# =============================================================================
# DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_prices(tickers: Tuple[str, ...]) -> Tuple[pd.DataFrame, List[str]]:
    frames, diagnostics = fetch_daily_ohlcv(tickers, period="5y")
    close = close_panel(frames, tickers, adjusted=True)
    if close.empty:
        return pd.DataFrame(), list(tickers)
    ordered = [ticker for ticker in tickers if ticker in close.columns]
    close = close.reindex(columns=ordered).dropna(axis=1, how="all").dropna(how="all")
    failed = diagnostics["Ticker"].astype(str).tolist() if not diagnostics.empty else []
    return close, failed


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_fred_series(start="2015-01-01")


def macro_series(panel: pd.DataFrame, key: str) -> pd.Series:
    if key not in panel.columns:
        return pd.Series(dtype=float)
    return clean_series(panel[key])


def build_net_liquidity(macro: pd.DataFrame) -> pd.Series:
    if not {"walcl", "tga", "rrp"}.issubset(macro.columns):
        return pd.Series(dtype=float)
    frame = macro[["walcl", "tga", "rrp"]].copy().sort_index().ffill()
    # WALCL / TGA are $mm. RRPONTSYD is $bn, so convert RRP to $mm.
    net = frame["walcl"] - frame["tga"] - frame["rrp"] * 1_000.0
    return clean_series(net.rename("net_liquidity"))


# =============================================================================
# REGIME ENGINE
# =============================================================================

def build_snapshot(prices: pd.DataFrame, macro: pd.DataFrame, asof: pd.Timestamp) -> dict:
    rsp_spy = safe_ratio(prices, "RSP", "SPY")
    iwm_spy = safe_ratio(prices, "IWM", "SPY")
    eem_spy = safe_ratio(prices, "EEM", "SPY")
    hyg_lqd = safe_ratio(prices, "HYG", "LQD")
    copper_gold = safe_ratio(prices, "HG=F", "GC=F")
    tip_ief = safe_ratio(prices, "TIP", "IEF")

    dollar_series = (
        clean_series(prices["DX-Y.NYB"])
        if "DX-Y.NYB" in prices and not clean_series(prices["DX-Y.NYB"]).empty
        else clean_series(prices["UUP"]) if "UUP" in prices else pd.Series(dtype=float)
    )

    growth_indicators = [
        pct_indicator("Breadth · RSP/SPY", rsp_spy, asof, threshold=0.50),
        pct_indicator("Small caps · IWM/SPY", iwm_spy, asof, threshold=1.00),
        pct_indicator("Copper/Gold", copper_gold, asof, threshold=2.00),
        pct_indicator("EM vs U.S. · EEM/SPY", eem_spy, asof, threshold=1.00),
        bp_indicator(
            "High Yield OAS",
            macro_series(macro, "hy_oas"),
            asof,
            threshold=10.0,
            orientation=-1,
        ),
    ]
    growth_state, growth_detail = classify_votes(
        growth_indicators, "Improving", "Weakening"
    )

    inflation_indicators = [
        bp_indicator(
            "10Y Breakeven",
            macro_series(macro, "t10yie"),
            asof,
            threshold=5.0,
            orientation=1,
        ),
        pct_indicator(
            "WTI Crude",
            clean_series(prices["CL=F"]) if "CL=F" in prices else pd.Series(dtype=float),
            asof,
            threshold=5.0,
            orientation=1,
        ),
        pct_indicator("Copper/Gold", copper_gold, asof, threshold=2.00, orientation=1),
        pct_indicator("TIPS vs Treasuries · TIP/IEF", tip_ief, asof, threshold=0.75),
    ]
    inflation_state, inflation_detail = classify_votes(
        inflation_indicators, "Rising", "Falling"
    )

    rates_indicators = [
        bp_indicator("U.S. 2Y", macro_series(macro, "dgs2"), asof, threshold=10.0),
        bp_indicator("U.S. 10Y", macro_series(macro, "dgs10"), asof, threshold=10.0),
        bp_indicator("U.S. 30Y", macro_series(macro, "dgs30"), asof, threshold=10.0),
        bp_indicator(
            "10Y Real Yield",
            macro_series(macro, "dfii10"),
            asof,
            threshold=10.0,
        ),
    ]
    rates_state, rates_detail = classify_votes(
        rates_indicators, "Rising", "Falling"
    )

    net_liquidity = build_net_liquidity(macro)
    liquidity_indicators = [
        pct_indicator("U.S. Dollar", dollar_series, asof, threshold=1.00, orientation=-1),
        bp_indicator(
            "10Y Real Yield",
            macro_series(macro, "dfii10"),
            asof,
            threshold=10.0,
            orientation=-1,
        ),
        bp_indicator(
            "High Yield OAS",
            macro_series(macro, "hy_oas"),
            asof,
            threshold=10.0,
            orientation=-1,
        ),
        bn_indicator(
            "Fed - TGA - RRP",
            net_liquidity,
            asof,
            threshold=100.0,
            orientation=1,
        ),
        pct_indicator(
            "EUR/USD",
            clean_series(prices["EURUSD=X"]) if "EURUSD=X" in prices else pd.Series(dtype=float),
            asof,
            threshold=1.00,
            orientation=1,
        ),
    ]
    liquidity_state, liquidity_detail = classify_votes(
        liquidity_indicators, "Easing", "Tightening"
    )

    risk_indicators = [
        pct_indicator(
            "S&P 500",
            clean_series(prices["SPY"]) if "SPY" in prices else pd.Series(dtype=float),
            asof,
            threshold=2.00,
        ),
        pct_indicator("Breadth · RSP/SPY", rsp_spy, asof, threshold=0.50),
        pct_indicator("Credit · HYG/LQD", hyg_lqd, asof, threshold=0.50),
        pct_indicator(
            "VIX",
            clean_series(prices["^VIX"]) if "^VIX" in prices else pd.Series(dtype=float),
            asof,
            threshold=10.0,
            orientation=-1,
        ),
        pct_indicator("EM vs U.S. · EEM/SPY", eem_spy, asof, threshold=1.00),
    ]
    base_risk_state, risk_detail = classify_votes(
        risk_indicators, "Constructive", "Defensive"
    )

    spy_1m = pct_change_days(prices["SPY"] if "SPY" in prices else None, asof, 30)
    breadth_1m = pct_change_days(rsp_spy, asof, 30)
    credit_1m = pct_change_days(hyg_lqd, asof, 30)
    vix_1m = pct_change_days(prices["^VIX"] if "^VIX" in prices else None, asof, 30)

    if (
        is_valid(spy_1m)
        and spy_1m > 2.0
        and is_valid(breadth_1m)
        and breadth_1m > 0.5
        and is_valid(credit_1m)
        and credit_1m > 0.5
    ):
        risk_state = "Broad risk-on"
    elif (
        is_valid(spy_1m)
        and spy_1m > 1.0
        and (
            (is_valid(breadth_1m) and breadth_1m < -0.5)
            or (is_valid(credit_1m) and credit_1m < -0.5)
        )
    ):
        risk_state = "Narrow risk-on"
    elif (
        (is_valid(spy_1m) and spy_1m < -2.0)
        and (
            (is_valid(credit_1m) and credit_1m < -0.5)
            or (is_valid(vix_1m) and vix_1m > 10.0)
        )
    ):
        risk_state = "Risk-off"
    else:
        risk_state = base_risk_state if base_risk_state != "Constructive" else "Mixed / constructive"

    regime = classify_macro_quadrant(growth_state, inflation_state)
    narrative = regime_narrative(
        regime, growth_state, inflation_state, rates_state, liquidity_state, risk_state
    )

    return {
        "asof": pd.Timestamp(asof),
        "regime": regime,
        "narrative": narrative,
        "growth_state": growth_state,
        "growth_detail": growth_detail,
        "growth_indicators": growth_indicators,
        "inflation_state": inflation_state,
        "inflation_detail": inflation_detail,
        "inflation_indicators": inflation_indicators,
        "rates_state": rates_state,
        "rates_detail": rates_detail,
        "rates_indicators": rates_indicators,
        "liquidity_state": liquidity_state,
        "liquidity_detail": liquidity_detail,
        "liquidity_indicators": liquidity_indicators,
        "risk_state": risk_state,
        "risk_detail": risk_detail,
        "risk_indicators": risk_indicators,
        "ratios": {
            "rsp_spy": rsp_spy,
            "iwm_spy": iwm_spy,
            "eem_spy": eem_spy,
            "hyg_lqd": hyg_lqd,
            "copper_gold": copper_gold,
            "tip_ief": tip_ief,
        },
        "dollar_series": dollar_series,
        "net_liquidity": net_liquidity,
    }


def classify_macro_quadrant(growth_state: str, inflation_state: str) -> str:
    if growth_state == "Improving" and inflation_state == "Falling":
        return "Goldilocks / disinflationary growth"
    if growth_state == "Improving" and inflation_state == "Rising":
        return "Reflation"
    if growth_state == "Weakening" and inflation_state == "Rising":
        return "Stagflation pressure"
    if growth_state == "Weakening" and inflation_state == "Falling":
        return "Growth scare / disinflation"
    return "Transition / mixed"


def regime_narrative(
    regime: str,
    growth_state: str,
    inflation_state: str,
    rates_state: str,
    liquidity_state: str,
    risk_state: str,
) -> str:
    base = {
        "Goldilocks / disinflationary growth":
            "Growth-sensitive markets are improving while inflation pressure is fading. This is the cleanest backdrop for duration-sensitive risk when credit and breadth confirm.",
        "Reflation":
            "Growth and inflation proxies are rising together. Cyclical risk can work, but the long end and real yields become the key constraint on valuation.",
        "Stagflation pressure":
            "Growth-sensitive markets are weakening while inflation pressure remains firm. This is the most difficult mix for duration and weak-balance-sheet risk.",
        "Growth scare / disinflation":
            "Growth-sensitive markets and inflation pressure are falling together. Duration should normally begin to confirm; if yields rise anyway, term premium or fiscal pressure is the dominant contradiction.",
        "Transition / mixed":
            "Growth and inflation signals do not yet define a clean quadrant. The useful information is in the cross-asset divergences, not a forced headline label.",
    }[regime]
    return (
        f"{base} Rates are {rates_state.lower()}, liquidity is {liquidity_state.lower()}, "
        f"and risk confirmation is {risk_state.lower()}."
    )


def governing_tension(snapshot: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> Tuple[str, str]:
    asof = snapshot["asof"]
    ten_y = abs_change_days(macro_series(macro, "dgs10"), asof, 30) * 100
    real_y = abs_change_days(macro_series(macro, "dfii10"), asof, 30) * 100
    breakeven = abs_change_days(macro_series(macro, "t10yie"), asof, 30) * 100
    spy = pct_change_days(prices["SPY"] if "SPY" in prices else None, asof, 30)
    credit = pct_change_days(snapshot["ratios"]["hyg_lqd"], asof, 30)
    breadth = pct_change_days(snapshot["ratios"]["rsp_spy"], asof, 30)
    dollar = pct_change_days(snapshot["dollar_series"], asof, 30)

    if snapshot["growth_state"] == "Weakening" and is_valid(ten_y) and ten_y > 10:
        return (
            "Growth is weakening but the long end is still selling off",
            f"10Y yields are {fmt_bp(ten_y)} over one month. The market is pricing a term-premium, fiscal or supply problem that is overpowering the softer growth tape.",
        )
    if snapshot["inflation_state"] == "Falling" and is_valid(ten_y) and ten_y > 10:
        return (
            "Disinflation is not translating into lower nominal yields",
            f"10Y breakevens are {fmt_bp(breakeven)} over one month while the 10Y nominal yield is {fmt_bp(ten_y)}. Real rates and term premium are doing the tightening.",
        )
    if is_valid(spy) and spy > 1.0 and is_valid(credit) and credit < -0.5:
        return (
            "Equities are running ahead of credit",
            f"SPY is {fmt_pct(spy)} over one month while HYG/LQD is {fmt_pct(credit)}. Equity beta is not receiving clean credit sponsorship.",
        )
    if is_valid(spy) and spy > 1.0 and is_valid(breadth) and breadth < -0.5:
        return (
            "Index strength is narrow",
            f"SPY is {fmt_pct(spy)} over one month while RSP/SPY is {fmt_pct(breadth)}. Cap-weight leadership is masking weaker participation.",
        )
    if is_valid(dollar) and dollar < -1.0 and is_valid(real_y) and real_y > 10:
        return (
            "Dollar liquidity is easing while real-rate pressure rises",
            f"The dollar is {fmt_pct(dollar)} over one month but 10Y real yields are {fmt_bp(real_y)}. Global liquidity relief is being offset by a higher discount rate.",
        )
    return (
        "No single cross-asset contradiction dominates",
        "The major growth, inflation, rates, liquidity and risk signals are either aligned or too mixed to elevate one tension above the rest.",
    )


# =============================================================================
# TABLES / READS
# =============================================================================

def directional_read(value: float, threshold: float, pos: str, neg: str, neutral: str = "Flat") -> str:
    if not is_valid(value):
        return "N/A"
    if float(value) >= threshold:
        return pos
    if float(value) <= -threshold:
        return neg
    return neutral


def read_class(read: str) -> str:
    positive = {
        "Broadening", "Improving", "Tighter", "Falling", "Weaker dollar",
        "Lower vol", "Risk-on", "Easing", "Outperforming",
    }
    negative = {
        "Narrowing", "Weakening", "Wider", "Rising", "Stronger dollar",
        "Higher vol", "Risk-off", "Tightening", "Underperforming",
    }
    if read in positive:
        return "read-pos"
    if read in negative:
        return "read-neg"
    return "read-mix"


def build_macro_tape(prices: pd.DataFrame, macro: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    asof = snapshot["asof"]
    rows: List[dict] = []

    def add_pct(group: str, name: str, series: pd.Series, threshold: float, pos: str, neg: str, level: str = "") -> None:
        one_w = pct_change_days(series, asof, 7)
        one_m = pct_change_days(series, asof, 30)
        three_m = pct_change_days(series, asof, 91)
        rows.append(
            {
                "Group": group,
                "Signal": name,
                "Level": level or fmt_level(value_asof(series, asof), 2),
                "1W": fmt_pct(one_w),
                "1M": fmt_pct(one_m),
                "3M": fmt_pct(three_m),
                "Read": directional_read(one_m, threshold, pos, neg, "Mixed"),
            }
        )

    def add_bp(group: str, name: str, series: pd.Series, threshold: float, pos: str, neg: str) -> None:
        one_w = abs_change_days(series, asof, 7) * 100
        one_m = abs_change_days(series, asof, 30) * 100
        three_m = abs_change_days(series, asof, 91) * 100
        rows.append(
            {
                "Group": group,
                "Signal": name,
                "Level": fmt_level(value_asof(series, asof), 2, "%"),
                "1W": fmt_bp(one_w),
                "1M": fmt_bp(one_m),
                "3M": fmt_bp(three_m),
                "Read": directional_read(one_m, threshold, pos, neg, "Mixed"),
            }
        )

    add_pct("Risk", "S&P 500", clean_series(prices["SPY"]) if "SPY" in prices else pd.Series(dtype=float), 2.0, "Risk-on", "Risk-off")
    add_pct("Risk", "Equal Weight / S&P", snapshot["ratios"]["rsp_spy"], 0.50, "Broadening", "Narrowing")
    add_pct("Risk", "Small Caps / S&P", snapshot["ratios"]["iwm_spy"], 1.00, "Improving", "Weakening")
    add_pct("Credit", "HYG / LQD", snapshot["ratios"]["hyg_lqd"], 0.50, "Improving", "Weakening")
    add_bp("Credit", "High Yield OAS", macro_series(macro, "hy_oas"), 10.0, "Wider", "Tighter")

    add_bp("Rates", "U.S. 2Y", macro_series(macro, "dgs2"), 10.0, "Rising", "Falling")
    add_bp("Rates", "U.S. 10Y", macro_series(macro, "dgs10"), 10.0, "Rising", "Falling")
    add_bp("Rates", "U.S. 30Y", macro_series(macro, "dgs30"), 10.0, "Rising", "Falling")
    add_bp("Rates", "10Y Real Yield", macro_series(macro, "dfii10"), 10.0, "Rising", "Falling")
    add_bp("Inflation", "10Y Breakeven", macro_series(macro, "t10yie"), 5.0, "Rising", "Falling")

    add_pct("FX / Liquidity", "U.S. Dollar", snapshot["dollar_series"], 1.00, "Stronger dollar", "Weaker dollar")
    add_pct("FX / Liquidity", "EUR/USD", clean_series(prices["EURUSD=X"]) if "EURUSD=X" in prices else pd.Series(dtype=float), 1.00, "Improving", "Weakening")
    add_pct("FX / Liquidity", "USD/JPY", clean_series(prices["JPY=X"]) if "JPY=X" in prices else pd.Series(dtype=float), 1.00, "Higher", "Lower")

    add_pct("Commodities", "WTI Crude", clean_series(prices["CL=F"]) if "CL=F" in prices else pd.Series(dtype=float), 5.00, "Rising", "Falling")
    add_pct("Commodities", "Gold", clean_series(prices["GC=F"]) if "GC=F" in prices else pd.Series(dtype=float), 3.00, "Rising", "Falling")
    add_pct("Growth", "Copper / Gold", snapshot["ratios"]["copper_gold"], 2.00, "Improving", "Weakening")
    add_pct("Global Growth", "EM / U.S.", snapshot["ratios"]["eem_spy"], 1.00, "Outperforming", "Underperforming")

    vix = clean_series(prices["^VIX"]) if "^VIX" in prices else pd.Series(dtype=float)
    add_pct("Volatility", "VIX", vix, 10.00, "Higher vol", "Lower vol")

    net = snapshot["net_liquidity"]
    one_w = abs_change_days(net, asof, 7) / 1_000.0
    one_m = abs_change_days(net, asof, 30) / 1_000.0
    three_m = abs_change_days(net, asof, 91) / 1_000.0
    rows.append(
        {
            "Group": "Liquidity",
            "Signal": "Fed - TGA - RRP",
            "Level": (
                f"${value_asof(net, asof) / 1_000.0:,.0f}bn"
                if is_valid(value_asof(net, asof))
                else "N/A"
            ),
            "1W": fmt_bn(one_w),
            "1M": fmt_bn(one_m),
            "3M": fmt_bn(three_m),
            "Read": directional_read(one_m, 100.0, "Easing", "Tightening", "Mixed"),
        }
    )

    return pd.DataFrame(rows)


def render_macro_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No macro tape data available.")
        return
    body: List[str] = []
    for _, row in frame.iterrows():
        read = str(row["Read"])
        body.append(
            "<tr>"
            f"<td class='macro-group'>{escape(str(row['Group']))}</td>"
            f"<td class='macro-name'>{escape(str(row['Signal']))}</td>"
            f"<td>{escape(str(row['Level']))}</td>"
            f"<td>{escape(str(row['1W']))}</td>"
            f"<td>{escape(str(row['1M']))}</td>"
            f"<td>{escape(str(row['3M']))}</td>"
            f"<td><span class='read-chip {read_class(read)}'>{escape(read)}</span></td>"
            "</tr>"
        )
    st.markdown(
        """
        <div class="macro-table-wrap">
            <table class="macro-table">
                <thead>
                    <tr>
                        <th>Theme</th>
                        <th>Signal</th>
                        <th>Level</th>
                        <th>1W</th>
                        <th>1M</th>
                        <th>3M</th>
                        <th>Read</th>
                    </tr>
                </thead>
                <tbody>
        """
        + "".join(body)
        + """
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_weekly_changes(prices: pd.DataFrame, macro: pd.DataFrame, snapshot: dict) -> List[dict]:
    asof = snapshot["asof"]
    candidates: List[dict] = []

    def add(name: str, move: float, threshold: float, formatter, read: str, color_hint: str) -> None:
        if not is_valid(move) or threshold <= 0:
            return
        candidates.append(
            {
                "name": name,
                "move": float(move),
                "move_text": formatter(move),
                "importance": abs(float(move)) / float(threshold),
                "read": read,
                "color": color_hint,
            }
        )

    ten_y = abs_change_days(macro_series(macro, "dgs10"), asof, 7) * 100
    two_y = abs_change_days(macro_series(macro, "dgs2"), asof, 7) * 100
    real_y = abs_change_days(macro_series(macro, "dfii10"), asof, 7) * 100
    breakeven = abs_change_days(macro_series(macro, "t10yie"), asof, 7) * 100
    hy_oas = abs_change_days(macro_series(macro, "hy_oas"), asof, 7) * 100
    dollar = pct_change_days(snapshot["dollar_series"], asof, 7)
    breadth = pct_change_days(snapshot["ratios"]["rsp_spy"], asof, 7)
    small_caps = pct_change_days(snapshot["ratios"]["iwm_spy"], asof, 7)
    copper_gold = pct_change_days(snapshot["ratios"]["copper_gold"], asof, 7)
    crude = pct_change_days(prices["CL=F"] if "CL=F" in prices else None, asof, 7)
    vix = pct_change_days(prices["^VIX"] if "^VIX" in prices else None, asof, 7)
    net_liq = abs_change_days(snapshot["net_liquidity"], asof, 7) / 1_000.0

    add("U.S. 10Y yield", ten_y, 8.0, fmt_bp, "Long-end tightening" if ten_y > 0 else "Long-end relief", PALETTE["red"] if ten_y > 0 else PALETTE["green"])
    add("U.S. 2Y yield", two_y, 8.0, fmt_bp, "Policy path repricing higher" if two_y > 0 else "Policy path repricing lower", PALETTE["red"] if two_y > 0 else PALETTE["green"])
    add("10Y real yield", real_y, 8.0, fmt_bp, "Discount-rate tightening" if real_y > 0 else "Real-rate relief", PALETTE["red"] if real_y > 0 else PALETTE["green"])
    add("10Y breakeven", breakeven, 5.0, fmt_bp, "Inflation compensation higher" if breakeven > 0 else "Inflation compensation lower", PALETTE["red"] if breakeven > 0 else PALETTE["green"])
    add("High Yield OAS", hy_oas, 10.0, fmt_bp, "Credit spreads wider" if hy_oas > 0 else "Credit spreads tighter", PALETTE["red"] if hy_oas > 0 else PALETTE["green"])
    add("U.S. dollar", dollar, 1.0, fmt_pct, "Dollar tightening" if dollar > 0 else "Dollar liquidity relief", PALETTE["red"] if dollar > 0 else PALETTE["green"])
    add("Breadth · RSP/SPY", breadth, 1.0, fmt_pct, "Participation broadening" if breadth > 0 else "Participation narrowing", PALETTE["green"] if breadth > 0 else PALETTE["red"])
    add("Small caps · IWM/SPY", small_caps, 1.25, fmt_pct, "Domestic cyclicals improving" if small_caps > 0 else "Domestic cyclicals weakening", PALETTE["green"] if small_caps > 0 else PALETTE["red"])
    add("Copper / Gold", copper_gold, 2.5, fmt_pct, "Growth/reflation signal improving" if copper_gold > 0 else "Growth/reflation signal weakening", PALETTE["green"] if copper_gold > 0 else PALETTE["red"])
    add("WTI crude", crude, 4.0, fmt_pct, "Energy inflation higher" if crude > 0 else "Energy inflation relief", PALETTE["red"] if crude > 0 else PALETTE["green"])
    add("VIX", vix, 12.0, fmt_pct, "Volatility pressure higher" if vix > 0 else "Volatility pressure lower", PALETTE["red"] if vix > 0 else PALETTE["green"])
    add("Fed - TGA - RRP", net_liq, 75.0, fmt_bn, "Net liquidity added" if net_liq > 0 else "Net liquidity drained", PALETTE["green"] if net_liq > 0 else PALETTE["red"])

    candidates = sorted(candidates, key=lambda item: item["importance"], reverse=True)
    meaningful = [item for item in candidates if item["importance"] >= 0.75]
    return (meaningful or candidates)[:6]


def build_tensions(prices: pd.DataFrame, macro: pd.DataFrame, snapshot: dict) -> List[dict]:
    asof = snapshot["asof"]
    items: List[dict] = []

    ten_y = abs_change_days(macro_series(macro, "dgs10"), asof, 30) * 100
    real_y = abs_change_days(macro_series(macro, "dfii10"), asof, 30) * 100
    breakeven = abs_change_days(macro_series(macro, "t10yie"), asof, 30) * 100
    spy = pct_change_days(prices["SPY"] if "SPY" in prices else None, asof, 30)
    breadth = pct_change_days(snapshot["ratios"]["rsp_spy"], asof, 30)
    credit = pct_change_days(snapshot["ratios"]["hyg_lqd"], asof, 30)
    dollar = pct_change_days(snapshot["dollar_series"], asof, 30)

    if snapshot["growth_state"] == "Weakening" and is_valid(ten_y) and ten_y > 10:
        items.append(
            {
                "status": "Divergence",
                "title": "Growth vs long-end yields",
                "copy": f"Growth-sensitive signals are weakening, but the 10Y yield is {fmt_bp(ten_y)} over one month. Term premium, fiscal supply or inflation risk is overpowering the growth slowdown.",
                "color": PALETTE["red"],
            }
        )
    else:
        items.append(
            {
                "status": "Aligned",
                "title": "Growth vs long-end yields",
                "copy": f"Growth is {snapshot['growth_state'].lower()} and the 10Y yield is {fmt_bp(ten_y)} over one month. Rates are broadly behaving consistently with the growth impulse.",
                "color": PALETTE["green"],
            }
        )

    if snapshot["inflation_state"] == "Falling" and is_valid(ten_y) and ten_y > 10:
        items.append(
            {
                "status": "Divergence",
                "title": "Inflation vs nominal yields",
                "copy": f"Inflation pressure is falling and 10Y breakevens are {fmt_bp(breakeven)}, yet nominal 10Y yields are {fmt_bp(ten_y)}. Real yields or term premium are driving the selloff.",
                "color": PALETTE["red"],
            }
        )
    else:
        items.append(
            {
                "status": "Check",
                "title": "Inflation vs nominal yields",
                "copy": f"10Y breakevens are {fmt_bp(breakeven)} and 10Y nominal yields are {fmt_bp(ten_y)} over one month. The split tells you whether inflation compensation or real rates are doing the work.",
                "color": PALETTE["amber"],
            }
        )

    if is_valid(spy) and spy > 1.0 and is_valid(credit) and credit < -0.5:
        items.append(
            {
                "status": "Divergence",
                "title": "Equities vs credit",
                "copy": f"SPY is {fmt_pct(spy)} while HYG/LQD is {fmt_pct(credit)} over one month. Equity beta is running without clean credit sponsorship.",
                "color": PALETTE["red"],
            }
        )
    else:
        items.append(
            {
                "status": "Aligned",
                "title": "Equities vs credit",
                "copy": f"SPY is {fmt_pct(spy)} and HYG/LQD is {fmt_pct(credit)} over one month. Credit is not materially contradicting the equity tape.",
                "color": PALETTE["green"],
            }
        )

    if is_valid(spy) and spy > 1.0 and is_valid(breadth) and breadth < -0.5:
        items.append(
            {
                "status": "Divergence",
                "title": "Index level vs breadth",
                "copy": f"SPY is {fmt_pct(spy)} but RSP/SPY is {fmt_pct(breadth)} over one month. The headline index is stronger than participation underneath it.",
                "color": PALETTE["red"],
            }
        )
    else:
        items.append(
            {
                "status": "Aligned",
                "title": "Index level vs breadth",
                "copy": f"SPY is {fmt_pct(spy)} and RSP/SPY is {fmt_pct(breadth)} over one month. Breadth is not materially diverging from the index.",
                "color": PALETTE["green"],
            }
        )

    if is_valid(dollar) and dollar < -1.0 and is_valid(real_y) and real_y > 10:
        items.append(
            {
                "status": "Offsetting",
                "title": "Dollar vs real yields",
                "copy": f"The dollar is {fmt_pct(dollar)} while 10Y real yields are {fmt_bp(real_y)}. Dollar liquidity is easing, but the discount rate is tightening.",
                "color": PALETTE["amber"],
            }
        )
    else:
        items.append(
            {
                "status": "Check",
                "title": "Dollar vs real yields",
                "copy": f"The dollar is {fmt_pct(dollar)} and 10Y real yields are {fmt_bp(real_y)} over one month. Together they define the external-liquidity and discount-rate mix.",
                "color": PALETTE["amber"],
            }
        )

    # Put the actual divergences first.
    order = {"Divergence": 0, "Offsetting": 1, "Check": 2, "Aligned": 3}
    return sorted(items, key=lambda item: order.get(item["status"], 4))[:4]


# =============================================================================
# PERFORMANCE
# =============================================================================

def performance_frame(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    rows: List[dict] = []
    for ticker in PERFORMANCE_TICKERS:
        if ticker not in prices.columns or clean_series(prices[ticker]).empty:
            continue
        series = clean_series(prices[ticker])
        rows.append(
            {
                "Ticker": ticker,
                "Asset": TICKERS.get(ticker, ticker),
                "1W": pct_change_days(series, asof, 7),
                "1M": pct_change_days(series, asof, 30),
                "3M": pct_change_days(series, asof, 91),
                "YTD": ytd_change_pct(series, asof),
            }
        )
    return pd.DataFrame(rows)


def performance_heatmap(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return go.Figure()

    cols = ["1W", "1M", "3M", "YTD"]
    raw = frame[cols].to_numpy(dtype=float)
    normalized = np.full_like(raw, np.nan, dtype=float)

    for j, col in enumerate(cols):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            continue
        lo = np.nanpercentile(finite, 10)
        hi = np.nanpercentile(finite, 90)
        scale = max(abs(lo), abs(hi), 1.0)
        normalized[:, j] = np.clip(values / scale, -1.0, 1.0)

    text = np.empty(raw.shape, dtype=object)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            text[i, j] = fmt_pct(raw[i, j]) if np.isfinite(raw[i, j]) else "N/A"

    labels = frame["Ticker"] + "  " + frame["Asset"]

    fig = go.Figure(
        go.Heatmap(
            z=normalized,
            x=cols,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.00, "#a85a50"],
                [0.38, "#dcc2bb"],
                [0.50, "#f7f6f3"],
                [0.62, "#c6d7bd"],
                [1.00, "#668a50"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        height=max(480, 26 * len(frame) + 75),
        margin=dict(l=10, r=18, t=18, b=20),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#334155", size=11),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This page is deliberately rule-based and transparent. It does **not** compress the market into a proprietary -1 to +1 score.

        The regime is built from five questions:

        **Growth:** Are breadth, small caps, copper/gold, EM and credit improving or weakening?

        **Inflation:** Are breakevens, crude and inflation-sensitive market ratios rising or falling?

        **Rates:** Are nominal and real yields moving higher or lower?

        **Liquidity:** Are the dollar, real yields, credit spreads and Fed/TGA/RRP liquidity easing or tightening?

        **Risk:** Are equities being confirmed by breadth, credit and volatility?

        Each call uses simple one-month thresholds. The underlying moves are shown directly on the page.
        """
    )


# =============================================================================
# LOAD
# =============================================================================

render_page_header(
    PageHeader(
        title=TITLE,
        description=SUBTITLE,
        eyebrow="ADFM Cross-Asset Regimes",
    )
)

prices, market_failed = fetch_market_prices(tuple(TICKERS.keys()))
macro, macro_diag = fetch_macro_data()

if prices.empty:
    st.error("No market data loaded. Check market-data connectivity.")
    st.stop()

latest_market_dt = max(
    [
        pd.Timestamp(clean_series(prices[col]).index[-1])
        for col in prices.columns
        if not clean_series(prices[col]).empty
    ],
    default=pd.Timestamp.today().normalize(),
)
asof = pd.Timestamp(latest_market_dt).tz_localize(None) if getattr(latest_market_dt, "tzinfo", None) else pd.Timestamp(latest_market_dt)

snapshot = build_snapshot(prices, macro, asof)
tension_title, tension_copy = governing_tension(snapshot, prices, macro)

loaded_market = sum(not clean_series(prices[col]).empty for col in prices.columns)
macro_ok = 0
if not macro_diag.empty and "status" in macro_diag.columns:
    macro_ok = int((macro_diag["status"] == "OK").sum())
macro_total = int(len(macro_diag)) if not macro_diag.empty else 0

failed_text = ", ".join(market_failed[:6]) + ("..." if len(market_failed) > 6 else "") if market_failed else "None"

st.markdown(
    f"""
    <div class="data-status">
        Latest market observation {escape(asof.strftime("%b %d, %Y"))}
        &middot; market proxies {loaded_market}/{len(TICKERS)}
        &middot; primary macro series {macro_ok}/{macro_total}
        &middot; unavailable market data: {escape(failed_text)}
    </div>
    """,
    unsafe_allow_html=True,
)

hero_color = {
    "Goldilocks / disinflationary growth": PALETTE["green"],
    "Reflation": PALETTE["amber"],
    "Stagflation pressure": PALETTE["red"],
    "Growth scare / disinflation": PALETTE["blue"],
    "Transition / mixed": PALETTE["amber"],
}.get(snapshot["regime"], PALETTE["blue"])

st.markdown(
    f"""
    <div class="regime-hero" style="--hero-accent:{hero_color}">
        <div>
            <div class="hero-kicker">Current macro regime</div>
            <div class="hero-title">{escape(snapshot["regime"])}</div>
            <div class="hero-copy">{escape(snapshot["narrative"])}</div>
        </div>
        <div class="hero-side">
            <div class="hero-side-label">Governing tension</div>
            <div class="hero-side-value">{escape(tension_title)}</div>
            <div class="hero-copy">{escape(tension_copy)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

state_cards = [
    (
        "Growth impulse",
        snapshot["growth_state"],
        f"{snapshot['growth_detail']}. {evidence_text(snapshot['growth_indicators'])}",
    ),
    (
        "Inflation impulse",
        snapshot["inflation_state"],
        f"{snapshot['inflation_detail']}. {evidence_text(snapshot['inflation_indicators'])}",
    ),
    (
        "Rates impulse",
        snapshot["rates_state"],
        f"{snapshot['rates_detail']}. {evidence_text(snapshot['rates_indicators'])}",
    ),
    (
        "Liquidity / FCI",
        snapshot["liquidity_state"],
        f"{snapshot['liquidity_detail']}. {evidence_text(snapshot['liquidity_indicators'])}",
    ),
    (
        "Risk confirmation",
        snapshot["risk_state"],
        f"{snapshot['risk_detail']}. {evidence_text(snapshot['risk_indicators'])}",
    ),
]

state_html: List[str] = []
for label, value, detail in state_cards:
    color = score_color(value)
    state_html.append(
        "<div class='state-card'>"
        f"<div class='state-label'>{escape(label)}</div>"
        f"<div class='state-value' style='color:{color}'>{escape(value)}</div>"
        f"<div class='state-evidence'>{escape(detail)}</div>"
        "</div>"
    )
st.markdown(
    "<div class='state-grid'>" + "".join(state_html) + "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# REGIME TRANSITION
# =============================================================================

st.markdown("<div class='section-title'>Regime Transition</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>The same transparent framework evaluated at prior month-end-like lookbacks. This shows whether the macro mix is stable or actually changing.</div>",
    unsafe_allow_html=True,
)

transition_specs = [
    ("3 months ago", asof - pd.Timedelta(days=91)),
    ("1 month ago", asof - pd.Timedelta(days=30)),
    ("Current", asof),
]
transition_html: List[str] = []
for label, snap_date in transition_specs:
    snap = build_snapshot(prices, macro, snap_date)
    transition_html.append(
        "<div class='transition-card'>"
        f"<div class='transition-date'>{escape(label)} · {escape(pd.Timestamp(snap_date).strftime('%b %d, %Y'))}</div>"
        f"<div class='transition-regime'>{escape(snap['regime'])}</div>"
        f"<div class='transition-detail'>Growth {escape(snap['growth_state'].lower())} · inflation {escape(snap['inflation_state'].lower())} · rates {escape(snap['rates_state'].lower())} · liquidity {escape(snap['liquidity_state'].lower())}</div>"
        "</div>"
    )
st.markdown(
    "<div class='transition-grid'>" + "".join(transition_html) + "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# WHAT CHANGED
# =============================================================================

st.markdown("<div class='section-title'>What Changed This Week</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Only concrete market moves are ranked here. The ranking compares each move with a practical threshold for that instrument; the displayed number is always the actual move.</div>",
    unsafe_allow_html=True,
)

weekly = build_weekly_changes(prices, macro, snapshot)
weekly_html: List[str] = []
for item in weekly:
    weekly_html.append(
        "<div class='change-card'>"
        f"<div class='change-name'>{escape(item['name'])}</div>"
        f"<div class='change-move' style='color:{item['color']}'>{escape(item['move_text'])}</div>"
        f"<div class='change-read'>{escape(item['read'])}</div>"
        "</div>"
    )
st.markdown(
    "<div class='change-grid'>" + "".join(weekly_html) + "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# MACRO TAPE
# =============================================================================

st.markdown("<div class='section-title'>Cross-Asset Macro Tape</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>The core signals behind the regime call. Returns are actual price moves; rates, breakevens and spreads are basis-point changes. No percentile or composite scoring.</div>",
    unsafe_allow_html=True,
)
render_macro_table(build_macro_tape(prices, macro, snapshot))


# =============================================================================
# TENSIONS
# =============================================================================

st.markdown("<div class='section-title'>Cross-Asset Tensions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>The places where one market is refusing to confirm another. These are usually more important than the headline regime label.</div>",
    unsafe_allow_html=True,
)

tensions = build_tensions(prices, macro, snapshot)
tension_html: List[str] = []
for item in tensions:
    tension_html.append(
        "<div class='tension-card'>"
        f"<div class='tension-status' style='color:{item['color']}'>{escape(item['status'])}</div>"
        f"<div class='tension-title'>{escape(item['title'])}</div>"
        f"<div class='tension-copy'>{escape(item['copy'])}</div>"
        "</div>"
    )
st.markdown(
    "<div class='tension-grid'>" + "".join(tension_html) + "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# CROSS-ASSET PERFORMANCE
# =============================================================================

st.markdown("<div class='section-title'>Cross-Asset Performance</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Actual returns across major equity, credit, rates, dollar and commodity proxies. Cell color is normalized within each horizon so one oil or volatility outlier does not wash out the rest of the tape.</div>",
    unsafe_allow_html=True,
)
perf = performance_frame(prices, asof)
st.plotly_chart(performance_heatmap(perf), width="stretch")


# =============================================================================
# METHODOLOGY
# =============================================================================

with st.expander("Methodology and thresholds"):
    st.markdown(
        """
        **Growth impulse** uses one-month moves in RSP/SPY, IWM/SPY, copper/gold, EEM/SPY and high-yield OAS. Breadth needs roughly ±0.50%, small caps and EM roughly ±1.00%, copper/gold ±2.00%, and high-yield spreads ±10 bp to register a directional vote.

        **Inflation impulse** uses the 10-year breakeven, WTI crude, copper/gold and TIP/IEF. A ±5 bp breakeven move, roughly ±5% crude move, ±2% copper/gold move or ±0.75% TIP/IEF move is treated as meaningful.

        **Rates impulse** uses the 2-year, 10-year, 30-year and 10-year real yield. A one-month move beyond roughly ±10 bp is directional.

        **Liquidity / financial conditions** uses the broad dollar, real yields, high-yield spreads, EUR/USD and a simple Fed balance-sheet liquidity measure: Federal Reserve assets minus the Treasury General Account minus overnight reverse repo. RRP is converted to the same units before subtraction.

        **Risk confirmation** uses SPY, breadth, HYG/LQD, VIX and EEM/SPY. The regime is not considered clean risk-on if the index rises without breadth or credit.

        No proprietary composite is shown. A state changes only when the balance of its underlying observable inputs moves decisively in one direction.
        """
    )

render_footer()
