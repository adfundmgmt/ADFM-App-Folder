from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


# ============================================================
# Page Config
# ============================================================

TITLE = "Credit Conditions Monitor"
SUBTITLE = "Yahoo Finance only: market-implied credit stress, bank pressure, loan appetite, EM credit, volatility, and duration confirmation."

MARKET_TICKERS: Tuple[str, ...] = (
    "HYG",   # iShares iBoxx High Yield Corporate Bond ETF
    "JNK",   # SPDR Bloomberg High Yield Bond ETF
    "LQD",   # iShares iBoxx Investment Grade Corporate Bond ETF
    "BKLN",  # Invesco Senior Loan ETF
    "SRLN",  # SPDR Blackstone Senior Loan ETF
    "EMB",   # iShares J.P. Morgan USD Emerging Markets Bond ETF
    "KRE",   # SPDR S&P Regional Banking ETF
    "XLF",   # Financial Select Sector SPDR Fund
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
    "IWM",   # Russell 2000 ETF
    "TLT",   # 20+ Year Treasury ETF
    "IEF",   # 7-10 Year Treasury ETF
    "SHY",   # 1-3 Year Treasury ETF
    "UUP",   # Dollar Index ETF proxy
    "GLD",   # Gold ETF, liquidity/risk hedge context
    "^VIX",  # CBOE VIX Index
    "^TNX",  # CBOE 10Y Treasury yield index, quoted as yield x 10
    "^IRX",  # CBOE 13-week T-bill yield index, quoted as yield x 10
)

DISPLAY_NAMES: Dict[str, str] = {
    "HYG": "HY Credit",
    "JNK": "HY Credit 2",
    "LQD": "IG Credit",
    "BKLN": "Leveraged Loans",
    "SRLN": "Senior Loans",
    "EMB": "EM USD Debt",
    "KRE": "Regional Banks",
    "XLF": "Financials",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Small Caps",
    "TLT": "Long Duration",
    "IEF": "Intermediate Duration",
    "SHY": "Bills / Front End",
    "UUP": "Dollar Proxy",
    "GLD": "Gold",
    "^VIX": "VIX",
    "^TNX": "10Y Yield",
    "^IRX": "3M Bill Yield",
}

PERIOD_OPTIONS: Dict[str, str] = {
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
}

LOOKBACKS: Dict[str, object] = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 91,
    "6M": 182,
    "YTD": "YTD",
}

COLORS = {
    "green": "#15803d",
    "soft_green": "#dcfce7",
    "red": "#b91c1c",
    "soft_red": "#fee2e2",
    "amber": "#b45309",
    "soft_amber": "#fef3c7",
    "blue": "#1d4ed8",
    "purple": "#6d28d9",
    "slate": "#334155",
    "muted": "#64748b",
    "border": "#e5e7eb",
    "grid": "#e2e8f0",
    "dark": "#0f172a",
    "light": "#f8fafc",
}

LINE_COLORS = [
    "#1d4ed8",
    "#b91c1c",
    "#15803d",
    "#6d28d9",
    "#b45309",
    "#0f766e",
    "#4338ca",
    "#be123c",
    "#475569",
]

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2.0rem;
            max-width: 1580px;
        }

        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        .adfm-title-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.95rem;
        }

        .adfm-title {
            font-size: 1.85rem;
            line-height: 1.12;
            font-weight: 760;
            margin: 0 0 0.22rem 0;
            color: #0f172a;
            letter-spacing: -0.025em;
        }

        .adfm-subtitle {
            font-size: 0.93rem;
            color: #64748b;
            line-height: 1.42;
            max-width: 950px;
        }

        .source-pill {
            white-space: nowrap;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            color: #475569;
            font-size: 0.78rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            margin-top: 0.12rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 13px 15px 12px 15px;
            min-height: 104px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.045);
        }

        .metric-label {
            font-size: 0.70rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            margin-bottom: 0.42rem;
            font-weight: 680;
        }

        .metric-value {
            font-size: 1.24rem;
            font-weight: 780;
            color: #0f172a;
            line-height: 1.18;
        }

        .metric-footnote {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 0.42rem;
            line-height: 1.35;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.85rem 0.95rem 0.95rem 0.95rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
            margin-top: 0.95rem;
        }

        .section-title {
            font-size: 1.00rem;
            font-weight: 760;
            color: #0f172a;
            margin: 0 0 0.55rem 0;
            letter-spacing: -0.01em;
        }

        .section-subtitle {
            font-size: 0.80rem;
            color: #64748b;
            margin-top: -0.35rem;
            margin-bottom: 0.65rem;
            line-height: 1.42;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 13px 15px;
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.48;
            margin-top: 0.95rem;
        }

        .warn-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 14px;
            padding: 11px 13px;
            color: #92400e;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 0.75rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

@dataclass(frozen=True)
class ProxySpec:
    name: str
    numerator: str
    denominator: str
    read_through: str
    higher_is_good: bool = True


PROXY_SPECS: Tuple[ProxySpec, ...] = (
    ProxySpec("HYG/LQD", "HYG", "LQD", "HY beta versus IG credit. Higher means credit risk appetite is improving."),
    ProxySpec("JNK/LQD", "JNK", "LQD", "Second HY confirmation versus IG credit."),
    ProxySpec("BKLN/LQD", "BKLN", "LQD", "Leveraged loan sponsorship versus IG credit."),
    ProxySpec("SRLN/LQD", "SRLN", "LQD", "Senior loan confirmation versus IG credit."),
    ProxySpec("EMB/LQD", "EMB", "LQD", "EM USD credit versus domestic IG credit."),
    ProxySpec("KRE/SPY", "KRE", "SPY", "Regional banks versus the market. Lower exposes credit transmission risk."),
    ProxySpec("XLF/SPY", "XLF", "SPY", "Financials versus the market. Lower means credit-sensitive equities are losing sponsorship."),
    ProxySpec("IWM/SPY", "IWM", "SPY", "Small caps versus large caps. Lower can flag tighter domestic liquidity."),
    ProxySpec("HYG/SHY", "HYG", "SHY", "HY credit versus front-end Treasury collateral."),
    ProxySpec("LQD/TLT", "LQD", "TLT", "IG credit versus pure duration. Higher can mean credit is absorbing rates better."),
)


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    clean = df.dropna(how="all")
    if clean.empty:
        return "N/A"
    return pd.to_datetime(clean.index[-1]).strftime("%Y-%m-%d")


def fmt_num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 1, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "N/A"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def fmt_price(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def fmt_yield_from_yahoo_index(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x / 10.0:.2f}%"


def stress_color(score: float) -> str:
    if score >= 0.35:
        return COLORS["red"]
    if score <= -0.35:
        return COLORS["green"]
    return COLORS["amber"]


def classify_credit(score: float) -> str:
    if score >= 0.60:
        return "Stress / De-risk"
    if score >= 0.25:
        return "Tightening"
    if score > -0.20:
        return "Neutral / Watch"
    if score > -0.60:
        return "Constructive"
    return "Easy / Risk-On"


def classify_component(score: float) -> str:
    if not np.isfinite(score):
        return "N/A"
    if score >= 0.60:
        return "Stress"
    if score >= 0.25:
        return "Tightening"
    if score <= -0.60:
        return "Easy"
    if score <= -0.25:
        return "Constructive"
    return "Neutral"


def asof_value(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    eligible = clean.loc[clean.index <= target]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1])


def lookback_target(label: str) -> pd.Timestamp:
    today = pd.Timestamp(date.today())
    if label == "YTD":
        return pd.Timestamp(date(today.year, 1, 1))
    days = LOOKBACKS.get(label, 30)
    if isinstance(days, int):
        return today - pd.Timedelta(days=days)
    return today - pd.Timedelta(days=30)


def pct_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    now = latest(clean)
    then = asof_value(clean, target)
    if not np.isfinite(now) or not np.isfinite(then) or then == 0:
        return np.nan
    return float(now / then - 1.0)


def absolute_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    now = latest(clean)
    then = asof_value(clean, target)
    if not np.isfinite(now) or not np.isfinite(then):
        return np.nan
    return float(now - then)


def drawdown_pct(series: pd.Series, window: int = 126) -> float:
    clean = series.dropna().tail(window)
    if clean.empty:
        return np.nan
    peak = clean.cummax().iloc[-1]
    if not np.isfinite(peak) or peak == 0:
        return np.nan
    return float((clean.iloc[-1] / peak - 1.0) * 100.0)


def rolling_drawdown(series: pd.Series, window: int = 126) -> pd.Series:
    clean = series.astype(float).copy()
    peak = clean.rolling(window=window, min_periods=30).max()
    return clean.divide(peak).subtract(1.0)


def zscore(series: pd.Series, window: int = 252) -> float:
    clean = series.dropna().tail(window)
    if len(clean) < 30:
        return np.nan
    std = clean.std()
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float((clean.iloc[-1] - clean.mean()) / std)


def rolling_percentile_last(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    clean = series.astype(float).copy()

    def pct_rank(values: np.ndarray) -> float:
        s = pd.Series(values).dropna()
        if len(s) < min_periods:
            return np.nan
        return float(s.rank(pct=True).iloc[-1])

    return clean.rolling(window=window, min_periods=min_periods).apply(pct_rank, raw=True)


def rebase(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.dropna(how="all").copy()
    if clean.empty:
        return clean
    clean = clean.ffill().bfill()
    first = clean.iloc[0].replace(0, np.nan)
    return clean.divide(first).multiply(100.0)


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def chart_layout(height: int = 390, showlegend: bool = True) -> Dict[str, object]:
    return dict(
        height=height,
        margin=dict(l=12, r=12, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color=COLORS["slate"]),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )


def apply_axis_style(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )
    return fig


# ============================================================
# Data Fetching
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    try:
        data = yf.download(
            list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = list(data.columns.get_level_values(0))
            level1 = list(data.columns.get_level_values(1))

            if "Close" in level0:
                close = data.xs("Close", axis=1, level=0).copy()
            elif "Close" in level1:
                close = data.xs("Close", axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            if "Close" not in data.columns:
                return pd.DataFrame()
            close = data[["Close"]].copy()
            close.columns = [tickers[0]]

        close.columns = [str(c).upper() for c in close.columns]
        close = clean_index(close)
        close = close.dropna(how="all").ffill()
        close = close.loc[:, close.notna().sum() > 20]
        return close

    except Exception:
        return pd.DataFrame()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Market-implied credit monitor built only from Yahoo Finance tickers.

        This page does not use FRED, cash OAS, NFCI, or external spread files. It reads the credit tape through liquid ETFs, relative ratios, banks, volatility, rates proxies, and drawdowns.
        """
    )

    st.divider()

    st.header("Controls")
    history_label = st.selectbox("History", list(PERIOD_OPTIONS.keys()), index=3)
    focus_window = st.selectbox("Focus window", ["1D", "1W", "1M", "3M", "6M", "YTD"], index=2)
    stress_window = st.selectbox(
        "Stress percentile window",
        [63, 126, 252, 504],
        index=2,
        format_func=lambda x: f"{x} trading days",
    )
    smoothing_days = st.selectbox("Composite smoothing", [1, 3, 5, 10], index=1, format_func=lambda x: "None" if x == 1 else f"{x} days")
    show_raw = st.checkbox("Show raw Yahoo data", value=False)


# ============================================================
# Header and Load Data
# ============================================================

market_period = PERIOD_OPTIONS[history_label]

market = fetch_market(MARKET_TICKERS, market_period)

st.markdown(
    f"""
    <div class="adfm-title-row">
        <div>
            <div class="adfm-title">{TITLE}</div>
            <div class="adfm-subtitle">{SUBTITLE}</div>
        </div>
        <div class="source-pill">Source: Yahoo Finance · Latest: {latest_date(market)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if market.empty:
    st.error("No Yahoo Finance market data loaded. Check yfinance availability or ticker access.")
    st.stop()

loaded_tickers = set(market.columns)
missing_tickers = [t for t in MARKET_TICKERS if t.upper() not in loaded_tickers]

if missing_tickers:
    st.markdown(
        f"""
        <div class="warn-box">
            <b>Partial Yahoo load:</b> {', '.join(missing_tickers)} did not load. The dashboard is using the available tickers and will skip unavailable signals automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Derived Proxies
# ============================================================

proxy = pd.DataFrame(index=market.index)
proxy_reads: Dict[str, str] = {}
proxy_higher_is_good: Dict[str, bool] = {}

for spec in PROXY_SPECS:
    if spec.numerator in market.columns and spec.denominator in market.columns:
        denom = market[spec.denominator].replace(0, np.nan)
        proxy[spec.name] = market[spec.numerator].divide(denom)
        proxy_reads[spec.name] = spec.read_through
        proxy_higher_is_good[spec.name] = spec.higher_is_good

proxy = proxy.dropna(how="all").ffill()

rates = pd.DataFrame(index=market.index)
if "^TNX" in market.columns:
    rates["10Y Yield"] = market["^TNX"] / 10.0
if "^IRX" in market.columns:
    rates["3M Bill Yield"] = market["^IRX"] / 10.0
if {"^TNX", "^IRX"}.issubset(market.columns):
    rates["10Y-3M Slope"] = (market["^TNX"] - market["^IRX"]) / 10.0
rates = rates.dropna(how="all").ffill()


# ============================================================
# Composite Credit Stress
# Higher = tighter / worse. Lower = easier / better.
# ============================================================

stress_components = pd.DataFrame(index=market.index)

for name in ["HYG/LQD", "JNK/LQD", "BKLN/LQD", "SRLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY", "IWM/SPY", "HYG/SHY", "LQD/TLT"]:
    if name in proxy.columns:
        pct = rolling_percentile_last(proxy[name], window=stress_window)
        stress_components[name] = -(pct * 2.0 - 1.0)

if "HYG" in market.columns:
    hyg_dd = rolling_drawdown(market["HYG"], window=126)
    stress_components["HYG Drawdown"] = np.clip(-hyg_dd * 8.0, -1.0, 1.0)

if "JNK" in market.columns:
    jnk_dd = rolling_drawdown(market["JNK"], window=126)
    stress_components["JNK Drawdown"] = np.clip(-jnk_dd * 8.0, -1.0, 1.0)

if "^VIX" in market.columns:
    vix_pct = rolling_percentile_last(market["^VIX"], window=stress_window)
    stress_components["VIX"] = vix_pct * 2.0 - 1.0

stress_components = stress_components.dropna(how="all").ffill()

if stress_components.empty:
    credit_score_ts = pd.Series(dtype=float)
    credit_score = 0.0
    current_components = pd.Series(dtype=float)
else:
    credit_score_ts = stress_components.mean(axis=1).dropna()
    if smoothing_days > 1:
        credit_score_ts = credit_score_ts.rolling(smoothing_days, min_periods=1).mean()
    credit_score = latest(credit_score_ts) if not credit_score_ts.empty else 0.0
    current_components = stress_components.dropna(how="all").iloc[-1].dropna().sort_values(ascending=False)

credit_regime = classify_credit(credit_score)
target = lookback_target(focus_window)

if len(current_components) > 0:
    tightening_count = int((current_components >= 0.25).sum())
    easing_count = int((current_components <= -0.25).sum())
    stress_breadth = tightening_count / len(current_components)
else:
    tightening_count = 0
    easing_count = 0
    stress_breadth = np.nan


# ============================================================
# Key Metrics
# ============================================================

hyg_lqd_move = pct_change_since(proxy["HYG/LQD"], target) * 100 if "HYG/LQD" in proxy.columns else np.nan
kre_spy_move = pct_change_since(proxy["KRE/SPY"], target) * 100 if "KRE/SPY" in proxy.columns else np.nan
bkl_lqd_move = pct_change_since(proxy["BKLN/LQD"], target) * 100 if "BKLN/LQD" in proxy.columns else np.nan
emb_lqd_move = pct_change_since(proxy["EMB/LQD"], target) * 100 if "EMB/LQD" in proxy.columns else np.nan
hyg_drawdown = drawdown_pct(market["HYG"], 126) if "HYG" in market.columns else np.nan
vix_level = latest(market["^VIX"]) if "^VIX" in market.columns else np.nan
vix_move = absolute_change_since(market["^VIX"], target) if "^VIX" in market.columns else np.nan

cards = [
    (
        "Credit Regime",
        credit_regime,
        f"Composite {credit_score:+.2f}",
        stress_color(credit_score),
    ),
    (
        "Stress Breadth",
        fmt_pct(stress_breadth * 100.0, 0),
        f"{tightening_count} tightening / {easing_count} easing",
        COLORS["red"] if np.isfinite(stress_breadth) and stress_breadth >= 0.45 else COLORS["green"] if np.isfinite(stress_breadth) and stress_breadth <= 0.20 else COLORS["amber"],
    ),
    (
        "HY vs IG",
        fmt_pct(hyg_lqd_move, signed=True),
        f"HYG/LQD {focus_window}",
        COLORS["green"] if np.isfinite(hyg_lqd_move) and hyg_lqd_move > 0 else COLORS["red"],
    ),
    (
        "Bank Beta",
        fmt_pct(kre_spy_move, signed=True),
        f"KRE/SPY {focus_window}",
        COLORS["green"] if np.isfinite(kre_spy_move) and kre_spy_move > 0 else COLORS["red"],
    ),
    (
        "Loan Appetite",
        fmt_pct(bkl_lqd_move, signed=True),
        f"BKLN/LQD {focus_window}",
        COLORS["green"] if np.isfinite(bkl_lqd_move) and bkl_lqd_move > 0 else COLORS["red"],
    ),
    (
        "HY Drawdown",
        fmt_pct(hyg_drawdown),
        "126D drawdown",
        COLORS["red"] if np.isfinite(hyg_drawdown) and hyg_drawdown < -3.0 else COLORS["amber"] if np.isfinite(hyg_drawdown) and hyg_drawdown < -1.0 else COLORS["green"],
    ),
]

metric_cols = st.columns(6)
for col, (label, value, footnote, color) in zip(metric_cols, cards):
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


# ============================================================
# Visuals First
# ============================================================

top_left, top_right = st.columns([1.15, 0.85])

with top_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Credit Stress Pulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Composite of Yahoo-traded credit, bank, loan, EM, drawdown, and volatility signals. Higher means tighter credit conditions.</div>', unsafe_allow_html=True)

    if credit_score_ts.empty:
        st.info("Composite stress could not be calculated with the available Yahoo data.")
    else:
        fig = go.Figure()
        fig.add_hrect(y0=0.60, y1=1.05, fillcolor=COLORS["soft_red"], opacity=0.45, line_width=0)
        fig.add_hrect(y0=0.25, y1=0.60, fillcolor=COLORS["soft_amber"], opacity=0.45, line_width=0)
        fig.add_hrect(y0=-1.05, y1=-0.25, fillcolor=COLORS["soft_green"], opacity=0.35, line_width=0)
        fig.add_trace(
            go.Scatter(
                x=credit_score_ts.index,
                y=credit_score_ts,
                mode="lines",
                name="Composite Stress",
                line=dict(color=stress_color(credit_score), width=2.8),
            )
        )
        fig.add_hline(y=0, line_width=1, line_color=COLORS["grid"])
        fig.add_hline(y=0.60, line_width=1, line_dash="dot", line_color=COLORS["red"])
        fig.add_hline(y=0.25, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        fig.add_hline(y=-0.25, line_width=1, line_dash="dot", line_color=COLORS["green"])
        fig.update_layout(**chart_layout(height=390, showlegend=False))
        fig.update_yaxes(title_text="Stress score", range=[-1.05, 1.05])
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Current component scores. Positive bars are tightening. Negative bars are easing.</div>', unsafe_allow_html=True)

    if current_components.empty:
        st.info("Pressure map unavailable.")
    else:
        bar_df = current_components.sort_values(ascending=True).reset_index()
        bar_df.columns = ["Signal", "Stress"]
        bar_colors = [stress_color(v) for v in bar_df["Stress"]]

        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=bar_df["Stress"],
                y=bar_df["Signal"],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(15, 23, 42, 0.10)", width=1)),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                name="Stress",
            )
        )
        bar_fig.add_vline(x=0, line_width=1, line_color=COLORS["grid"])
        bar_fig.add_vline(x=0.25, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        bar_fig.add_vline(x=0.60, line_width=1, line_dash="dot", line_color=COLORS["red"])
        bar_fig.add_vline(x=-0.25, line_width=1, line_dash="dot", line_color=COLORS["green"])
        bar_fig.update_layout(**chart_layout(height=390, showlegend=False))
        bar_fig.update_xaxes(title_text="Stress score", range=[-1.05, 1.05])
        apply_axis_style(bar_fig)
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Proxy Leadership and Drawdowns
# ============================================================

mid_left, mid_right = st.columns([1.05, 0.95])

with mid_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Credit Risk Appetite</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Relative proxy basket rebased to 100. Rising lines usually mean credit sponsorship is improving.</div>', unsafe_allow_html=True)

    proxy_cols = [c for c in ["HYG/LQD", "JNK/LQD", "BKLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY"] if c in proxy.columns]
    if not proxy_cols:
        st.info("Relative credit proxy data unavailable.")
    else:
        proxy_rebased = rebase(proxy[proxy_cols])
        proxy_fig = go.Figure()
        for i, col_name in enumerate(proxy_rebased.columns):
            proxy_fig.add_trace(
                go.Scatter(
                    x=proxy_rebased.index,
                    y=proxy_rebased[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                )
            )
        proxy_fig.add_hline(y=100, line_width=1, line_color=COLORS["grid"])
        proxy_fig.update_layout(**chart_layout(height=400, showlegend=True))
        proxy_fig.update_yaxes(title_text="Rebased to 100")
        apply_axis_style(proxy_fig)
        st.plotly_chart(proxy_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with mid_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Credit ETF Drawdowns</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Drawdown pressure across HY, IG, loans, EM debt, banks, and equities.</div>', unsafe_allow_html=True)

    drawdown_cols = [c for c in ["HYG", "JNK", "LQD", "BKLN", "EMB", "KRE", "SPY"] if c in market.columns]
    if not drawdown_cols:
        st.info("Drawdown data unavailable.")
    else:
        dd = pd.DataFrame(index=market.index)
        for c in drawdown_cols:
            dd[DISPLAY_NAMES.get(c, c)] = rolling_drawdown(market[c], window=126) * 100.0

        dd_fig = go.Figure()
        for i, col_name in enumerate(dd.columns):
            dd_fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                )
            )
        dd_fig.add_hline(y=0, line_width=1, line_color=COLORS["grid"])
        dd_fig.add_hline(y=-3, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        dd_fig.add_hline(y=-6, line_width=1, line_dash="dot", line_color=COLORS["red"])
        dd_fig.update_layout(**chart_layout(height=400, showlegend=True))
        dd_fig.update_yaxes(title_text="126D drawdown %")
        apply_axis_style(dd_fig)
        st.plotly_chart(dd_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Rates, Volatility, and Asset Confirmation
# ============================================================

lower_left, lower_right = st.columns([1.0, 1.0])

with lower_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rates and Volatility Backdrop</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Yahoo rate indexes are quoted as yield x 10. The chart converts them back to percent.</div>', unsafe_allow_html=True)

    if rates.empty and "^VIX" not in market.columns:
        st.info("Rates and volatility proxies unavailable.")
    else:
        rv_fig = make_subplots(specs=[[{"secondary_y": True}]])

        if "10Y Yield" in rates.columns:
            rv_fig.add_trace(
                go.Scatter(
                    x=rates.index,
                    y=rates["10Y Yield"],
                    mode="lines",
                    name="10Y Yield",
                    line=dict(color=COLORS["blue"], width=2),
                ),
                secondary_y=False,
            )

        if "3M Bill Yield" in rates.columns:
            rv_fig.add_trace(
                go.Scatter(
                    x=rates.index,
                    y=rates["3M Bill Yield"],
                    mode="lines",
                    name="3M Bill Yield",
                    line=dict(color=COLORS["slate"], width=2),
                ),
                secondary_y=False,
            )

        if "^VIX" in market.columns:
            rv_fig.add_trace(
                go.Scatter(
                    x=market.index,
                    y=market["^VIX"],
                    mode="lines",
                    name="VIX",
                    line=dict(color=COLORS["red"], width=2),
                ),
                secondary_y=True,
            )

        rv_fig.update_layout(**chart_layout(height=385, showlegend=True))
        rv_fig.update_yaxes(title_text="Yield %", secondary_y=False)
        rv_fig.update_yaxes(title_text="VIX", secondary_y=True, showgrid=False)
        apply_axis_style(rv_fig)
        st.plotly_chart(rv_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with lower_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Macro Confirmation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Cross-asset confirmation around credit: duration, dollar, gold, equities, and small caps.</div>', unsafe_allow_html=True)

    macro_cols = [c for c in ["SPY", "QQQ", "IWM", "TLT", "IEF", "UUP", "GLD"] if c in market.columns]
    if not macro_cols:
        st.info("Macro confirmation data unavailable.")
    else:
        macro_rebased = rebase(market[macro_cols].rename(columns=DISPLAY_NAMES))
        macro_fig = go.Figure()
        for i, col_name in enumerate(macro_rebased.columns):
            macro_fig.add_trace(
                go.Scatter(
                    x=macro_rebased.index,
                    y=macro_rebased[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                )
            )
        macro_fig.add_hline(y=100, line_width=1, line_color=COLORS["grid"])
        macro_fig.update_layout(**chart_layout(height=385, showlegend=True))
        macro_fig.update_yaxes(title_text="Rebased to 100")
        apply_axis_style(macro_fig)
        st.plotly_chart(macro_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Credit Tape
# ============================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Credit Tape</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Decision table across ratio proxies, ETFs, rates, and volatility. Ratio moves are percentage changes; rates and VIX are absolute changes.</div>', unsafe_allow_html=True)

tape_rows: List[Dict[str, object]] = []


def add_ratio_tape(name: str, series: pd.Series, read: str, higher_is_good: bool = True) -> None:
    current_stress = latest(stress_components[name]) if name in stress_components.columns else np.nan
    tape_rows.append(
        {
            "Signal": name,
            "Latest": fmt_num(latest(series), 3),
            "1D": fmt_pct(pct_change_since(series, lookback_target("1D")) * 100, signed=True),
            "1W": fmt_pct(pct_change_since(series, lookback_target("1W")) * 100, signed=True),
            "1M": fmt_pct(pct_change_since(series, lookback_target("1M")) * 100, signed=True),
            "3M": fmt_pct(pct_change_since(series, lookback_target("3M")) * 100, signed=True),
            "YTD": fmt_pct(pct_change_since(series, lookback_target("YTD")) * 100, signed=True),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(current_stress, 2),
            "Regime": classify_component(current_stress),
            "Read-through": read,
        }
    )


def add_price_tape(ticker: str, series: pd.Series, read: str) -> None:
    current_stress = latest(stress_components[f"{ticker} Drawdown"]) if f"{ticker} Drawdown" in stress_components.columns else np.nan
    tape_rows.append(
        {
            "Signal": DISPLAY_NAMES.get(ticker, ticker),
            "Latest": fmt_price(latest(series)),
            "1D": fmt_pct(pct_change_since(series, lookback_target("1D")) * 100, signed=True),
            "1W": fmt_pct(pct_change_since(series, lookback_target("1W")) * 100, signed=True),
            "1M": fmt_pct(pct_change_since(series, lookback_target("1M")) * 100, signed=True),
            "3M": fmt_pct(pct_change_since(series, lookback_target("3M")) * 100, signed=True),
            "YTD": fmt_pct(pct_change_since(series, lookback_target("YTD")) * 100, signed=True),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(current_stress, 2),
            "Regime": classify_component(current_stress) if np.isfinite(current_stress) else "Price",
            "Read-through": read,
        }
    )


def add_absolute_tape(name: str, series: pd.Series, latest_formatter, read: str) -> None:
    tape_rows.append(
        {
            "Signal": name,
            "Latest": latest_formatter(latest(series)),
            "1D": fmt_num(absolute_change_since(series, lookback_target("1D")), 2),
            "1W": fmt_num(absolute_change_since(series, lookback_target("1W")), 2),
            "1M": fmt_num(absolute_change_since(series, lookback_target("1M")), 2),
            "3M": fmt_num(absolute_change_since(series, lookback_target("3M")), 2),
            "YTD": fmt_num(absolute_change_since(series, lookback_target("YTD")), 2),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(latest(stress_components[name]) if name in stress_components.columns else np.nan, 2),
            "Regime": classify_component(latest(stress_components[name])) if name in stress_components.columns else "Macro",
            "Read-through": read,
        }
    )


for name in ["HYG/LQD", "JNK/LQD", "BKLN/LQD", "SRLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY", "IWM/SPY", "HYG/SHY", "LQD/TLT"]:
    if name in proxy.columns:
        add_ratio_tape(name, proxy[name], proxy_reads.get(name, "Relative credit proxy"), proxy_higher_is_good.get(name, True))

for ticker, read in [
    ("HYG", "HY ETF price. Persistent drawdown is a market-implied spread warning."),
    ("JNK", "Second high-yield ETF confirmation."),
    ("LQD", "IG credit price. Separates credit and duration stress imperfectly but usefully."),
    ("BKLN", "Loan ETF price. Tracks floating-rate credit appetite."),
    ("EMB", "EM hard-currency debt price. Flags global credit pressure."),
    ("KRE", "Regional bank equity proxy. Often leads tightening in credit transmission."),
    ("SPY", "Equity beta confirmation."),
]:
    if ticker in market.columns:
        add_price_tape(ticker, market[ticker], read)

if "^VIX" in market.columns:
    add_absolute_tape("VIX", market["^VIX"], fmt_price, "Equity volatility. Higher means tighter risk appetite.")

if "10Y Yield" in rates.columns:
    add_absolute_tape("10Y Yield", rates["10Y Yield"], lambda x: f"{x:.2f}%" if np.isfinite(x) else "N/A", "Rates pressure. Move shown in percentage points.")

if "10Y-3M Slope" in rates.columns:
    add_absolute_tape("10Y-3M Slope", rates["10Y-3M Slope"], lambda x: f"{x:.2f}%" if np.isfinite(x) else "N/A", "Curve slope. Move shown in percentage points.")

if tape_rows:
    tape = pd.DataFrame(tape_rows)
    st.dataframe(tape, use_container_width=True, hide_index=True, height=520)
else:
    st.info("No tape signals loaded.")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Interpretation Box
# ============================================================

if credit_score >= 0.60:
    readthrough = (
        "Credit proxies are in stress mode. The signal is no longer isolated to one ETF or one bank proxy. "
        "When HY underperforms IG, KRE loses to SPY, drawdowns deepen, and VIX rises together, the tape is saying de-risking pressure has moved into credit transmission."
    )
elif credit_score >= 0.25:
    readthrough = (
        "Credit is tightening. Equity rallies need confirmation from HYG/LQD and KRE/SPY here. "
        "If those ratios fail while VIX stays bid or HY drawdowns deepen, the rally is losing credit sponsorship."
    )
elif credit_score <= -0.25:
    readthrough = (
        "Credit is supportive. HY, loans, banks, or volatility proxies are not confirming broad funding stress. "
        "That keeps the burden of proof on equity bears unless the bank and HY ratios roll over."
    )
else:
    readthrough = (
        "Credit is neutral. The composite is not strong enough by itself. "
        "The next useful break should come from HYG/LQD, KRE/SPY, BKLN/LQD, and VIX moving in the same direction."
    )

st.markdown(
    f"""
    <div class="note-box">
        <b>Read-through:</b> {readthrough}<br>
        <b>Data discipline:</b> Yahoo Finance only. No FRED, no NFCI, no cash OAS, and no bps spread labels. Outputs are market-implied proxy signals based on liquid instruments available through yfinance.<br>
        <b>Latest Yahoo close:</b> {latest_date(market)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Raw Data
# ============================================================

if show_raw:
    with st.expander("Raw Yahoo Close Data", expanded=False):
        renamed_market = market.rename(columns=DISPLAY_NAMES)
        st.dataframe(renamed_market.tail(500), use_container_width=True)

    with st.expander("Derived Proxy Data", expanded=False):
        if proxy.empty:
            st.info("No proxy data calculated.")
        else:
            st.dataframe(proxy.tail(500), use_container_width=True)

    with st.expander("Stress Components", expanded=False):
        if stress_components.empty:
            st.info("No stress components calculated.")
        else:
            st.dataframe(stress_components.tail(500), use_container_width=True)
