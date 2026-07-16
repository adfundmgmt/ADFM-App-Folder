import math
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from adfm_core.market_data import configure_yfinance_cache
from adfm_core.regime_math import grouped_weighted_composite
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_selection_note,
)
import yfinance as yf
from plotly.subplots import make_subplots

configure_yfinance_cache()


# ============================================================
# PAGE SETUP
# ============================================================

TITLE = "Liquidity Conditions Monitor"

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2.35rem;
        padding-bottom: 2.6rem;
        max-width: 1580px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f3f6fa;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2d3748;
        font-weight: 750;
        letter-spacing: -0.01em;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div {
        color: #3f4a5a;
    }

    .tool-divider {
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        border-top: 1px solid #d8dee8;
    }

    .hero-title {
        font-size: 2.55rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #2d3142;
        margin-bottom: 0.25rem;
    }

    .hero-caption {
        color: #8b949e;
        font-size: 1.00rem;
        line-height: 1.48;
        margin-bottom: 1.55rem;
        max-width: 1180px;
    }

    .section-title {
        font-size: 1.08rem;
        line-height: 1.25;
        font-weight: 750;
        letter-spacing: -0.015em;
        color: #2d3142;
        margin-top: 1.15rem;
        margin-bottom: 0.28rem;
    }

    .section-subtitle {
        font-size: 0.91rem;
        color: #8b949e;
        margin-bottom: 0.85rem;
        line-height: 1.42;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 12px 14px 10px 14px;
        min-height: 88px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.045);
    }

    .metric-label {
        font-size: 0.69rem;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.42rem;
        white-space: nowrap;
    }

    .metric-value {
        font-size: 1.22rem;
        font-weight: 760;
        color: #111827;
        line-height: 1.12;
        white-space: nowrap;
    }

    .metric-footnote {
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: 0.40rem;
        line-height: 1.25;
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 0.85rem;
        color: #334155;
        font-size: 0.91rem;
        line-height: 1.48;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 0.90rem;
        color: #7c2d12;
        font-size: 0.91rem;
        line-height: 1.45;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] label {
        color: #536171;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    div[data-baseweb="input"] {
        border-radius: 12px;
        background-color: #f5f7fb;
        border: 1px solid #e0e6ef;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        background-color: #f5f7fb;
        border-color: #e0e6ef;
    }

    .stPlotlyChart {
        background: #ffffff;
        border-radius: 12px;
    }

    .stDataFrame {
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    .stDownloadButton button {
        border-radius: 11px;
        font-weight: 700;
    }

    .js-plotly-plot .plotly .modebar {
        opacity: 0.35;
    }

    .js-plotly-plot .plotly .modebar:hover {
        opacity: 0.75;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
inject_explorer_style()


PLOTLY_FONT = "Arial, sans-serif"
PLOTLY_GRID = "rgba(226, 232, 240, 0.72)"
PLOTLY_AXIS = "#d7dde8"
PLOTLY_TEXT = "#334155"
PLOTLY_TITLE = "#111827"


def apply_adfm_plot_layout(
    fig: go.Figure,
    height: int,
    margin: Optional[Dict[str, int]] = None,
    showlegend: bool = True,
    legend_y: float = 1.065,
    hovermode: str = "x unified",
) -> go.Figure:
    """Shared chart styling used across the ADFM Streamlit pages."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=margin or dict(l=42, r=28, t=54, b=42),
        font=dict(color=PLOTLY_TEXT, family=PLOTLY_FONT),
        hovermode=hovermode,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    return fig


def clean_axis(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.55)",
        showline=True,
        linewidth=1,
        linecolor=PLOTLY_AXIS,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        showline=False,
        zeroline=False,
    )
    return fig


# ============================================================
# FED FCI-G DATA
# ============================================================

FED_FCIG_SOURCES = {
    "FCI-G Baseline": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_3yr.csv",
    "FCI-G 1Y Lookback": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_1yr.csv",
}


# ============================================================
# MARKET-IMPLIED LIQUIDITY COMPONENTS
# ============================================================

COMPONENTS = [
    {
        "name": "HY Credit / IG Credit",
        "category": "Credit",
        "numerator": "HYG",
        "denominator": "LQD",
        "orientation": 1,
        "weight": 1.35,
        "description": "High yield outperforming investment grade means credit risk appetite is improving.",
    },
    {
        "name": "Junk Credit / IG Credit",
        "category": "Credit",
        "numerator": "JNK",
        "denominator": "LQD",
        "orientation": 1,
        "weight": 1.00,
        "description": "Second credit confirmation line for risk appetite.",
    },
    {
        "name": "HY Credit / Cash Proxy",
        "category": "Credit",
        "numerator": "HYG",
        "denominator": "SHY",
        "orientation": 1,
        "weight": 1.00,
        "description": "Credit beta versus cash-like short duration.",
    },
    {
        "name": "Equal-Weight S&P / S&P 500",
        "category": "Equity Breadth",
        "numerator": "RSP",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 1.10,
        "description": "Broad S&P participation beneath cap-weight leadership.",
    },
    {
        "name": "Small Caps / S&P 500",
        "category": "Equity Breadth",
        "numerator": "IWM",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 1.05,
        "description": "Small-cap participation is a domestic cyclicality and liquidity check.",
    },
    {
        "name": "Equal-Weight Nasdaq / QQQ",
        "category": "Equity Breadth",
        "numerator": "QQQE",
        "denominator": "QQQ",
        "orientation": 1,
        "weight": 1.00,
        "description": "Nasdaq breadth beneath mega-cap leadership.",
    },
    {
        "name": "Disruptive Growth / QQQ",
        "category": "Speculation",
        "numerator": "ARKK",
        "denominator": "QQQ",
        "orientation": 1,
        "weight": 0.90,
        "description": "Speculative duration appetite relative to large-cap growth.",
    },
    {
        "name": "Biotech / QQQ",
        "category": "Speculation",
        "numerator": "XBI",
        "denominator": "QQQ",
        "orientation": 1,
        "weight": 1.00,
        "description": "Biotech is a clean animal-spirits and financing-conditions ratio.",
    },
    {
        "name": "IPO Basket / S&P 500",
        "category": "Speculation",
        "numerator": "IPO",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.65,
        "description": "Recent-issuance risk appetite relative to the broad market.",
    },
    {
        "name": "Regional Banks / S&P 500",
        "category": "Funding",
        "numerator": "KRE",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 1.05,
        "description": "Bank equity confirmation for funding and credit creation.",
    },
    {
        "name": "Financials / S&P 500",
        "category": "Funding",
        "numerator": "XLF",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.85,
        "description": "Broad financials confirmation.",
    },
    {
        "name": "Semis / Nasdaq",
        "category": "Leadership",
        "numerator": "SMH",
        "denominator": "QQQ",
        "orientation": 1,
        "weight": 0.75,
        "description": "AI and capex leadership relative to Nasdaq beta.",
    },
    {
        "name": "NVDA / Semis",
        "category": "Leadership",
        "numerator": "NVDA",
        "denominator": "SMH",
        "orientation": 1,
        "weight": 0.40,
        "description": "Leadership concentration and AI reflexivity check. Lower weight by design.",
    },
    {
        "name": "Bitcoin ETF / S&P 500",
        "category": "Crypto",
        "numerator": "IBIT",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.75,
        "description": "Crypto ETF beta relative to broad equities.",
    },
    {
        "name": "Bitcoin / S&P 500",
        "category": "Crypto",
        "numerator": "BTC-USD",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.75,
        "description": "Native crypto liquidity impulse relative to equities.",
    },
    {
        "name": "Emerging Markets / S&P 500",
        "category": "Global Dollar Liquidity",
        "numerator": "EEM",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.85,
        "description": "Global dollar-liquidity sensitivity.",
    },
    {
        "name": "China Internet / S&P 500",
        "category": "Global Dollar Liquidity",
        "numerator": "KWEB",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.55,
        "description": "High-beta global liquidity and China risk appetite.",
    },
    {
        "name": "Long Duration / Cash Proxy",
        "category": "Rates",
        "numerator": "TLT",
        "denominator": "SHY",
        "orientation": 1,
        "weight": 0.75,
        "description": "Long-duration bid versus short-duration cash proxy.",
    },
    {
        "name": "Intermediate Duration / Cash Proxy",
        "category": "Rates",
        "numerator": "IEF",
        "denominator": "SHY",
        "orientation": 1,
        "weight": 0.60,
        "description": "Less volatile duration confirmation.",
    },
    {
        "name": "Dollar Pressure",
        "category": "Dollar",
        "ticker": "UUP",
        "orientation": -1,
        "weight": 1.20,
        "description": "Lower dollar pressure is positive for global liquidity.",
    },
    {
        "name": "Volatility Pressure",
        "category": "Volatility",
        "ticker": "^VIX",
        "orientation": -1,
        "weight": 1.35,
        "description": "Lower volatility mechanically eases risk budgets.",
    },
]

CORE_BENCHMARKS = ["SPY", "QQQ"]

LIQUIDITY_SLEEVE_WEIGHTS = {
    "Credit": 0.20,
    "Equity Breadth": 0.15,
    "Funding": 0.15,
    "Speculation": 0.12,
    "Global Dollar Liquidity": 0.10,
    "Dollar": 0.10,
    "Volatility": 0.10,
    "Rates": 0.08,
    "Crypto": 0.05,
    "Leadership": 0.05,
}


# ============================================================
# FORMATTING HELPERS
# ============================================================


def metric_card(label: str, value: str, footnote: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def fmt_delta(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{digits}f}"


def classify_regime(score: float, breadth: float) -> Tuple[str, str]:
    if pd.isna(score):
        return "Unavailable", "Insufficient component coverage."

    if score >= 0.90 and breadth >= 65:
        return (
            "Liquidity Expansion",
            "Easier credit, lower pressure, and broad participation.",
        )
    if score >= 0.35:
        return "Improving", "Liquidity impulse is positive, but breadth still matters."
    if score > -0.35:
        return "Neutral / Mixed", "Cross-currents dominate; no clean liquidity regime."
    if score > -0.90:
        return (
            "Deteriorating",
            "Liquidity impulse is weakening across enough components to matter.",
        )
    return (
        "Liquidity Contraction",
        "Tighter risk budgets, weaker breadth, or pressure from dollar/vol/credit.",
    )


def signed_signal_word(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value > 0:
        return "Easing"
    if value < 0:
        return "Tightening"
    return "Flat"


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(periods=periods, fill_method=None) * 100.0


def latest_valid(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def obs_change(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def obs_pct_change(series: pd.Series, periods: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return np.nan
    base = s.iloc[-1 - periods]
    if pd.isna(base) or base == 0:
        return np.nan
    return float((s.iloc[-1] / base - 1.0) * 100.0)


def rebase(series: pd.Series, base_value: float = 100.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(index=s.index, dtype="float64")

    base = valid.iloc[0]

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, dtype="float64")

    return s / base * base_value


def zscore_trailing(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def dynamic_axis_range(
    series: pd.Series, floor_abs: float = 1.25, pad: float = 0.20
) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return -floor_abs, floor_abs

    lo = float(s.quantile(0.02))
    hi = float(s.quantile(0.98))
    lo = min(lo, -floor_abs)
    hi = max(hi, floor_abs)

    span = hi - lo
    if span <= 0:
        return -floor_abs, floor_abs

    return lo - span * pad, hi + span * pad


def score_to_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score >= 0.90:
        return "Strong easing"
    if score >= 0.35:
        return "Easing"
    if score > -0.35:
        return "Mixed"
    if score > -0.90:
        return "Tightening"
    return "Strong tightening"


def color_score(val: object) -> str:
    try:
        x = float(val)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    if x >= 0.90:
        return "background-color: #dcfce7; color: #14532d;"
    if x >= 0.35:
        return "background-color: #ecfdf5; color: #065f46;"
    if x > -0.35:
        return "background-color: #f9fafb; color: #374151;"
    if x > -0.90:
        return "background-color: #fff7ed; color: #9a3412;"
    return "background-color: #fee2e2; color: #7f1d1d;"


def month_end_last(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()

    for freq in ("ME", "M"):
        try:
            return out.resample(freq).last()
        except ValueError:
            continue

    raise ValueError("Could not resample to month-end with either ME or M.")


def visible_axis_range(
    data: pd.Series | pd.DataFrame | List[pd.Series],
    floor_abs: Optional[float] = None,
    pad: float = 0.08,
    include_zero: bool = False,
) -> Optional[List[float]]:
    series_list: List[pd.Series] = []

    if isinstance(data, pd.Series):
        series_list = [data]
    elif isinstance(data, pd.DataFrame):
        series_list = [data[col] for col in data.columns]
    elif isinstance(data, list):
        series_list = [x for x in data if isinstance(x, pd.Series)]

    values = []
    for series in series_list:
        clean = pd.to_numeric(series, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        if not clean.empty:
            values.append(clean)

    if not values:
        return None

    combined = pd.concat(values)

    if include_zero:
        combined = pd.concat([combined, pd.Series([0.0])])

    y_min = float(combined.min())
    y_max = float(combined.max())

    if floor_abs is not None:
        y_min = min(y_min, -float(floor_abs))
        y_max = max(y_max, float(floor_abs))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None

    if y_min == y_max:
        base = abs(y_min) if y_min != 0 else 1.0
        cushion = base * max(pad, 0.05)
        return [y_min - cushion, y_max + cushion]

    spread = y_max - y_min
    cushion = spread * pad
    return [y_min - cushion, y_max + cushion]


def _clean_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    idx = idx[idx.notna()]

    try:
        if idx.tz is not None:
            idx = idx.tz_convert(None)
    except Exception:
        pass

    return pd.DatetimeIndex(idx).sort_values()


def build_trading_session_axis(
    *objects: object,
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, int]]:
    dates: List[pd.Timestamp] = []

    for obj in objects:
        if obj is None:
            continue

        if isinstance(obj, (pd.Series, pd.DataFrame)):
            idx = _clean_datetime_index(obj.index)
            if len(idx) > 0:
                dates.extend(pd.Timestamp(x) for x in idx)

    if not dates:
        return pd.DatetimeIndex([]), {}

    trading_index = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values()
    session_map = {pd.Timestamp(dt): i for i, dt in enumerate(trading_index)}
    return trading_index, session_map


def session_x(
    index: pd.Index, session_map: Dict[pd.Timestamp, int]
) -> List[Optional[int]]:
    idx = _clean_datetime_index(index)
    return [session_map.get(pd.Timestamp(dt)) for dt in idx]


def session_dates(index: pd.Index) -> List[str]:
    idx = _clean_datetime_index(index)
    return [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in idx]


def make_session_ticks(
    trading_index: pd.DatetimeIndex, max_ticks: int = 10
) -> Tuple[List[int], List[str]]:
    if trading_index.empty:
        return [], []

    tick_count = min(max_ticks, max(2, len(trading_index)))
    positions = np.linspace(0, len(trading_index) - 1, tick_count).round().astype(int)
    positions = np.unique(positions)

    tickvals = positions.tolist()

    if len(trading_index) > 900:
        ticktext = trading_index[positions].strftime("%Y").tolist()
    elif len(trading_index) > 260:
        ticktext = trading_index[positions].strftime("%b-%y").tolist()
    else:
        ticktext = trading_index[positions].strftime("%b %d").tolist()

    return tickvals, ticktext


def apply_trading_session_xaxis(
    fig: go.Figure,
    trading_index: pd.DatetimeIndex,
    tickvals: List[int],
    ticktext: List[str],
    rows: List[int],
    title_row: Optional[int] = None,
) -> go.Figure:
    if trading_index.empty:
        return fig

    x_range = [-0.5, max(len(trading_index) - 0.5, 0.5)]

    for row in rows:
        fig.update_xaxes(
            type="linear",
            range=x_range,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.45)",
            row=row,
            col=1,
        )

    if title_row is not None:
        fig.update_xaxes(title_text="Date", row=title_row, col=1)

    return fig


# ============================================================
# DATA LOADERS
# ============================================================


def required_tickers(
    components: List[Dict[str, object]], benchmarks: List[str]
) -> List[str]:
    tickers = set(benchmarks)

    for component in components:
        if "ticker" in component:
            tickers.add(str(component["ticker"]))
        else:
            tickers.add(str(component["numerator"]))
            tickers.add(str(component["denominator"]))

    return sorted(tickers)


def extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))

        if "Close" in level0:
            close = df["Close"].copy()
        elif "Adj Close" in level0:
            close = df["Adj Close"].copy()
        elif "Close" in level1:
            close = df.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in level1:
            close = df.xs("Adj Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" in df.columns:
            close = df[["Close"]].copy()
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]].copy()
        else:
            numeric_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            close = df[numeric_cols].copy()

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.columns = [str(c).strip() for c in close.columns]
    close = close.sort_index()
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.loc[close.index.notna()]
    close.index = (
        close.index.tz_localize(None)
        if getattr(close.index, "tz", None) is not None
        else close.index
    )
    close = close.loc[:, ~close.columns.duplicated(keep="last")]

    for col in close.columns:
        close[col] = pd.to_numeric(close[col], errors="coerce")

    close = close.dropna(how="all")
    return close


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def load_yahoo_prices(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    close = extract_close_prices(raw)

    if close.empty:
        return pd.DataFrame()

    available = [c for c in close.columns if close[c].notna().sum() > 45]
    close = close[available].copy()

    return close


def choose_fcig_column(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = []

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() >= 12:
            numeric_cols.append(col)

    if not numeric_cols:
        return None

    priority_terms = ["fci-g index", "fci-g", "fcig", "fci_g", "fci g"]

    for term in priority_terms:
        for col in numeric_cols:
            lower = str(col).lower()
            if term in lower and "contribution" not in lower and "cont" not in lower:
                return col

    return numeric_cols[0]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_fed_fcig() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    frames = []
    contribution_frame = pd.DataFrame()
    errors: Dict[str, str] = {}

    headers = {
        "User-Agent": "ADFM-Liquidity-Conditions-Monitor/2.0",
        "Accept": "text/csv,*/*;q=0.8",
    }

    for label, url in FED_FCIG_SOURCES.items():
        try:
            response = requests.get(url, headers=headers, timeout=(4, 20))
            response.raise_for_status()

            temp = pd.read_csv(BytesIO(response.content))
            temp.columns = [str(c).strip() for c in temp.columns]

            if temp.empty:
                errors[label] = "Fed CSV returned an empty file."
                continue

            date_col = None
            for col in temp.columns:
                lower = str(col).lower()
                if "date" in lower or "month" in lower or "time" in lower:
                    date_col = col
                    break

            if date_col is None:
                date_col = temp.columns[0]

            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col]).set_index(date_col).sort_index()

            value_col = choose_fcig_column(temp)

            if value_col is None:
                errors[label] = "Could not identify a numeric FCI-G value column."
                continue

            out = temp[[value_col]].rename(columns={value_col: label})
            frames.append(out)

            if label == "FCI-G Baseline":
                possible_contribs = []
                for col in temp.columns:
                    if col == value_col:
                        continue
                    temp[col] = pd.to_numeric(temp[col], errors="coerce")
                    if temp[col].notna().sum() >= 12:
                        possible_contribs.append(col)

                if possible_contribs:
                    contribution_frame = temp[possible_contribs].copy()

        except Exception as exc:
            errors[label] = str(exc)

    if not frames:
        return pd.DataFrame(), contribution_frame, errors

    fcig = pd.concat(frames, axis=1).sort_index()
    fcig = fcig[~fcig.index.duplicated(keep="last")]

    for col in fcig.columns:
        fcig[col] = pd.to_numeric(fcig[col], errors="coerce")

    fcig = fcig.dropna(how="all")

    return fcig, contribution_frame, errors


# ============================================================
# SIGNAL ENGINE
# ============================================================


def build_component_series(
    prices: pd.DataFrame, components: List[Dict[str, object]]
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    series_map: Dict[str, pd.Series] = {}
    available_specs: List[Dict[str, object]] = []

    for spec in components:
        name = str(spec["name"])

        if "ticker" in spec:
            ticker = str(spec["ticker"])

            if ticker not in prices.columns:
                continue

            s = pd.to_numeric(prices[ticker], errors="coerce").copy()
            display_ticker = ticker
        else:
            numerator = str(spec["numerator"])
            denominator = str(spec["denominator"])

            if numerator not in prices.columns or denominator not in prices.columns:
                continue

            den = pd.to_numeric(prices[denominator], errors="coerce")
            num = pd.to_numeric(prices[numerator], errors="coerce")

            s = num / den.replace(0, np.nan)
            display_ticker = f"{numerator}/{denominator}"

        if s.dropna().shape[0] < 90:
            continue

        series_map[name] = s
        new_spec = dict(spec)
        new_spec["display_ticker"] = display_ticker
        available_specs.append(new_spec)

    if not series_map:
        return pd.DataFrame(), []

    component_df = pd.DataFrame(series_map).sort_index()
    component_df = component_df.dropna(how="all")

    return component_df, available_specs


def build_scores(
    component_df: pd.DataFrame,
    available_specs: List[Dict[str, object]],
    z_window: int,
    min_z_periods: int,
    smoothing_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    score_df = pd.DataFrame(index=component_df.index)
    raw_impulse_df = pd.DataFrame(index=component_df.index)

    for spec in available_specs:
        name = str(spec["name"])
        orientation = float(spec.get("orientation", 1.0))
        raw = pd.to_numeric(component_df[name], errors="coerce")

        impulse_21d = safe_pct_change(raw, 21) * orientation
        impulse_63d = safe_pct_change(raw, 63) * orientation
        impulse_126d = safe_pct_change(raw, 126) * orientation

        z_21d = zscore_trailing(impulse_21d, window=z_window, min_periods=min_z_periods)
        z_63d = zscore_trailing(impulse_63d, window=z_window, min_periods=min_z_periods)
        z_126d = zscore_trailing(
            impulse_126d, window=z_window, min_periods=min_z_periods
        )

        score = (0.50 * z_21d + 0.35 * z_63d + 0.15 * z_126d).clip(
            lower=-3.0, upper=3.0
        )
        raw_impulse = 0.60 * impulse_21d + 0.30 * impulse_63d + 0.10 * impulse_126d

        score_df[name] = score
        raw_impulse_df[name] = raw_impulse

    min_sleeves = max(
        4,
        int(
            math.ceil(
                len(set(str(spec.get("category", "Other")) for spec in available_specs))
                * 0.45
            )
        ),
    )
    sleeve_score_df, composite, breadth, coverage = grouped_weighted_composite(
        score_df,
        available_specs,
        group_weights=LIQUIDITY_SLEEVE_WEIGHTS,
        min_groups=min_sleeves,
    )

    if smoothing_window > 1:
        composite = composite.rolling(smoothing_window, min_periods=1).mean()

    return score_df, raw_impulse_df, sleeve_score_df, composite, breadth, coverage


def build_scorecard(
    component_df: pd.DataFrame,
    score_df: pd.DataFrame,
    raw_impulse_df: pd.DataFrame,
    available_specs: List[Dict[str, object]],
) -> pd.DataFrame:
    rows = []

    for spec in available_specs:
        name = str(spec["name"])
        orientation = float(spec.get("orientation", 1.0))
        raw = component_df[name]

        latest_score = latest_valid(score_df[name])
        latest_level = latest_valid(raw)
        latest_21d = latest_valid(safe_pct_change(raw, 21) * orientation)
        latest_63d = latest_valid(safe_pct_change(raw, 63) * orientation)
        latest_126d = latest_valid(safe_pct_change(raw, 126) * orientation)
        latest_raw_impulse = latest_valid(raw_impulse_df[name])

        rows.append(
            {
                "Component": name,
                "Category": spec.get("category", ""),
                "Ticker / Ratio": spec.get("display_ticker", ""),
                "Latest": latest_level,
                "21D Move": latest_21d,
                "63D Move": latest_63d,
                "126D Move": latest_126d,
                "Raw Impulse": latest_raw_impulse,
                "Score": latest_score,
                "Signal": score_to_bucket(latest_score),
                "Weight": float(spec.get("weight", 1.0)),
                "Description": spec.get("description", ""),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values("Score", ascending=False, na_position="last").reset_index(
        drop=True
    )
    return out


def build_category_scores(sleeve_score_df: pd.DataFrame) -> pd.DataFrame:
    if sleeve_score_df.empty:
        return pd.DataFrame()
    rows = []

    for category in sleeve_score_df.columns:
        series = sleeve_score_df[category]
        rows.append(
            {
                "Category": category,
                "Latest Score": latest_valid(series),
                "1W Change": obs_change(series, 5),
                "1M Change": obs_change(series, 21),
                "3M Change": obs_change(series, 63),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("Latest Score", ascending=False)
        .reset_index(drop=True)
    )
    return df


def filter_by_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()

    if lookback == "max":
        return out.copy()

    latest = out.index.max()

    if lookback == "6m":
        start = latest - pd.DateOffset(months=6)
    elif lookback == "1y":
        start = latest - pd.DateOffset(years=1)
    elif lookback == "2y":
        start = latest - pd.DateOffset(years=2)
    elif lookback == "3y":
        start = latest - pd.DateOffset(years=3)
    elif lookback == "5y":
        start = latest - pd.DateOffset(years=5)
    elif lookback == "10y":
        start = latest - pd.DateOffset(years=10)
    else:
        start = out.index.min()

    return out[out.index >= start].copy()


# ============================================================
# HEADER
# ============================================================

render_page_header(
    PageHeader(
        title=TITLE,
        description="Daily traded-liquidity impulse from market ratios, with the Federal Reserve FCI-G as the official financial-conditions overlay.",
        eyebrow="ADFM Liquidity Regimes",
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Liquidity regime monitor for tracking whether credit, equity breadth, speculative beta, banks, crypto, rates, dollar pressure, and volatility are confirming easier or tighter traded liquidity.

        **What this page shows**

        - Yahoo Finance market-implied liquidity composite.
        - Sleeve breadth and sleeve attribution.
        - Fed FCI-G financial-conditions overlay.
        - Component scorecard for what is driving the impulse.

        **Data source**

        - Yahoo Finance adjusted daily prices through `yfinance`.
        - Federal Reserve FCI-G monthly CSVs.
        """
    )

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Display Controls")

    lookback = st.selectbox(
        "Display lookback",
        ["6m", "1y", "2y", "3y", "5y", "10y", "max"],
        index=2,
    )

    yahoo_period = st.selectbox(
        "Yahoo download period",
        ["1y", "2y", "3y", "5y", "10y", "max"],
        index=3,
        help="Use at least 3y if you want stable impulse z-scores.",
    )

    benchmark = st.selectbox(
        "Overlay benchmark",
        ["SPY", "QQQ"],
        index=0,
    )

    z_window = st.number_input(
        "Impulse z-score lookback, trading days",
        min_value=126,
        max_value=1260,
        value=504,
        step=21,
    )

    min_z_periods = st.number_input(
        "Minimum z-score observations",
        min_value=63,
        max_value=756,
        value=252,
        step=21,
    )

    smoothing_window = st.number_input(
        "Composite smoothing, trading days",
        min_value=1,
        max_value=21,
        value=3,
        step=1,
    )

    show_benchmark = st.checkbox("Show benchmark overlay", value=True)
    show_category_chart = st.checkbox("Show category pressure chart", value=True)
    show_component_bars = st.checkbox("Show component score bars", value=True)
    show_fed_fcig = st.checkbox("Show Fed FCI-G overlay", value=True)
    show_scorecard = st.checkbox("Show component scorecard", value=True)
    show_raw_download = st.checkbox("Show download section", value=True)

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)
    st.caption(
        "Composite score = weighted average of component impulse z-scores. Each component uses 21D, 63D, and 126D liquidity moves after direction adjustment."
    )


# ============================================================
# DATA PIPELINE
# ============================================================

tickers = required_tickers(COMPONENTS, CORE_BENCHMARKS)
prices = load_yahoo_prices(tuple(tickers), yahoo_period)

if prices.empty:
    st.error("Yahoo Finance returned no usable price data.")
    st.stop()

component_df, available_specs = build_component_series(prices, COMPONENTS)

if component_df.empty or not available_specs:
    st.error(
        "No usable liquidity components could be built from the Yahoo Finance download."
    )
    st.stop()

(
    score_df,
    raw_impulse_df,
    sleeve_score_df,
    composite,
    breadth_series,
    coverage_series,
) = build_scores(
    component_df=component_df,
    available_specs=available_specs,
    z_window=int(z_window),
    min_z_periods=int(min_z_periods),
    smoothing_window=int(smoothing_window),
)

scorecard = build_scorecard(component_df, score_df, raw_impulse_df, available_specs)
category_scores = build_category_scores(sleeve_score_df)

display_component_df = filter_by_lookback(component_df, lookback)
display_score_df = filter_by_lookback(score_df, lookback)
display_raw_impulse_df = filter_by_lookback(raw_impulse_df, lookback)
display_composite = filter_by_lookback(
    composite.to_frame("Liquidity Composite"), lookback
)["Liquidity Composite"]
display_breadth = filter_by_lookback(breadth_series.to_frame("Breadth"), lookback)[
    "Breadth"
]
display_coverage = filter_by_lookback(coverage_series.to_frame("Coverage"), lookback)[
    "Coverage"
]
display_prices = filter_by_lookback(prices, lookback)

latest_date = (
    display_composite.dropna().index.max()
    if display_composite.notna().any()
    else prices.index.max()
)
latest_score = latest_valid(display_composite)
latest_breadth = latest_valid(display_breadth)
latest_coverage = latest_valid(display_coverage)
regime, regime_description = classify_regime(latest_score, latest_breadth)

composite_1w = obs_change(display_composite, 5)
composite_1m = obs_change(display_composite, 21)
composite_3m = obs_change(display_composite, 63)

available_component_count = len(available_specs)
total_component_count = len(COMPONENTS)


# ============================================================
# SNAPSHOT
# ============================================================

st.markdown(
    "<div class='section-title'>Liquidity Regime Snapshot</div>", unsafe_allow_html=True
)
st.markdown(
    f"<div class='section-subtitle'>Latest Yahoo-derived signal: {pd.Timestamp(latest_date).strftime('%b %d, %Y')}. Positive means easier traded liquidity; negative means tightening pressure.</div>",
    unsafe_allow_html=True,
)

render_kpi_cards(
    [
        ("Regime", regime, regime_description),
        ("Composite", fmt_num(latest_score, 2), "Sleeve-weighted impulse z-score"),
        ("1W change", fmt_delta(composite_1w, 2), signed_signal_word(composite_1w)),
        ("1M change", fmt_delta(composite_1m, 2), signed_signal_word(composite_1m)),
        ("3M change", fmt_delta(composite_3m, 2), signed_signal_word(composite_3m)),
        (
            "Sleeve breadth",
            fmt_pct(latest_breadth, 0),
            f"{fmt_pct(latest_coverage, 0)} coverage · {available_component_count}/{total_component_count} proxies",
        ),
    ]
)

if not category_scores.empty:
    strongest_sleeve = category_scores.iloc[0]
    weakest_sleeve = category_scores.iloc[-1]
    sleeve_read = (
        f"{strongest_sleeve['Category']} is the strongest easing sleeve at {strongest_sleeve['Latest Score']:+.2f}; "
        f"{weakest_sleeve['Category']} is the main tightening sleeve at {weakest_sleeve['Latest Score']:+.2f}."
    )
else:
    sleeve_read = "Sleeve attribution is unavailable with the current data coverage."

render_selection_note(
    "Active liquidity read",
    f"{regime}: {regime_description} {sleeve_read} The composite weights sleeves first, so categories with more available proxies do not dominate by construction.",
)


# ============================================================
# MAIN CHART
# ============================================================

st.markdown(
    "<div class='section-title'>Market-Implied Liquidity Impulse</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-subtitle'>A daily composite of 21D, 63D, and 126D moves across credit, breadth, speculation, banks, crypto, rates, dollar, and volatility. The composite shows traded-liquidity impulse; the benchmark overlay is separately rebased to preserve signal shape.</div>",
    unsafe_allow_html=True,
)

bench_rebased = None
if show_benchmark and benchmark in display_prices.columns:
    bench_rebased = rebase(display_prices[benchmark]).dropna()

trading_index, session_map = build_trading_session_axis(
    display_composite,
    display_breadth,
    bench_rebased,
)
tickvals, ticktext = make_session_ticks(trading_index, max_ticks=10)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.075,
    row_heights=[0.72, 0.28],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    subplot_titles=("Liquidity impulse", "Sleeve breadth"),
)

comp_y_range = visible_axis_range(
    display_composite, floor_abs=1.00, pad=0.10, include_zero=True
)
if comp_y_range is None:
    comp_y0, comp_y1 = dynamic_axis_range(display_composite, floor_abs=1.15, pad=0.16)
else:
    comp_y0, comp_y1 = comp_y_range

fig.add_hrect(
    y0=0.90,
    y1=max(comp_y1, 0.90),
    fillcolor="rgba(5, 95, 70, 0.07)",
    line_width=0,
    row=1,
    col=1,
)

fig.add_hrect(
    y0=min(comp_y0, -0.90),
    y1=-0.90,
    fillcolor="rgba(153, 27, 27, 0.07)",
    line_width=0,
    row=1,
    col=1,
)

fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#8b949e", row=1, col=1)
fig.add_hline(y=0.90, line_width=1, line_dash="dot", line_color="#cbd5e1", row=1, col=1)
fig.add_hline(
    y=-0.90, line_width=1, line_dash="dot", line_color="#cbd5e1", row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=session_x(display_composite.index, session_map),
        y=display_composite,
        customdata=session_dates(display_composite.index),
        name="Liquidity Composite",
        mode="lines",
        line=dict(width=2.9, color="#111827"),
        hovertemplate="%{customdata}<br>Composite: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
    secondary_y=False,
)

if bench_rebased is not None and not bench_rebased.empty:
    fig.add_trace(
        go.Scatter(
            x=session_x(bench_rebased.index, session_map),
            y=bench_rebased,
            customdata=session_dates(bench_rebased.index),
            name=f"{benchmark}, rebased",
            mode="lines",
            line=dict(width=1.9, color="#2563eb"),
            opacity=0.62,
            hovertemplate=f"%{{customdata}}<br>{benchmark}: %{{y:.1f}}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="#8b949e", row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=session_x(display_breadth.index, session_map),
        y=display_breadth,
        customdata=session_dates(display_breadth.index),
        name="% Sleeves Easing",
        mode="lines",
        line=dict(width=2.0, color="#475569"),
        fill="tozeroy",
        opacity=0.85,
        hovertemplate="%{customdata}<br>Sleeves easing: %{y:.0f}%<extra></extra>",
    ),
    row=2,
    col=1,
)

apply_adfm_plot_layout(
    fig,
    height=690,
    margin=dict(l=46, r=58, t=54, b=48),
    showlegend=True,
    legend_y=1.065,
)
clean_axis(fig)
apply_trading_session_xaxis(
    fig,
    trading_index=trading_index,
    tickvals=tickvals,
    ticktext=ticktext,
    rows=[1, 2],
    title_row=2,
)

fig.update_yaxes(
    title_text="Composite z-score",
    range=[comp_y0, comp_y1],
    showgrid=True,
    gridcolor=PLOTLY_GRID,
    zeroline=False,
    row=1,
    col=1,
    secondary_y=False,
)

benchmark_y_range = (
    visible_axis_range(bench_rebased, pad=0.07) if bench_rebased is not None else None
)
fig.update_yaxes(
    title_text=f"{benchmark}, rebased",
    range=benchmark_y_range,
    showgrid=False,
    zeroline=False,
    row=1,
    col=1,
    secondary_y=True,
)

fig.update_yaxes(
    title_text="Breadth",
    range=[0, 100],
    ticksuffix="%",
    showgrid=True,
    gridcolor=PLOTLY_GRID,
    zeroline=False,
    row=2,
    col=1,
)

st.plotly_chart(fig, width="stretch")


# ============================================================
# CATEGORY PRESSURE CHART
# ============================================================

if show_category_chart and not category_scores.empty:
    st.markdown(
        "<div class='section-title'>Liquidity Pressure by Sleeve</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>The composite is only useful if the internals explain it. This table and chart show where the liquidity impulse is coming from.</div>",
        unsafe_allow_html=True,
    )

    cat_display = category_scores.copy()
    cat_display = cat_display.sort_values("Latest Score", ascending=True)

    fig_cat = go.Figure()
    fig_cat.add_vline(x=0, line_width=1, line_dash="dot", line_color="#8b949e")
    fig_cat.add_trace(
        go.Bar(
            x=cat_display["Latest Score"],
            y=cat_display["Category"],
            orientation="h",
            name="Latest Score",
            hovertemplate="%{y}<br>Score: %{x:.2f}<extra></extra>",
        )
    )

    apply_adfm_plot_layout(
        fig_cat,
        height=max(360, 42 * len(cat_display) + 110),
        margin=dict(l=135, r=28, t=24, b=42),
        showlegend=False,
        hovermode="closest",
    )
    clean_axis(fig_cat)

    fig_cat.update_xaxes(title_text="Latest sleeve score", zeroline=False)
    fig_cat.update_yaxes(title_text="", showgrid=False)
    st.plotly_chart(fig_cat, width="stretch")


# ============================================================
# COMPONENT SCORE BARS
# ============================================================

if show_component_bars and not scorecard.empty:
    st.markdown(
        "<div class='section-title'>Component Score Stack</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-subtitle'>Current component scores after direction adjustment. Positive means the component is contributing to easier liquidity; negative means it is tightening the tape.</div>",
        unsafe_allow_html=True,
    )

    bar_df = (
        scorecard[["Component", "Score", "Category", "Ticker / Ratio"]]
        .dropna(subset=["Score"])
        .copy()
    )
    bar_df = bar_df.sort_values("Score", ascending=True)

    fig_bar = go.Figure()
    fig_bar.add_vline(x=0, line_width=1, line_dash="dot", line_color="#8b949e")
    fig_bar.add_trace(
        go.Bar(
            x=bar_df["Score"],
            y=bar_df["Component"],
            orientation="h",
            text=bar_df["Ticker / Ratio"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Score: %{x:.2f}<extra></extra>",
        )
    )

    apply_adfm_plot_layout(
        fig_bar,
        height=max(560, 31 * len(bar_df) + 110),
        margin=dict(l=220, r=118, t=24, b=42),
        showlegend=False,
        hovermode="closest",
    )
    clean_axis(fig_bar)

    fig_bar.update_xaxes(title_text="Component score", zeroline=False)
    fig_bar.update_yaxes(title_text="", showgrid=False)
    st.plotly_chart(fig_bar, width="stretch")


# ============================================================
# FED FCI-G OVERLAY
# ============================================================

if show_fed_fcig:
    fcig_df, fcig_contribs, fcig_errors = load_fed_fcig()

    st.markdown(
        "<div class='section-title'>Official Fed Financial Conditions Overlay</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Federal Reserve FCI-G. Positive FCI-G readings indicate financial conditions are a growth headwind; negative readings indicate a growth tailwind. The Yahoo composite remains a traded-market impulse, not an official macro series.</div>",
        unsafe_allow_html=True,
    )

    if fcig_df.empty:
        st.markdown(
            """
            <div class="warning-box">
            Fed FCI-G data could not be loaded from the Federal Reserve CSV endpoints. The Yahoo liquidity composite above still works independently.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if fcig_errors:
            with st.expander("Fed FCI-G source errors"):
                for key, value in fcig_errors.items():
                    st.write(f"**{key}:** {value}")
    else:
        fcig_plot = filter_by_lookback(fcig_df, lookback)
        composite_monthly = month_end_last(composite.dropna())
        composite_monthly = filter_by_lookback(
            composite_monthly.to_frame("Yahoo Composite"), lookback
        )["Yahoo Composite"]

        fig_fcig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.075,
            row_heights=[0.68, 0.32],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=(
                "Fed FCI-G and Yahoo traded-liquidity impulse",
                "Easing direction comparison",
            ),
        )

        fig_fcig.add_hline(
            y=0, line_width=1, line_dash="dot", line_color="#8b949e", row=1, col=1
        )

        if "FCI-G Baseline" in fcig_plot.columns:
            fig_fcig.add_trace(
                go.Scatter(
                    x=fcig_plot.index,
                    y=fcig_plot["FCI-G Baseline"],
                    name="FCI-G Baseline",
                    mode="lines",
                    line=dict(width=2.7, color="#111827"),
                    hovertemplate="%{x|%Y-%m}<br>FCI-G Baseline: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        if "FCI-G 1Y Lookback" in fcig_plot.columns:
            fig_fcig.add_trace(
                go.Scatter(
                    x=fcig_plot.index,
                    y=fcig_plot["FCI-G 1Y Lookback"],
                    name="FCI-G 1Y Lookback",
                    mode="lines",
                    line=dict(width=2.1, color="#b45309"),
                    opacity=0.82,
                    hovertemplate="%{x|%Y-%m}<br>FCI-G 1Y: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

        if not composite_monthly.empty:
            fig_fcig.add_trace(
                go.Scatter(
                    x=composite_monthly.index,
                    y=composite_monthly,
                    name="Yahoo Composite",
                    mode="lines",
                    line=dict(width=2.0, color="#2563eb"),
                    opacity=0.78,
                    hovertemplate="%{x|%Y-%m}<br>Yahoo Composite: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        if "FCI-G Baseline" in fcig_df.columns and not composite_monthly.empty:
            fed_easing = -fcig_df["FCI-G Baseline"].dropna()
            fed_easing = filter_by_lookback(
                fed_easing.to_frame("Fed Easing Direction"), lookback
            )["Fed Easing Direction"]

            fig_fcig.add_hline(
                y=0, line_width=1, line_dash="dot", line_color="#8b949e", row=2, col=1
            )
            fig_fcig.add_trace(
                go.Scatter(
                    x=fed_easing.index,
                    y=fed_easing,
                    name="Fed FCI-G, inverted",
                    mode="lines",
                    line=dict(width=2.2, color="#0f172a"),
                    hovertemplate="%{x|%Y-%m}<br>Fed easing direction: %{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig_fcig.add_trace(
                go.Scatter(
                    x=composite_monthly.index,
                    y=composite_monthly,
                    name="Yahoo Composite, monthly",
                    mode="lines",
                    line=dict(width=2.0, color="#2563eb"),
                    hovertemplate="%{x|%Y-%m}<br>Yahoo Composite: %{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        apply_adfm_plot_layout(
            fig_fcig,
            height=690,
            margin=dict(l=46, r=58, t=54, b=48),
            showlegend=True,
            legend_y=1.065,
        )
        clean_axis(fig_fcig)

        fig_fcig.update_yaxes(
            title_text="FCI-G growth impulse",
            showgrid=True,
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            row=1,
            col=1,
            secondary_y=False,
        )

        fig_fcig.update_yaxes(
            title_text="Yahoo composite",
            showgrid=False,
            zeroline=False,
            row=1,
            col=1,
            secondary_y=True,
        )

        fig_fcig.update_yaxes(
            title_text="Positive = easier",
            showgrid=True,
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            row=2,
            col=1,
        )

        fig_fcig.update_xaxes(
            tickformat="%b-%y", showgrid=False, row=2, col=1, title_text="Date"
        )

        st.plotly_chart(fig_fcig, width="stretch")

        latest_fcig_row = fcig_df.dropna(how="all").iloc[-1]
        latest_fcig_date = fcig_df.dropna(how="all").index[-1]

        baseline = latest_fcig_row.get("FCI-G Baseline", np.nan)
        one_year = latest_fcig_row.get("FCI-G 1Y Lookback", np.nan)
        fcig_3m_change = (
            obs_change(fcig_df["FCI-G Baseline"], 3)
            if "FCI-G Baseline" in fcig_df.columns
            else np.nan
        )
        official_signal = (
            "Growth headwind"
            if baseline > 0
            else "Growth tailwind"
            if baseline < 0
            else "Neutral"
        )

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            metric_card(
                "FCI-G Baseline",
                fmt_num(baseline, 2),
                f"Latest: {latest_fcig_date:%b %Y}",
            )

        with f2:
            metric_card("FCI-G 1Y", fmt_num(one_year, 2), "Faster lookback window")

        with f3:
            metric_card(
                "3M FCI-G Change",
                fmt_delta(fcig_3m_change, 2),
                "Positive = conditions tightened",
            )

        with f4:
            metric_card(
                "Official Signal", official_signal, "Positive FCI-G = GDP headwind"
            )

        if not fcig_contribs.empty:
            contrib_latest = fcig_contribs.dropna(how="all").iloc[-1].dropna()
            if not contrib_latest.empty:
                st.markdown(
                    "<div class='section-title'>Fed FCI-G Contribution Stack</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='section-subtitle'>Latest baseline FCI-G contribution by input. Positive values tighten financial conditions; negative values ease them.</div>",
                    unsafe_allow_html=True,
                )

                contrib_df = contrib_latest.reset_index()
                contrib_df.columns = ["Input", "Contribution"]
                contrib_df = contrib_df.sort_values("Contribution", ascending=True)

                fig_contrib = go.Figure()
                fig_contrib.add_vline(
                    x=0, line_width=1, line_dash="dot", line_color="#8b949e"
                )
                fig_contrib.add_trace(
                    go.Bar(
                        x=contrib_df["Contribution"],
                        y=contrib_df["Input"],
                        orientation="h",
                        hovertemplate="%{y}<br>Contribution: %{x:.2f}<extra></extra>",
                    )
                )

                apply_adfm_plot_layout(
                    fig_contrib,
                    height=max(340, 38 * len(contrib_df) + 100),
                    margin=dict(l=155, r=35, t=24, b=42),
                    showlegend=False,
                    hovermode="closest",
                )
                clean_axis(fig_contrib)
                fig_contrib.update_xaxes(
                    title_text="FCI-G contribution", zeroline=False
                )
                fig_contrib.update_yaxes(title_text="", showgrid=False)
                st.plotly_chart(fig_contrib, width="stretch")

        if fcig_errors:
            with st.expander("Fed FCI-G source notes"):
                for key, value in fcig_errors.items():
                    st.write(f"**{key}:** {value}")


# ============================================================
# SCORECARD TABLE
# ============================================================

if show_scorecard:
    st.markdown(
        "<div class='section-title'>Component Scorecard</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-subtitle'>Moves are direction-adjusted: positive means easier liquidity for that component; negative means tighter liquidity pressure.</div>",
        unsafe_allow_html=True,
    )

    table = scorecard.copy()
    numeric_cols = [
        "Latest",
        "21D Move",
        "63D Move",
        "126D Move",
        "Raw Impulse",
        "Score",
        "Weight",
    ]
    for col in numeric_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    display_cols = [
        "Component",
        "Category",
        "Ticker / Ratio",
        "21D Move",
        "63D Move",
        "126D Move",
        "Score",
        "Signal",
        "Weight",
        "Description",
    ]
    display_cols = [col for col in display_cols if col in table.columns]

    styled = (
        table[display_cols]
        .style.format(
            {
                "21D Move": "{:+.1f}%",
                "63D Move": "{:+.1f}%",
                "126D Move": "{:+.1f}%",
                "Score": "{:+.2f}",
                "Weight": "{:.2f}",
            },
            na_rep="N/A",
        )
        .map(color_score, subset=["Score"] if "Score" in display_cols else None)
    )

    st.dataframe(styled, width="stretch", height=560)


# ============================================================
# DOWNLOAD AND METHODOLOGY
# ============================================================

if show_raw_download:
    left, right = st.columns([1.10, 0.90])

    with left:
        st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Export the composite, breadth, component scores, and raw component series.</div>",
            unsafe_allow_html=True,
        )

        export = pd.concat(
            [
                composite.rename("Liquidity Composite"),
                breadth_series.rename("Sleeve Breadth"),
                coverage_series.rename("Sleeve Coverage"),
                sleeve_score_df.add_prefix("Sleeve Score | "),
                score_df.add_prefix("Score | "),
                raw_impulse_df.add_prefix("Raw Impulse | "),
                component_df.add_prefix("Series | "),
            ],
            axis=1,
        ).sort_index()
        export.index.name = "Date"

        csv = export.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Download Liquidity Monitor CSV",
            data=csv,
            file_name="adfm_liquidity_conditions_monitor.csv",
            mime="text/csv",
            width="stretch",
        )

    with right:
        st.markdown(
            "<div class='section-title'>Methodology</div>", unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="info-box">
            <b>Market-implied signal</b><br>
            Each Yahoo component is converted into a direction-adjusted liquidity move. For ratios, the numerator is divided by the denominator. For pressure variables like UUP and VIX, the sign is inverted because lower dollar pressure and lower volatility are liquidity-positive.
            <br><br>
            <b>Composite construction</b><br>
            Component score = 50% of 21D move z-score + 35% of 63D move z-score + 15% of 126D move z-score. Component weights are normalized inside each sleeve, then explicit sleeve weights are applied. This prevents proxy-rich sleeves such as credit or speculation from receiving an accidental extra vote.
            <br><br>
            <b>Fed overlay</b><br>
            FCI-G is official Federal Reserve data. Positive FCI-G means financial conditions are a headwind to growth; negative FCI-G means they are a tailwind. The Yahoo composite is faster and market-implied; FCI-G is slower and macro-official.
            <br><br>
            <b>Important limitation</b><br>
            This is not Fed net liquidity. It is a traded-market liquidity impulse. That distinction is intentional.
            </div>
            """,
            unsafe_allow_html=True,
        )

render_footer()
