import math
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


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
        padding-top: 0.95rem !important;
        padding-bottom: 2rem;
        max-width: 1740px;
    }

    .adfm-header-wrap {
        margin-top: 0.10rem;
        margin-bottom: 1.10rem;
        padding-bottom: 0.15rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .adfm-title {
        font-size: 1.82rem;
        line-height: 1.20;
        font-weight: 780;
        color: #111827;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.025em;
    }

    .adfm-subtitle {
        font-size: 0.93rem;
        line-height: 1.42;
        color: #6b7280;
        margin: 0 0 0.80rem 0;
        max-width: 1180px;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 760;
        color: #111827;
        margin-top: 0.35rem;
        margin-bottom: 0.20rem;
    }

    .section-subtitle {
        font-size: 0.86rem;
        color: #6b7280;
        margin-bottom: 0.72rem;
        line-height: 1.38;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        padding: 12px 14px 10px 14px;
        min-height: 88px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.035);
    }

    .metric-label {
        font-size: 0.70rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.052em;
        margin-bottom: 0.42rem;
        white-space: nowrap;
    }

    .metric-value {
        font-size: 1.28rem;
        font-weight: 760;
        color: #111827;
        line-height: 1.10;
        white-space: nowrap;
    }

    .metric-footnote {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 0.40rem;
        line-height: 1.22;
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        padding: 13px 15px;
        margin-bottom: 0.85rem;
        color: #111827;
        font-size: 0.90rem;
        line-height: 1.42;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 13px;
        padding: 12px 14px;
        margin-bottom: 0.85rem;
        color: #7c2d12;
        font-size: 0.89rem;
        line-height: 1.40;
    }

    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 650;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        padding: 10px 12px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# FED FCI-G DATA
# ============================================================

FED_FCIG_SOURCES = {
    "FCI-G Baseline": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_3yr.csv",
    "FCI-G 1Y Lookback": "https://www.federalreserve.gov/econres/notes/feds-notes/fci_g_public_monthly_1yr.csv",
}


# ============================================================
# YAHOO LIQUIDITY COMPONENTS
# ============================================================

COMPONENTS = [
    {
        "name": "HY Credit / IG Credit",
        "category": "Credit",
        "numerator": "HYG",
        "denominator": "LQD",
        "orientation": 1,
        "weight": 1.25,
        "description": "High yield outperforming investment grade means credit risk appetite is improving.",
    },
    {
        "name": "Junk Credit / IG Credit",
        "category": "Credit",
        "numerator": "JNK",
        "denominator": "LQD",
        "orientation": 1,
        "weight": 1.00,
        "description": "Second credit confirmation line; useful when HYG is distorted by ETF flows.",
    },
    {
        "name": "Small Caps / S&P 500",
        "category": "Equity Breadth",
        "numerator": "IWM",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 1.00,
        "description": "Small-cap participation is a liquidity and domestic cyclicality check.",
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
        "description": "One of the cleaner animal-spirits ratios.",
    },
    {
        "name": "Regional Banks / S&P 500",
        "category": "Funding",
        "numerator": "KRE",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 1.00,
        "description": "Bank equity confirmation for funding and credit creation.",
    },
    {
        "name": "Financials / S&P 500",
        "category": "Funding",
        "numerator": "XLF",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.75,
        "description": "Broad financials confirmation.",
    },
    {
        "name": "Semis / Nasdaq",
        "category": "Leadership",
        "numerator": "SMH",
        "denominator": "QQQ",
        "orientation": 1,
        "weight": 0.85,
        "description": "AI and capex leadership relative to Nasdaq beta.",
    },
    {
        "name": "NVDA / Semis",
        "category": "Leadership",
        "numerator": "NVDA",
        "denominator": "SMH",
        "orientation": 1,
        "weight": 0.45,
        "description": "Leadership concentration and AI reflexivity check. Lower weight by design.",
    },
    {
        "name": "Bitcoin ETF / S&P 500",
        "category": "Crypto",
        "numerator": "IBIT",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.80,
        "description": "Crypto ETF beta relative to broad equities.",
    },
    {
        "name": "Bitcoin / S&P 500",
        "category": "Crypto",
        "numerator": "BTC-USD",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.80,
        "description": "Native crypto liquidity impulse relative to equities.",
    },
    {
        "name": "Emerging Markets / S&P 500",
        "category": "Global Dollar Liquidity",
        "numerator": "EEM",
        "denominator": "SPY",
        "orientation": 1,
        "weight": 0.90,
        "description": "Global dollar-liquidity sensitivity.",
    },
    {
        "name": "Long Duration / Cash Proxy",
        "category": "Rates",
        "numerator": "TLT",
        "denominator": "SHY",
        "orientation": 1,
        "weight": 0.80,
        "description": "Long-duration bid versus short-duration cash proxy.",
    },
    {
        "name": "Intermediate Duration / Cash Proxy",
        "category": "Rates",
        "numerator": "IEF",
        "denominator": "SHY",
        "orientation": 1,
        "weight": 0.65,
        "description": "Less volatile duration confirmation.",
    },
    {
        "name": "Dollar Pressure",
        "category": "Dollar",
        "ticker": "UUP",
        "orientation": -1,
        "weight": 1.10,
        "description": "Lower dollar pressure is positive for global liquidity.",
    },
    {
        "name": "Volatility Pressure",
        "category": "Volatility",
        "ticker": "^VIX",
        "orientation": -1,
        "weight": 1.25,
        "description": "Lower volatility mechanically eases risk budgets.",
    },
]

CORE_BENCHMARKS = ["SPY", "QQQ"]


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


def classify_regime(score: float) -> Tuple[str, str]:
    if pd.isna(score):
        return "Unavailable", "Insufficient component coverage"

    if score >= 0.75:
        return "Liquidity Expansion", "Risk appetite, breadth, credit, dollar, and vol are broadly confirming easier traded liquidity."

    if score >= 0.25:
        return "Improving", "Liquidity impulse is improving, but confirmation is not broad enough to call full expansion."

    if score > -0.25:
        return "Neutral / Mixed", "Signals are cross-current. The tape is not giving a clean liquidity message."

    if score > -0.75:
        return "Deteriorating", "The liquidity impulse is weakening. Watch credit, dollar, volatility, and breadth confirmation."

    return "Liquidity Contraction", "Risk budgets are tightening across the market-implied liquidity stack."


def regime_color(score: float) -> str:
    if pd.isna(score):
        return "#6b7280"
    if score >= 0.75:
        return "#065f46"
    if score >= 0.25:
        return "#047857"
    if score > -0.25:
        return "#6b7280"
    if score > -0.75:
        return "#b45309"
    return "#991b1b"


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


def score_to_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score >= 0.75:
        return "Strong easing"
    if score >= 0.25:
        return "Easing"
    if score > -0.25:
        return "Mixed"
    if score > -0.75:
        return "Tightening"
    return "Strong tightening"


def color_score(val: object) -> str:
    try:
        x = float(val)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    if x >= 0.75:
        return "background-color: #dcfce7; color: #14532d;"
    if x >= 0.25:
        return "background-color: #ecfdf5; color: #065f46;"
    if x > -0.25:
        return "background-color: #f9fafb; color: #374151;"
    if x > -0.75:
        return "background-color: #fff7ed; color: #9a3412;"
    return "background-color: #fee2e2; color: #7f1d1d;"


# ============================================================
# DATA LOADERS
# ============================================================

def required_tickers(components: List[Dict[str, object]], benchmarks: List[str]) -> List[str]:
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
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            close = df[numeric_cols].copy()

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.columns = [str(c).strip() for c in close.columns]
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
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

    available = [c for c in close.columns if close[c].notna().sum() > 20]
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

    priority_terms = ["fci-g", "fcig", "fci_g", "fci g", "fci"]

    for term in priority_terms:
        for col in numeric_cols:
            lower = str(col).lower()
            if term in lower and "contribution" not in lower and "cont" not in lower:
                return col

    return numeric_cols[0]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_fed_fcig() -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    errors: Dict[str, str] = {}

    headers = {
        "User-Agent": "ADFM-Liquidity-Conditions-Monitor/1.0",
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

        except Exception as exc:
            errors[label] = str(exc)

    if not frames:
        return pd.DataFrame(), errors

    fcig = pd.concat(frames, axis=1).sort_index()
    fcig = fcig[~fcig.index.duplicated(keep="last")]

    for col in fcig.columns:
        fcig[col] = pd.to_numeric(fcig[col], errors="coerce")

    fcig = fcig.dropna(how="all")

    return fcig, errors


# ============================================================
# SIGNAL ENGINE
# ============================================================

def build_component_series(prices: pd.DataFrame, components: List[Dict[str, object]]) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
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

        if s.dropna().shape[0] < 60:
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
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    score_df = pd.DataFrame(index=component_df.index)
    weighted_scores = pd.DataFrame(index=component_df.index)
    weights = {}

    for spec in available_specs:
        name = str(spec["name"])
        orientation = float(spec.get("orientation", 1.0))
        weight = float(spec.get("weight", 1.0))

        z = zscore_trailing(component_df[name], window=z_window, min_periods=min_z_periods)
        score = (z * orientation).clip(lower=-3.0, upper=3.0)

        score_df[name] = score
        weighted_scores[name] = score * weight
        weights[name] = weight

    weights_series = pd.Series(weights)

    def weighted_mean(row: pd.Series) -> float:
        valid = row.dropna()

        if valid.empty:
            return np.nan

        active_weights = weights_series.loc[valid.index]
        denom = active_weights.sum()

        if denom == 0:
            return np.nan

        return float((valid * active_weights).sum() / denom)

    min_components = max(4, int(math.ceil(len(available_specs) * 0.45)))
    valid_count = score_df.notna().sum(axis=1)

    composite = score_df.apply(weighted_mean, axis=1)
    composite = composite.where(valid_count >= min_components)

    if smoothing_window > 1:
        composite = composite.rolling(smoothing_window, min_periods=1).mean()

    return score_df, composite, weighted_scores


def build_scorecard(
    component_df: pd.DataFrame,
    score_df: pd.DataFrame,
    available_specs: List[Dict[str, object]],
) -> pd.DataFrame:
    rows = []

    for spec in available_specs:
        name = str(spec["name"])
        orientation = float(spec.get("orientation", 1.0))
        raw = component_df[name]
        adj_21d = safe_pct_change(raw, 21) * orientation
        adj_63d = safe_pct_change(raw, 63) * orientation

        latest_score = latest_valid(score_df[name])
        latest_level = latest_valid(raw)
        latest_21d = latest_valid(adj_21d)
        latest_63d = latest_valid(adj_63d)

        rows.append(
            {
                "Component": name,
                "Category": spec.get("category", ""),
                "Ticker / Ratio": spec.get("display_ticker", ""),
                "Latest": latest_level,
                "21D Liquidity Move": latest_21d,
                "63D Liquidity Move": latest_63d,
                "Score": latest_score,
                "Signal": score_to_bucket(latest_score),
                "Weight": float(spec.get("weight", 1.0)),
                "Description": spec.get("description", ""),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)
    return out


def build_category_scores(score_df: pd.DataFrame, available_specs: List[Dict[str, object]]) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()

    name_to_category = {str(spec["name"]): str(spec.get("category", "Other")) for spec in available_specs}
    rows = []

    for category in sorted(set(name_to_category.values())):
        names = [name for name, cat in name_to_category.items() if cat == category and name in score_df.columns]

        if not names:
            continue

        series = score_df[names].mean(axis=1)
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

    df = pd.DataFrame(rows).sort_values("Latest Score", ascending=False).reset_index(drop=True)
    return df


def filter_by_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    if df.empty:
        return df

    if lookback == "max":
        return df.copy()

    latest = df.index.max()

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
        start = df.index.min()

    return df[df.index >= start].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="adfm-header-wrap">
        <div class="adfm-title">{TITLE}</div>
        <div class="adfm-subtitle">
            Daily market-implied liquidity impulse from Yahoo Finance ratios, with the Federal Reserve FCI-G as the official financial-conditions overlay.
            Positive composite readings indicate easier traded liquidity; negative readings indicate tightening pressure.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Framework")
    st.markdown(
        """
This page is not a Fed balance-sheet scraper.

It uses Yahoo Finance to track whether liquidity is actually reaching traded markets through credit, breadth, high beta, crypto, dollar pressure, volatility, duration, and banks.

The Fed FCI-G overlay is pulled directly from the Federal Reserve's public monthly CSVs.
        """
    )

    st.divider()
    st.markdown("### Controls")

    lookback = st.selectbox(
        "Display lookback",
        ["6m", "1y", "2y", "3y", "5y", "10y", "max"],
        index=2,
    )

    yahoo_period = st.selectbox(
        "Yahoo download period",
        ["1y", "2y", "3y", "5y", "10y", "max"],
        index=3,
        help="Use at least 3y if you want stable z-scores.",
    )

    benchmark = st.selectbox(
        "Overlay benchmark",
        ["SPY", "QQQ"],
        index=0,
    )

    z_window = st.number_input(
        "Z-score lookback, trading days",
        min_value=126,
        max_value=1260,
        value=504,
        step=21,
    )

    min_z_periods = st.number_input(
        "Minimum z-score observations",
        min_value=63,
        max_value=504,
        value=126,
        step=21,
    )

    smoothing_window = st.number_input(
        "Composite smoothing, trading days",
        min_value=1,
        max_value=21,
        value=5,
        step=1,
    )

    st.divider()
    st.markdown("### Display")

    show_fed_fcig = st.checkbox("Show Fed FCI-G overlay", value=True)
    show_category_scores = st.checkbox("Show category pressure map", value=True)
    show_raw_components = st.checkbox("Show component table", value=True)
    show_download = st.checkbox("Show download buttons", value=True)

    st.divider()
    st.caption("Data: Yahoo Finance via yfinance; Federal Reserve FCI-G monthly CSVs.")


# ============================================================
# LOAD DATA
# ============================================================

tickers = required_tickers(COMPONENTS, CORE_BENCHMARKS)
prices = load_yahoo_prices(tuple(tickers), yahoo_period)

if prices.empty:
    st.error("Yahoo Finance returned no usable price data.")
    st.stop()

component_df, available_specs = build_component_series(prices, COMPONENTS)

if component_df.empty or not available_specs:
    st.error("No usable liquidity components could be built from the Yahoo Finance download.")
    st.stop()

score_df, composite, weighted_scores = build_scores(
    component_df=component_df,
    available_specs=available_specs,
    z_window=int(z_window),
    min_z_periods=int(min_z_periods),
    smoothing_window=int(smoothing_window),
)

scorecard = build_scorecard(component_df, score_df, available_specs)
category_scores = build_category_scores(score_df, available_specs)

display_component_df = filter_by_lookback(component_df, lookback)
display_score_df = filter_by_lookback(score_df, lookback)
display_composite = filter_by_lookback(composite.to_frame("Liquidity Composite"), lookback)["Liquidity Composite"]

display_prices = filter_by_lookback(prices, lookback)

latest_date = display_composite.dropna().index.max() if display_composite.notna().any() else prices.index.max()
latest_score = latest_valid(display_composite)
regime, regime_description = classify_regime(latest_score)

composite_1w = obs_change(display_composite, 5)
composite_1m = obs_change(display_composite, 21)
composite_3m = obs_change(display_composite, 63)

latest_breadth = np.nan
breadth_series = pd.Series(index=score_df.index, dtype="float64")

if not score_df.empty:
    positive_count = (score_df > 0).sum(axis=1)
    valid_count = score_df.notna().sum(axis=1)
    breadth_series = positive_count / valid_count.replace(0, np.nan) * 100.0
    latest_breadth = latest_valid(filter_by_lookback(breadth_series.to_frame("Breadth"), lookback)["Breadth"])

available_component_count = len(available_specs)
total_component_count = len(COMPONENTS)


# ============================================================
# SNAPSHOT
# ============================================================

st.markdown("<div class='section-title'>Liquidity Regime Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='section-subtitle'>Latest Yahoo-derived signal: {pd.Timestamp(latest_date).strftime('%b %d, %Y')}. Composite is a weighted average of trailing z-scores after direction adjustment.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    metric_card("Regime", regime, regime_description)

with c2:
    metric_card("Composite", fmt_num(latest_score, 2), "Positive = easier traded liquidity")

with c3:
    metric_card("1W Change", fmt_delta(composite_1w, 2), signed_signal_word(composite_1w))

with c4:
    metric_card("1M Change", fmt_delta(composite_1m, 2), signed_signal_word(composite_1m))

with c5:
    metric_card("3M Change", fmt_delta(composite_3m, 2), signed_signal_word(composite_3m))

with c6:
    metric_card("Signal Breadth", fmt_pct(latest_breadth, 0), f"{available_component_count}/{total_component_count} components active")


# ============================================================
# MAIN CHART
# ============================================================

st.markdown("<div class='section-title'>Yahoo Market-Implied Liquidity Composite</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Credit, breadth, speculation, banks, crypto, dollar pressure, volatility, and duration compressed into one daily liquidity impulse.</div>",
    unsafe_allow_html=True,
)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_hrect(
    y0=0.75,
    y1=3.0,
    fillcolor="rgba(5, 95, 70, 0.07)",
    line_width=0
)

fig.add_hrect(
    y0=-3.0,
    y1=-0.75,
    fillcolor="rgba(153, 27, 27, 0.07)",
    line_width=0
)

fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af")
fig.add_hline(y=0.75, line_width=1, line_dash="dot", line_color="#d1d5db")
fig.add_hline(y=-0.75, line_width=1, line_dash="dot", line_color="#d1d5db")

fig.add_trace(
    go.Scatter(
        x=display_composite.index,
        y=display_composite,
        name="Liquidity Composite",
        mode="lines",
        line=dict(width=2.8, color="#111827"),
        hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.2f}<extra></extra>",
    )
)

if benchmark in display_prices.columns:
    bench_rebased = rebase(display_prices[benchmark])

    fig.add_trace(
        go.Scatter(
            x=bench_rebased.index,
            y=bench_rebased,
            name=f"{benchmark}, rebased",
            mode="lines",
            line=dict(width=1.9, color="#2563eb"),
            opacity=0.72,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark}: %{{y:.1f}}<extra></extra>",
        ),
        secondary_y=True,
    )

fig.update_layout(
    template="plotly_white",
    height=560,
    margin=dict(l=55, r=55, t=25, b=35),
    legend=dict(
        orientation="h",
        x=0,
        y=1.08,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.78)",
    ),
    hovermode="x unified",
)

fig.update_yaxes(
    title_text="Liquidity z-score",
    range=[-3.0, 3.0],
    showgrid=True,
    gridcolor="#edf0f4",
    zeroline=False
)

fig.update_yaxes(
    title_text=f"{benchmark}, rebased",
    showgrid=False,
    zeroline=False,
    secondary_y=True,
)

fig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date")

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# FED FCI-G OVERLAY
# ============================================================

if show_fed_fcig:
    fcig_df, fcig_errors = load_fed_fcig()

    st.markdown("<div class='section-title'>Official Fed Financial Conditions Overlay</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Federal Reserve FCI-G. Positive readings indicate financial conditions are a headwind to future GDP growth; negative readings indicate a tailwind.</div>",
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

        fig_fcig = make_subplots(specs=[[{"secondary_y": True}]])

        fig_fcig.add_hline(
            y=0,
            line_width=1,
            line_dash="dot",
            line_color="#9ca3af"
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
                secondary_y=False,
            )

        if "FCI-G 1Y Lookback" in fcig_plot.columns:
            fig_fcig.add_trace(
                go.Scatter(
                    x=fcig_plot.index,
                    y=fcig_plot["FCI-G 1Y Lookback"],
                    name="FCI-G 1Y Lookback",
                    mode="lines",
                    line=dict(width=2.2, color="#b45309"),
                    hovertemplate="%{x|%Y-%m}<br>FCI-G 1Y: %{y:.2f}<extra></extra>",
                ),
                secondary_y=False,
            )

        composite_monthly = composite.dropna().resample("M").last()

        if not composite_monthly.empty:
            composite_monthly = filter_by_lookback(composite_monthly.to_frame("Yahoo Composite"), lookback)["Yahoo Composite"]

            fig_fcig.add_trace(
                go.Scatter(
                    x=composite_monthly.index,
                    y=composite_monthly,
                    name="Yahoo Liquidity Composite",
                    mode="lines",
                    line=dict(width=2.0, color="#2563eb"),
                    opacity=0.76,
                    hovertemplate="%{x|%Y-%m}<br>Yahoo Composite: %{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

        fig_fcig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=55, r=55, t=25, b=35),
            legend=dict(
                orientation="h",
                x=0,
                y=1.08,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.78)",
            ),
            hovermode="x unified",
        )

        fig_fcig.update_yaxes(
            title_text="FCI-G growth impulse",
            showgrid=True,
            gridcolor="#edf0f4",
            zeroline=False
        )

        fig_fcig.update_yaxes(
            title_text="Yahoo composite",
            showgrid=False,
            zeroline=False,
            secondary_y=True,
        )

        fig_fcig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date")

        st.plotly_chart(fig_fcig, use_container_width=True)

        latest_fcig_row = fcig_df.dropna(how="all").iloc[-1]
        latest_fcig_date = fcig_df.dropna(how="all").index[-1]

        f1, f2, f3, f4 = st.columns(4)

        baseline = latest_fcig_row.get("FCI-G Baseline", np.nan)
        one_year = latest_fcig_row.get("FCI-G 1Y Lookback", np.nan)

        with f1:
            metric_card("FCI-G Baseline", fmt_num(baseline, 2), f"Latest: {latest_fcig_date:%b %Y}")

        with f2:
            metric_card("FCI-G 1Y", fmt_num(one_year, 2), "Faster lookback window")

        with f3:
            fed_signal = "Growth headwind" if pd.notna(baseline) and baseline > 0 else "Growth tailwind"
            metric_card("Fed Signal", fed_signal, "Positive = tighter conditions")

        with f4:
            spread = latest_score - (-baseline if pd.notna(baseline) else np.nan)
            metric_card("Tape vs Fed Gap", fmt_delta(spread, 2), "Yahoo composite less inverted FCI-G")


# ============================================================
# CATEGORY MAP
# ============================================================

if show_category_scores and not category_scores.empty:
    st.markdown("<div class='section-title'>Category Pressure Map</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Latest score by liquidity sleeve. Positive means easing pressure; negative means tightening pressure.</div>",
        unsafe_allow_html=True,
    )

    fig_cat = go.Figure()

    fig_cat.add_trace(
        go.Bar(
            x=category_scores["Latest Score"],
            y=category_scores["Category"],
            orientation="h",
            name="Latest Score",
            hovertemplate="%{y}<br>Score: %{x:.2f}<extra></extra>",
        )
    )

    fig_cat.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
        line_color="#9ca3af",
    )

    fig_cat.update_layout(
        template="plotly_white",
        height=max(360, 48 * len(category_scores)),
        margin=dict(l=130, r=25, t=20, b=35),
        showlegend=False,
    )

    fig_cat.update_xaxes(
        title_text="Liquidity score",
        showgrid=True,
        gridcolor="#edf0f4",
        zeroline=False,
        range=[
            min(-2.0, float(category_scores["Latest Score"].min()) - 0.25),
            max(2.0, float(category_scores["Latest Score"].max()) + 0.25),
        ],
    )

    fig_cat.update_yaxes(title_text="")

    st.plotly_chart(fig_cat, use_container_width=True)


# ============================================================
# COMPONENT SCORECARD
# ============================================================

if show_raw_components:
    st.markdown("<div class='section-title'>Component Scorecard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Sorted by latest liquidity score. 21D and 63D moves are direction-adjusted: positive means easier liquidity pressure.</div>",
        unsafe_allow_html=True,
    )

    display_cols = [
        "Component",
        "Category",
        "Ticker / Ratio",
        "Latest",
        "21D Liquidity Move",
        "63D Liquidity Move",
        "Score",
        "Signal",
        "Weight",
        "Description",
    ]

    display_scorecard = scorecard[display_cols].copy()

    numeric_cols = ["Latest", "21D Liquidity Move", "63D Liquidity Move", "Score", "Weight"]

    for col in numeric_cols:
        display_scorecard[col] = pd.to_numeric(display_scorecard[col], errors="coerce")

    styled = (
        display_scorecard.style
        .format(
            {
                "Latest": "{:.3f}",
                "21D Liquidity Move": "{:+.1f}%",
                "63D Liquidity Move": "{:+.1f}%",
                "Score": "{:+.2f}",
                "Weight": "{:.2f}",
            },
            na_rep="N/A",
        )
        .applymap(color_score, subset=["Score"])
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=520,
        hide_index=True,
    )


# ============================================================
# DOWNLOADS
# ============================================================

if show_download:
    st.markdown("<div class='section-title'>Downloads</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the series behind the dashboard.</div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    composite_export = pd.DataFrame(
        {
            "Liquidity Composite": composite,
            "Signal Breadth": breadth_series,
        }
    )
    composite_export.index.name = "Date"

    components_export = component_df.copy()
    components_export.index.name = "Date"

    scores_export = score_df.copy()
    scores_export.index.name = "Date"

    with d1:
        st.download_button(
            "Download Composite CSV",
            data=composite_export.to_csv(index=True).encode("utf-8"),
            file_name="adfm_liquidity_composite.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download Component Levels CSV",
            data=components_export.to_csv(index=True).encode("utf-8"),
            file_name="adfm_liquidity_component_levels.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            "Download Component Scores CSV",
            data=scores_export.to_csv(index=True).encode("utf-8"),
            file_name="adfm_liquidity_component_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("Methodology", expanded=False):
    st.markdown(
        """
**Market-implied liquidity composite**

Each component is converted into a trailing z-score. The sign is adjusted so that positive always means easier liquidity pressure and negative always means tighter liquidity pressure. The final composite is a weighted average across available components.

**Why this is different from a Fed balance-sheet tracker**

This is not trying to replicate reserve balances, the Treasury General Account, or reverse repos. It tracks whether liquidity is reaching the tape through credit, volatility, dollar pressure, equity breadth, banks, crypto, duration, and high-beta participation.

**Fed FCI-G overlay**

The Fed FCI-G is shown separately because it measures the effect of financial conditions on future growth. Positive FCI-G readings imply a growth headwind from tighter conditions. Negative readings imply a tailwind.

**Interpretation**

The cleanest easing regime is when the Yahoo composite is rising while FCI-G is falling. The most dangerous regime is when the Yahoo composite rolls over while FCI-G is still positive, because the tape is losing liquidity confirmation before the official macro drag has cleared.
        """
    )

st.caption("© 2026 AD Fund Management LP")
