# sp_subsector_rotation_monitor_v10.py
# ADFM | S&P 500 Sector + Subsector Breadth & Rotation Monitor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="S&P 500 Sector + Subsector Breadth & Rotation Monitor",
    layout="wide",
)

st.title("S&P 500 Sector + Subsector Breadth & Rotation Monitor")


# =============================================================================
# Constants
# =============================================================================

MAJOR_SECTORS: Dict[str, str] = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

SUBSECTOR_ROWS: List[Dict[str, str]] = [
    # Communication Services
    {"Ticker": "IYZ", "Name": "Telecom", "Sector Group": "Communication Services", "Tier": "Core"},
    {"Ticker": "FDN", "Name": "Internet / platform growth", "Sector Group": "Communication Services", "Tier": "Core"},
    {"Ticker": "PBS", "Name": "Media / entertainment", "Sector Group": "Communication Services", "Tier": "Core"},
    {"Ticker": "SOCL", "Name": "Social media / online networks", "Sector Group": "Communication Services", "Tier": "Thematic"},
    {"Ticker": "ESPO", "Name": "Video games / esports", "Sector Group": "Communication Services", "Tier": "Thematic"},

    # Consumer Discretionary
    {"Ticker": "XRT", "Name": "Retail", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "RTH", "Name": "Large-cap retail", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "XHB", "Name": "Homebuilders", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "ITB", "Name": "Home construction", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "JETS", "Name": "Airlines", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "PEJ", "Name": "Travel / leisure", "Sector Group": "Consumer Discretionary", "Tier": "Core"},
    {"Ticker": "CARZ", "Name": "Autos", "Sector Group": "Consumer Discretionary", "Tier": "Thematic"},
    {"Ticker": "IBUY", "Name": "Online retail / e-commerce", "Sector Group": "Consumer Discretionary", "Tier": "Thematic"},
    {"Ticker": "BJK", "Name": "Gaming / casinos", "Sector Group": "Consumer Discretionary", "Tier": "Thematic"},

    # Consumer Staples
    {"Ticker": "PBJ", "Name": "Food & beverage", "Sector Group": "Consumer Staples", "Tier": "Core"},
    {"Ticker": "RHS", "Name": "Equal-weight staples", "Sector Group": "Consumer Staples", "Tier": "Core"},
    {"Ticker": "IYK", "Name": "Broad staples confirmation", "Sector Group": "Consumer Staples", "Tier": "Core"},
    {"Ticker": "MOO", "Name": "Agribusiness / food supply chain", "Sector Group": "Consumer Staples", "Tier": "Thematic"},

    # Energy
    {"Ticker": "XOP", "Name": "E&P", "Sector Group": "Energy", "Tier": "Core"},
    {"Ticker": "OIH", "Name": "Oil services", "Sector Group": "Energy", "Tier": "Core"},
    {"Ticker": "XES", "Name": "Oil equipment & services equal-weight", "Sector Group": "Energy", "Tier": "Core"},
    {"Ticker": "FCG", "Name": "Natural gas equities", "Sector Group": "Energy", "Tier": "Core"},
    {"Ticker": "AMLP", "Name": "Midstream / MLPs", "Sector Group": "Energy", "Tier": "Core"},
    {"Ticker": "ICLN", "Name": "Clean energy", "Sector Group": "Energy", "Tier": "Thematic"},
    {"Ticker": "TAN", "Name": "Solar", "Sector Group": "Energy", "Tier": "Thematic"},
    {"Ticker": "URA", "Name": "Uranium / nuclear fuel cycle", "Sector Group": "Energy", "Tier": "Thematic"},

    # Financials
    {"Ticker": "KBE", "Name": "Banks", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "KRE", "Name": "Regional banks", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "KBWB", "Name": "Money-center banks", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "KIE", "Name": "Insurance", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "KCE", "Name": "Capital markets / brokers", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "IAI", "Name": "Broker-dealers / investment banks", "Sector Group": "Financials", "Tier": "Core"},
    {"Ticker": "BIZD", "Name": "BDCs / private credit beta", "Sector Group": "Financials", "Tier": "Thematic"},
    {"Ticker": "PSP", "Name": "Private equity / alt managers", "Sector Group": "Financials", "Tier": "Thematic"},
    {"Ticker": "FINX", "Name": "Fintech", "Sector Group": "Financials", "Tier": "Thematic"},

    # Health Care
    {"Ticker": "XBI", "Name": "Biotech equal-weight", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "IBB", "Name": "Biotech cap-weight", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "XPH", "Name": "Pharmaceuticals", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "IHE", "Name": "Large pharma", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "IHI", "Name": "Medical devices", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "XHE", "Name": "Health care equipment equal-weight", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "IHF", "Name": "Health care providers", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "XHS", "Name": "Health care services", "Sector Group": "Health Care", "Tier": "Core"},
    {"Ticker": "ARKG", "Name": "Genomics / speculative health innovation", "Sector Group": "Health Care", "Tier": "Thematic"},

    # Industrials
    {"Ticker": "ITA", "Name": "Aerospace & defense", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "XAR", "Name": "Aerospace & defense equal-weight", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "PPA", "Name": "Defense primes / contractor basket", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "IYT", "Name": "Transportation", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "XTN", "Name": "Transportation equal-weight", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "PAVE", "Name": "Infrastructure", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "AIRR", "Name": "Small/mid industrial cyclicals", "Sector Group": "Industrials", "Tier": "Core"},
    {"Ticker": "PHO", "Name": "Water infrastructure", "Sector Group": "Industrials", "Tier": "Thematic"},
    {"Ticker": "BOTZ", "Name": "Robotics / automation", "Sector Group": "Industrials", "Tier": "Thematic"},

    # Materials
    {"Ticker": "XME", "Name": "Metals & mining", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "SLX", "Name": "Steel", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "COPX", "Name": "Copper miners", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "PICK", "Name": "Broad global mining", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "GDX", "Name": "Gold miners", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "GDXJ", "Name": "Junior gold miners", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "SIL", "Name": "Silver miners", "Sector Group": "Materials", "Tier": "Core"},
    {"Ticker": "LIT", "Name": "Lithium / battery materials", "Sector Group": "Materials", "Tier": "Thematic"},
    {"Ticker": "REMX", "Name": "Rare earths / strategic materials", "Sector Group": "Materials", "Tier": "Thematic"},
    {"Ticker": "WOOD", "Name": "Timber / forest products", "Sector Group": "Materials", "Tier": "Thematic"},

    # Real Estate
    {"Ticker": "VNQ", "Name": "Broad REITs", "Sector Group": "Real Estate", "Tier": "Core"},
    {"Ticker": "IYR", "Name": "U.S. real estate", "Sector Group": "Real Estate", "Tier": "Core"},
    {"Ticker": "REZ", "Name": "Residential / multisector REITs", "Sector Group": "Real Estate", "Tier": "Core"},
    {"Ticker": "REM", "Name": "Mortgage REITs", "Sector Group": "Real Estate", "Tier": "Core"},
    {"Ticker": "KBWY", "Name": "High-yield REITs", "Sector Group": "Real Estate", "Tier": "Thematic"},
    {"Ticker": "HOMZ", "Name": "Housing ecosystem", "Sector Group": "Real Estate", "Tier": "Thematic"},

    # Technology
    {"Ticker": "SMH", "Name": "Semiconductors", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "SOXX", "Name": "Semiconductors broader", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "XSD", "Name": "Semiconductors equal-weight", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "IGV", "Name": "Software", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "SKYY", "Name": "Cloud infrastructure", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "CIBR", "Name": "Cybersecurity", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "HACK", "Name": "Cybersecurity high beta", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "IGM", "Name": "Expanded technology", "Sector Group": "Technology", "Tier": "Core"},
    {"Ticker": "WCLD", "Name": "Cloud software", "Sector Group": "Technology", "Tier": "Thematic"},
    {"Ticker": "AIQ", "Name": "AI / automation", "Sector Group": "Technology", "Tier": "Thematic"},
    {"Ticker": "ROBO", "Name": "Robotics", "Sector Group": "Technology", "Tier": "Thematic"},

    # Utilities
    {"Ticker": "RYU", "Name": "Equal-weight utilities", "Sector Group": "Utilities", "Tier": "Core"},
    {"Ticker": "PUI", "Name": "Utilities momentum / defensive confirmation", "Sector Group": "Utilities", "Tier": "Core"},
    {"Ticker": "GRID", "Name": "Grid / electrification", "Sector Group": "Utilities", "Tier": "Thematic"},
    {"Ticker": "NLR", "Name": "Nuclear energy", "Sector Group": "Utilities", "Tier": "Thematic"},
]

BENCHMARKS: Dict[str, str] = {
    "SPY": "SPDR S&P 500",
    "RSP": "Invesco S&P 500 Equal Weight",
    "QQQ": "Invesco QQQ",
    "IWM": "iShares Russell 2000",
    "DIA": "SPDR Dow Jones Industrial Average",
    "TLT": "iShares 20+ Year Treasury Bond",
    "IEF": "iShares 7-10 Year Treasury Bond",
    "UUP": "Invesco DB US Dollar Index Bullish Fund",
}

LOOKBACK_PERIOD = "2y"
INTERVAL = "1d"

WINDOW_PRESETS = {
    "Fast (1M vs 3M)": {
        "short": 21,
        "long": 63,
        "label_short": "1M",
        "label_long": "3M",
    },
    "Intermediate (3M vs 6M)": {
        "short": 63,
        "long": 126,
        "label_short": "3M",
        "label_long": "6M",
    },
    "Trend (6M vs 12M)": {
        "short": 126,
        "long": 252,
        "label_short": "6M",
        "label_long": "12M",
    },
}

TRAIL_OPTIONS = {
    "None": 0,
    "4 weeks": 4,
    "8 weeks": 8,
    "12 weeks": 12,
}

ROTATION_MODES = {
    "Benchmark-relative rotation": "relative",
    "Absolute sector/subsector rotation": "absolute",
}

UNIVERSE_SCOPES = [
    "Major sectors only",
    "Core subsectors",
    "Core + thematic subsectors",
    "Major sectors + core subsectors",
    "Major sectors + all subsectors",
]

LABEL_MODES = [
    "Top ranked only",
    "All tickers",
    "No labels",
]

SECTOR_GROUP_COLORS = {
    "Communication Services": "#4E79A7",
    "Consumer Discretionary": "#F28E2B",
    "Consumer Staples": "#59A14F",
    "Energy": "#E15759",
    "Financials": "#76B7B2",
    "Health Care": "#EDC948",
    "Industrials": "#B07AA1",
    "Materials": "#FF9DA7",
    "Real Estate": "#9C755F",
    "Technology": "#2F5597",
    "Utilities": "#BAB0AC",
}

QUADRANT_LABELS = {
    "Q1": "Leading",
    "Q2": "Improving",
    "Q3": "Lagging",
    "Q4": "Weakening",
    "Neutral": "Neutral",
}

STATE_COLORS = {
    "Leading": "#16a34a",
    "Improving": "#2563eb",
    "Lagging": "#dc2626",
    "Weakening": "#f59e0b",
    "Neutral": "#6b7280",
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RotationConfig:
    short_window: int
    long_window: int
    short_label: str
    long_label: str


# =============================================================================
# Universe helpers
# =============================================================================

def major_sector_rows() -> List[Dict[str, str]]:
    return [
        {
            "Ticker": ticker,
            "Name": name,
            "Sector Group": name,
            "Tier": "Major Sector",
        }
        for ticker, name in MAJOR_SECTORS.items()
    ]


def build_universe(scope: str) -> pd.DataFrame:
    major = major_sector_rows()
    core_subsectors = [row for row in SUBSECTOR_ROWS if row["Tier"] == "Core"]
    all_subsectors = SUBSECTOR_ROWS.copy()

    if scope == "Major sectors only":
        rows = major
    elif scope == "Core subsectors":
        rows = core_subsectors
    elif scope == "Core + thematic subsectors":
        rows = all_subsectors
    elif scope == "Major sectors + core subsectors":
        rows = major + core_subsectors
    elif scope == "Major sectors + all subsectors":
        rows = major + all_subsectors
    else:
        rows = core_subsectors

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    df["Sort Group"] = df["Sector Group"].map({group: i for i, group in enumerate(SECTOR_GROUP_COLORS.keys())})
    df["Sort Tier"] = df["Tier"].map({"Major Sector": 0, "Core": 1, "Thematic": 2}).fillna(9)
    df = df.sort_values(["Sort Group", "Sort Tier", "Ticker"]).drop(columns=["Sort Group", "Sort Tier"])
    return df.reset_index(drop=True)


def filter_universe_by_groups(universe_df: pd.DataFrame, selected_groups: List[str]) -> pd.DataFrame:
    if not selected_groups:
        return universe_df.iloc[0:0].copy()

    return universe_df[universe_df["Sector Group"].isin(selected_groups)].reset_index(drop=True)


def filter_universe_by_data(
    universe_df: pd.DataFrame,
    prices: pd.DataFrame,
    min_valid_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = build_price_diagnostics(
        prices=prices,
        tickers=universe_df["Ticker"].tolist(),
        min_valid_rows=min_valid_rows,
    )

    eligible = diagnostics[
        diagnostics["Status"].isin(["OK", "Stale"])
    ]["Ticker"].tolist()

    filtered = universe_df[universe_df["Ticker"].isin(eligible)].reset_index(drop=True)
    return filtered, diagnostics


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("Settings")

    universe_scope = st.selectbox(
        "Universe",
        options=UNIVERSE_SCOPES,
        index=1,
        help="Use subsectors for granular rotation work; add major sectors when you want top-down confirmation.",
    )

    base_universe = build_universe(universe_scope)
    group_options = base_universe["Sector Group"].drop_duplicates().tolist()

    selected_groups = st.multiselect(
        "Sector group filter",
        options=group_options,
        default=group_options,
    )

    selected_universe = filter_universe_by_groups(base_universe, selected_groups)

    benchmark = st.selectbox(
        "Benchmark",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{x} | {BENCHMARKS[x]}",
        index=0,
    )

    rotation_mode_label = st.radio(
        "Rotation mode",
        options=list(ROTATION_MODES.keys()),
        index=0,
        help="Benchmark-relative rotation uses sector/subsector performance versus the selected benchmark.",
    )
    rotation_mode = ROTATION_MODES[rotation_mode_label]

    window_choice = st.selectbox(
        "Rotation window",
        options=list(WINDOW_PRESETS.keys()),
        index=0,
    )

    trail_choice = st.selectbox(
        "Rotation trail",
        options=list(TRAIL_OPTIONS.keys()),
        index=1,
    )

    label_mode = st.selectbox(
        "Ticker labels on map",
        options=LABEL_MODES,
        index=0,
    )

    label_top_n = st.slider(
        "Label top N",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
        disabled=label_mode != "Top ranked only",
    )

    show_diagnostics = st.checkbox("Show data diagnostics", value=False)


# =============================================================================
# Helpers
# =============================================================================

def get_rotation_config(choice: str) -> RotationConfig:
    cfg = WINDOW_PRESETS[choice]
    return RotationConfig(
        short_window=cfg["short"],
        long_window=cfg["long"],
        short_label=cfg["label_short"],
        long_label=cfg["label_long"],
    )


def _safe_to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    try:
        if out.index.tz is not None:
            out.index = out.index.tz_convert(None)
    except Exception:
        pass

    return out


def _normalize_download(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Normalize yfinance output into a clean wide dataframe of adjusted closes.
    Handles MultiIndex and single-index outputs.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    field_candidates = ["Adj Close", "Close"]
    out = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            selected = None

            for field in field_candidates:
                if (field, ticker) in data.columns:
                    selected = data[(field, ticker)]
                    break

                if (ticker, field) in data.columns:
                    selected = data[(ticker, field)]
                    break

            if selected is not None:
                selected = pd.to_numeric(selected, errors="coerce").dropna()
                if not selected.empty:
                    out[ticker] = selected

        df = pd.DataFrame(out)
        return _safe_to_datetime_index(df)

    if len(tickers) == 1:
        ticker = tickers[0]
        for field in field_candidates:
            if field in data.columns:
                s = pd.to_numeric(data[field], errors="coerce").dropna()
                if not s.empty:
                    return _safe_to_datetime_index(pd.DataFrame({ticker: s}))

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(
    tickers: List[str],
    period: str = LOOKBACK_PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    unique_tickers = list(dict.fromkeys([t for t in tickers if t]))

    if not unique_tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=unique_tickers,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    prices = _normalize_download(raw, unique_tickers)

    if prices.empty:
        return prices

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.ffill()

    return prices


def validate_benchmark(prices: pd.DataFrame, benchmark_ticker: str, min_valid_rows: int) -> Tuple[bool, str]:
    if prices.empty:
        return False, "No market data returned."

    if benchmark_ticker not in prices.columns:
        return False, f"Benchmark data unavailable for {benchmark_ticker}."

    s = pd.to_numeric(prices[benchmark_ticker], errors="coerce").dropna()

    if s.empty:
        return False, f"Benchmark data unavailable for {benchmark_ticker}."

    if len(s) < min_valid_rows:
        return False, f"Benchmark has only {len(s)} valid rows; need at least {min_valid_rows}."

    return True, ""


def build_price_diagnostics(
    prices: pd.DataFrame,
    tickers: List[str],
    min_valid_rows: int = 100,
) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Status",
        "First Date",
        "Latest Date",
        "Valid Rows",
        "Missing Values",
        "Last Price",
        "Stale Days",
    ]

    if prices.empty:
        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Status": "Missing",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": "",
                    "Last Price": np.nan,
                    "Stale Days": np.nan,
                }
                for ticker in tickers
            ],
            columns=columns,
        )

    max_dt = pd.to_datetime(prices.index.max()).date()
    rows = []

    for ticker in tickers:
        if ticker not in prices.columns:
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": "Missing",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": "",
                    "Last Price": np.nan,
                    "Stale Days": np.nan,
                }
            )
            continue

        s = pd.to_numeric(prices[ticker], errors="coerce").dropna()

        if s.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": "No valid data",
                    "First Date": "",
                    "Latest Date": "",
                    "Valid Rows": 0,
                    "Missing Values": int(prices[ticker].isna().sum()),
                    "Last Price": np.nan,
                    "Stale Days": np.nan,
                }
            )
            continue

        first_dt = pd.to_datetime(s.index.min()).date()
        last_dt = pd.to_datetime(s.index.max()).date()
        stale_days = (max_dt - last_dt).days

        if len(s) < min_valid_rows:
            status = "Thin history"
        elif stale_days > 5:
            status = "Stale"
        else:
            status = "OK"

        rows.append(
            {
                "Ticker": ticker,
                "Status": status,
                "First Date": first_dt,
                "Latest Date": last_dt,
                "Valid Rows": int(len(s)),
                "Missing Values": int(prices[ticker].isna().sum()),
                "Last Price": float(s.iloc[-1]),
                "Stale Days": int(stale_days),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def compute_relative_strength(
    prices: pd.DataFrame,
    item_tickers: List[str],
    benchmark_ticker: str,
) -> pd.DataFrame:
    rs = prices[item_tickers].div(prices[benchmark_ticker], axis=0)
    rs = rs.replace([np.inf, -np.inf], np.nan)
    return rs.dropna(how="all")


def compute_zscore(frame: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    rolling_mean = frame.rolling(window).mean()
    rolling_std = frame.rolling(window).std().replace(0, np.nan)
    z = (frame - rolling_mean) / rolling_std
    return z.replace([np.inf, -np.inf], np.nan)


def pct_change_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    if len(frame) <= periods:
        return pd.Series(index=frame.columns, dtype=float)

    out = frame.pct_change(periods).iloc[-1]
    return out.replace([np.inf, -np.inf], np.nan)


def slope_last(frame: pd.DataFrame, periods: int) -> pd.Series:
    if len(frame) <= periods:
        return pd.Series(index=frame.columns, dtype=float)

    out = frame.iloc[-1] / frame.shift(periods).iloc[-1] - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def classify_quadrant(long_ret: float, short_ret: float) -> str:
    if pd.isna(long_ret) or pd.isna(short_ret):
        return "Neutral"

    if long_ret > 0 and short_ret > 0:
        return "Q1"

    if long_ret < 0 and short_ret > 0:
        return "Q2"

    if long_ret < 0 and short_ret < 0:
        return "Q3"

    if long_ret > 0 and short_ret < 0:
        return "Q4"

    return "Neutral"


def pct_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    ranked = series.rank(pct=True)
    return ranked.fillna(0.5)


def build_snapshot(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
) -> pd.DataFrame:
    item_tickers = universe_df["Ticker"].tolist()
    item_prices = prices[item_tickers].copy()

    rs = compute_relative_strength(prices, item_tickers, benchmark_ticker)

    if rotation_basis == "relative":
        rotation_frame = rs
        rotation_label = "Relative"
    else:
        rotation_frame = item_prices
        rotation_label = "Absolute"

    abs_short = pct_change_last(item_prices, cfg.short_window)
    abs_long = pct_change_last(item_prices, cfg.long_window)

    rs_short = pct_change_last(rs, cfg.short_window)
    rs_long = pct_change_last(rs, cfg.long_window)

    rotation_short = pct_change_last(rotation_frame, cfg.short_window)
    rotation_long = pct_change_last(rotation_frame, cfg.long_window)

    rs_slope_10d = slope_last(rs, 10)
    rs_slope_5d = slope_last(rs, 5)
    rs_accel = (rs_slope_5d - rs_slope_10d).replace([np.inf, -np.inf], np.nan)

    rotation_slope_10d = slope_last(rotation_frame, 10)
    rotation_slope_5d = slope_last(rotation_frame, 5)
    rotation_accel = (rotation_slope_5d - rotation_slope_10d).replace([np.inf, -np.inf], np.nan)

    rs_z_window = min(63, max(20, cfg.long_window))
    rs_z = compute_zscore(rs, window=rs_z_window)
    rs_z_last = rs_z.iloc[-1] if not rs_z.empty else pd.Series(index=item_tickers, dtype=float)

    angle = np.degrees(np.arctan2(rotation_short, rotation_long))
    speed = np.sqrt(rotation_short.pow(2) + rotation_long.pow(2))

    composite = (
        0.25 * pct_rank(rs_short)
        + 0.25 * pct_rank(rs_long)
        + 0.20 * pct_rank(rs_slope_10d)
        + 0.15 * pct_rank(rs_accel)
        + 0.10 * pct_rank(abs_short)
        + 0.05 * pct_rank(abs_long)
    ) * 100.0

    meta = universe_df.set_index("Ticker")

    snap = pd.DataFrame(
        {
            "Ticker": item_tickers,
            "Name": meta.reindex(item_tickers)["Name"].values,
            "Sector Group": meta.reindex(item_tickers)["Sector Group"].values,
            "Tier": meta.reindex(item_tickers)["Tier"].values,
            "Rotation Mode": rotation_label,
            f"Rotation {cfg.short_label}": rotation_short.reindex(item_tickers).values,
            f"Rotation {cfg.long_label}": rotation_long.reindex(item_tickers).values,
            f"Abs {cfg.short_label}": abs_short.reindex(item_tickers).values,
            f"Abs {cfg.long_label}": abs_long.reindex(item_tickers).values,
            f"RS {cfg.short_label}": rs_short.reindex(item_tickers).values,
            f"RS {cfg.long_label}": rs_long.reindex(item_tickers).values,
            "RS Slope 10D": rs_slope_10d.reindex(item_tickers).values,
            "RS Slope 5D": rs_slope_5d.reindex(item_tickers).values,
            "RS Accel": rs_accel.reindex(item_tickers).values,
            "Rotation Slope 10D": rotation_slope_10d.reindex(item_tickers).values,
            "Rotation Accel": rotation_accel.reindex(item_tickers).values,
            "RS Z": rs_z_last.reindex(item_tickers).values,
            "Angle (deg)": angle.reindex(item_tickers).values,
            "Speed": speed.reindex(item_tickers).values,
            "Composite Score": composite.reindex(item_tickers).values,
        }
    )

    snap["QuadrantCode"] = [
        classify_quadrant(lr, sr)
        for lr, sr in zip(
            snap[f"Rotation {cfg.long_label}"],
            snap[f"Rotation {cfg.short_label}"],
        )
    ]
    snap["Quadrant"] = snap["QuadrantCode"].map(QUADRANT_LABELS)

    snap["State"] = np.select(
        [
            (snap["Quadrant"] == "Leading") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Leading") & (snap["Rotation Accel"] <= 0),
            (snap["Quadrant"] == "Improving") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Improving") & (snap["Rotation Accel"] <= 0),
            (snap["Quadrant"] == "Weakening") & (snap["Rotation Accel"] < 0),
            (snap["Quadrant"] == "Weakening") & (snap["Rotation Accel"] >= 0),
            (snap["Quadrant"] == "Lagging") & (snap["Rotation Accel"] > 0),
            (snap["Quadrant"] == "Lagging") & (snap["Rotation Accel"] <= 0),
        ],
        [
            "Leading and accelerating",
            "Leading but cooling",
            "Improving fast",
            "Improving but uneven",
            "Weakening fast",
            "Weakening but stabilizing",
            "Lagging but stabilizing",
            "Lagging and deteriorating",
        ],
        default=snap["Quadrant"],
    )

    snap["MarkerSize"] = 16 + (snap["Speed"].rank(pct=True).fillna(0.5) * 18)
    snap["Color"] = snap["Sector Group"].map(SECTOR_GROUP_COLORS).fillna("#4b5563")

    snap = snap.sort_values("Composite Score", ascending=False).reset_index(drop=True)
    snap["Rank"] = np.arange(1, len(snap) + 1)

    return snap


def compute_rank_delta(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    rotation_basis: str,
    compare_days: int = 5,
) -> pd.DataFrame:
    if len(prices) <= cfg.long_window + compare_days + 15:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Prior Rank",
                "Rank Δ",
                "Prior Quadrant",
                "Prior Composite Score",
            ]
        )

    past_prices = prices.iloc[:-compare_days].copy()
    current_prices = prices.copy()

    past_snap = build_snapshot(
        past_prices,
        universe_df,
        benchmark_ticker,
        cfg,
        rotation_basis,
    )[["Ticker", "Rank", "Quadrant", "Composite Score"]].rename(
        columns={
            "Rank": "Prior Rank",
            "Quadrant": "Prior Quadrant",
            "Composite Score": "Prior Composite Score",
        }
    )

    curr_snap = build_snapshot(
        current_prices,
        universe_df,
        benchmark_ticker,
        cfg,
        rotation_basis,
    )[["Ticker", "Rank", "Quadrant", "Composite Score"]]

    merged = curr_snap.merge(past_snap, on="Ticker", how="left")
    merged["Rank Δ"] = merged["Prior Rank"] - merged["Rank"]

    return merged[
        [
            "Ticker",
            "Prior Rank",
            "Rank Δ",
            "Prior Quadrant",
            "Prior Composite Score",
        ]
    ]


def build_rotation_trails(
    prices: pd.DataFrame,
    universe_df: pd.DataFrame,
    benchmark_ticker: str,
    cfg: RotationConfig,
    trail_weeks: int,
    rotation_basis: str,
) -> Dict[str, pd.DataFrame]:
    if trail_weeks <= 0:
        return {}

    item_tickers = universe_df["Ticker"].tolist()
    needed = item_tickers + [benchmark_ticker]
    available = [t for t in needed if t in prices.columns]

    weekly_prices = prices[available].resample("W-FRI").last().dropna(how="all")

    if weekly_prices.empty:
        return {}

    available_items = [ticker for ticker in item_tickers if ticker in weekly_prices.columns]

    if not available_items:
        return {}

    if rotation_basis == "relative":
        if benchmark_ticker not in weekly_prices.columns:
            return {}

        rotation_frame = weekly_prices[available_items].div(
            weekly_prices[benchmark_ticker],
            axis=0,
        )
    else:
        rotation_frame = weekly_prices[available_items]

    rotation_frame = rotation_frame.replace([np.inf, -np.inf], np.nan)

    short_weeks = max(1, round(cfg.short_window / 5))
    long_weeks = max(2, round(cfg.long_window / 5))

    trail_points = {}

    for ticker in available_items:
        if ticker not in rotation_frame.columns:
            continue

        s = rotation_frame[ticker].dropna()

        if s.empty or len(s) <= long_weeks:
            continue

        short_ret = s.pct_change(short_weeks)
        long_ret = s.pct_change(long_weeks)

        df = pd.DataFrame(
            {
                "x": long_ret,
                "y": short_ret,
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            continue

        trail_points[ticker] = df.tail(trail_weeks)

    return trail_points


def make_rs_chart(
    rs_series: pd.Series,
    item_name: str,
    benchmark_ticker: str,
) -> go.Figure:
    rs_20 = rs_series.rolling(20).mean()
    rs_50 = rs_series.rolling(50).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rs_series.index,
            y=rs_series.values,
            mode="lines",
            name="RS Ratio",
            line=dict(width=2.4, color="#1f77b4"),
            hovertemplate="%{x|%Y-%m-%d}<br>RS: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rs_20.index,
            y=rs_20.values,
            mode="lines",
            name="20D MA",
            line=dict(width=1.4, dash="dot", color="#2ca02c"),
            hovertemplate="%{x|%Y-%m-%d}<br>20D MA: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rs_50.index,
            y=rs_50.values,
            mode="lines",
            name="50D MA",
            line=dict(width=1.4, dash="dash", color="#9467bd"),
            hovertemplate="%{x|%Y-%m-%d}<br>50D MA: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Relative Strength | {item_name} vs {benchmark_ticker}",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Date",
        yaxis_title="RS Ratio",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
        ),
    )

    return fig


def make_rotation_scatter(
    snap: pd.DataFrame,
    cfg: RotationConfig,
    trail_points: Dict[str, pd.DataFrame],
    rotation_basis: str,
    benchmark_ticker: str,
    label_mode: str,
    label_top_n: int,
) -> go.Figure:
    x_col = f"Rotation {cfg.long_label}"
    y_col = f"Rotation {cfg.short_label}"

    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, opacity=0.45)
    fig.add_vline(x=0, line_width=1, opacity=0.45)

    plot_df = snap.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid rotation data for the selected universe/window.",
            showarrow=False,
        )
        fig.update_layout(height=560)
        return fig

    label_set = set()
    if label_mode == "All tickers":
        label_set = set(plot_df["Ticker"])
    elif label_mode == "Top ranked only":
        label_set = set(plot_df.nsmallest(label_top_n, "Rank")["Ticker"])

    for ticker, trail_df in trail_points.items():
        if ticker not in plot_df["Ticker"].values or trail_df.empty:
            continue

        group = plot_df.loc[plot_df["Ticker"] == ticker, "Sector Group"].iloc[0]
        color = SECTOR_GROUP_COLORS.get(group, "#4b5563")

        fig.add_trace(
            go.Scatter(
                x=trail_df["x"],
                y=trail_df["y"],
                mode="lines+markers",
                line=dict(color=color, width=1.2),
                marker=dict(size=4, color=color, opacity=0.5),
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for _, row in plot_df.iterrows():
        ticker = row["Ticker"]
        border_color = STATE_COLORS.get(row["Quadrant"], "#111827")
        text = ticker if ticker in label_set else ""

        hovertemplate = (
            f"<b>{row['Name']} ({ticker})</b><br>"
            f"Group: {row['Sector Group']}<br>"
            f"Tier: {row['Tier']}<br>"
            f"Rotation mode: {row['Rotation Mode']}<br>"
            f"Rotation {cfg.short_label}: {row[f'Rotation {cfg.short_label}']:.2%}<br>"
            f"Rotation {cfg.long_label}: {row[f'Rotation {cfg.long_label}']:.2%}<br>"
            f"Abs {cfg.short_label}: {row[f'Abs {cfg.short_label}']:.2%}<br>"
            f"Abs {cfg.long_label}: {row[f'Abs {cfg.long_label}']:.2%}<br>"
            f"RS {cfg.short_label}: {row[f'RS {cfg.short_label}']:.2%}<br>"
            f"RS {cfg.long_label}: {row[f'RS {cfg.long_label}']:.2%}<br>"
            f"RS Slope 10D: {row['RS Slope 10D']:.2%}<br>"
            f"RS Accel: {row['RS Accel']:.2%}<br>"
            f"Composite Score: {row['Composite Score']:.1f}<br>"
            f"Quadrant: {row['Quadrant']}<br>"
            f"State: {row['State']}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=[row[x_col]],
                y=[row[y_col]],
                mode="markers+text" if text else "markers",
                text=[text] if text else None,
                textposition="top center",
                marker=dict(
                    size=float(row["MarkerSize"]),
                    color=row["Color"],
                    line=dict(width=1.8, color=border_color),
                    opacity=0.90,
                ),
                name=ticker,
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    x_min = float(plot_df[x_col].min())
    x_max = float(plot_df[x_col].max())
    y_min = float(plot_df[y_col].min())
    y_max = float(plot_df[y_col].max())

    x_pad = max(0.02, abs(x_max - x_min) * 0.20, plot_df[x_col].abs().max() * 0.15)
    y_pad = max(0.02, abs(y_max - y_min) * 0.20, plot_df[y_col].abs().max() * 0.15)

    if x_min == x_max:
        x_min -= x_pad
        x_max += x_pad

    if y_min == y_max:
        y_min -= y_pad
        y_max += y_pad

    fig.add_annotation(
        x=x_max + x_pad * 0.45,
        y=y_max + y_pad * 0.45,
        text="Leading",
        showarrow=False,
        font=dict(color=STATE_COLORS["Leading"], size=12),
    )

    fig.add_annotation(
        x=x_min - x_pad * 0.45,
        y=y_max + y_pad * 0.45,
        text="Improving",
        showarrow=False,
        font=dict(color=STATE_COLORS["Improving"], size=12),
    )

    fig.add_annotation(
        x=x_min - x_pad * 0.45,
        y=y_min - y_pad * 0.45,
        text="Lagging",
        showarrow=False,
        font=dict(color=STATE_COLORS["Lagging"], size=12),
    )

    fig.add_annotation(
        x=x_max + x_pad * 0.45,
        y=y_min - y_pad * 0.45,
        text="Weakening",
        showarrow=False,
        font=dict(color=STATE_COLORS["Weakening"], size=12),
    )

    mode_title = (
        f"Relative to {benchmark_ticker}"
        if rotation_basis == "relative"
        else "Absolute return"
    )

    fig.update_layout(
        title=f"Rotation Map | {mode_title} | {cfg.short_label} vs {cfg.long_label}",
        height=610,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(
            title=f"{cfg.long_label} rotation return",
            tickformat=".1%",
            zeroline=False,
            range=[x_min - x_pad, x_max + x_pad],
        ),
        yaxis=dict(
            title=f"{cfg.short_label} rotation return",
            tickformat=".1%",
            zeroline=False,
            range=[y_min - y_pad, y_max + y_pad],
        ),
    )

    return fig


def format_delta(val: float) -> str:
    if pd.isna(val):
        return ""

    sign = "+" if val > 0 else ""
    return f"{sign}{int(val)}"


def build_display_table(
    snap: pd.DataFrame,
    rank_delta: pd.DataFrame,
    cfg: RotationConfig,
) -> pd.DataFrame:
    table = snap.merge(rank_delta, on="Ticker", how="left")

    table["Quadrant Δ"] = np.where(
        table["Prior Quadrant"].fillna("") == table["Quadrant"].fillna(""),
        "",
        table["Prior Quadrant"].fillna("") + " → " + table["Quadrant"].fillna(""),
    )

    table["Rank Δ"] = table["Rank Δ"].apply(format_delta)

    cols = [
        "Rank",
        "Ticker",
        "Name",
        "Sector Group",
        "Tier",
        "Quadrant",
        "State",
        "Composite Score",
        "Rank Δ",
        "Quadrant Δ",
        f"Rotation {cfg.short_label}",
        f"Rotation {cfg.long_label}",
        f"Abs {cfg.short_label}",
        f"Abs {cfg.long_label}",
        f"RS {cfg.short_label}",
        f"RS {cfg.long_label}",
        "RS Slope 10D",
        "RS Accel",
        "RS Z",
        "Angle (deg)",
        "Speed",
    ]

    return table[cols].copy()


def style_snapshot_table(df: pd.DataFrame, cfg: RotationConfig):
    fmt_map = {
        "Composite Score": "{:.1f}",
        f"Rotation {cfg.short_label}": "{:.2%}",
        f"Rotation {cfg.long_label}": "{:.2%}",
        f"Abs {cfg.short_label}": "{:.2%}",
        f"Abs {cfg.long_label}": "{:.2%}",
        f"RS {cfg.short_label}": "{:.2%}",
        f"RS {cfg.long_label}": "{:.2%}",
        "RS Slope 10D": "{:.2%}",
        "RS Accel": "{:.2%}",
        "RS Z": "{:.2f}",
        "Angle (deg)": "{:.1f}",
        "Speed": "{:.2%}",
    }

    fmt_map = {k: v for k, v in fmt_map.items() if k in df.columns}

    gradient_cols_primary = [
        c
        for c in [
            "Composite Score",
            f"Rotation {cfg.short_label}",
            f"Rotation {cfg.long_label}",
            f"RS {cfg.short_label}",
            f"RS {cfg.long_label}",
            "RS Slope 10D",
            "RS Accel",
        ]
        if c in df.columns
    ]

    gradient_cols_speed = [c for c in ["Speed"] if c in df.columns]

    styler = df.style.format(fmt_map)

    if gradient_cols_primary:
        styler = styler.background_gradient(
            subset=gradient_cols_primary,
            cmap="RdYlGn",
        )

    if gradient_cols_speed:
        styler = styler.background_gradient(
            subset=gradient_cols_speed,
            cmap="PuBuGn",
        )

    return styler


# =============================================================================
# Load data
# =============================================================================

cfg = get_rotation_config(window_choice)
min_required_rows = cfg.long_window + 20

if selected_universe.empty:
    st.warning("No sector groups selected.")
    st.stop()

all_tickers = list(
    dict.fromkeys(
        selected_universe["Ticker"].tolist()
        + list(BENCHMARKS.keys())
    )
)

with st.spinner("Loading market data..."):
    prices = fetch_prices(all_tickers)

benchmark_ok, benchmark_error = validate_benchmark(prices, benchmark, min_required_rows)

if not benchmark_ok:
    st.error(benchmark_error)
    st.stop()

prices = prices.sort_index().ffill()

eligible_universe, universe_diagnostics = filter_universe_by_data(
    selected_universe,
    prices,
    min_valid_rows=min_required_rows,
)

benchmark_diagnostics = build_price_diagnostics(
    prices=prices,
    tickers=[benchmark],
    min_valid_rows=min_required_rows,
)

diagnostics_df = pd.concat(
    [universe_diagnostics, benchmark_diagnostics],
    ignore_index=True,
).drop_duplicates(subset=["Ticker"], keep="first")

if eligible_universe.empty:
    st.error("No eligible sector/subsector ETFs have enough valid history for the selected rotation window.")
    st.stop()

missing_or_thin = universe_diagnostics[~universe_diagnostics["Status"].isin(["OK", "Stale"])]

if not missing_or_thin.empty:
    with st.expander(
        f"Dropped {len(missing_or_thin)} ticker(s) with missing or thin data",
        expanded=False,
    ):
        st.dataframe(
            missing_or_thin,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Analytics
# =============================================================================

item_tickers = eligible_universe["Ticker"].tolist()

rs = compute_relative_strength(prices, item_tickers, benchmark)

snap = build_snapshot(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
)

rank_delta = compute_rank_delta(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    rotation_basis=rotation_mode,
    compare_days=5,
)

trail_points = build_rotation_trails(
    prices=prices,
    universe_df=eligible_universe,
    benchmark_ticker=benchmark,
    cfg=cfg,
    trail_weeks=TRAIL_OPTIONS[trail_choice],
    rotation_basis=rotation_mode,
)

display_df = build_display_table(snap, rank_delta, cfg)


# =============================================================================
# Optional diagnostics
# =============================================================================

if show_diagnostics:
    with st.expander("Data diagnostics", expanded=True):
        st.dataframe(
            diagnostics_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Main charts
# =============================================================================

left, right = st.columns([1.0, 1.22])

with left:
    st.subheader("Relative strength")

    rs_options = snap.sort_values("Rank")["Ticker"].tolist()
    rs_default = rs_options.index("SMH") if "SMH" in rs_options else 0

    rs_item = st.selectbox(
        "Ticker for RS chart",
        options=rs_options,
        format_func=lambda x: (
            f"{x} | "
            f"{eligible_universe.set_index('Ticker').loc[x, 'Name']} | "
            f"{eligible_universe.set_index('Ticker').loc[x, 'Sector Group']}"
        ),
        index=rs_default,
    )

    rs_meta = eligible_universe.set_index("Ticker").loc[rs_item]

    rs_fig = make_rs_chart(
        rs_series=rs[rs_item].dropna(),
        item_name=f"{rs_meta['Name']} ({rs_item})",
        benchmark_ticker=benchmark,
    )

    st.plotly_chart(rs_fig, use_container_width=True)

with right:
    st.subheader("Rotation map")

    rot_fig = make_rotation_scatter(
        snap=snap,
        cfg=cfg,
        trail_points=trail_points,
        rotation_basis=rotation_mode,
        benchmark_ticker=benchmark,
        label_mode=label_mode,
        label_top_n=label_top_n,
    )

    st.plotly_chart(rot_fig, use_container_width=True)


# =============================================================================
# Snapshot table
# =============================================================================

st.subheader("Rotation snapshot")

filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.0])

with filter_col1:
    sort_col = st.selectbox(
        "Sort by",
        options=[
            "Rank",
            "Ticker",
            "Name",
            "Sector Group",
            "Tier",
            "Quadrant",
            "State",
            "Composite Score",
            f"Rotation {cfg.short_label}",
            f"Rotation {cfg.long_label}",
            f"Abs {cfg.short_label}",
            f"Abs {cfg.long_label}",
            f"RS {cfg.short_label}",
            f"RS {cfg.long_label}",
            "RS Slope 10D",
            "RS Accel",
            "RS Z",
            "Angle (deg)",
            "Speed",
        ],
        index=7,
    )

with filter_col2:
    table_group_options = display_df["Sector Group"].drop_duplicates().tolist()

    table_groups = st.multiselect(
        "Group filter",
        options=table_group_options,
        default=table_group_options,
    )

with filter_col3:
    quadrant_options = [
        q
        for q in ["Leading", "Improving", "Weakening", "Lagging", "Neutral"]
        if q in display_df["Quadrant"].unique()
    ]

    selected_quadrants = st.multiselect(
        "Quadrant filter",
        options=quadrant_options,
        default=quadrant_options,
    )

with filter_col4:
    state_options = sorted(display_df["State"].dropna().unique().tolist())

    selected_states = st.multiselect(
        "State filter",
        options=state_options,
        default=state_options,
    )

with filter_col5:
    ascending = st.toggle("Ascending", value=False)
    only_accelerating = st.toggle("RS accel only", value=False)

    tier_options = display_df["Tier"].drop_duplicates().tolist()
    selected_tiers = st.multiselect(
        "Tier",
        options=tier_options,
        default=tier_options,
    )

table_filtered = display_df.copy()

if table_groups:
    table_filtered = table_filtered[table_filtered["Sector Group"].isin(table_groups)]

if selected_tiers:
    table_filtered = table_filtered[table_filtered["Tier"].isin(selected_tiers)]

if selected_quadrants:
    table_filtered = table_filtered[table_filtered["Quadrant"].isin(selected_quadrants)]

if selected_states:
    table_filtered = table_filtered[table_filtered["State"].isin(selected_states)]

if only_accelerating:
    table_filtered = table_filtered[table_filtered["RS Accel"] > 0]

if table_filtered.empty:
    st.info("No sectors/subsectors match the current filter selection.")
else:
    table_sorted = table_filtered.sort_values(
        sort_col,
        ascending=ascending,
    ).reset_index(drop=True)

    row_height = 34
    table_height = min(780, max(180, row_height * (len(table_sorted) + 1)))

    st.dataframe(
        style_snapshot_table(table_sorted, cfg),
        use_container_width=True,
        height=table_height,
        hide_index=True,
    )

    csv = table_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download snapshot CSV",
        data=csv,
        file_name="adfm_subsector_rotation_snapshot.csv",
        mime="text/csv",
    )


# =============================================================================
# Footer
# =============================================================================

coverage = len([c for c in item_tickers if c in prices.columns and not prices[c].dropna().empty])
latest_dt = pd.to_datetime(prices.index.max()).date()

st.caption(
    f"Data through: {latest_dt} | Benchmark: {benchmark} | Rotation mode: {rotation_mode_label} | "
    f"Universe: {universe_scope} | Coverage: {coverage}/{len(selected_universe)} eligible tickers"
)

st.caption("© 2026 AD Fund Management LP")
