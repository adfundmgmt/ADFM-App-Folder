import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape as html_escape
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, render_footer, render_page_header, render_sidebar_about
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

# ============================== Page config ==============================
st.set_page_config(layout="wide", page_title="Cross-Asset Ratio Chartbook")
render_page_header(
    PageHeader(
        title="Cross-Asset Ratio Chartbook",
        description=(
            "Inspect the trend, technical condition, and historical extremes of fifty specific "
            "cross-asset, macro, thematic, and single-stock relationships."
        ),
        eyebrow="ADFM Technicals + Analogs",
    )
)

st.markdown(
    """
    <style>
    .ratio-chart-heading {
        margin: 1.1rem 0 .45rem;
        color: #000000;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.25;
    }

    @media (max-width: 760px) {
        .ratio-chart-heading {
            margin-top: .9rem;
            font-size: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== Defaults =================================
DEFAULT_LOOKBACK = "3 Years"
DEFAULT_RSI_WINDOW = 14
DEFAULT_STALE_DAYS = 7

MA_DEFAULTS = {
    8: False,
    21: True,
    50: True,
    100: False,
    200: True,
}

MA_COLORS = {
    8: "#6c757d",
    21: PASTEL["lavender"],
    50: PASTEL["blue"],
    100: PASTEL["coral"],
    200: PASTEL["rose"],
}

# ============================== Ratio specs ==============================
@dataclass(frozen=True)
class RatioSpec:
    ticker_1: str
    ticker_2: str
    label: str
    note: str = ""


CORE_RATIO_SPECS: List[RatioSpec] = [
    RatioSpec(
        "SPY",
        "IEF",
        "Equities / Intermediate Treasuries",
        "Primary risk-assets-versus-rates ratio.",
    ),
    RatioSpec(
        "UUP",
        "SPY",
        "Dollar / S&P 500",
        "Dollar tightness versus equity risk.",
    ),
    RatioSpec(
        "FXY",
        "SPY",
        "Yen / S&P 500",
        "Yen strength versus equity risk; useful for carry-unwind pressure.",
    ),
    RatioSpec(
        "GLD",
        "SPY",
        "Gold / S&P 500",
        "Monetary hedge versus equity risk.",
    ),
    RatioSpec(
        "DBC",
        "SPY",
        "Broad Commodities / S&P 500",
        "Commodity leadership versus equities.",
    ),
    RatioSpec(
        "XLE",
        "SPY",
        "Energy / S&P 500",
        "Energy and inflation beta versus broad equities.",
    ),
    RatioSpec(
        "QQQ",
        "SPY",
        "Nasdaq 100 / S&P 500",
        "Mega-cap growth leadership.",
    ),
    RatioSpec(
        "QQQE",
        "QQQ",
        "Equal Weight Nasdaq 100 / Nasdaq 100",
        "Nasdaq breadth versus mega-cap concentration.",
    ),
    RatioSpec(
        "NVDA",
        "SMH",
        "Nvidia / Semiconductors",
        "Nvidia leadership versus the broader semiconductor complex.",
    ),
    RatioSpec(
        "IBIT",
        "SPY",
        "Bitcoin ETF / S&P 500",
        "Bitcoin and crypto risk appetite versus broad equity risk.",
    ),
    RatioSpec(
        "XBI",
        "QQQ",
        "Biotech / Nasdaq 100",
        "Speculative biotech and funding-cycle sensitivity versus mega-cap growth.",
    ),
    RatioSpec(
        "KRE",
        "XLRE",
        "Regional Banks / Real Estate",
        "Credit creation and bank stress versus real-estate duration sensitivity.",
    ),
    RatioSpec(
        "BX",
        "XLF",
        "Blackstone / Financials",
        "Private-assets leadership versus broad financials.",
    ),
    RatioSpec(
        "SPY",
        "VXUS",
        "US Equities / Ex-US Equities",
        "US equity leadership versus the rest of the world.",
    ),
    RatioSpec(
        "TLT",
        "SHY",
        "Long Duration Treasuries / Treasury Bills",
        "Duration impulse versus cash; useful for easing, slowdown, and long-end stress regimes.",
    ),
    RatioSpec(
        "^VIX",
        "SPY",
        "VIX / S&P 500",
        "Volatility stress versus equity risk; useful for de-risking and carry-unwind pressure.",
    ),
    RatioSpec(
        "EEM",
        "VEA",
        "Emerging Markets / Developed Ex-US",
        "EM leadership versus developed ex-US; global liquidity and dollar-cycle signal.",
    ),
    RatioSpec(
        "FXI",
        "EEM",
        "China Large Caps / Emerging Markets",
        "China beta versus broader EM; isolates whether China is leading or dragging the complex.",
    ),
    RatioSpec(
        "TLT",
        "GLD",
        "Long Treasuries / Gold",
        "Long-duration sovereign confidence versus monetary and fiscal defense.",
    ),
    RatioSpec(
        "UUP",
        "GLD",
        "Dollar / Gold",
        "Separates dollar strength from gold-led monetary and credibility demand.",
    ),
    RatioSpec(
        "FXY",
        "UUP",
        "Yen / Dollar",
        "Direct policy-divergence and carry-unwind signal between the yen and dollar.",
    ),
    RatioSpec(
        "FXA",
        "UUP",
        "Australian Dollar / US Dollar",
        "Commodity, China, global-growth, and carry appetite versus dollar defensiveness.",
    ),
    RatioSpec(
        "BKLN",
        "IEF",
        "Senior Loans / Intermediate Treasuries",
        "Floating-rate credit risk versus duration; useful for funding and default pressure.",
    ),
    RatioSpec(
        "IEF",
        "SHY",
        "Intermediate Treasuries / Treasury Bills",
        "Intermediate-duration demand versus cash; a cleaner easing and growth-scare signal.",
    ),
    RatioSpec(
        "GDX",
        "GLD",
        "Gold Miners / Gold",
        "Mining operating leverage, cost pressure, and equity beta versus bullion.",
    ),
    RatioSpec(
        "SIL",
        "SLV",
        "Silver Miners / Silver",
        "Silver-miner operating leverage and margin pressure versus the underlying metal.",
    ),
    RatioSpec(
        "UNG",
        "USO",
        "Natural Gas / Oil",
        "Domestic weather and storage scarcity versus the global crude-demand cycle.",
    ),
    RatioSpec(
        "DBA",
        "DBC",
        "Agriculture / Broad Commodities",
        "Food inflation and crop stress versus the broader energy-and-metals complex.",
    ),
    RatioSpec(
        "XBI",
        "XLV",
        "Biotech / Healthcare",
        "Early-stage funding and innovation appetite versus established healthcare.",
    ),
    RatioSpec(
        "ARKK",
        "QQQ",
        "Speculative Innovation / Nasdaq 100",
        "Unprofitable long-duration growth versus profitable mega-cap technology.",
    ),
    RatioSpec(
        "IPO",
        "QQQ",
        "Recent IPOs / Nasdaq 100",
        "New-issue risk appetite versus established large-cap growth leadership.",
    ),
    RatioSpec(
        "XRT",
        "XLP",
        "Retail / Consumer Staples",
        "Consumer breadth and discretionary demand versus defensive consumption.",
    ),
    RatioSpec(
        "XME",
        "XLI",
        "Metals & Mining / Industrials",
        "Upstream materials and inflation beta versus downstream industrial capex.",
    ),
    RatioSpec(
        "PAVE",
        "SPY",
        "US Infrastructure / S&P 500",
        "Infrastructure, reshoring, and physical-capex leadership versus the broad market.",
    ),
    RatioSpec(
        "INDA",
        "EEM",
        "India / Emerging Markets",
        "India's domestic-growth premium versus the broader emerging-market complex.",
    ),
    RatioSpec(
        "NVDA",
        "AVGO",
        "Nvidia / Broadcom",
        "GPU-platform leadership versus custom silicon, networking, and hyperscaler exposure.",
    ),
    RatioSpec(
        "AMD",
        "NVDA",
        "AMD / Nvidia",
        "AI-accelerator challenger momentum versus the incumbent GPU platform.",
    ),
    RatioSpec(
        "MU",
        "SMH",
        "Micron / Semiconductors",
        "Memory-cycle and high-bandwidth-memory leadership versus the chip complex.",
    ),
    RatioSpec(
        "ANET",
        "SMH",
        "Arista / Semiconductors",
        "AI networking and Ethernet scale-out demand versus compute silicon.",
    ),
    RatioSpec(
        "COHR",
        "SMH",
        "Coherent / Semiconductors",
        "Optical interconnect and photonics demand versus compute silicon.",
    ),
    RatioSpec(
        "PLTR",
        "IGV",
        "Palantir / Software",
        "AI application monetization and government demand versus broad software.",
    ),
    RatioSpec(
        "ORCL",
        "MSFT",
        "Oracle / Microsoft",
        "AI infrastructure and cloud challenger momentum versus the incumbent platform.",
    ),
    RatioSpec(
        "VST",
        "XLU",
        "Vistra / Utilities",
        "Merchant-power and data-center scarcity value versus regulated utilities.",
    ),
    RatioSpec(
        "ETN",
        "XLI",
        "Eaton / Industrials",
        "Electrification, grid equipment, and data-center power exposure versus industrials.",
    ),
    RatioSpec(
        "JPM",
        "KRE",
        "JPMorgan / Regional Banks",
        "Money-center balance-sheet strength versus regional-bank credit and funding risk.",
    ),
    RatioSpec(
        "GS",
        "JPM",
        "Goldman Sachs / JPMorgan",
        "Capital-markets and advisory activity versus diversified money-center banking.",
    ),
    RatioSpec(
        "COST",
        "WMT",
        "Costco / Walmart",
        "Membership-led and higher-income consumption versus mass-market defensive retail.",
    ),
    RatioSpec(
        "LLY",
        "NVO",
        "Eli Lilly / Novo Nordisk",
        "Direct obesity-market, pipeline, pricing, and execution leadership.",
    ),
    RatioSpec(
        "COIN",
        "IBIT",
        "Coinbase / Bitcoin ETF",
        "Crypto trading, custody, and regulatory beta versus spot Bitcoin exposure.",
    ),
    RatioSpec(
        "MSTR",
        "IBIT",
        "Strategy / Bitcoin ETF",
        "Balance-sheet leverage and equity premium versus spot Bitcoin exposure.",
    ),
]

RATIO_FAMILIES: Dict[str, List[RatioSpec]] = {
    "Cross-Asset Regime": (
        CORE_RATIO_SPECS[0:6]
        + [CORE_RATIO_SPECS[9], CORE_RATIO_SPECS[15]]
    ),
    "Equity Breadth / Global Leadership": (
        CORE_RATIO_SPECS[6:9]
        + CORE_RATIO_SPECS[10:14]
        + CORE_RATIO_SPECS[16:18]
        + [CORE_RATIO_SPECS[34]]
    ),
    "Rates / Credit / FX": (
        [CORE_RATIO_SPECS[14]] + CORE_RATIO_SPECS[18:24]
    ),
    "Commodities / Inflation": CORE_RATIO_SPECS[24:28],
    "Equity Themes / Cyclicals": CORE_RATIO_SPECS[28:34],
    "Single-Stock Relative Value": CORE_RATIO_SPECS[35:50],
}

# ============================== Sidebar ==================================
with st.sidebar:
    render_sidebar_about("11_Cross-Asset_Ratio_Chartbook.py")
    st.header("Chartbook")
    selected_families = st.multiselect(
        "Chart families",
        options=list(RATIO_FAMILIES.keys()),
        default=list(RATIO_FAMILIES.keys()),
    )

    st.markdown("---")
    st.header("Lookback")
    spans = {
        "3 Months": 90,
        "6 Months": 180,
        "9 Months": 270,
        "YTD": None,
        "1 Year": 365,
        "3 Years": 365 * 3,
        "5 Years": 365 * 5,
        "10 Years": 365 * 10,
        "20 Years": 365 * 20,
    }

    span_key = st.selectbox(
        "Period",
        list(spans.keys()),
        index=list(spans.keys()).index(DEFAULT_LOOKBACK),
    )

    st.markdown("---")
    with st.expander("Chart Settings", expanded=False):
        show_signal_strip = st.checkbox("Show signal strip", value=True)
        rsi_window = st.slider(
            "RSI window",
            min_value=5,
            max_value=30,
            value=DEFAULT_RSI_WINDOW,
            step=1,
        )
        show_rsi = st.checkbox("Show RSI pane", value=False)

        selected_mas = {}
        st.caption("Moving averages")
        for ma_len, default_value in MA_DEFAULTS.items():
            selected_mas[ma_len] = st.checkbox(f"{ma_len} DMA", value=default_value)

    st.markdown("---")
    st.header("Custom Ratios")
    custom_ratio_text = st.text_area(
        "Enter one or more ratios",
        value="",
        help="Use formats like GLD/SPY, SMH/QQQ, FXE/FXB, or one pair per line.",
        height=90,
    )

# ============================== Dates ====================================
market_date = date.today()
yf_end = market_date + timedelta(days=1)

if span_key == "YTD":
    display_days = max(1, (market_date - date(market_date.year, 1, 1)).days)
else:
    display_days = int(spans[span_key])

# Keep enough warm-up history for the 200-day average without downloading
# 25 years of prices on the default three-year view. Date-only bounds also
# make the Streamlit cache key stable throughout the trading day.
history_days = max(900, display_days + 450)
hist_start = market_date - timedelta(days=history_days)

if span_key == "YTD":
    disp_start = pd.Timestamp(date(market_date.year, 1, 1))
else:
    disp_start = pd.Timestamp(market_date - timedelta(days=display_days))

# ============================== Helpers ==================================
def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = clean_ticker(item)

        if item and item not in seen:
            seen.add(item)
            out.append(item)

    return out


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def parse_custom_ratio_text(text: str) -> List[RatioSpec]:
    if not text or not text.strip():
        return []

    raw_parts = re.split(r"[\n,;]+", text)
    specs: List[RatioSpec] = []

    for part in raw_parts:
        part = part.strip().upper()

        if not part:
            continue

        if "/" in part:
            pieces = [p.strip() for p in part.split("/") if p.strip()]
        else:
            pieces = [p.strip() for p in re.split(r"\s+", part) if p.strip()]

        if len(pieces) < 2:
            continue

        a = clean_ticker(pieces[0])
        b = clean_ticker(pieces[1])

        if a and b and a != b:
            specs.append(RatioSpec(a, b, f"{a} / {b}"))

    deduped = []
    seen = set()

    for spec in specs:
        key = (spec.ticker_1, spec.ticker_2)

        if key not in seen:
            seen.add(key)
            deduped.append(spec)

    return deduped


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    ticker_list = unique_keep_order(tickers)

    if not ticker_list:
        return pd.DataFrame()

    def _extract_close(block: pd.DataFrame) -> Optional[pd.Series]:
        if block is None or block.empty:
            return None

        for field in ("Close", "Adj Close"):
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce")

        numeric = block.select_dtypes(include=[np.number])

        if numeric.empty:
            return None

        return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    def _normalize(raw: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).astype(str)
            level1 = raw.columns.get_level_values(1).astype(str)

            tickers_in_level0 = set(level0).intersection(requested)
            tickers_in_level1 = set(level1).intersection(requested)

            if tickers_in_level0:
                for ticker in requested:
                    if ticker not in tickers_in_level0:
                        continue

                    try:
                        close = _extract_close(raw[ticker])

                        if close is not None:
                            out[ticker] = close
                    except Exception:
                        continue

            elif tickers_in_level1:
                for ticker in requested:
                    if ticker not in tickers_in_level1:
                        continue

                    for field in ("Close", "Adj Close"):
                        try:
                            if (field, ticker) in raw.columns:
                                out[ticker] = pd.to_numeric(
                                    raw[(field, ticker)],
                                    errors="coerce",
                                )
                                break
                        except Exception:
                            continue

        else:
            if len(requested) == 1:
                close = _extract_close(raw)

                if close is not None:
                    out[requested[0]] = close

        if out.empty:
            return pd.DataFrame()

        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.ffill()
        out = out.dropna(how="all")

        return out

    frames = []

    for batch in chunked(ticker_list, 30):
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = _normalize(raw, batch)

            if not normalized.empty:
                frames.append(normalized)
                continue
        except Exception:
            pass

        try:
            raw = yf.download(
                tickers=batch,
                period="max",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = _normalize(raw, batch)

            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.sort_index().ffill().dropna(how="all")

    return out


def first_valid_on_or_after(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[ts:]

    if not sub.empty:
        return sub.index[0], float(sub.iloc[0])

    return s.index[-1], float(s.iloc[-1])


def last_valid_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[:ts]

    if not sub.empty:
        return sub.index[-1], float(sub.iloc[-1])

    return s.index[0], float(s.iloc[0])


def rebase_series(series: pd.Series, base_date: pd.Timestamp, base: float = 100.0) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return pd.Series(dtype=float)

    base_ts, base_val = last_valid_on_or_before(s, base_date)

    if pd.isna(base_ts) or not np.isfinite(base_val) or base_val == 0:
        return pd.Series(dtype=float)

    return s / base_val * base


def compute_price_ratio(
    s1: pd.Series,
    s2: pd.Series,
    base_date: pd.Timestamp,
    base: float = 100.0,
) -> pd.Series:
    a, b = s1.align(s2, join="inner")
    raw_ratio = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
    return rebase_series(raw_ratio, base_date=base_date, base=base)


def rsi_wilder(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return pd.Series(dtype=float)

    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def period_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    prev = s.iloc[-periods - 1]
    latest = s.iloc[-1]

    if not np.isfinite(prev) or prev == 0:
        return np.nan

    return latest / prev - 1.0


def ytd_change(series: pd.Series) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    latest_date = s.index[-1]
    year_start = pd.Timestamp(datetime(latest_date.year, 1, 1))

    _, base_val = first_valid_on_or_after(s, year_start)
    latest = s.iloc[-1]

    if not np.isfinite(base_val) or base_val == 0:
        return np.nan

    return latest / base_val - 1.0


def days_since_window_extreme(series: pd.Series, window: int, kind: str) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    view = s.tail(window)

    if view.empty:
        return np.nan

    extreme_date = view.idxmax() if kind == "high" else view.idxmin()

    return float(len(s.loc[extreme_date:]) - 1)


def fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{value:+.1%}"


def fmt_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{value:,.1f}"


def fmt_days(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{int(value)}d"


def make_display_title(spec: RatioSpec) -> str:
    return f"{spec.label} ({spec.ticker_1}/{spec.ticker_2})"


def ratio_signal_line(
    ratio: pd.Series,
    rsi_len: int,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> str:
    s = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return "No usable data."

    latest = float(s.iloc[-1])
    latest_date = pd.Timestamp(s.index[-1]).date()
    today = pd.Timestamp(datetime.today()).date()
    stale = (today - latest_date).days > stale_days

    rsi = rsi_wilder(s, window=rsi_len).dropna()
    rsi_latest = float(rsi.iloc[-1]) if not rsi.empty else np.nan

    ma50 = s.rolling(50, min_periods=20).mean().dropna()
    ma100 = s.rolling(100, min_periods=40).mean().dropna()
    ma200 = s.rolling(200, min_periods=80).mean().dropna()

    vs_ma50 = latest / float(ma50.iloc[-1]) - 1.0 if not ma50.empty and ma50.iloc[-1] else np.nan
    vs_ma100 = latest / float(ma100.iloc[-1]) - 1.0 if not ma100.empty and ma100.iloc[-1] else np.nan
    vs_ma200 = latest / float(ma200.iloc[-1]) - 1.0 if not ma200.empty and ma200.iloc[-1] else np.nan

    parts = [
        f"Last {fmt_num(latest)}",
        f"1M {fmt_pct(period_change(s, 21))}",
        f"3M {fmt_pct(period_change(s, 63))}",
        f"6M {fmt_pct(period_change(s, 126))}",
        f"YTD {fmt_pct(ytd_change(s))}",
        f"RSI {fmt_num(rsi_latest)}",
        f"vs 50D {fmt_pct(vs_ma50)}",
        f"vs 100D {fmt_pct(vs_ma100)}",
        f"vs 200D {fmt_pct(vs_ma200)}",
        f"3M high {fmt_days(days_since_window_extreme(s, 63, 'high'))} ago",
        f"3M low {fmt_days(days_since_window_extreme(s, 63, 'low'))} ago",
        f"Last data {latest_date}",
    ]

    if stale:
        parts.append("data may be stale")

    return " | ".join(parts)


def make_empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(
        height=280,
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_fig(
    ratio: pd.Series,
    title: str,
    display_start: pd.Timestamp,
    ma_settings: Dict[int, bool],
    show_rsi_flag: bool,
    rsi_len: int,
    compact: bool = False,
) -> go.Figure:
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if ratio.empty:
        return make_empty_fig("No data")

    ratio_view = ratio.loc[display_start:].copy()

    if ratio_view.empty:
        ratio_view = ratio.copy()

    x_start = ratio_view.index.min()
    x_end = ratio_view.index.max()

    rows = 2 if show_rsi_flag else 1
    row_heights = [0.78, 0.22] if show_rsi_flag else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=ratio_view.index,
            y=ratio_view.values,
            mode="lines",
            name="Ratio",
            line={"color": "black", "width": 2.0},
            hovertemplate="%{y:.2f}<extra>Ratio</extra>",
        ),
        row=1,
        col=1,
    )

    visible_y = [ratio_view]

    for ma_len, enabled in sorted(ma_settings.items()):
        if not enabled:
            continue

        min_obs = max(2, min(ma_len, int(ma_len * 0.40)))
        ma = ratio.rolling(ma_len, min_periods=min_obs).mean()
        ma_view = ma.loc[x_start:x_end].dropna()

        if ma_view.empty:
            continue

        visible_y.append(ma_view)

        fig.add_trace(
            go.Scatter(
                x=ma_view.index,
                y=ma_view.values,
                mode="lines",
                name=f"{ma_len}D",
                line={"color": MA_COLORS.get(ma_len, "#555555"), "width": 1.2},
                hovertemplate=f"%{{y:.2f}}<extra>{ma_len}D</extra>",
            ),
            row=1,
            col=1,
        )

    latest_x = ratio_view.index[-1]
    latest_y = ratio_view.iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[latest_x],
            y=[latest_y],
            mode="markers",
            name="Last",
            marker={"color": "black", "size": 6},
            showlegend=False,
            hovertemplate="%{y:.2f}<extra>Last</extra>",
        ),
        row=1,
        col=1,
    )

    y_all = pd.concat(visible_y).replace([np.inf, -np.inf], np.nan).dropna()

    if not y_all.empty:
        ymin = float(y_all.min())
        ymax = float(y_all.max())
        pad = (ymax - ymin) * 0.06 if ymax != ymin else max(abs(ymin) * 0.05, 1.0)
        fig.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=1)

    if show_rsi_flag:
        rsi = rsi_wilder(ratio, window=rsi_len)
        rsi_view = rsi.loc[x_start:x_end].dropna()

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y2",
            x0=x_start,
            x1=x_end,
            y0=30,
            y1=70,
            fillcolor="gray",
            opacity=0.08,
            line_width=0,
        )

        fig.add_trace(
            go.Scatter(
                x=rsi_view.index,
                y=rsi_view.values,
                mode="lines",
                name="RSI",
                line={"color": "black", "width": 1.1},
                showlegend=False,
                hovertemplate="%{y:.1f}<extra>RSI</extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[70, 70],
                mode="lines",
                name="RSI 70",
                line={"color": "#b22222", "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[30, 30],
                mode="lines",
                name="RSI 30",
                line={"color": "#2e8b57", "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    fig.update_layout(
        title_text="",
        height=(430 if show_rsi_flag else 315) if compact else (620 if show_rsi_flag else 465),
        margin={"l": 42, "r": 20, "t": 42, "b": 34},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
    )

    fig.update_xaxes(
        range=[x_start, x_end],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Ratio Index",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        row=1,
        col=1,
    )

    return fig


def render_ratio_block(
    ratio: pd.Series,
    title: str,
    note: str,
    display_start: pd.Timestamp,
    ma_settings: Dict[int, bool],
    show_rsi_flag: bool,
    rsi_len: int,
    signal_strip: bool,
    chart_key: str,
    compact: bool = False,
):
    clean = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty:
        st.warning(f"No usable data for {title}.")
        return

    try:
        st.markdown(
            f'<div class="ratio-chart-heading">{html_escape(title)}</div>',
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)
        fig = make_fig(
            ratio=clean,
            title=title,
            display_start=display_start,
            ma_settings=ma_settings,
            show_rsi_flag=show_rsi_flag,
            rsi_len=rsi_len,
            compact=compact,
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=chart_key,
        )

    except Exception as e:
        st.warning(f"Chart rendering failed for {title}. Data loaded, but Plotly rejected the chart configuration.")
        st.caption(f"Error: {type(e).__name__}: {e}")

    if signal_strip:
        st.caption(ratio_signal_line(clean, rsi_len=rsi_len))

    st.markdown("---")


# ============================== Selected universe =========================
custom_specs = parse_custom_ratio_text(custom_ratio_text)

selected_groups: Dict[str, List[RatioSpec]] = {
    family: RATIO_FAMILIES[family]
    for family in selected_families
    if family in RATIO_FAMILIES
}

if not selected_groups:
    st.warning("Select at least one chart family in the sidebar.")
    st.stop()

static_specs = [spec for specs in selected_groups.values() for spec in specs]

static_tickers = unique_keep_order(
    [ticker for spec in static_specs for ticker in (spec.ticker_1, spec.ticker_2)]
)

custom_tickers = unique_keep_order(
    [ticker for spec in custom_specs for ticker in (spec.ticker_1, spec.ticker_2)]
)

# ============================== Fetch static data =========================
with st.spinner("Loading ratio data..."):
    closes_static = fetch_closes(tuple(static_tickers), hist_start, yf_end)

if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

# ============================== Render core ratios ========================
failed_pairs = []

def render_spec(spec: RatioSpec, compact: bool = False) -> None:
    a = spec.ticker_1
    b = spec.ticker_2
    if a not in closes_static.columns or b not in closes_static.columns:
        failed_pairs.append(f"{a}/{b}")
        return

    ratio = compute_price_ratio(
        closes_static[a],
        closes_static[b],
        base_date=disp_start,
        base=100.0,
    )
    if ratio.empty:
        failed_pairs.append(f"{a}/{b}")
        return

    render_ratio_block(
        ratio,
        make_display_title(spec),
        spec.note,
        disp_start,
        selected_mas,
        show_rsi,
        rsi_window,
        show_signal_strip,
        chart_key=f"core-ratio-{a}-{b}",
        compact=compact,
    )


for family, specs in selected_groups.items():
    st.subheader(family)
    for row_start in range(0, len(specs), 2):
        columns = st.columns(2, gap="large")
        for column_index, spec in enumerate(specs[row_start : row_start + 2]):
            with columns[column_index]:
                render_spec(spec, compact=True)

if failed_pairs:
    st.caption("Unavailable this session: " + ", ".join(sorted(set(failed_pairs))))

# ============================== Custom ratios =============================
if custom_specs:
    st.subheader("Custom Ratios")

    closes_custom = fetch_closes(tuple(custom_tickers), hist_start, yf_end)

    if closes_custom.empty:
        st.warning("No custom ratio data available.")
    else:
        custom_failed = []

        for custom_index, spec in enumerate(custom_specs):
            a = spec.ticker_1
            b = spec.ticker_2

            if a not in closes_custom.columns or b not in closes_custom.columns:
                custom_failed.append(f"{a}/{b}")
                continue

            ratio = compute_price_ratio(
                closes_custom[a],
                closes_custom[b],
                base_date=disp_start,
                base=100.0,
            )

            if ratio.empty:
                custom_failed.append(f"{a}/{b}")
                continue

            render_ratio_block(
                ratio,
                make_display_title(spec),
                spec.note,
                disp_start,
                selected_mas,
                show_rsi,
                rsi_window,
                show_signal_strip,
                chart_key=f"custom-ratio-{custom_index}-{a}-{b}",
                compact=False,
            )

        if custom_failed:
            st.caption("Unavailable custom ratios: " + ", ".join(sorted(set(custom_failed))))

render_footer()
