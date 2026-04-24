import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

# ============================== Page config ==============================
st.set_page_config(layout="wide", page_title="Ratio Charts")
st.title("Ratio Charts")
st.caption("Cross-asset and equity leadership ratios using Yahoo Finance adjusted daily close data.")

# ============================== Defaults =================================
DEFAULT_LOOKBACK = "5 Years"
DEFAULT_RSI_WINDOW = 14
DEFAULT_STALE_DAYS = 7
DEFAULT_MIN_BASKET_COVERAGE = 0.60

MA_DEFAULTS = {
    8: True,
    21: False,
    50: True,
    100: True,
    200: True,
}

MA_COLORS = {
    8: "#6c757d",
    21: "#9467bd",
    50: "#1f77b4",
    100: "#ff7f0e",
    200: "#d62728",
}

# ============================== Ratio specs ==============================
@dataclass(frozen=True)
class RatioSpec:
    ticker_1: str
    ticker_2: str
    label: str
    group: str
    note: str = ""


CYCLICALS = ["XLK", "XLI", "XLF", "XLY", "XLC", "XLB", "XLE"]
DEFENSIVES = ["XLP", "XLV", "XLU", "XLRE"]

PRESET_GROUPS: Dict[str, List[RatioSpec]] = {
    "Macro Regime": [
        RatioSpec("SPY", "TLT", "Equities / Long Treasuries", "Macro Regime", "Risk assets versus duration."),
        RatioSpec("HYG", "LQD", "High Yield / Investment Grade Credit", "Macro Regime", "Credit risk appetite."),
        RatioSpec("HYG", "IEF", "High Yield / Intermediate Treasuries", "Macro Regime", "Credit beta versus safer duration."),
        RatioSpec("TIP", "TLT", "TIPS / Nominal Long Treasuries", "Macro Regime", "Inflation-linked duration versus nominal duration."),
        RatioSpec("GLD", "TLT", "Gold / Long Treasuries", "Macro Regime", "Real-rate and hard-asset pressure."),
        RatioSpec("DBC", "TLT", "Broad Commodities / Long Treasuries", "Macro Regime", "Inflation assets versus duration."),
        RatioSpec("UUP", "GLD", "Dollar / Gold", "Macro Regime", "Dollar tightness versus gold."),
        RatioSpec("UUP", "SPY", "Dollar / S&P 500", "Macro Regime", "Dollar strength versus equity risk."),
        RatioSpec("XLE", "SPY", "Energy / S&P 500", "Macro Regime", "Energy leadership versus broad equities."),
    ],
    "Global Equity Leadership": [
        RatioSpec("VXUS", "SPY", "Ex-US Equities / US Equities", "Global Equity Leadership"),
        RatioSpec("EFA", "SPY", "Developed Ex-US / US Equities", "Global Equity Leadership"),
        RatioSpec("EEM", "SPY", "Emerging Markets / US Equities", "Global Equity Leadership"),
        RatioSpec("EZU", "SPY", "Europe / US Equities", "Global Equity Leadership"),
        RatioSpec("EWJ", "SPY", "Japan / US Equities", "Global Equity Leadership"),
        RatioSpec("FXI", "SPY", "China Large Caps / US Equities", "Global Equity Leadership"),
        RatioSpec("INDA", "SPY", "India / US Equities", "Global Equity Leadership"),
        RatioSpec("EWZ", "SPY", "Brazil / US Equities", "Global Equity Leadership"),
        RatioSpec("EWW", "SPY", "Mexico / US Equities", "Global Equity Leadership"),
    ],
    "Equity Internal Leadership": [
        RatioSpec("RSP", "SPY", "Equal Weight S&P 500 / Cap Weight S&P 500", "Equity Internal Leadership", "Breadth and concentration."),
        RatioSpec("IWM", "SPY", "Small Caps / S&P 500", "Equity Internal Leadership", "Domestic cyclicality and risk appetite."),
        RatioSpec("MDY", "SPY", "Mid Caps / S&P 500", "Equity Internal Leadership"),
        RatioSpec("QQQ", "IWM", "Nasdaq 100 / Small Caps", "Equity Internal Leadership", "Mega-cap growth versus domestic beta."),
        RatioSpec("XLY", "XLP", "Consumer Discretionary / Staples", "Equity Internal Leadership"),
        RatioSpec("XLF", "XLU", "Financials / Utilities", "Equity Internal Leadership"),
        RatioSpec("XLI", "XLU", "Industrials / Utilities", "Equity Internal Leadership"),
        RatioSpec("XLK", "SPY", "Technology / S&P 500", "Equity Internal Leadership"),
        RatioSpec("XLC", "SPY", "Communication Services / S&P 500", "Equity Internal Leadership"),
        RatioSpec("SMH", "QQQ", "Semis / Nasdaq 100", "Equity Internal Leadership"),
        RatioSpec("SMH", "IGV", "Semis / Software", "Equity Internal Leadership"),
        RatioSpec("IGV", "QQQ", "Software / Nasdaq 100", "Equity Internal Leadership"),
    ],
    "Factor Leadership": [
        RatioSpec("IWF", "IWD", "Growth / Value", "Factor Leadership"),
        RatioSpec("MTUM", "VLUE", "Momentum / Value", "Factor Leadership"),
        RatioSpec("QUAL", "SPY", "Quality / S&P 500", "Factor Leadership"),
        RatioSpec("USMV", "SPY", "Minimum Volatility / S&P 500", "Factor Leadership"),
        RatioSpec("SPLV", "SPY", "Low Volatility / S&P 500", "Factor Leadership"),
        RatioSpec("RPV", "RPG", "Pure Value / Pure Growth", "Factor Leadership"),
    ],
    "Rates and Housing Sensitivity": [
        RatioSpec("TLT", "IEF", "Long Duration / Intermediate Duration", "Rates and Housing Sensitivity"),
        RatioSpec("TLT", "SHY", "Long Duration / Short Duration", "Rates and Housing Sensitivity"),
        RatioSpec("KRE", "XLF", "Regional Banks / Financials", "Rates and Housing Sensitivity"),
        RatioSpec("ITB", "XLRE", "Homebuilders / Real Estate", "Rates and Housing Sensitivity"),
        RatioSpec("XHB", "XLRE", "Housing Basket / Real Estate", "Rates and Housing Sensitivity"),
        RatioSpec("VNQ", "SPY", "REITs / S&P 500", "Rates and Housing Sensitivity"),
    ],
    "Commodities and Hard Assets": [
        RatioSpec("GDX", "GLD", "Gold Miners / Gold", "Commodities and Hard Assets"),
        RatioSpec("SLV", "GLD", "Silver / Gold", "Commodities and Hard Assets"),
        RatioSpec("COPX", "GLD", "Copper Miners / Gold", "Commodities and Hard Assets"),
        RatioSpec("XME", "SPY", "Metals and Mining / S&P 500", "Commodities and Hard Assets"),
        RatioSpec("XOP", "XLE", "E&P / Energy Sector", "Commodities and Hard Assets"),
        RatioSpec("OIH", "XLE", "Oil Services / Energy Sector", "Commodities and Hard Assets"),
        RatioSpec("DBA", "DBC", "Agriculture / Broad Commodities", "Commodities and Hard Assets"),
        RatioSpec("URA", "SPY", "Uranium / S&P 500", "Commodities and Hard Assets"),
    ],
    "AI and Growth Stack": [
        RatioSpec("SMH", "SPY", "Semiconductors / S&P 500", "AI and Growth Stack"),
        RatioSpec("SOXX", "QQQ", "Semiconductors / Nasdaq 100", "AI and Growth Stack"),
        RatioSpec("NVDA", "SMH", "Nvidia / Semiconductors", "AI and Growth Stack"),
        RatioSpec("AMD", "SMH", "AMD / Semiconductors", "AI and Growth Stack"),
        RatioSpec("MSFT", "QQQ", "Microsoft / Nasdaq 100", "AI and Growth Stack"),
        RatioSpec("META", "QQQ", "Meta / Nasdaq 100", "AI and Growth Stack"),
        RatioSpec("WCLD", "QQQ", "Cloud Software / Nasdaq 100", "AI and Growth Stack"),
        RatioSpec("ARKK", "QQQ", "Speculative Growth / Nasdaq 100", "AI and Growth Stack"),
    ],
    "Currency Proxy Leadership": [
        RatioSpec("FXY", "UUP", "Yen ETF / Dollar Index ETF", "Currency Proxy Leadership"),
        RatioSpec("FXE", "UUP", "Euro ETF / Dollar Index ETF", "Currency Proxy Leadership"),
        RatioSpec("FXF", "UUP", "Swiss Franc ETF / Dollar Index ETF", "Currency Proxy Leadership"),
        RatioSpec("FXB", "UUP", "British Pound ETF / Dollar Index ETF", "Currency Proxy Leadership"),
        RatioSpec("CEW", "UUP", "EM Currency Basket / Dollar Index ETF", "Currency Proxy Leadership"),
    ],
}

GROUP_DESCRIPTIONS = {
    "Macro Regime": "Core cross-asset ratios for risk appetite, inflation pressure, dollar tightness, gold, credit, and duration.",
    "Global Equity Leadership": "Regional leadership versus US equities.",
    "Equity Internal Leadership": "Breadth, concentration, cyclicals versus defensives, and growth leadership inside the US equity market.",
    "Factor Leadership": "Growth, value, momentum, quality, and defensive factor leadership.",
    "Rates and Housing Sensitivity": "Duration, banks, housing, and real estate leadership.",
    "Commodities and Hard Assets": "Gold, miners, energy, uranium, metals, and commodity sub-leadership.",
    "AI and Growth Stack": "Semis, software, mega-cap AI, and speculative growth leadership.",
    "Currency Proxy Leadership": "ETF proxies for currency leadership versus the US dollar ETF.",
}

# ============================== Sidebar ==================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Ratio chart workspace for cross-asset regime framing, leadership analysis, and tape confirmation.

        **How to read it**
        - A rising ratio means the first ticker is outperforming the second ticker.
        - Ratios are rebased to 100 at the selected lookback start date.
        - The signal line below each chart gives quick trend, momentum, and stale-data context.

        **Data source:** Yahoo Finance adjusted daily close history.
        """
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
    st.header("Coverage")
    selected_groups = st.multiselect(
        "Preset groups",
        list(PRESET_GROUPS.keys()),
        default=list(PRESET_GROUPS.keys()),
    )

    st.markdown("---")
    with st.expander("Chart Settings", expanded=False):
        show_signal_strip = st.checkbox("Show signal strip", value=True)
        rsi_window = st.slider("RSI window", min_value=5, max_value=30, value=DEFAULT_RSI_WINDOW, step=1)
        show_rsi = st.checkbox("Show RSI pane", value=True)

        selected_mas = {}
        st.caption("Moving averages")
        for ma_len, default_value in MA_DEFAULTS.items():
            selected_mas[ma_len] = st.checkbox(f"{ma_len} DMA", value=default_value)

        min_basket_coverage = st.slider(
            "Minimum basket coverage",
            min_value=0.30,
            max_value=1.00,
            value=DEFAULT_MIN_BASKET_COVERAGE,
            step=0.05,
            help="Minimum share of basket constituents required before an equal-weight basket value is calculated.",
        )

    st.markdown("---")
    st.header("Custom Ratios")
    custom_ratio_text = st.text_area(
        "Enter one or more ratios",
        value="GLD/TLT",
        help="Use formats like GLD/TLT, SMH/QQQ, or one pair per line.",
        height=90,
    )

# ============================== Dates ====================================
now = datetime.today()
yf_end = now + timedelta(days=1)
hist_start = now - timedelta(days=365 * 25)

if span_key == "YTD":
    disp_start = pd.Timestamp(datetime(now.year, 1, 1))
else:
    disp_start = pd.Timestamp(now - timedelta(days=spans[span_key]))

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
            specs.append(RatioSpec(a, b, f"{a} / {b}", "Custom Ratios"))

    deduped = []
    seen = set()
    for spec in specs:
        key = (spec.ticker_1, spec.ticker_2)
        if key not in seen:
            seen.add(key)
            deduped.append(spec)

    return deduped


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers: Tuple[str, ...], start: datetime, end: datetime) -> pd.DataFrame:
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
                                out[ticker] = pd.to_numeric(raw[(field, ticker)], errors="coerce")
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


def normalize_prices_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(index=prices.index)

    out = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        s = pd.to_numeric(prices[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        first = s.first_valid_index()

        if first is None:
            continue

        base_val = s.loc[first]

        if not np.isfinite(base_val) or base_val == 0:
            continue

        out[col] = s / base_val * 100.0

    return out


def equal_weight_basket(prices: pd.DataFrame, min_coverage: float = 0.60) -> pd.Series:
    normalized = normalize_prices_to_100(prices)

    if normalized.empty:
        return pd.Series(dtype=float)

    required = max(1, int(ceil(len(normalized.columns) * min_coverage)))
    counts = normalized.notna().sum(axis=1)
    basket = normalized.mean(axis=1, skipna=True)
    basket = basket.where(counts >= required)

    return basket.dropna()


def compute_basket_ratio(
    basket_1: pd.Series,
    basket_2: pd.Series,
    base_date: pd.Timestamp,
    base: float = 100.0,
) -> pd.Series:
    a, b = basket_1.align(basket_2, join="inner")
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


def ratio_signal_line(ratio: pd.Series, rsi_len: int, stale_days: int = DEFAULT_STALE_DAYS) -> str:
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
        title_text=title,
        height=540 if show_rsi_flag else 380,
        margin={"l": 40, "r": 22, "t": 54, "b": 34},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x",
        showlegend=True,
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
    display_start: pd.Timestamp,
    ma_settings: Dict[int, bool],
    show_rsi_flag: bool,
    rsi_len: int,
    signal_strip: bool,
    note: str = "",
):
    clean = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if clean.empty:
        st.warning(f"No usable data for {title}.")
        return

    if note:
        st.caption(note)

    try:
        fig = make_fig(
            ratio=clean,
            title=title,
            display_start=display_start,
            ma_settings=ma_settings,
            show_rsi_flag=show_rsi_flag,
            rsi_len=rsi_len,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Chart rendering failed for {title}. Data loaded, but Plotly rejected the chart configuration.")
        st.caption(f"Error: {type(e).__name__}: {e}")

    if signal_strip:
        st.caption(ratio_signal_line(clean, rsi_len=rsi_len))

    st.markdown("---")


# ============================== Selected universe =========================
custom_specs = parse_custom_ratio_text(custom_ratio_text)

selected_presets: List[RatioSpec] = []
for group in selected_groups:
    selected_presets.extend(PRESET_GROUPS.get(group, []))

static_tickers = unique_keep_order(
    CYCLICALS
    + DEFENSIVES
    + [ticker for spec in selected_presets for ticker in (spec.ticker_1, spec.ticker_2)]
)

custom_tickers = unique_keep_order(
    [ticker for spec in custom_specs for ticker in (spec.ticker_1, spec.ticker_2)]
)

# ============================== Fetch static data =========================
with st.spinner("Downloading price history..."):
    closes_static = fetch_closes(tuple(static_tickers), hist_start, yf_end)

if closes_static.empty:
    st.error("Failed to download price data.")
    st.stop()

# ============================== Headline basket ===========================
st.subheader("Headline Basket")

available_cyc = [t for t in CYCLICALS if t in closes_static.columns]
available_def = [t for t in DEFENSIVES if t in closes_static.columns]

if available_cyc and available_def:
    cyc_basket = equal_weight_basket(closes_static[available_cyc], min_coverage=min_basket_coverage)
    def_basket = equal_weight_basket(closes_static[available_def], min_coverage=min_basket_coverage)
    cyc_def_ratio = compute_basket_ratio(cyc_basket, def_basket, base_date=disp_start, base=100.0)

    render_ratio_block(
        cyc_def_ratio,
        "Cyclicals / Defensives Equal Weight Basket",
        disp_start,
        selected_mas,
        show_rsi,
        rsi_window,
        show_signal_strip,
        note=f"Cyclicals: {', '.join(available_cyc)} | Defensives: {', '.join(available_def)}",
    )
else:
    st.warning("Unable to build the cyclicals / defensives basket with current data.")

# ============================== Preset ratios =============================
st.subheader("Preset Ratio Charts")

failed_pairs = []

if not selected_presets:
    st.info("Select at least one preset group in the sidebar.")
else:
    for group in selected_groups:
        group_specs = [spec for spec in selected_presets if spec.group == group]

        if not group_specs:
            continue

        st.markdown(f"### {group}")
        st.caption(GROUP_DESCRIPTIONS.get(group, ""))

        for spec in group_specs:
            a = spec.ticker_1
            b = spec.ticker_2

            if a in closes_static.columns and b in closes_static.columns:
                ratio = compute_price_ratio(
                    closes_static[a],
                    closes_static[b],
                    base_date=disp_start,
                    base=100.0,
                )

                if ratio.empty:
                    failed_pairs.append(f"{a}/{b}")
                    continue

                render_ratio_block(
                    ratio,
                    make_display_title(spec),
                    disp_start,
                    selected_mas,
                    show_rsi,
                    rsi_window,
                    show_signal_strip,
                    note=spec.note,
                )

            else:
                failed_pairs.append(f"{a}/{b}")

if failed_pairs:
    st.caption("Unavailable this session: " + ", ".join(sorted(set(failed_pairs))))

# ============================== Custom ratios =============================
st.subheader("Custom Ratios")

if not custom_specs:
    st.info("Enter custom pairs in the sidebar using the format TICKER1/TICKER2.")
else:
    closes_custom = fetch_closes(tuple(custom_tickers), hist_start, yf_end)

    if closes_custom.empty:
        st.warning("No custom ratio data available.")
    else:
        custom_failed = []

        for spec in custom_specs:
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
                disp_start,
                selected_mas,
                show_rsi,
                rsi_window,
                show_signal_strip,
            )

        if custom_failed:
            st.caption("Unavailable custom ratios: " + ", ".join(sorted(set(custom_failed))))

st.caption("© 2026 AD Fund Management LP")
