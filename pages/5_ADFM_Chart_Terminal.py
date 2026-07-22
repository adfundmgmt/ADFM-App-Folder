import streamlit as st

from adfm_core.chart_patterns import PatternDetection, PatternLine, detect_chart_patterns
from adfm_core.ui import render_footer
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from dataclasses import dataclass
from html import escape as html_escape
from textwrap import dedent

from plotly.subplots import make_subplots


# ============================================================
# Page
# ============================================================

TITLE = "ADFM Chart Terminal"

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants
# ============================================================

PERIOD_OPTIONS = ["5d", "1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "10y", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]
CHART_TYPES = ["Candles", "Line"]

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
CAP_MAX_ROWS = 250_000
APP_VERSION = "2026-07-22-automatic-chart-patterns"

WATCHLISTS = {
    "Index": ["^SPX", "^NDX", "SPY", "QQQ", "IWM", "DIA"],
    "Mag 7": ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA"],
    "Rates / Credit": ["TLT", "IEF", "SHY", "HYG", "LQD", "JNK"],
    "Commodities": ["USO", "BNO", "GLD", "SLV", "CPER", "UNG"],
    "FX": ["DX-Y.NYB", "USDJPY=X", "EURUSD=X", "EURGBP=X", "GBPUSD=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "IBIT", "ETHE"],
}

COLORS = {
    "up": "rgba(46,139,87,0.82)",
    "down": "rgba(178,34,34,0.82)",
    "sma8": "#6c757d",
    "sma20": "#9467bd",
    "sma50": "#1f77b4",
    "sma100": "#ff7f0e",
    "sma200": "#d62728",
    "sma65": "#17becf",
    "sma130": "#bcbd22",
    "sma195": "#8c564b",
    "sma260": "#e377c2",
    "bb": "rgba(70,70,70,0.38)",
    "bb_fill": "rgba(120,120,120,0.05)",
    "grid": "rgba(0,0,0,0.08)",
    "text": "#1f2933",
    "muted": "#6b7280",
    "volume_up": "rgba(46,139,87,0.38)",
    "volume_down": "rgba(178,34,34,0.34)",
    "volume_neutral": "rgba(150,150,150,0.32)",
    "rsi": "#111111",
    "macd": "#1f77b4",
    "signal": "#ff7f0e",
    "hist_up": "rgba(46,139,87,0.48)",
    "hist_down": "rgba(178,34,34,0.45)",
    "range": "rgba(55,65,81,0.55)",
    "last": "rgba(17,24,39,0.78)",
    "fib_extension": "rgba(46,139,87,0.78)",
    "fib_retracement": "rgba(55,65,81,0.58)",
    "fib_anchor": "rgba(17,24,39,0.55)",
    "pattern_bullish": "#0f9d75",
    "pattern_bearish": "#d6455d",
    "pattern_bilateral": "#3b82f6",
    "pattern_support": "rgba(14,116,144,0.86)",
    "pattern_resistance": "rgba(147,51,234,0.82)",
    "pattern_muted": "rgba(71,85,105,0.72)",
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    dedent(
        """
        <style>
        .stApp {
            background: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background: #f0f2f6;
        }

        .block-container {
            padding-top: 1.85rem;
            padding-bottom: 1.0rem;
            max-width: 100%;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }

        .adfm-header {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 14px;
            padding: 14px 16px 12px 16px;
            background: #ffffff;
            margin-bottom: 4px;
        }

        .adfm-title {
            font-size: 1.35rem;
            line-height: 1.25;
            font-weight: 750;
            color: #111827;
            margin: 0;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .adfm-subtitle {
            font-size: 0.82rem;
            color: #6b7280;
            margin-top: 4px;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(8, minmax(95px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }

        .metric-card {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 12px;
            padding: 8px 10px;
            background: #ffffff;
            min-height: 58px;
        }

        .metric-label {
            font-size: 0.70rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 2px;
        }

        .metric-value {
            font-size: 1.02rem;
            line-height: 1.20;
            font-weight: 700;
            color: #111827;
            white-space: nowrap;
        }

        .metric-note {
            font-size: 0.70rem;
            color: #6b7280;
            margin-top: 2px;
        }

        .signal-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }

        .signal-table th {
            text-align: left;
            color: #6b7280;
            font-weight: 650;
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            padding: 7px 8px;
        }

        .signal-table td {
            border-bottom: 1px solid rgba(49, 51, 63, 0.08);
            padding: 8px;
            vertical-align: top;
        }

        .memo-box {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 12px;
            padding: 12px 14px;
            background: #ffffff;
            font-size: 0.93rem;
            line-height: 1.5;
            color: #1f2933;
        }

        .chart-top-gap {
            height: 18px;
        }

        .compare-chart-top-gap {
            height: 10px;
        }

        .pattern-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(210px, 1fr));
            gap: 9px;
            margin: -2px 0 12px 0;
        }

        .pattern-card {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-left: 4px solid var(--pattern-accent);
            border-radius: 12px;
            padding: 10px 12px;
            background: linear-gradient(135deg, var(--pattern-tint), #ffffff 72%);
            min-height: 92px;
        }

        .pattern-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }

        .pattern-card-name {
            color: #111827;
            font-size: 0.88rem;
            font-weight: 750;
        }

        .pattern-card-score {
            color: var(--pattern-accent);
            background: rgba(255,255,255,0.82);
            border: 1px solid color-mix(in srgb, var(--pattern-accent) 22%, transparent);
            border-radius: 999px;
            padding: 2px 7px;
            font-size: 0.68rem;
            font-weight: 750;
            white-space: nowrap;
        }

        .pattern-card-meta {
            color: #475569;
            font-size: 0.72rem;
            font-weight: 650;
            margin-top: 5px;
        }

        .pattern-card-levels {
            color: #64748b;
            font-size: 0.70rem;
            margin-top: 5px;
        }

        .pattern-empty {
            border: 1px dashed rgba(49, 51, 63, 0.16);
            border-radius: 12px;
            color: #64748b;
            font-size: 0.78rem;
            padding: 10px 12px;
            margin: -2px 0 12px 0;
            background: #fafafa;
        }

        @media (max-width: 1200px) {
            .metric-strip {
                grid-template-columns: repeat(4, minmax(95px, 1fr));
            }

            .pattern-strip {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 700px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(95px, 1fr));
            }
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)

st.title(TITLE)


# ============================================================
# Settings
# ============================================================

@dataclass(frozen=True)
class ChartSettings:
    ticker: str
    period: str
    interval: str
    chart_type: str
    auto_adjust: bool
    log_scale: bool
    tight_y_axis: bool

    show_sma8: bool
    show_sma20: bool
    show_sma50: bool
    show_sma100: bool
    show_sma200: bool
    show_sma65: bool
    show_sma130: bool
    show_sma195: bool
    show_sma260: bool

    show_bbands: bool
    show_last_price: bool
    show_range_levels: bool

    show_volume: bool
    show_rsi: bool
    show_macd: bool
    show_elliott_wave: bool
    show_fibonacci: bool
    show_chart_patterns: bool

    compare_tickers: str
    compare_mode: str


def ensure_session_defaults() -> None:
    defaults = {
        "ticker_input": "^SPX",
        "period_input": "1y",
        "interval_input": "1d",
        "chart_type_input": "Candles",
        "compare_input": "",
    }

    # Reset once after this fix is deployed so stale Streamlit session state
    # does not keep the old YTD default / broken render state alive.
    if st.session_state.get("_adfm_chart_terminal_version") != APP_VERSION:
        for key, value in defaults.items():
            st.session_state[key] = value
        st.session_state["_adfm_chart_terminal_version"] = APP_VERSION
        return

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def parse_compare_tickers(raw: str, primary: str) -> list[str]:
    if not raw:
        return []

    tickers = []
    seen = {normalize_ticker(primary)}

    for part in raw.replace(";", ",").split(","):
        ticker = normalize_ticker(part)
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    return tickers[:8]


def queue_watchlist_ticker(ticker: str) -> None:
    # Do not write directly into ticker_input inside the same run after the
    # text_input widget has been instantiated. Streamlit disallows that.
    # Queue the value, then apply it at the top of the next run before the
    # widget is created.
    st.session_state["_queued_ticker_input"] = normalize_ticker(ticker)


def apply_queued_ticker_update() -> None:
    queued_ticker = st.session_state.pop("_queued_ticker_input", None)

    if queued_ticker:
        st.session_state["ticker_input"] = queued_ticker


def read_settings() -> ChartSettings:
    ensure_session_defaults()
    apply_queued_ticker_update()

    with st.sidebar:
        st.header("About This Tool")
        st.markdown(
            """
            **Purpose:** Single-name technical chart workspace for regime framing, trend confirmation, momentum checks, and trade invalidation.

            **How to read it**
            - The header gives price, return, drawdown, and volatility context.
            - The chart shows price, moving averages, Bollinger Bands, volume, RSI, MACD, and optional automatic pattern recognition.
            - The signal matrix turns the tape into trend, momentum, volatility, structure, and risk-level context.

            **Data source:** Yahoo Finance OHLCV history.
            """
        )

        st.markdown("---")
        st.header("Chart Controls")

        ticker = st.text_input(
            "Ticker",
            key="ticker_input",
            placeholder="^SPX, NVDA, TLT, USDJPY=X",
        )

        period = st.selectbox(
            "Window",
            PERIOD_OPTIONS,
            index=PERIOD_OPTIONS.index(st.session_state.get("period_input", "1y")),
            key="period_input",
        )

        interval = st.selectbox(
            "Interval",
            INTERVAL_OPTIONS,
            index=INTERVAL_OPTIONS.index(st.session_state.get("interval_input", "1d")),
            key="interval_input",
        )

        chart_type = st.selectbox(
            "Chart Type",
            CHART_TYPES,
            index=CHART_TYPES.index(st.session_state.get("chart_type_input", "Candles")),
            key="chart_type_input",
        )

        compare_tickers = st.text_input(
            "Compare",
            key="compare_input",
            placeholder="SPY, QQQ, TLT",
            help="Optional. Enter tickers separated by commas. The relative chart is indexed to 100.",
        )

        st.markdown("---")
        st.header("Watchlists")

        selected_watchlist = st.selectbox(
            "Group",
            list(WATCHLISTS.keys()),
            index=0,
        )

        cols = st.columns(2)
        for i, watch_ticker in enumerate(WATCHLISTS[selected_watchlist]):
            with cols[i % 2]:
                st.button(
                    watch_ticker,
                    key=f"watch_{selected_watchlist}_{watch_ticker}",
                    use_container_width=True,
                    on_click=queue_watchlist_ticker,
                    args=(watch_ticker,),
                )

        st.markdown("---")
        with st.expander("Chart Settings", expanded=False):
            show_last_price = st.checkbox("Last price line", value=False)
            show_bbands = st.checkbox("Bollinger Bands", value=True)

            st.caption("Moving averages")
            show_sma8 = st.checkbox("8 DMA", value=True)
            show_sma20 = st.checkbox("20 DMA", value=True)
            show_sma50 = st.checkbox("50 DMA", value=True)
            show_sma100 = st.checkbox("100 DMA", value=True)
            show_sma200 = st.checkbox("200 DMA", value=True)
            show_sma65 = st.checkbox("1Q MA (65 DMA)", value=False)
            show_sma130 = st.checkbox("2Q MA (130 DMA)", value=False)
            show_sma195 = st.checkbox("3Q MA (195 DMA)", value=False)
            show_sma260 = st.checkbox("4Q MA (260 DMA)", value=False)

            st.caption("Panels and overlays")
            show_chart_patterns = st.toggle(
                "Automatic chart patterns",
                value=False,
                help="Ranks up to three volatility-adjusted reversal, continuation, and breakout structures. Confirmation requires a close through the inferred trigger.",
            )
            show_volume = st.checkbox("Volume", value=True)
            show_rsi = st.checkbox("RSI", value=True)
            show_macd = st.checkbox("MACD", value=True)
            show_elliott_wave = st.checkbox(
                "Elliott Wave pivot map",
                value=False,
                help="Heuristic swing-pivot overlay. Use it as pattern context, not a deterministic signal.",
            )
            show_fibonacci = st.checkbox(
                "Fibonacci levels",
                value=False,
                help="Auto-draws retracement and extension levels from the dominant qualified Elliott swing.",
            )

        auto_adjust = False
        log_scale = False
        tight_y_axis = True
        show_range_levels = False

    compare_mode = "Indexed to 100"

    return ChartSettings(
        ticker=normalize_ticker(ticker),
        period=period,
        interval=interval,
        chart_type=chart_type,
        auto_adjust=auto_adjust,
        log_scale=log_scale,
        tight_y_axis=tight_y_axis,
        show_sma8=show_sma8,
        show_sma20=show_sma20,
        show_sma50=show_sma50,
        show_sma100=show_sma100,
        show_sma200=show_sma200,
        show_sma65=show_sma65,
        show_sma130=show_sma130,
        show_sma195=show_sma195,
        show_sma260=show_sma260,
        show_bbands=show_bbands,
        show_last_price=show_last_price,
        show_range_levels=show_range_levels,
        show_volume=show_volume,
        show_rsi=show_rsi,
        show_macd=show_macd,
        show_elliott_wave=show_elliott_wave,
        show_fibonacci=show_fibonacci,
        show_chart_patterns=show_chart_patterns,
        compare_tickers=compare_tickers,
        compare_mode=compare_mode,
    )


# ============================================================
# Dates
# ============================================================

def today_normalized() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def start_date_from_period(period: str) -> pd.Timestamp | None:
    today = today_normalized()

    if period == "max":
        return None

    if period == "YTD":
        return pd.Timestamp(year=today.year, month=1, day=1)

    if period.endswith("d"):
        days = int(period[:-1])
        return today - pd.DateOffset(days=days)

    if period.endswith("mo"):
        months = int(period[:-2])
        return today - pd.DateOffset(months=months)

    if period.endswith("y"):
        years = int(period[:-1])
        return today - pd.DateOffset(years=years)

    return None


def warmup_start_date(period: str, interval: str) -> pd.Timestamp | None:
    display_start = start_date_from_period(period)

    if display_start is None:
        return None

    if interval == "1d":
        return display_start - pd.DateOffset(years=3)

    if interval == "1wk":
        return display_start - pd.DateOffset(years=6)

    return display_start - pd.DateOffset(years=15)


def format_last_bar(index_value: pd.Timestamp) -> str:
    try:
        return pd.Timestamp(index_value).strftime("%b %d, %Y")
    except Exception:
        return "Unavailable"


# ============================================================
# Data Fetching and Cleaning
# ============================================================

def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    required = set(REQUIRED_PRICE_COLUMNS)

    level_0 = list(df.columns.get_level_values(0))
    level_1 = list(df.columns.get_level_values(1))

    if required.issubset(set(level_0)):
        out = df.copy()
        out.columns = df.columns.get_level_values(0)
        return out

    if required.issubset(set(level_1)):
        out = df.copy()
        out.columns = df.columns.get_level_values(1)
        return out

    out = df.copy()
    out.columns = ["_".join(str(x) for x in col if str(x) != "") for col in df.columns]
    return out


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = flatten_yfinance_columns(df).copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    rename_map = {}
    for col in out.columns:
        clean = str(col).strip()
        if clean in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            rename_map[col] = clean

    out = out.rename(columns=rename_map)

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in out.columns]
    if missing:
        return pd.DataFrame()

    for col in REQUIRED_PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    else:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.dropna(subset=REQUIRED_PRICE_COLUMNS)

    if len(out) > CAP_MAX_ROWS:
        out = out.tail(CAP_MAX_ROWS)

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_with_yfinance_download(
    ticker: str,
    interval: str,
    auto_adjust: bool,
    fetch_start: str | None,
) -> pd.DataFrame:
    if fetch_start is None:
        raw = yf.download(
            tickers=ticker,
            period="max",
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
            progress=False,
            threads=False,
        )
    else:
        raw = yf.download(
            tickers=ticker,
            start=fetch_start,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
            progress=False,
            threads=False,
        )

    return clean_price_data(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_with_ticker_history(
    ticker: str,
    interval: str,
    auto_adjust: bool,
    fetch_start: str | None,
    fallback_period: str,
) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)

    if fetch_start is None:
        raw = ticker_obj.history(
            period="max",
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )
    else:
        raw = ticker_obj.history(
            start=fetch_start,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )

    cleaned = clean_price_data(raw)

    if cleaned.empty:
        fallback_period_clean = "1y" if fallback_period == "YTD" else fallback_period
        raw = ticker_obj.history(
            period=fallback_period_clean,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )
        cleaned = clean_price_data(raw)

    return cleaned


def fetch_history(settings: ChartSettings) -> tuple[pd.DataFrame, str | None]:
    fetch_start = warmup_start_date(settings.period, settings.interval)
    fetch_start_str = fetch_start.strftime("%Y-%m-%d") if fetch_start is not None else None

    try:
        df = fetch_with_yfinance_download(
            ticker=settings.ticker,
            interval=settings.interval,
            auto_adjust=settings.auto_adjust,
            fetch_start=fetch_start_str,
        )

        if not df.empty:
            return df, None

    except Exception as exc:
        download_error = f"{type(exc).__name__}: {exc}"
    else:
        download_error = "Primary price fetch returned no data."

    try:
        df = fetch_with_ticker_history(
            ticker=settings.ticker,
            interval=settings.interval,
            auto_adjust=settings.auto_adjust,
            fetch_start=fetch_start_str,
            fallback_period=settings.period,
        )

        if not df.empty:
            return df, None

        return pd.DataFrame(), download_error

    except Exception as exc:
        history_error = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), f"{download_error} | fallback failed: {history_error}"


def prepare_display_window(df: pd.DataFrame, settings: ChartSettings) -> pd.DataFrame:
    out = df.copy()

    if settings.interval == "1d":
        out = out[out.index.weekday < 5].copy()

    start = start_date_from_period(settings.period)

    if start is None:
        return out

    display = out[out.index >= start].copy()

    if display.empty:
        return out.tail(300).copy()

    return display


def has_usable_volume(df: pd.DataFrame) -> bool:
    if "Volume" not in df.columns:
        return False

    vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()

    if vol.empty:
        return False

    positive_count = int((vol > 0).sum())
    required_count = max(5, int(len(vol) * 0.10))

    return positive_count >= required_count


# ============================================================
# Indicators
# ============================================================

def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)

    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line

    return macd, signal_line, hist


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)

    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()

    upper = mid + mult * std
    lower = mid - mult * std

    return mid, upper, lower


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in [8, 20, 50, 65, 100, 130, 195, 200, 260]:
        out[f"SMA{window}"] = out["Close"].rolling(
            window=window,
            min_periods=window,
        ).mean()

    out["RSI14"] = compute_rsi(out["Close"], length=14)

    macd, signal, hist = compute_macd(out["Close"])
    out["MACD"] = macd
    out["MACD_SIGNAL"] = signal
    out["MACD_HIST"] = hist

    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(out["Close"])
    out["BB_MID"] = bb_mid
    out["BB_UPPER"] = bb_upper
    out["BB_LOWER"] = bb_lower

    out["ATR14"] = compute_atr(out)
    out["ATR14_PCT"] = out["ATR14"] / out["Close"].replace(0, np.nan)

    out["ROLLING_VOL_20"] = out["Close"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)

    out["DRAWDOWN_252"] = (out["Close"] / out["Close"].rolling(252, min_periods=30).max()) - 1.0

    return out


# ============================================================
# Metrics and Signals
# ============================================================

def fmt_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    if abs(value) >= 10:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:,.3f}"

    return f"{value:,.4f}"


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:+.2f}%"


def fmt_pct_abs(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def close_asof(df: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    if df.empty:
        return None

    eligible = df[df.index <= target_date]

    if eligible.empty:
        eligible = df[df.index >= target_date]

    if eligible.empty:
        return None

    return float(eligible["Close"].iloc[-1])


def return_from_close(latest_close: float, base_close: float | None) -> float | None:
    if base_close is None or pd.isna(base_close) or base_close == 0:
        return None

    return (latest_close / float(base_close)) - 1.0


def interval_bar_unit(interval: str) -> tuple[str, str]:
    if interval == "1wk":
        return "W", "week"

    if interval == "1mo":
        return "M", "month"

    return "D", "day"


def interval_bar_label(interval: str, bars: int) -> str:
    suffix, _ = interval_bar_unit(interval)
    return f"{bars}{suffix}"


def interval_bar_note(interval: str, bars: int) -> str:
    _, unit = interval_bar_unit(interval)
    plural = "" if bars == 1 else "s"
    return f"{bars} {unit}{plural}"


def atr_note_for_interval(interval: str) -> str:
    _, unit = interval_bar_unit(interval)
    return f"14-{unit}"


def drawdown_note_for_interval(interval: str) -> str:
    if interval == "1mo":
        return "From 12M high"

    return "From 52W high"


def header_horizon_keys(interval: str) -> list[tuple[str, str]]:
    if interval == "1mo":
        return [("3M", "3M"), ("1Y", "1Y")]

    return [("1M", "1M"), ("3M", "3M")]


def compute_return_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty:
        return {}

    close = df["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    latest_date = pd.Timestamp(df.index[-1])

    one_bar_return = return_from_close(latest_close, float(close.iloc[-2])) if len(close) >= 2 else None
    five_bar_return = return_from_close(latest_close, float(close.iloc[-6])) if len(close) >= 6 else None

    out = {
        "1BAR": one_bar_return,
        "5BAR": five_bar_return,
        "1M": return_from_close(latest_close, close_asof(df, latest_date - pd.DateOffset(months=1))),
        "3M": return_from_close(latest_close, close_asof(df, latest_date - pd.DateOffset(months=3))),
        "YTD": None,
        "1Y": return_from_close(latest_close, close_asof(df, latest_date - pd.DateOffset(years=1))),
    }

    prior_year_end = pd.Timestamp(year=latest_date.year - 1, month=12, day=31)
    ytd_base = close_asof(df, prior_year_end)

    if ytd_base is None:
        year_data = df[df.index >= pd.Timestamp(year=latest_date.year, month=1, day=1)]
        if not year_data.empty:
            ytd_base = float(year_data["Close"].iloc[0])

    out["YTD"] = return_from_close(latest_close, ytd_base)

    return out


def compute_header_stats(df: pd.DataFrame) -> dict[str, str]:
    close = df["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    latest_date = pd.Timestamp(df.index[-1])

    ret = compute_return_metrics(df)

    one_year = df[df.index >= latest_date - pd.DateOffset(years=1)]
    if one_year.empty:
        one_year = df.tail(min(len(df), 252))

    high_1y = float(one_year["High"].max()) if not one_year.empty else np.nan
    low_1y = float(one_year["Low"].min()) if not one_year.empty else np.nan

    drawdown_1y = (latest_close / high_1y) - 1.0 if high_1y and not pd.isna(high_1y) else np.nan
    upside_from_1y_low = (latest_close / low_1y) - 1.0 if low_1y and not pd.isna(low_1y) else np.nan

    last_row = df.iloc[-1]

    atr_pct = last_row.get("ATR14_PCT", np.nan)
    rsi = last_row.get("RSI14", np.nan)

    return {
        "Last": fmt_price(latest_close),
        "1BAR": fmt_pct(ret.get("1BAR")),
        "5BAR": fmt_pct(ret.get("5BAR")),
        "1M": fmt_pct(ret.get("1M")),
        "3M": fmt_pct(ret.get("3M")),
        "YTD": fmt_pct(ret.get("YTD")),
        "1Y": fmt_pct(ret.get("1Y")),
        "Drawdown": fmt_pct(drawdown_1y),
        "ATR": fmt_pct_abs(atr_pct),
        "1Y Low Dist": fmt_pct(upside_from_1y_low),
        "RSI": "N/A" if pd.isna(rsi) else f"{float(rsi):.1f}",
    }


def latest_float(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None

    value = df[column].iloc[-1]

    if pd.isna(value):
        return None

    return float(value)


def classify_trend(df: pd.DataFrame) -> tuple[str, str]:
    close = latest_float(df, "Close")
    sma20 = latest_float(df, "SMA20")
    sma50 = latest_float(df, "SMA50")
    sma100 = latest_float(df, "SMA100")
    sma200 = latest_float(df, "SMA200")

    if close is None:
        return "Unavailable", "No closing price available."

    checks = [
        ("20DMA", sma20),
        ("50DMA", sma50),
        ("100DMA", sma100),
        ("200DMA", sma200),
    ]

    valid = [(name, value) for name, value in checks if value is not None]

    if not valid:
        return "Unavailable", "Not enough data for moving-average trend classification."

    above = [name for name, value in valid if close > value]
    below = [name for name, value in valid if close < value]

    if len(above) == len(valid):
        return "Bullish", "Price is above the major moving averages."

    if len(below) == len(valid):
        return "Bearish", "Price is below the major moving averages."

    if close > (sma50 or close) and close > (sma200 or close):
        return "Constructive", "Price is above the core 50DMA / 200DMA trend filter, but the stack is not fully clean."

    if close < (sma50 or close) and close < (sma200 or close):
        return "Damaged", "Price is below the core 50DMA / 200DMA trend filter."

    return "Mixed", "Trend filters are split across short and long horizons."


def classify_momentum(df: pd.DataFrame) -> tuple[str, str]:
    rsi = latest_float(df, "RSI14")
    hist = latest_float(df, "MACD_HIST")

    if rsi is None and hist is None:
        return "Unavailable", "Not enough data for RSI or MACD."

    rsi_text = "RSI unavailable" if rsi is None else f"RSI {rsi:.1f}"
    hist_text = "MACD histogram unavailable" if hist is None else f"MACD histogram {hist:.2f}"

    if rsi is not None and hist is not None:
        if rsi >= 60 and hist > 0:
            return "Positive", f"{rsi_text}; {hist_text}. Momentum confirms the trend."
        if rsi >= 50 and hist > 0:
            return "Constructive", f"{rsi_text}; {hist_text}. Momentum is positive but not stretched."
        if rsi < 45 and hist < 0:
            return "Negative", f"{rsi_text}; {hist_text}. Momentum is deteriorating."
        if rsi > 70:
            return "Extended", f"{rsi_text}; {hist_text}. Momentum is strong but extended."

    return "Mixed", f"{rsi_text}; {hist_text}."


def classify_volatility(df: pd.DataFrame) -> tuple[str, str]:
    if "ATR14_PCT" not in df.columns:
        return "Unavailable", "ATR unavailable."

    atr_series = pd.to_numeric(df["ATR14_PCT"], errors="coerce").dropna()

    if len(atr_series) < 30:
        return "Unavailable", "Not enough ATR history."

    latest_atr = float(atr_series.iloc[-1])
    lookback = atr_series.tail(min(len(atr_series), 252))
    percentile = float((lookback <= latest_atr).mean())

    if percentile >= 0.80:
        return "Elevated", f"ATR is {latest_atr * 100:.2f}% of price, in the {percentile * 100:.0f}th percentile of recent history."
    if percentile <= 0.25:
        return "Compressed", f"ATR is {latest_atr * 100:.2f}% of price, in the {percentile * 100:.0f}th percentile of recent history."

    return "Normal", f"ATR is {latest_atr * 100:.2f}% of price, near the middle of recent history."


def classify_structure(df: pd.DataFrame) -> tuple[str, str]:
    if len(df) < 30:
        return "Unavailable", "Not enough price history."

    close = float(df["Close"].iloc[-1])
    high_20 = float(df["High"].tail(20).max())
    low_20 = float(df["Low"].tail(20).min())
    high_60 = float(df["High"].tail(min(len(df), 60)).max())
    low_60 = float(df["Low"].tail(min(len(df), 60)).min())

    if close >= high_20 * 0.995:
        return "Breakout pressure", f"Price is pressing the 20-bar high at {high_20:,.2f}."
    if close <= low_20 * 1.005:
        return "Breakdown pressure", f"Price is pressing the 20-bar low at {low_20:,.2f}."

    range_position = (close - low_60) / max(high_60 - low_60, 1e-9)

    if range_position >= 0.70:
        return "Upper range", "Price sits in the upper part of the 60-bar range."
    if range_position <= 0.30:
        return "Lower range", "Price sits in the lower part of the 60-bar range."

    return "Range", "Price is inside the recent range without a clean breakout or breakdown."


def nearest_level_below(price: float, levels: list[tuple[str, float | None]]) -> tuple[str, float] | None:
    valid = [(label, value) for label, value in levels if value is not None and value < price]

    if not valid:
        return None

    return max(valid, key=lambda item: item[1])


def nearest_level_above(price: float, levels: list[tuple[str, float | None]]) -> tuple[str, float] | None:
    valid = [(label, value) for label, value in levels if value is not None and value > price]

    if not valid:
        return None

    return min(valid, key=lambda item: item[1])


def risk_levels(df: pd.DataFrame) -> tuple[str, str]:
    close = latest_float(df, "Close")

    if close is None:
        return "Unavailable", "No close available."

    levels = [
        ("8DMA", latest_float(df, "SMA8")),
        ("20DMA", latest_float(df, "SMA20")),
        ("50DMA", latest_float(df, "SMA50")),
        ("100DMA", latest_float(df, "SMA100")),
        ("200DMA", latest_float(df, "SMA200")),
        ("20-bar low", float(df["Low"].tail(20).min()) if len(df) >= 20 else None),
        ("60-bar low", float(df["Low"].tail(60).min()) if len(df) >= 60 else None),
    ]

    below = nearest_level_below(close, levels)
    above = nearest_level_above(close, levels)

    if below and above:
        support_label, support_value = below
        resistance_label, resistance_value = above
        return (
            support_label,
            f"Nearest support is {support_label} at {support_value:,.2f}. Nearest resistance is {resistance_label} at {resistance_value:,.2f}.",
        )

    if below:
        support_label, support_value = below
        return support_label, f"Nearest support is {support_label} at {support_value:,.2f}."

    if above:
        resistance_label, resistance_value = above
        return resistance_label, f"Nearest visible resistance is {resistance_label} at {resistance_value:,.2f}."

    return "Unavailable", "No nearby moving-average or range level is available."



def elliott_swing_window(df: pd.DataFrame) -> int:
    n = len(df)

    if n < 45:
        return 0

    return max(3, min(12, int(round(n / 40))))


def elliott_min_swing_pct(df: pd.DataFrame) -> float:
    close = latest_float(df, "Close")

    if close is None or close == 0:
        return 0.012

    atr = latest_float(df, "ATR14_PCT")
    atr_component = 1.35 * atr if atr is not None and not pd.isna(atr) else 0.0

    visible_high = float(pd.to_numeric(df["High"], errors="coerce").max())
    visible_low = float(pd.to_numeric(df["Low"], errors="coerce").min())
    range_component = ((visible_high - visible_low) / abs(close)) * 0.035 if close else 0.0

    return max(0.008, min(0.040, max(atr_component, range_component)))


def find_elliott_pivots(df: pd.DataFrame, max_pivots: int = 9) -> list[dict[str, object]]:
    if df.empty or len(df) < 45:
        return []

    window = elliott_swing_window(df)

    if window <= 0 or len(df) < (window * 2 + 5):
        return []

    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    index = list(df.index)

    candidates: list[dict[str, object]] = []

    for i in range(window, len(df) - window):
        high_window = highs[i - window : i + window + 1]
        low_window = lows[i - window : i + window + 1]

        if np.isfinite(highs[i]) and highs[i] >= np.nanmax(high_window):
            candidates.append(
                {
                    "pos": i,
                    "index": index[i],
                    "price": float(highs[i]),
                    "kind": "H",
                    "current": False,
                }
            )

        if np.isfinite(lows[i]) and lows[i] <= np.nanmin(low_window):
            candidates.append(
                {
                    "pos": i,
                    "index": index[i],
                    "price": float(lows[i]),
                    "kind": "L",
                    "current": False,
                }
            )

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: int(item["pos"]))
    min_swing = elliott_min_swing_pct(df)
    pivots: list[dict[str, object]] = []

    for candidate in candidates:
        if not pivots:
            pivots.append(candidate)
            continue

        last = pivots[-1]

        if int(candidate["pos"]) == int(last["pos"]):
            continue

        if str(candidate["kind"]) == str(last["kind"]):
            candidate_price = float(candidate["price"])
            last_price = float(last["price"])
            is_more_extreme_high = str(candidate["kind"]) == "H" and candidate_price > last_price
            is_more_extreme_low = str(candidate["kind"]) == "L" and candidate_price < last_price

            if is_more_extreme_high or is_more_extreme_low:
                pivots[-1] = candidate

            continue

        last_price = float(last["price"])
        candidate_price = float(candidate["price"])
        move = abs(candidate_price / last_price - 1.0) if last_price else 0.0

        if move >= min_swing or len(pivots) < 2:
            pivots.append(candidate)

    if pivots:
        last = pivots[-1]
        current_price = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])
        current_kind = "H" if current_price >= float(last["price"]) else "L"
        move_from_last = abs(current_price / float(last["price"]) - 1.0) if float(last["price"]) else 0.0

        if current_kind != str(last["kind"]) and move_from_last >= min_swing * 0.75:
            pivots.append(
                {
                    "pos": len(df) - 1,
                    "index": df.index[-1],
                    "price": current_price,
                    "kind": current_kind,
                    "current": True,
                }
            )

    return pivots[-max_pivots:]


def elliott_display_points(df: pd.DataFrame) -> list[dict[str, object]]:
    pivots = find_elliott_pivots(df, max_pivots=9)

    if len(pivots) >= 5:
        points = pivots[-5:]
        labels = ["1", "2", "3", "4", "5"]
    elif len(pivots) >= 3:
        points = pivots[-3:]
        labels = ["A", "B", "C"]
    else:
        return []

    for point, label in zip(points, labels):
        point["label"] = label

    return points


def elliott_wave_state(df: pd.DataFrame) -> tuple[str, str]:
    points = elliott_display_points(df)

    if not points:
        return "Unavailable", "Not enough qualified swing pivots for a useful Elliott Wave map."

    latest = points[-1]
    latest_label = str(latest.get("label", ""))
    latest_kind = str(latest.get("kind", ""))
    latest_price = float(latest["price"])

    prior_low = next((point for point in reversed(points[:-1]) if str(point.get("kind")) == "L"), None)
    prior_high = next((point for point in reversed(points[:-1]) if str(point.get("kind")) == "H"), None)

    if latest_label == "5":
        reading = "Provisional impulse"
    elif latest_label == "C":
        reading = "Corrective map"
    else:
        reading = "Wave map"

    if latest_kind == "H" and prior_low is not None:
        anchor = f"prior swing low at {float(prior_low['price']):,.2f}"
        note = f"Latest qualified pivot is wave {latest_label} high at {latest_price:,.2f}; count weakens below the {anchor}."
    elif latest_kind == "L" and prior_high is not None:
        anchor = f"prior swing high at {float(prior_high['price']):,.2f}"
        note = f"Latest qualified pivot is wave {latest_label} low at {latest_price:,.2f}; count improves through the {anchor}."
    else:
        note = f"Latest qualified pivot is wave {latest_label} at {latest_price:,.2f}."

    return reading, note



# ============================================================
# Fibonacci Levels
# ============================================================

FIB_RETRACEMENT_RATIOS = [0.382, 0.500, 0.618, 0.786]
FIB_EXTENSION_RATIOS = [1.000, 1.272, 1.382, 1.618, 2.618]


def format_fib_ratio(ratio: float) -> str:
    text = f"{ratio:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fibonacci_anchor_points(df: pd.DataFrame) -> dict[str, object] | None:
    if df.empty or len(df) < 20:
        return None

    pivots = find_elliott_pivots(df, max_pivots=9)

    if len(pivots) >= 2:
        best_pair: tuple[dict[str, object], dict[str, object]] | None = None
        best_score = -np.inf

        for i in range(len(pivots) - 1):
            first = pivots[i]
            first_price = float(first["price"])

            if first_price == 0 or not np.isfinite(first_price):
                continue

            for j in range(i + 1, len(pivots)):
                second = pivots[j]

                if str(first.get("kind")) == str(second.get("kind")):
                    continue

                second_price = float(second["price"])

                if not np.isfinite(second_price):
                    continue

                move = abs(second_price / first_price - 1.0)
                recency_bonus = 1.0 + (0.035 * j)
                score = move * recency_bonus

                if score > best_score:
                    best_pair = (first, second)
                    best_score = score

        if best_pair is None:
            best_pair = (pivots[-2], pivots[-1])

        start, end = best_pair
        start_price = float(start["price"])
        end_price = float(end["price"])

        if start_price == end_price:
            return None

        return {
            "start_index": start["index"],
            "end_index": end["index"],
            "start_price": start_price,
            "end_price": end_price,
            "direction": "up" if end_price > start_price else "down",
            "source": "qualified Elliott swing",
        }

    high_idx = df["High"].idxmax()
    low_idx = df["Low"].idxmin()
    high_price = float(df.loc[high_idx, "High"])
    low_price = float(df.loc[low_idx, "Low"])

    if high_price == low_price:
        return None

    if pd.Timestamp(low_idx) < pd.Timestamp(high_idx):
        return {
            "start_index": low_idx,
            "end_index": high_idx,
            "start_price": low_price,
            "end_price": high_price,
            "direction": "up",
            "source": "visible range",
        }

    return {
        "start_index": high_idx,
        "end_index": low_idx,
        "start_price": high_price,
        "end_price": low_price,
        "direction": "down",
        "source": "visible range",
    }


def fibonacci_levels(df: pd.DataFrame) -> dict[str, object] | None:
    anchor = fibonacci_anchor_points(df)

    if anchor is None:
        return None

    start_price = float(anchor["start_price"])
    end_price = float(anchor["end_price"])
    swing = abs(end_price - start_price)

    if swing <= 0:
        return None

    direction = str(anchor["direction"])
    retracements = []
    extensions = []

    for ratio in FIB_RETRACEMENT_RATIOS:
        if direction == "up":
            price = end_price - (ratio * swing)
        else:
            price = end_price + (ratio * swing)

        retracements.append(
            {
                "ratio": ratio,
                "price": float(price),
                "kind": "retracement",
            }
        )

    for ratio in FIB_EXTENSION_RATIOS:
        if direction == "up":
            price = start_price + (ratio * swing)
        else:
            price = start_price - (ratio * swing)

        extensions.append(
            {
                "ratio": ratio,
                "price": float(price),
                "kind": "extension",
            }
        )

    return {
        "anchor": anchor,
        "retracements": retracements,
        "extensions": extensions,
        "all_levels": retracements + extensions,
    }


def fibonacci_state(df: pd.DataFrame) -> tuple[str, str]:
    data = fibonacci_levels(df)
    close = latest_float(df, "Close")

    if data is None or close is None:
        return "Unavailable", "Not enough qualified swing structure for Fibonacci levels."

    anchor = data["anchor"]
    direction = str(anchor["direction"])
    source = str(anchor["source"])
    start_price = float(anchor["start_price"])
    end_price = float(anchor["end_price"])

    all_levels = [
        (f"{format_fib_ratio(float(level['ratio']))} {level['kind']}", float(level["price"]))
        for level in data["all_levels"]
        if np.isfinite(float(level["price"]))
    ]

    below = nearest_level_below(close, all_levels)
    above = nearest_level_above(close, all_levels)

    if direction == "up":
        reading = "Upside swing map"
    else:
        reading = "Downside swing map"

    range_text = f"Anchor is the {source} from {start_price:,.2f} to {end_price:,.2f}."

    if below and above:
        below_label, below_price = below
        above_label, above_price = above
        return reading, f"{range_text} Nearest Fib support is {below_label} at {below_price:,.2f}; nearest Fib resistance is {above_label} at {above_price:,.2f}."

    if below:
        below_label, below_price = below
        return reading, f"{range_text} Nearest Fib support is {below_label} at {below_price:,.2f}."

    if above:
        above_label, above_price = above
        return reading, f"{range_text} Nearest Fib resistance is {above_label} at {above_price:,.2f}."

    return reading, range_text


def build_signal_rows(
    df: pd.DataFrame,
    include_elliott: bool = False,
    include_fibonacci: bool = False,
) -> list[dict[str, str]]:
    trend, trend_note = classify_trend(df)
    momentum, momentum_note = classify_momentum(df)
    volatility, volatility_note = classify_volatility(df)
    structure, structure_note = classify_structure(df)
    risk, risk_note = risk_levels(df)

    rows = [
        {"Signal": "Trend", "Reading": trend, "Interpretation": trend_note},
        {"Signal": "Momentum", "Reading": momentum, "Interpretation": momentum_note},
        {"Signal": "Volatility", "Reading": volatility, "Interpretation": volatility_note},
        {"Signal": "Structure", "Reading": structure, "Interpretation": structure_note},
        {"Signal": "Risk level", "Reading": risk, "Interpretation": risk_note},
    ]

    if include_elliott:
        wave, wave_note = elliott_wave_state(df)
        rows.append({"Signal": "Elliott Wave", "Reading": wave, "Interpretation": wave_note})

    if include_fibonacci:
        fib, fib_note = fibonacci_state(df)
        rows.append({"Signal": "Fibonacci", "Reading": fib, "Interpretation": fib_note})

    return rows


def build_technical_memo(
    df: pd.DataFrame,
    ticker: str,
    include_elliott: bool = False,
    include_fibonacci: bool = False,
) -> str:
    trend, trend_note = classify_trend(df)
    momentum, momentum_note = classify_momentum(df)
    volatility, volatility_note = classify_volatility(df)
    structure, structure_note = classify_structure(df)
    risk, risk_note = risk_levels(df)

    close = latest_float(df, "Close")
    sma50 = latest_float(df, "SMA50")
    sma200 = latest_float(df, "SMA200")

    invalidation = "the next major moving-average break"
    if close is not None and sma50 is not None and close > sma50:
        invalidation = "a close below the 50DMA"
    elif close is not None and sma200 is not None and close > sma200:
        invalidation = "a close below the 200DMA"
    elif close is not None and sma50 is not None and close < sma50:
        invalidation = "a reclaim of the 50DMA"

    memo = (
        f"{ticker} screens as {trend.lower()} on trend and {momentum.lower()} on momentum. "
        f"{trend_note} {momentum_note} Volatility is {volatility.lower()}: {volatility_note} "
        f"Structure reads as {structure.lower()}. {structure_note} {risk_note} "
        f"The practical invalidation level is {invalidation}."
    )

    if include_elliott:
        wave, wave_note = elliott_wave_state(df)
        memo += f" Elliott Wave overlay reads as {wave.lower()}: {wave_note}"

    if include_fibonacci:
        fib, fib_note = fibonacci_state(df)
        memo += f" Fibonacci overlay reads as {fib.lower()}: {fib_note}"

    return memo


def signal_table_html(rows: list[dict[str, str]]) -> str:
    body = []

    for row in rows:
        signal = html_escape(str(row.get("Signal", "")))
        reading = html_escape(str(row.get("Reading", "")))
        interpretation = html_escape(str(row.get("Interpretation", "")))
        body.append(
            "<tr>"
            f"<td>{signal}</td>"
            f"<td><strong>{reading}</strong></td>"
            f"<td>{interpretation}</td>"
            "</tr>"
        )

    return (
        '<table class="signal-table">'
        '<thead>'
        '<tr>'
        '<th>Signal</th>'
        '<th>Reading</th>'
        '<th>Interpretation</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        + "".join(body)
        + '</tbody>'
        + '</table>'
    )


# ============================================================
# Plot Helpers
# ============================================================

def build_rangebreaks(index: pd.DatetimeIndex, interval: str) -> list[dict]:
    if interval != "1d" or len(index) == 0:
        return []

    normalized = pd.DatetimeIndex(index).normalize().unique().sort_values()

    if len(normalized) == 0:
        return []

    full_business_days = pd.date_range(
        start=normalized.min(),
        end=normalized.max(),
        freq="B",
    )

    missing_business_days = full_business_days.difference(normalized)
    rangebreaks = [dict(bounds=["sat", "mon"])]

    if len(missing_business_days) > 0:
        rangebreaks.append(
            dict(values=missing_business_days.strftime("%Y-%m-%d").tolist())
        )

    return rangebreaks


def attach_plot_x(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_PLOT_X"] = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    return out


def plot_x_values(df: pd.DataFrame) -> pd.Series:
    if "_PLOT_X" in df.columns:
        return df["_PLOT_X"]

    return pd.Series(pd.DatetimeIndex(df.index).strftime("%Y-%m-%d"), index=df.index)


def x_tick_spec(df: pd.DataFrame, max_ticks: int = 8) -> tuple[list[str], list[str]]:
    if df.empty:
        return [], []

    x_values = plot_x_values(df).astype(str).tolist()
    dates = pd.DatetimeIndex(df.index)
    n = len(x_values)

    if n <= max_ticks:
        positions = list(range(n))
    else:
        positions = np.linspace(0, n - 1, max_ticks).round().astype(int).tolist()

    positions = sorted(dict.fromkeys(int(pos) for pos in positions if 0 <= int(pos) < n))

    tickvals = [x_values[pos] for pos in positions]
    ticktext = []

    for pos in positions:
        dt = dates[pos]
        label = dt.strftime("%b %d")
        if pos == positions[0] or pos == positions[-1] or dt.year != dates[positions[max(0, positions.index(pos) - 1)]].year:
            label = dt.strftime("%b %d<br>%Y")
        ticktext.append(label)

    return tickvals, ticktext


def active_panels(settings: ChartSettings, usable_volume: bool) -> list[str]:
    panels = []

    if settings.show_volume and usable_volume:
        panels.append("volume")

    if settings.show_rsi:
        panels.append("rsi")

    if settings.show_macd:
        panels.append("macd")

    return panels


def panel_layout(panels: list[str]) -> tuple[int, list[float], int]:
    row_count = 1 + len(panels)

    if row_count == 1:
        return row_count, [1.0], 670

    if row_count == 2:
        return row_count, [0.80, 0.20], 790

    if row_count == 3:
        return row_count, [0.70, 0.15, 0.15], 890

    return row_count, [0.64, 0.12, 0.12, 0.12], 980


def volume_tick_count(volume: pd.Series) -> int:
    clean = pd.to_numeric(volume, errors="coerce").dropna()

    if clean.empty:
        return 4

    maximum = float(clean.max())

    if maximum >= 1e10:
        return 7

    if maximum >= 1e9:
        return 6

    return 5


def price_axis_range(df: pd.DataFrame, settings: ChartSettings) -> list[float] | None:
    if df.empty:
        return None

    columns = ["High", "Low"]

    overlay_columns = [
        ("SMA8", settings.show_sma8),
        ("SMA20", settings.show_sma20),
        ("SMA50", settings.show_sma50),
        ("SMA100", settings.show_sma100),
        ("SMA200", settings.show_sma200),
        ("BB_UPPER", settings.show_bbands),
        ("BB_LOWER", settings.show_bbands),
    ]

    for column, enabled in overlay_columns:
        if enabled and column in df.columns:
            columns.append(column)

    values = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not series.empty:
            values.append(series)

    if settings.show_fibonacci:
        fib_data = fibonacci_levels(df)
        if fib_data is not None:
            fib_prices = [
                float(level["price"])
                for level in fib_data["all_levels"]
                if np.isfinite(float(level["price"]))
            ]
            if fib_prices:
                values.append(pd.Series(fib_prices, dtype=float))

    if not values:
        return None

    combined = pd.concat(values)
    ymin = float(combined.min())
    ymax = float(combined.max())

    visible_range = ymax - ymin

    if visible_range <= 0:
        close = float(df["Close"].iloc[-1])
        pad = max(abs(close) * 0.02, 1e-6)
        return [close - pad, close + pad]

    atr = latest_float(df, "ATR14")
    atr_pad = 1.25 * atr if atr is not None else 0.0
    pct_pad = 0.045 * visible_range
    price_pad = abs(float(df["Close"].iloc[-1])) * 0.003

    pad = max(atr_pad, pct_pad, price_pad)

    lower = ymin - pad
    upper = ymax + pad

    return [lower, upper]


def add_last_price_line(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    last_close = float(df["Close"].iloc[-1])
    previous_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
    change_pct = ((last_close / previous_close) - 1.0) * 100 if previous_close != 0 else 0.0

    fig.add_hline(
        y=last_close,
        row=row,
        col=1,
        line_color=COLORS["last"],
        line_dash="dot",
        line_width=1,
        annotation_text=f"Last {last_close:,.2f} ({change_pct:+.2f}%)",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["last"], size=11),
    )


def add_range_levels(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    visible_high = float(df["High"].max())
    visible_low = float(df["Low"].min())

    fig.add_hline(
        y=visible_high,
        row=row,
        col=1,
        line_color=COLORS["range"],
        line_dash="dot",
        line_width=1,
        annotation_text=f"Range high {visible_high:,.2f}",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )

    fig.add_hline(
        y=visible_low,
        row=row,
        col=1,
        line_color=COLORS["range"],
        line_dash="dot",
        line_width=1,
        annotation_text=f"Range low {visible_low:,.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )




def add_elliott_wave_overlay(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    points = elliott_display_points(df)

    if not points:
        return

    x_values = [pd.Timestamp(point["index"]).strftime("%Y-%m-%d") for point in points]
    y_values = [float(point["price"]) for point in points]
    labels = [str(point.get("label", "")) for point in points]
    text_positions = [
        "top center" if str(point.get("kind")) == "H" else "bottom center"
        for point in points
    ]

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers+text",
            line=dict(color="rgba(17,24,39,0.75)", width=1.5, dash="dash"),
            marker=dict(size=7, color="rgba(17,24,39,0.85)", line=dict(width=1, color="#ffffff")),
            text=labels,
            textposition=text_positions,
            textfont=dict(size=12, color="#111827"),
            name="Elliott Wave",
            hovertemplate="Wave %{text}: %{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=row,
        col=1,
    )


def add_fibonacci_overlay(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    data = fibonacci_levels(df)

    if data is None:
        return

    anchor = data["anchor"]
    start_x = pd.Timestamp(anchor["start_index"]).strftime("%Y-%m-%d")
    end_x = pd.Timestamp(anchor["end_index"]).strftime("%Y-%m-%d")
    x_start = str(plot_x_values(df).iloc[0])
    x_end = str(plot_x_values(df).iloc[-1])

    fig.add_trace(
        go.Scatter(
            x=[start_x, end_x],
            y=[float(anchor["start_price"]), float(anchor["end_price"])],
            mode="lines+markers",
            line=dict(color=COLORS["fib_anchor"], width=1.2, dash="dash"),
            marker=dict(size=6, color=COLORS["fib_anchor"]),
            name="Fib anchor",
            hovertemplate="Fib anchor: %{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=row,
        col=1,
    )

    for level in data["retracements"]:
        ratio = float(level["ratio"])
        price = float(level["price"])
        label = f"{format_fib_ratio(ratio)} ({price:,.2f})"

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[price, price],
                mode="lines+text",
                line=dict(color=COLORS["fib_retracement"], width=1.0, dash="dot"),
                text=["", label],
                textposition="middle right",
                textfont=dict(size=10, color=COLORS["fib_retracement"]),
                name="Fib retracement",
                legendgroup="fib_retracement",
                hovertemplate=f"Fib retracement {format_fib_ratio(ratio)}: %{{y:.2f}}<extra></extra>",
                showlegend=ratio == FIB_RETRACEMENT_RATIOS[0],
            ),
            row=row,
            col=1,
        )

    for level in data["extensions"]:
        ratio = float(level["ratio"])
        price = float(level["price"])
        label = f"{format_fib_ratio(ratio)} ({price:,.2f})"

        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[price, price],
                mode="lines+text",
                line=dict(color=COLORS["fib_extension"], width=1.15),
                text=["", label],
                textposition="middle right",
                textfont=dict(size=10, color=COLORS["fib_extension"]),
                name="Fib extension",
                legendgroup="fib_extension",
                hovertemplate=f"Fib extension {format_fib_ratio(ratio)}: %{{y:.2f}}<extra></extra>",
                showlegend=ratio == FIB_EXTENSION_RATIOS[0],
            ),
            row=row,
            col=1,
        )


def pattern_accent(pattern: PatternDetection) -> str:
    if pattern.bias == "bullish":
        return COLORS["pattern_bullish"]
    if pattern.bias == "bearish":
        return COLORS["pattern_bearish"]
    return COLORS["pattern_bilateral"]


def pattern_line_style(line: PatternLine, pattern: PatternDetection) -> tuple[str, str, float]:
    if line.role == "support":
        return COLORS["pattern_support"], "solid", 1.8
    if line.role == "resistance":
        return COLORS["pattern_resistance"], "solid", 1.8
    if line.role == "breakout":
        return pattern_accent(pattern), "dash", 1.45
    if line.role == "pole":
        return pattern_accent(pattern), "solid", 2.2
    return COLORS["pattern_muted"], "dot", 1.25


def add_chart_pattern_overlay(
    fig: go.Figure,
    df: pd.DataFrame,
    patterns: list[PatternDetection],
    row: int,
) -> None:
    if not patterns or df.empty:
        return

    x_values = plot_x_values(df).astype(str).tolist()
    visible_low = float(pd.to_numeric(df["Low"], errors="coerce").min())
    visible_high = float(pd.to_numeric(df["High"], errors="coerce").max())
    visible_range = max(visible_high - visible_low, abs(float(df["Close"].iloc[-1])) * 0.01)

    for pattern_index, pattern in enumerate(patterns):
        accent = pattern_accent(pattern)
        valid_points = [point for point in pattern.points if 0 <= point.pos < len(x_values)]

        if len(valid_points) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[x_values[point.pos] for point in valid_points],
                    y=[point.price for point in valid_points],
                    mode="lines+markers",
                    line=dict(color=accent, width=1.45),
                    marker=dict(
                        size=6,
                        color="#ffffff",
                        line=dict(color=accent, width=1.5),
                    ),
                    opacity=0.78,
                    name=pattern.name,
                    legendgroup="chart_patterns",
                    hovertemplate=(
                        f"<b>{html_escape(pattern.name)}</b><br>"
                        f"{html_escape(pattern.status)} | {pattern.confidence:.0f}% confidence<br>"
                        "Pivot %{y:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

        for line in pattern.lines:
            start_pos = max(0, min(len(x_values) - 1, line.start_pos))
            end_pos = max(0, min(len(x_values) - 1, line.end_pos))
            color, dash, width = pattern_line_style(line, pattern)
            fig.add_trace(
                go.Scatter(
                    x=[x_values[start_pos], x_values[end_pos]],
                    y=[line.start_price, line.end_price],
                    mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    opacity=0.92,
                    name=f"{pattern.name} {line.role}",
                    legendgroup="chart_patterns",
                    hovertemplate=f"{html_escape(pattern.name)} | {line.role}: %{{y:.2f}}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

        anchor_pos = max(0, min(len(x_values) - 1, pattern.end_pos))
        if pattern.breakout_level is not None and np.isfinite(pattern.breakout_level):
            anchor_y = float(pattern.breakout_level)
        elif valid_points:
            anchor_y = float(valid_points[-1].price)
        else:
            anchor_y = float(df["Close"].iloc[-1])

        yshift = 16 + pattern_index * 20 if pattern.bias != "bearish" else -(18 + pattern_index * 20)
        fig.add_annotation(
            x=x_values[anchor_pos],
            y=anchor_y,
            text=f"<b>{html_escape(pattern.name)}</b> | {pattern.status}",
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=accent,
            ax=0,
            ay=-yshift,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=accent,
            borderwidth=1,
            borderpad=4,
            font=dict(size=10, color="#111827"),
            align="left",
            row=row,
            col=1,
        )

        if pattern.status == "Confirmed" and pattern.target_level is not None:
            target = float(pattern.target_level)
            if visible_low - visible_range * 0.20 <= target <= visible_high + visible_range * 0.20:
                target_start = max(0, len(x_values) - max(18, len(x_values) // 7))
                fig.add_trace(
                    go.Scatter(
                        x=[x_values[target_start], x_values[-1]],
                        y=[target, target],
                        mode="lines+text",
                        line=dict(color=accent, width=1.15, dash="dot"),
                        text=["", f"Target {target:,.2f}"],
                        textposition="middle right",
                        textfont=dict(size=9, color=accent),
                        hovertemplate=f"{html_escape(pattern.name)} measured target: %{{y:.2f}}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )


def pattern_summary_html(patterns: list[PatternDetection]) -> str:
    if not patterns:
        return (
            '<div class="pattern-empty">'
            '<b>No high-confidence current pattern.</b> The overlay is intentionally selective; '
            'short windows and noisy price action may not produce a qualified structure.'
            '</div>'
        )

    cards = []
    for pattern in patterns:
        accent = pattern_accent(pattern)
        tint = (
            "rgba(15,157,117,0.055)" if pattern.bias == "bullish"
            else "rgba(214,69,93,0.055)" if pattern.bias == "bearish"
            else "rgba(59,130,246,0.055)"
        )
        breakout = fmt_price(pattern.breakout_level) if pattern.breakout_level is not None else "Two-sided"
        target = fmt_price(pattern.target_level) if pattern.target_level is not None else "After confirmation"
        bias = pattern.bias.capitalize()
        cards.append(
            f'<div class="pattern-card" style="--pattern-accent:{accent};--pattern-tint:{tint}">'
            '<div class="pattern-card-top">'
            f'<div class="pattern-card-name">{html_escape(pattern.name)}</div>'
            f'<div class="pattern-card-score">{pattern.confidence:.0f}%</div>'
            '</div>'
            f'<div class="pattern-card-meta">{bias} &middot; {html_escape(pattern.status)} &middot; {html_escape(pattern.family.title())}</div>'
            f'<div class="pattern-card-levels">Trigger {breakout} &nbsp;&middot;&nbsp; Target {target}</div>'
            '</div>'
        )
    return f'<div class="pattern-strip">{"".join(cards)}</div>'

def hover_price_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(["N/A"] * len(df), index=df.index)

    return pd.to_numeric(df[column], errors="coerce").map(fmt_price).astype(str)


def price_hover_customdata(df: pd.DataFrame) -> np.ndarray:
    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "SMA8",
        "SMA20",
        "SMA50",
        "SMA100",
        "SMA200",
        "SMA65",
        "SMA130",
        "SMA195",
        "SMA260",
        "BB_UPPER",
        "BB_MID",
        "BB_LOWER",
    ]

    return np.column_stack([hover_price_column(df, column) for column in columns])


def add_price_hover_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    settings: ChartSettings,
    row: int,
) -> None:
    bb_section = ""

    if settings.show_bbands and all(col in df.columns for col in ["BB_UPPER", "BB_MID", "BB_LOWER"]):
        bb_section = (
            "<br><br>"
            "<span style='color:#6b7280'>Bollinger Bands</span><br>"
            "Upper %{customdata[13]} | Mid %{customdata[14]} | Lower %{customdata[15]}"
        )

    hovertemplate = (
        "<b>%{x}</b><br>"
        "<span style='color:#6b7280'>OHLC</span><br>"
        "Open %{customdata[0]} | High %{customdata[1]}<br>"
        "Low %{customdata[2]} | Close %{customdata[3]}"
        "<br><br>"
        "<span style='color:#6b7280'>Moving averages</span><br>"
        "8DMA %{customdata[4]} | 20DMA %{customdata[5]}<br>"
        "50DMA %{customdata[6]} | 100DMA %{customdata[7]} | 200DMA %{customdata[8]}<br>"
        "1Q %{customdata[9]} | 2Q %{customdata[10]} | 3Q %{customdata[11]} | 4Q %{customdata[12]}"
        + bb_section
        + "<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_values(df),
            y=df["Close"],
            mode="markers",
            marker=dict(
                size=18,
                color="rgba(17,24,39,0.001)",
                line=dict(width=0),
            ),
            name="Technical read",
            customdata=price_hover_customdata(df),
            hovertemplate=hovertemplate,
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor="rgba(17,24,39,0.18)",
                font=dict(size=12, color="#111827", family="Arial, sans-serif"),
                align="left",
            ),
            showlegend=False,
            cliponaxis=False,
        ),
        row=row,
        col=1,
    )

def add_price_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    settings: ChartSettings,
    patterns: list[PatternDetection],
    row: int,
) -> None:
    if settings.chart_type == "Candles":
        fig.add_trace(
            go.Candlestick(
                x=plot_x_values(df),
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLORS["up"],
                increasing_fillcolor=COLORS["up"],
                decreasing_line_color=COLORS["down"],
                decreasing_fillcolor=COLORS["down"],
                name="Price",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_x_values(df),
                y=df["Close"],
                mode="lines",
                line=dict(color="#111827", width=1.7),
                name="Close",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    ma_config = [
        ("SMA8", "8DMA", COLORS["sma8"], settings.show_sma8, 1.30),
        ("SMA20", "20DMA", COLORS["sma20"], settings.show_sma20, 1.45),
        ("SMA50", "50DMA", COLORS["sma50"], settings.show_sma50, 1.55),
        ("SMA100", "100DMA", COLORS["sma100"], settings.show_sma100, 1.45),
        ("SMA200", "200DMA", COLORS["sma200"], settings.show_sma200, 1.75),
        ("SMA65", "1Q MA (65DMA)", COLORS["sma65"], settings.show_sma65, 1.40),
        ("SMA130", "2Q MA (130DMA)", COLORS["sma130"], settings.show_sma130, 1.40),
        ("SMA195", "3Q MA (195DMA)", COLORS["sma195"], settings.show_sma195, 1.40),
        ("SMA260", "4Q MA (260DMA)", COLORS["sma260"], settings.show_sma260, 1.40),
    ]

    for column, label, color, enabled, width in ma_config:
        if enabled and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_x_values(df),
                    y=df[column],
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=label,
                    hoverinfo="skip",
                    showlegend=True,
                ),
                row=row,
                col=1,
            )

    if settings.show_bbands and all(col in df.columns for col in ["BB_UPPER", "BB_LOWER", "BB_MID"]):
        fig.add_trace(
            go.Scatter(
                x=plot_x_values(df),
                y=df["BB_UPPER"],
                mode="lines",
                line=dict(color=COLORS["bb"], width=1.0, dash="dot"),
                name="Bollinger Bands",
                legendgroup="bbands",
                hoverinfo="skip",
                showlegend=True,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x_values(df),
                y=df["BB_LOWER"],
                mode="lines",
                line=dict(color=COLORS["bb"], width=1.0, dash="dot"),
                fill="tonexty",
                fillcolor=COLORS["bb_fill"],
                name="BB lower",
                legendgroup="bbands",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x_values(df),
                y=df["BB_MID"],
                mode="lines",
                line=dict(color="rgba(90,90,90,0.60)", width=1.0, dash="dot"),
                name="BB mid",
                legendgroup="bbands",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    if settings.show_last_price:
        add_last_price_line(fig, df, row)

    if settings.show_elliott_wave:
        add_elliott_wave_overlay(fig, df, row)

    if settings.show_fibonacci:
        add_fibonacci_overlay(fig, df, row)

    if settings.show_chart_patterns:
        add_chart_pattern_overlay(fig, df, patterns, row)

    add_price_hover_trace(fig, df, settings, row)


def add_volume_panel(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    closes = df["Close"].astype(float).values
    colors = []

    for i in range(len(df)):
        if i == 0:
            colors.append(COLORS["volume_neutral"])
        elif closes[i] >= closes[i - 1]:
            colors.append(COLORS["volume_up"])
        else:
            colors.append(COLORS["volume_down"])

    fig.add_trace(
        go.Bar(
            x=plot_x_values(df),
            y=df["Volume"],
            marker_color=colors,
            name="Volume",
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def add_rsi_panel(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    x_start = str(plot_x_values(df).iloc[0])
    x_end = str(plot_x_values(df).iloc[-1])

    fig.add_shape(
        type="rect",
        xref=f"x{row}" if row > 1 else "x",
        yref=f"y{row}" if row > 1 else "y",
        x0=x_start,
        x1=x_end,
        y0=20,
        y1=80,
        fillcolor="gray",
        opacity=0.08,
        line_width=0,
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_values(df),
            y=df["RSI14"],
            mode="lines",
            line=dict(color=COLORS["rsi"], width=1.1),
            name="RSI",
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[80, 80],
            mode="lines",
            line=dict(color="#b22222", width=1, dash="dot"),
            name="RSI 80",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[20, 20],
            mode="lines",
            line=dict(color="#2e8b57", width=1, dash="dot"),
            name="RSI 20",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )


def add_macd_panel(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    hist_colors = [
        COLORS["hist_up"] if value >= 0 else COLORS["hist_down"]
        for value in df["MACD_HIST"].fillna(0.0)
    ]

    fig.add_trace(
        go.Bar(
            x=plot_x_values(df),
            y=df["MACD_HIST"],
            marker_color=hist_colors,
            name="MACD histogram",
            hovertemplate="Hist: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_values(df),
            y=df["MACD"],
            mode="lines",
            line=dict(color=COLORS["macd"], width=1.4),
            name="MACD",
            hovertemplate="MACD: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_values(df),
            y=df["MACD_SIGNAL"],
            mode="lines",
            line=dict(color=COLORS["signal"], width=1.2),
            name="Signal",
            hovertemplate="Signal: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_color="rgba(120,120,120,0.28)",
        line_width=1,
        row=row,
        col=1,
    )


def build_chart(
    df: pd.DataFrame,
    settings: ChartSettings,
    usable_volume: bool,
    patterns: list[PatternDetection] | None = None,
) -> go.Figure:
    df = attach_plot_x(df)
    panels = active_panels(settings, usable_volume)
    row_count, row_heights, fig_height = panel_layout(panels)

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        row_heights=row_heights,
        specs=[[{"type": "xy"}] for _ in range(row_count)],
    )

    row_map = {"price": 1}
    next_row = 2

    for panel in panels:
        row_map[panel] = next_row
        next_row += 1

    add_price_panel(fig, df, settings, patterns or [], row=row_map["price"])

    if "volume" in row_map:
        add_volume_panel(fig, df, row=row_map["volume"])

    if "rsi" in row_map:
        add_rsi_panel(fig, df, row=row_map["rsi"])

    if "macd" in row_map:
        add_macd_panel(fig, df, row=row_map["macd"])

    tickvals, ticktext = x_tick_spec(df)
    x_categories = plot_x_values(df).astype(str).tolist()

    for row in range(1, row_count + 1):
        fig.update_yaxes(
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            fixedrange=False,
            automargin=True,
            row=row,
            col=1,
        )

    price_range = price_axis_range(df, settings)

    fig.update_yaxes(
        title_text="Price",
        nticks=9,
        type="linear",
        range=price_range,
        row=row_map["price"],
        col=1,
    )

    if "volume" in row_map:
        fig.update_yaxes(
            title_text="Vol",
            nticks=volume_tick_count(df["Volume"]),
            row=row_map["volume"],
            col=1,
        )

    if "rsi" in row_map:
        fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            nticks=6,
            row=row_map["rsi"],
            col=1,
        )

    if "macd" in row_map:
        fig.update_yaxes(
            title_text="MACD",
            nticks=5,
            row=row_map["macd"],
            col=1,
        )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_categories,
        range=[-0.5, max(len(x_categories) - 0.5, 0.5)],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(17,24,39,0.22)",
    )

    fig.update_layout(
        height=fig_height,
        title=dict(text=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        hoverdistance=70,
        spikedistance=-1,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(17,24,39,0.18)",
            font=dict(size=12, color="#111827", family="Arial, sans-serif"),
            align="left",
        ),
        margin=dict(l=40, r=22, t=66, b=30),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.075,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#374151"),
            title=dict(text=""),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        bargap=0.08,
        xaxis_rangeslider_visible=False,
    )

    # Defensive clean-up for Plotly / Streamlit builds that render a blank
    # layout title as the literal word "undefined".
    fig.layout.title.text = ""
    fig.layout.legend.title.text = ""

    return fig


# ============================================================
# Compare Chart
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_compare_close(
    ticker: str,
    interval: str,
    auto_adjust: bool,
    fetch_start: str | None,
) -> pd.Series:
    df = fetch_with_yfinance_download(
        ticker=ticker,
        interval=interval,
        auto_adjust=auto_adjust,
        fetch_start=fetch_start,
    )

    if df.empty:
        return pd.Series(dtype=float, name=ticker)

    out = df["Close"].astype(float).copy()
    out.name = ticker

    return out


def build_compare_chart(
    primary_df: pd.DataFrame,
    settings: ChartSettings,
    compare_tickers: list[str],
) -> go.Figure | None:
    if primary_df.empty or not compare_tickers:
        return None

    fetch_start = warmup_start_date(settings.period, settings.interval)
    fetch_start_str = fetch_start.strftime("%Y-%m-%d") if fetch_start is not None else None

    series_list = [primary_df["Close"].astype(float).rename(settings.ticker)]

    for ticker in compare_tickers:
        try:
            close = fetch_compare_close(
                ticker=ticker,
                interval=settings.interval,
                auto_adjust=settings.auto_adjust,
                fetch_start=fetch_start_str,
            )

            if not close.empty:
                series_list.append(close)

        except Exception:
            continue

    if len(series_list) <= 1:
        return None

    combined = pd.concat(series_list, axis=1).dropna(how="all")
    start = start_date_from_period(settings.period)

    if start is not None:
        combined = combined[combined.index >= start]

    combined = combined.ffill().dropna(how="any")

    if combined.empty or len(combined) < 2:
        return None

    indexed = combined / combined.iloc[0] * 100.0

    fig = go.Figure()

    for column in indexed.columns:
        width = 2.4 if column == settings.ticker else 1.55
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=indexed[column],
                mode="lines",
                line=dict(width=width),
                name=column,
                hovertemplate=f"{column}: " + "%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=390,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=44, r=20, t=78, b=24),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.115,
            xanchor="left",
            x=0.18,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            title=dict(text=""),
            font=dict(size=11, color="#374151"),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        title=dict(
            text="<b>Relative performance, indexed to 100</b>",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=14, color=COLORS["text"]),
        ),
    )

    fig.layout.legend.title.text = ""

    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        rangebreaks=build_rangebreaks(indexed.index, settings.interval),
    )

    fig.update_yaxes(
        title_text="Indexed price",
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        automargin=True,
    )

    return fig


# ============================================================
# Render Helpers
# ============================================================

def metric_card(label: str, value: str, note: str = "") -> str:
    label = html_escape(str(label))
    value = html_escape(str(value))
    note = html_escape(str(note))

    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        '</div>'
    )


def render_header(
    settings: ChartSettings,
    display_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    stats_source = metrics_df if metrics_df is not None and not metrics_df.empty else display_df
    stats = compute_header_stats(stats_source)
    last_bar = format_last_bar(display_df.index[-1])

    ticker = html_escape(settings.ticker)
    interval = html_escape(settings.interval)
    period = html_escape(settings.period)
    last_bar_text = html_escape(last_bar)

    bar_one_label = interval_bar_label(settings.interval, 1)
    bar_five_label = interval_bar_label(settings.interval, 5)
    horizon_cards = [
        metric_card(label, stats[key])
        for label, key in header_horizon_keys(settings.interval)
    ]

    cards = "".join(
        [
            metric_card("Last", stats["Last"]),
            metric_card(bar_one_label, stats["1BAR"], interval_bar_note(settings.interval, 1)),
            metric_card(bar_five_label, stats["5BAR"], interval_bar_note(settings.interval, 5)),
            *horizon_cards,
            metric_card("YTD", stats["YTD"]),
            metric_card("Drawdown", stats["Drawdown"], drawdown_note_for_interval(settings.interval)),
            metric_card("ATR", stats["ATR"], atr_note_for_interval(settings.interval)),
        ]
    )

    html = (
        '<div class="adfm-header">'
        f'<div class="adfm-title">{ticker} Technical Regime</div>'
        f'<div class="adfm-subtitle">Interval: {interval} | Window: {period} | Last bar: {last_bar_text}</div>'
        f'<div class="metric-strip">{cards}</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# App
# ============================================================

settings = read_settings()

if not settings.ticker:
    st.error("Enter a ticker.")
    st.stop()

with st.spinner("Loading chart data..."):
    raw_df, fetch_error = fetch_history(settings)

if raw_df.empty:
    st.error(f"No data available for {settings.ticker}.")
    if fetch_error:
        st.caption(fetch_error)
    st.stop()

indicator_df = add_indicators(raw_df)
display_df = prepare_display_window(indicator_df, settings)

if display_df.empty:
    st.error("No display data available after filtering.")
    st.stop()

volume_is_usable = has_usable_volume(display_df)
chart_patterns = detect_chart_patterns(display_df, max_patterns=3) if settings.show_chart_patterns else []

if settings.show_volume and not volume_is_usable:
    st.sidebar.info("Volume panel hidden because this symbol does not have usable volume data.")

render_header(settings, display_df, indicator_df)
st.markdown('<div class="chart-top-gap"></div>', unsafe_allow_html=True)

chart = build_chart(
    df=display_df,
    settings=settings,
    usable_volume=volume_is_usable,
    patterns=chart_patterns,
)

st.plotly_chart(
    chart,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
        ],
    },
)

if settings.show_chart_patterns:
    st.markdown(pattern_summary_html(chart_patterns), unsafe_allow_html=True)

compare_tickers = parse_compare_tickers(settings.compare_tickers, settings.ticker)
compare_fig = build_compare_chart(
    primary_df=indicator_df,
    settings=settings,
    compare_tickers=compare_tickers,
)

if compare_fig is not None:
    st.markdown('<div class="compare-chart-top-gap"></div>', unsafe_allow_html=True)
    st.plotly_chart(
        compare_fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
            ],
        },
    )

signal_rows = build_signal_rows(
    indicator_df,
    include_elliott=settings.show_elliott_wave,
    include_fibonacci=settings.show_fibonacci,
)
memo = build_technical_memo(
    indicator_df,
    settings.ticker,
    include_elliott=settings.show_elliott_wave,
    include_fibonacci=settings.show_fibonacci,
)

left, right = st.columns([1.35, 1.00])

with left:
    st.markdown("#### Signal Matrix")
    st.markdown(signal_table_html(signal_rows), unsafe_allow_html=True)

with right:
    st.markdown("#### Technical Memo")
    st.markdown(f'<div class="memo-box">{html_escape(memo)}</div>', unsafe_allow_html=True)

render_footer()

