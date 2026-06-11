import streamlit as st
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
APP_VERSION = "2026-06-11-sidebar-controls-elliott-wave"

WATCHLISTS = {
    "Index": ["^SPX", "^NDX", "SPY", "QQQ", "IWM", "DIA"],
    "Mag 7": ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA"],
    "Rates / Credit": ["TLT", "IEF", "SHY", "HYG", "LQD", "JNK"],
    "Commodities": ["USO", "BNO", "GLD", "SLV", "CPER", "UNG"],
    "FX": ["DX-Y.NYB", "USDJPY=X", "EURUSD=X", "EURGBP=X", "GBPUSD=X"],
    "Crypto": ["BTC-USD", "ETH-USD", "IBIT", "ETHE"],
}

COLORS = {
    "up": "#1f9d75",
    "down": "#d64545",
    "sma8": "#00A6FB",
    "sma20": "#2962FF",
    "sma50": "#7E57C2",
    "sma100": "#B7791F",
    "sma200": "#111111",
    "bb": "rgba(90,90,90,0.72)",
    "bb_fill": "rgba(120,120,120,0.08)",
    "grid": "rgba(100,100,100,0.16)",
    "text": "#1f2933",
    "muted": "#6b7280",
    "volume_up": "rgba(31,157,117,0.50)",
    "volume_down": "rgba(214,69,69,0.50)",
    "volume_neutral": "rgba(150,150,150,0.40)",
    "rsi": "#5E35B1",
    "macd": "#1E88E5",
    "signal": "#FB8C00",
    "hist_up": "rgba(31,157,117,0.65)",
    "hist_down": "rgba(214,69,69,0.65)",
    "range": "rgba(55,65,81,0.65)",
    "last": "#0F4C81",
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

        @media (max-width: 1200px) {
            .metric-strip {
                grid-template-columns: repeat(4, minmax(95px, 1fr));
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

    show_bbands: bool
    show_last_price: bool
    show_range_levels: bool

    show_volume: bool
    show_rsi: bool
    show_macd: bool
    show_elliott_wave: bool

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


def read_settings() -> ChartSettings:
    ensure_session_defaults()

    with st.sidebar:
        st.header("About This Tool")
        st.markdown(
            """
            **Purpose:** Single-name technical chart workspace for regime framing, trend confirmation, momentum checks, and trade invalidation.

            **How to read it**
            - The header gives price, return, drawdown, and volatility context.
            - The chart shows price, moving averages, Bollinger Bands, volume, RSI, MACD, and optional Elliott Wave swing pivots.
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
                if st.button(watch_ticker, key=f"watch_{selected_watchlist}_{watch_ticker}", use_container_width=True):
                    st.session_state["ticker_input"] = watch_ticker
                    st.rerun()

        st.markdown("---")
        with st.expander("Chart Settings", expanded=False):
            show_last_price = st.checkbox("Last price line", value=True)
            show_bbands = st.checkbox("Bollinger Bands", value=True)

            st.caption("Moving averages")
            show_sma8 = st.checkbox("8 DMA", value=True)
            show_sma20 = st.checkbox("20 DMA", value=True)
            show_sma50 = st.checkbox("50 DMA", value=True)
            show_sma100 = st.checkbox("100 DMA", value=True)
            show_sma200 = st.checkbox("200 DMA", value=True)

            st.caption("Panels and overlays")
            show_volume = st.checkbox("Volume", value=True)
            show_rsi = st.checkbox("RSI", value=True)
            show_macd = st.checkbox("MACD", value=True)
            show_elliott_wave = st.checkbox(
                "Elliott Wave pivot map",
                value=False,
                help="Heuristic swing-pivot overlay. Use it as pattern context, not a deterministic signal.",
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
        show_bbands=show_bbands,
        show_last_price=show_last_price,
        show_range_levels=show_range_levels,
        show_volume=show_volume,
        show_rsi=show_rsi,
        show_macd=show_macd,
        show_elliott_wave=show_elliott_wave,
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

    for window in [8, 20, 50, 100, 200]:
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


def compute_return_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty:
        return {}

    close = df["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    latest_date = pd.Timestamp(df.index[-1])

    out = {
        "1D": return_from_close(latest_close, float(close.iloc[-2])) if len(close) >= 2 else None,
        "5D": return_from_close(latest_close, float(close.iloc[-6])) if len(close) >= 6 else None,
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
        one_year = df.tail(252)

    high_52w = float(one_year["High"].max()) if not one_year.empty else np.nan
    low_52w = float(one_year["Low"].min()) if not one_year.empty else np.nan

    drawdown_52w = (latest_close / high_52w) - 1.0 if high_52w and not pd.isna(high_52w) else np.nan
    upside_from_52w_low = (latest_close / low_52w) - 1.0 if low_52w and not pd.isna(low_52w) else np.nan

    last_row = df.iloc[-1]

    atr_pct = last_row.get("ATR14_PCT", np.nan)
    rsi = last_row.get("RSI14", np.nan)

    return {
        "Last": fmt_price(latest_close),
        "1D": fmt_pct(ret.get("1D")),
        "5D": fmt_pct(ret.get("5D")),
        "1M": fmt_pct(ret.get("1M")),
        "3M": fmt_pct(ret.get("3M")),
        "YTD": fmt_pct(ret.get("YTD")),
        "Drawdown": fmt_pct(drawdown_52w),
        "ATR": fmt_pct_abs(atr_pct),
        "52w Low Dist": fmt_pct(upside_from_52w_low),
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
        return "Upper range", f"Price sits in the upper part of the 60-bar range."
    if range_position <= 0.30:
        return "Lower range", f"Price sits in the lower part of the 60-bar range."

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


def build_signal_rows(df: pd.DataFrame, include_elliott: bool = False) -> list[dict[str, str]]:
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

    return rows


def build_technical_memo(df: pd.DataFrame, ticker: str, include_elliott: bool = False) -> str:
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

    x_values = [point["index"] for point in points]
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

def add_price_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    settings: ChartSettings,
    row: int,
) -> None:
    if settings.chart_type == "Candles":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
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
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Open: %{open:.2f}<br>"
                    "High: %{high:.2f}<br>"
                    "Low: %{low:.2f}<br>"
                    "Close: %{close:.2f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                line=dict(color="#111827", width=1.7),
                name="Close",
                hovertemplate="Close: %{y:.2f}<extra></extra>",
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
    ]

    for column, label, color, enabled, width in ma_config:
        if enabled and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=label,
                    hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=1,
            )

    if settings.show_bbands and all(col in df.columns for col in ["BB_UPPER", "BB_LOWER", "BB_MID"]):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_UPPER"],
                mode="lines",
                line=dict(color=COLORS["bb"], width=1.0, dash="dot"),
                name="Bollinger Bands",
                legendgroup="bbands",
                hovertemplate="BB upper: %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_LOWER"],
                mode="lines",
                line=dict(color=COLORS["bb"], width=1.0, dash="dot"),
                fill="tonexty",
                fillcolor=COLORS["bb_fill"],
                name="BB lower",
                legendgroup="bbands",
                hovertemplate="BB lower: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_MID"],
                mode="lines",
                line=dict(color="rgba(90,90,90,0.60)", width=1.0, dash="dot"),
                name="BB mid",
                legendgroup="bbands",
                hovertemplate="BB mid: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    if settings.show_last_price:
        add_last_price_line(fig, df, row)

    if settings.show_elliott_wave:
        add_elliott_wave_overlay(fig, df, row)


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
            x=df.index,
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
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI14"],
            mode="lines",
            line=dict(color=COLORS["rsi"], width=1.5),
            name="RSI 14",
            hovertemplate="RSI 14: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    for level in [70, 50, 30]:
        fig.add_hline(
            y=level,
            line_dash="dash" if level != 50 else "dot",
            line_color="rgba(120,120,120,0.45)",
            line_width=1,
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
            x=df.index,
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
            x=df.index,
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
            x=df.index,
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
) -> go.Figure:
    panels = active_panels(settings, usable_volume)
    row_count, row_heights, fig_height = panel_layout(panels)

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
        specs=[[{"type": "xy"}] for _ in range(row_count)],
    )

    row_map = {"price": 1}
    next_row = 2

    for panel in panels:
        row_map[panel] = next_row
        next_row += 1

    add_price_panel(fig, df, settings, row=row_map["price"])

    if "volume" in row_map:
        add_volume_panel(fig, df, row=row_map["volume"])

    if "rsi" in row_map:
        add_rsi_panel(fig, df, row=row_map["rsi"])

    if "macd" in row_map:
        add_macd_panel(fig, df, row=row_map["macd"])

    rangebreaks = build_rangebreaks(df.index, settings.interval)

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
        type="date",
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        showline=False,
        rangeslider_visible=False,
        rangebreaks=rangebreaks,
        tickformat="%b '%y" if settings.interval in ["1wk", "1mo"] else "%b %d\n%Y",
    )

    fig.update_layout(
        height=fig_height,
        title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=44, r=20, t=26, b=12),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.012,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#374151"),
            title_text="",
        ),
        bargap=0.08,
        xaxis_rangeslider_visible=False,
    )

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
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=44, r=20, t=28, b=18),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            title_text="",
        ),
        title=dict(
            text="Relative performance, indexed to 100",
            x=0.0,
            xanchor="left",
            font=dict(size=14),
        ),
    )

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

    cards = "".join(
        [
            metric_card("Last", stats["Last"]),
            metric_card("1D", stats["1D"]),
            metric_card("5D", stats["5D"]),
            metric_card("1M", stats["1M"]),
            metric_card("3M", stats["3M"]),
            metric_card("YTD", stats["YTD"]),
            metric_card("Drawdown", stats["Drawdown"], "From 52w high"),
            metric_card("ATR", stats["ATR"], "14-period"),
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

if settings.show_volume and not volume_is_usable:
    st.sidebar.info("Volume panel hidden because this symbol does not have usable volume data.")

render_header(settings, display_df, indicator_df)

chart = build_chart(
    df=display_df,
    settings=settings,
    usable_volume=volume_is_usable,
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

compare_tickers = parse_compare_tickers(settings.compare_tickers, settings.ticker)
compare_fig = build_compare_chart(
    primary_df=indicator_df,
    settings=settings,
    compare_tickers=compare_tickers,
)

if compare_fig is not None:
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

signal_rows = build_signal_rows(indicator_df, include_elliott=settings.show_elliott_wave)
memo = build_technical_memo(indicator_df, settings.ticker, include_elliott=settings.show_elliott_wave)

left, right = st.columns([1.35, 1.00])

with left:
    st.markdown("#### Signal Matrix")
    st.markdown(signal_table_html(signal_rows), unsafe_allow_html=True)

with right:
    st.markdown("#### Technical Memo")
    st.markdown(f'<div class="memo-box">{html_escape(memo)}</div>', unsafe_allow_html=True)

st.caption("© 2026 AD Fund Management LP")
