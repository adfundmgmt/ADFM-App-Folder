import datetime as dt
import json
import re
import time
from html import escape
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from adfm_core.market_data import configure_yfinance_cache
from adfm_core.regime_math import rolling_percentile_previous
from adfm_core.ui import render_footer
import yfinance as yf
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

configure_yfinance_cache()


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Volume Based Sentiment Indicator",
    layout="wide",
)


# =============================================================================
# CONSTANTS
# =============================================================================

NY_TZ = ZoneInfo("America/New_York")

CACHE_TTL_SECONDS = 3600
LOCAL_CACHE_DIR = Path(".volume_regime_cache")
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_SYMBOLS = [
    "QQQ",
    "SPY",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
]

US_MARKET_HOLIDAYS_STATIC = [
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-03-26",
    "2027-05-31",
    "2027-06-18",
    "2027-07-05",
    "2027-09-06",
    "2027-11-25",
    "2027-12-24",
]

PASTEL_GREEN = "#4f765f"
PASTEL_RED = "#a06452"
PASTEL_GREY = "#94a3b8"
AMBER = "#b08958"
BLUE = "#526f8f"

TITLE = "Volume Based Sentiment Indicator"
SUBTITLE = (
    "Conviction, participation, and sentiment through volume-regime signals."
)


# =============================================================================
# STREAMLIT STYLE
# =============================================================================

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1 {
        font-weight: 650;
        letter-spacing: -0.025em;
        margin-bottom: 0.9rem;
    }

    div[data-testid="stCaptionContainer"] {
        color: #64748b;
    }

    section[data-testid="stSidebar"] {
        background: #fafaf9;
        border-right: 1px solid rgba(100, 116, 139, 0.16);
    }

    .vbsi-title {
        font-size: 1.42rem;
        font-weight: 760;
        letter-spacing: -0.025em;
        margin: 0.2rem 0 0.05rem;
        color: inherit;
    }

    .vbsi-subtitle {
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.45rem;
    }

    .vbsi-banner {
        border: 1px solid rgba(100, 116, 139, 0.18);
        border-left: 3px solid #4f765f;
        border-radius: 6px;
        background: rgba(100, 116, 139, 0.045);
        padding: 9px 12px;
        margin: 0.45rem 0 0.55rem;
        font-size: 0.89rem;
        line-height: 1.45;
    }

    .vbsi-banner.negative { border-left-color: #a06452; }
    .vbsi-banner.neutral {
        border-left-color: #cbd5e1;
        background: rgba(100, 116, 139, 0.025);
    }

    .vbsi-kicker {
        font-size: 0.64rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 800;
        color: #4f765f;
        margin-right: 10px;
    }

    .vbsi-banner.negative .vbsi-kicker { color: #a06452; }
    .vbsi-banner.neutral .vbsi-kicker {
        color: inherit;
        background: rgba(100, 116, 139, 0.15);
        padding: 3px 7px;
        border-radius: 2px;
    }

    .vbsi-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid rgba(100, 116, 139, 0.16);
        border-radius: 7px;
        overflow: hidden;
        margin: 0.55rem 0 0.85rem;
    }

    .vbsi-kpi {
        padding: 10px 13px 11px;
        min-height: 66px;
        border-right: 1px solid rgba(100, 116, 139, 0.14);
    }

    .vbsi-kpi:last-child { border-right: 0; }

    .vbsi-kpi-label {
        font-size: 0.61rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 5px;
    }

    .vbsi-kpi-value {
        font-size: 1.13rem;
        line-height: 1;
        font-weight: 780;
        color: inherit;
    }

    .vbsi-kpi-note {
        font-size: 0.68rem;
        line-height: 1.25;
        color: #64748b;
        margin-top: 5px;
    }

    .vbsi-kpi.positive .vbsi-kpi-value { color: #4f765f; }
    .vbsi-kpi.negative .vbsi-kpi-value { color: #a06452; }

    .vbsi-section-title {
        font-size: 1.04rem;
        font-weight: 760;
        letter-spacing: -0.015em;
        margin: 1rem 0 0.08rem;
        color: inherit;
    }

    .vbsi-section-subtitle {
        font-size: 0.74rem;
        line-height: 1.42;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(100, 116, 139, 0.16);
        border-radius: 7px;
        overflow: hidden;
        background: #ffffff;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(100, 116, 139, 0.16);
        border-radius: 7px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
    .stTabs [data-baseweb="tab"] {
        height: 2.2rem;
        padding: 0 0.72rem;
        font-size: 0.82rem;
    }

    .js-plotly-plot .table .cell {
        font-size: 12px;
    }

    @media (max-width: 900px) {
        .vbsi-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .vbsi-kpi:nth-child(2) { border-right: 0; }
        .vbsi-kpi:nth-child(-n+2) { border-bottom: 1px solid rgba(100, 116, 139, 0.14); }
    }

    @media (prefers-color-scheme: dark) {
        .vbsi-subtitle, .vbsi-kpi-label, .vbsi-kpi-note, .vbsi-section-subtitle {
            color: #94a3b8;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_lens_header(symbol: str, volume_mode: str) -> None:
    st.markdown(
        "<div class='vbsi-title'>Volume Sentiment</div>"
        f"<div class='vbsi-subtitle'>{escape(symbol)} &middot; {escape(volume_mode)} &middot; hover, scan, compare</div>",
        unsafe_allow_html=True,
    )


def render_signal_banner(label: str, text: str, tone: str = "neutral") -> None:
    safe_tone = tone if tone in {"positive", "negative", "neutral"} else "neutral"
    st.markdown(
        f"<div class='vbsi-banner {safe_tone}'>"
        f"<span class='vbsi-kicker'>{escape(label)}</span>"
        f"{escape(text)}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_metric_grid(cards: list[tuple[str, str, str, str]]) -> None:
    body = []
    for label, value, note, tone in cards:
        safe_tone = tone if tone in {"positive", "negative", "neutral"} else "neutral"
        body.append(
            f"<div class='vbsi-kpi {safe_tone}'>"
            f"<div class='vbsi-kpi-label'>{escape(label)}</div>"
            f"<div class='vbsi-kpi-value'>{escape(value)}</div>"
            f"<div class='vbsi-kpi-note'>{escape(note)}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='vbsi-kpi-grid'>" + "".join(body) + "</div>",
        unsafe_allow_html=True,
    )


def render_volume_section(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='vbsi-section-title'>{escape(title)}</div>"
        f"<div class='vbsi-section-subtitle'>{escape(subtitle)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_symbol_for_path(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol.upper().strip())


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip().replace(" ", "")


def now_ny() -> dt.datetime:
    return dt.datetime.now(NY_TZ)


def current_market_date() -> pd.Timestamp:
    return pd.Timestamp(now_ny().date()).normalize()


def to_unix_seconds(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    return int(t.timestamp())


def normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out[~out.index.isna()].copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()

    return out


def fmt_large_number(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"

    x = float(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)

    if ax >= 1_000_000_000_000:
        return f"{sign}{ax / 1_000_000_000_000:.2f}T"
    if ax >= 1_000_000_000:
        return f"{sign}{ax / 1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{sign}{ax / 1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{sign}{ax / 1_000:.1f}K"

    return f"{x:,.0f}"


def fmt_price(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"${x:,.2f}"


def fmt_pct(x: Optional[float], digits: int = 2, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"

    prefix = "+" if signed and x > 0 else ""
    return f"{prefix}{x:.{digits}f}%"


def fmt_ratio(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.2f}x"


def fmt_pctl(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.0f}"


def fmt_volume_value(x: Optional[float], vol_label: str) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"

    if "Turnover" in vol_label:
        return f"{x:.2f}%"

    if "Dollar" in vol_label:
        return f"${fmt_large_number(x)}"

    return fmt_large_number(x)


def request_text(url: str, timeout: int = 20, retries: int = 3) -> str:
    last_err = None

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.80 * (attempt + 1))

    raise last_err


# =============================================================================
# MARKET CALENDAR
# =============================================================================


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_holiday_values(start_date_str: str, end_date_str: str) -> list:
    start_date = pd.Timestamp(start_date_str).normalize()
    end_date = pd.Timestamp(end_date_str).normalize()

    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")
        schedule = cal.schedule(start_date=start_date.date(), end_date=end_date.date())

        all_bdays = pd.bdate_range(start_date, end_date).normalize()
        trading_days = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()

        holidays = sorted(set(all_bdays.date) - set(trading_days.date))
        return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in holidays]

    except Exception:
        static = []

        for x in US_MARKET_HOLIDAYS_STATIC:
            ts = pd.Timestamp(x).normalize()
            if start_date <= ts <= end_date:
                static.append(ts.strftime("%Y-%m-%d"))

        return static


def is_market_open_now() -> bool:
    n = now_ny()

    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")
        schedule = cal.schedule(start_date=n.date(), end_date=n.date())

        if schedule.empty:
            return False

        row = schedule.iloc[0]
        market_open = row["market_open"].tz_convert(NY_TZ).to_pydatetime()
        market_close = row["market_close"].tz_convert(NY_TZ).to_pydatetime()

        return market_open <= n <= market_close

    except Exception:
        if n.weekday() >= 5:
            return False

        today_str = pd.Timestamp(n.date()).strftime("%Y-%m-%d")

        if today_str in US_MARKET_HOLIDAYS_STATIC:
            return False

        return dt.time(9, 30) <= n.time() <= dt.time(16, 0)


# =============================================================================
# LOCAL CACHE
# =============================================================================


def cache_path_for_symbol(symbol: str) -> Path:
    return LOCAL_CACHE_DIR / f"{safe_symbol_for_path(symbol)}.csv"


def load_local_cache(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> Optional[pd.DataFrame]:
    path = cache_path_for_symbol(symbol)

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"])

        if df.empty or "Date" not in df.columns:
            return None

        df = df.set_index("Date")
        df = normalize_dt_index(df)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]

        if missing:
            return None

        keep = needed + (["Raw_Close"] if "Raw_Close" in df.columns else [])
        df = df[keep].copy()
        df = df.loc[
            (df.index >= start_date.normalize()) & (df.index <= end_date.normalize())
        ].copy()

        if df.empty:
            return None

        return df

    except Exception:
        return None


def save_local_cache(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return

    path = cache_path_for_symbol(symbol)

    try:
        out = normalize_dt_index(df)
        keep = ["Open", "High", "Low", "Close", "Volume"]
        if "Raw_Close" in out.columns:
            keep.append("Raw_Close")
        out = out[keep].copy()
        out.to_csv(path, index_label="Date")
    except Exception:
        pass


# =============================================================================
# DATA FETCH
# =============================================================================


def validate_ohlcv(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_name} returned an empty DataFrame.")

    out = normalize_dt_index(df)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in out.columns]

    if missing:
        raise ValueError(f"{source_name} missing columns: {missing}")

    keep = needed + (["Raw_Close"] if "Raw_Close" in out.columns else [])
    out = out[keep].copy()

    for col in keep:
        out[col] = safe_numeric(out[col])

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    out = out[(out["Close"] > 0) & (out["High"] > 0) & (out["Low"] > 0)].copy()
    out = out[out["Volume"] >= 0].copy()

    if out.empty:
        raise ValueError(f"{source_name} had no valid OHLCV rows after cleaning.")

    return out


def fetch_from_yahoo_chart_api(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    period1 = to_unix_seconds(start_date)
    period2 = to_unix_seconds(end_date + pd.Timedelta(days=1))

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}"
        f"&interval=1d&includePrePost=false&events=div%2Csplits"
    )

    text = request_text(url)
    payload = json.loads(text)

    chart = payload.get("chart", {})
    result = chart.get("result", [])

    if not result:
        raise ValueError(f"No chart data returned. Error: {chart.get('error')}")

    result0 = result[0]
    ts = result0.get("timestamp", [])
    quote = result0.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result0.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")

    raw_closes = quote.get("close")
    closes = adjclose if adjclose is not None else raw_closes

    if not ts:
        raise ValueError("Yahoo chart API returned no timestamps.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(ts, unit="s", utc=True),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": closes,
            "Raw_Close": raw_closes,
            "Volume": quote.get("volume"),
        }
    )

    df = df.set_index("Date")
    return validate_ohlcv(df, "Yahoo Chart API")


def fetch_from_yfinance(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    if "Close" in df.columns:
        df["Raw_Close"] = df["Close"]
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return validate_ohlcv(df, "yfinance")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_ohlcv(
    symbol: str, start_date_str: str, end_date_str: str
) -> Tuple[pd.DataFrame, str]:
    start_date = pd.Timestamp(start_date_str).normalize()
    end_date = pd.Timestamp(end_date_str).normalize()

    errors = []

    try:
        df = fetch_from_yahoo_chart_api(symbol, start_date, end_date)
        save_local_cache(symbol, df)
        return df, "Yahoo Chart API"
    except Exception as e:
        errors.append(f"Yahoo Chart API: {e}")

    try:
        df = fetch_from_yfinance(symbol, start_date, end_date)
        save_local_cache(symbol, df)
        return df, "yfinance"
    except Exception as e:
        errors.append(f"yfinance: {e}")

    cached = load_local_cache(symbol, start_date, end_date)

    if cached is not None and not cached.empty:
        cache_mtime = dt.datetime.fromtimestamp(
            cache_path_for_symbol(symbol).stat().st_mtime
        )
        source = f"Local cache saved {cache_mtime.strftime('%Y-%m-%d %H:%M')}"
        return cached, source

    raise RuntimeError(" | ".join(errors))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_shares_outstanding(symbol: str) -> Optional[float]:
    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)

        if fast_info:
            try:
                shares = fast_info.get("shares")
            except Exception:
                shares = getattr(fast_info, "shares", None)

            if shares and np.isfinite(shares):
                return float(shares)
    except Exception:
        pass

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        shares = info.get("sharesOutstanding")

        if shares and np.isfinite(shares):
            return float(shares)
    except Exception:
        pass

    return None


# =============================================================================
# SIGNAL ENGINE
# =============================================================================


def compute_percentile_of_last(values: pd.Series) -> float:
    s = pd.Series(values).dropna()

    if len(s) == 0:
        return np.nan

    return float(s.rank(pct=True).iloc[-1] * 100.0)


def classify_state(pctl: float, high_cutoff: float, low_cutoff: float) -> str:
    if not np.isfinite(pctl):
        return "Unavailable"
    if pctl >= high_cutoff:
        return "Heavy"
    if pctl <= low_cutoff:
        return "Quiet"
    return "Normal"


def classify_setup(row: pd.Series, high_cutoff: float, low_cutoff: float) -> str:
    pctl = row.get("Volume_Pctl", np.nan)
    ret = row.get("Ret_1D", np.nan)
    close_loc = row.get("Close_Location", np.nan)
    close = row.get("Close", np.nan)
    ma20 = row.get("Price_MA20", np.nan)
    ma50 = row.get("Price_MA50", np.nan)
    range_atr = row.get("Range_ATR20", np.nan)

    if not np.isfinite(pctl):
        return "Unavailable"

    is_above_20 = np.isfinite(close) and np.isfinite(ma20) and close >= ma20
    is_below_20 = np.isfinite(close) and np.isfinite(ma20) and close < ma20
    is_uptrend = np.isfinite(ma20) and np.isfinite(ma50) and ma20 >= ma50
    is_downtrend = np.isfinite(ma20) and np.isfinite(ma50) and ma20 < ma50

    if pctl >= high_cutoff:
        if np.isfinite(ret) and np.isfinite(close_loc):
            if ret > 0 and close_loc >= 0.65 and is_above_20:
                return "Heavy Accumulation"
            if ret < 0 and close_loc <= 0.35:
                return "Heavy Distribution"
            if abs(ret) <= 0.35 and np.isfinite(range_atr) and range_atr >= 0.90:
                return "High Effort, Low Progress"
            if ret > 0 and close_loc <= 0.45:
                return "Upside Rejection"
            if ret < 0 and close_loc >= 0.55:
                return "Downside Reversal"

        return "Heavy Participation"

    if pctl <= low_cutoff:
        if np.isfinite(ret):
            if abs(ret) <= 0.35:
                return "Quiet Compression"
            if ret > 0 and is_above_20:
                return "Quiet Drift Up"
            if ret < 0 and is_below_20:
                return "Quiet Drift Down"

        return "Quiet Session"

    if is_above_20 and is_uptrend:
        return "Normal Uptrend"

    if is_below_20 and is_downtrend:
        return "Normal Downtrend"

    return "Normal"


def setup_color(setup: str, ret_1d: Optional[float] = None) -> str:
    setup = str(setup)

    constructive = {
        "Heavy Accumulation",
        "Downside Reversal",
        "Quiet Drift Up",
        "Normal Uptrend",
    }

    negative = {
        "Heavy Distribution",
        "Upside Rejection",
        "Quiet Drift Down",
        "Normal Downtrend",
    }

    amber = {
        "High Effort, Low Progress",
        "Heavy Participation",
    }

    quiet = {
        "Quiet Compression",
        "Quiet Session",
    }

    if setup in constructive:
        return PASTEL_GREEN

    if setup in negative:
        return PASTEL_RED

    if setup in amber:
        return AMBER

    if setup in quiet:
        return PASTEL_GREY

    if ret_1d is not None and np.isfinite(ret_1d):
        if ret_1d > 0:
            return PASTEL_GREEN
        if ret_1d < 0:
            return PASTEL_RED

    return PASTEL_GREY


def compute_forward_outcomes(
    df: pd.DataFrame, horizons: Tuple[int, ...] = (5, 10, 20, 63)
) -> pd.DataFrame:
    out = df.copy()

    for h in horizons:
        out[f"Fwd_{h}D"] = (out["Close"].shift(-h) / out["Close"] - 1.0) * 100.0

    close_arr = out["Close"].to_numpy(dtype=float)
    low_arr = out["Low"].to_numpy(dtype=float)
    high_arr = out["High"].to_numpy(dtype=float)

    max_dd_20 = np.full(len(out), np.nan)
    max_up_20 = np.full(len(out), np.nan)

    for i in range(len(out)):
        if not np.isfinite(close_arr[i]) or close_arr[i] <= 0:
            continue

        forward_low = low_arr[i + 1 : i + 21]
        forward_high = high_arr[i + 1 : i + 21]

        if len(forward_low) > 0 and np.isfinite(forward_low).any():
            max_dd_20[i] = (np.nanmin(forward_low) / close_arr[i] - 1.0) * 100.0

        if len(forward_high) > 0 and np.isfinite(forward_high).any():
            max_up_20[i] = (np.nanmax(forward_high) / close_arr[i] - 1.0) * 100.0

    out["Max_DD_20D"] = max_dd_20
    out["Max_Up_20D"] = max_up_20

    return out


def compute_volume_framework(
    df: pd.DataFrame,
    volume_mode: str,
    shares_outstanding: Optional[float],
    percentile_window: int,
    smooth_window: int,
    high_cutoff: float,
    low_cutoff: float,
) -> Tuple[pd.DataFrame, str, bool]:
    out = df.copy()

    raw_close = out.get("Raw_Close", out["Close"])
    out["Dollar_Volume"] = raw_close * out["Volume"]

    turnover_fallback = False

    if volume_mode == "Turnover %" and shares_outstanding and shares_outstanding > 0:
        out["Volume_Display"] = out["Volume"] / shares_outstanding * 100.0
        vol_label = "Turnover (% of shares outstanding)"
    elif volume_mode == "Dollar volume":
        out["Volume_Display"] = out["Dollar_Volume"].astype(float)
        vol_label = "Dollar volume"
    else:
        if volume_mode == "Turnover %":
            turnover_fallback = True

        out["Volume_Display"] = out["Volume"].astype(float)
        vol_label = "Volume (shares)"

    min_smooth = max(10, smooth_window // 2)

    out["Volume_Baseline"] = (
        out["Volume_Display"]
        .shift(1)
        .rolling(
            smooth_window,
            min_periods=min_smooth,
        )
        .median()
    )

    out["Volume_Ratio"] = out["Volume_Display"] / out["Volume_Baseline"].replace(
        0, np.nan
    )

    out["RVOL_20D"] = out["Volume_Display"] / out["Volume_Display"].shift(1).rolling(
        20,
        min_periods=10,
    ).median().replace(0, np.nan)

    out["RVOL_60D"] = out["Volume_Display"] / out["Volume_Display"].shift(1).rolling(
        60,
        min_periods=30,
    ).median().replace(0, np.nan)

    out["Volume_Pctl"] = rolling_percentile_previous(
        out["Volume_Ratio"],
        window=percentile_window,
        min_periods=max(40, percentile_window // 3),
        scale=100.0,
    )

    out["Ret_1D"] = out["Close"].pct_change() * 100.0
    out["Ret_5D"] = out["Close"].pct_change(5) * 100.0
    out["Ret_20D"] = out["Close"].pct_change(20) * 100.0

    out["Price_MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["Price_MA50"] = out["Close"].rolling(50, min_periods=50).mean()
    out["Price_MA100"] = out["Close"].rolling(100, min_periods=100).mean()

    daily_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["Close_Location"] = ((out["Close"] - out["Low"]) / daily_range).clip(0, 1)
    out["Close_Location"] = out["Close_Location"].fillna(0.50)

    prev_close = out["Close"].shift(1)

    tr_components = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    )

    out["True_Range"] = tr_components.max(axis=1)
    out["ATR20"] = out["True_Range"].rolling(20, min_periods=14).mean()
    out["Range_ATR20"] = out["True_Range"] / out["ATR20"].replace(0, np.nan)

    out["State"] = out["Volume_Pctl"].apply(
        lambda x: classify_state(x, high_cutoff, low_cutoff)
    )
    out["Setup"] = out.apply(
        lambda row: classify_setup(row, high_cutoff, low_cutoff), axis=1
    )

    out["Is_Heavy"] = out["State"].eq("Heavy")
    out["Is_Quiet"] = out["State"].eq("Quiet")
    out["Is_Extreme"] = out["State"].isin(["Heavy", "Quiet"])

    out = compute_forward_outcomes(out)

    return out, vol_label, turnover_fallback


# =============================================================================
# CHARTING
# =============================================================================


def volume_bar_color(row: pd.Series) -> str:
    state = row.get("State", "Normal")
    ret = row.get("Ret_1D", np.nan)
    close_loc = row.get("Close_Location", np.nan)

    if state == "Heavy":
        if np.isfinite(ret) and np.isfinite(close_loc):
            if ret >= 0 and close_loc >= 0.50:
                return "rgba(79,118,95,0.72)"
            if ret < 0 or close_loc < 0.45:
                return "rgba(160,100,82,0.72)"

        return "rgba(176,137,88,0.68)"

    if state == "Quiet":
        return "rgba(148,163,184,0.26)"

    return "rgba(148,163,184,0.14)"


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    vol_label: str,
    show_price_mas: bool,
    holiday_values: list,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.74, 0.26],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=symbol,
            line=dict(width=2.0, color="#111827"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if show_price_mas:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA20"],
                mode="lines",
                name="20D",
                line=dict(width=1.15, color="rgba(82,111,143,0.72)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>20D: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Price_MA50"],
                mode="lines",
                name="50D",
                line=dict(width=1.15, color="rgba(107,114,128,0.80)"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>50D: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    events = df[df["Is_Extreme"]].copy()

    if not events.empty:
        marker_colors = [
            setup_color(row["Setup"], row.get("Ret_1D", np.nan))
            for _, row in events.iterrows()
        ]

        marker_sizes = np.where(events["State"].eq("Heavy"), 8.0, 6.0)

        customdata = list(
            zip(
                events["Setup"].astype(str),
                events["Volume_Pctl"],
                events["Volume_Ratio"],
                events["Ret_1D"],
                events["Close_Location"] * 100.0,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=events.index,
                y=events["Close"],
                mode="markers",
                name="Extreme sessions",
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    line=dict(width=0.8, color="white"),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "%{customdata[0]}"
                    "<br>Volume percentile: %{customdata[1]:.0f}"
                    "<br>Vs baseline: %{customdata[2]:.2f}x"
                    "<br>1D move: %{customdata[3]:+.2f}%"
                    "<br>Close location: %{customdata[4]:.0f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    bar_colors = [volume_bar_color(row) for _, row in df.iterrows()]

    bar_customdata = list(
        zip(
            df["Setup"].astype(str),
            df["Volume_Pctl"],
            df["Volume_Ratio"],
            df["Ret_1D"],
        )
    )

    if "Turnover" in vol_label:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Turnover: %{y:.2f}%"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:.2f}%<extra></extra>"
    elif "Dollar" in vol_label:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Dollar volume: $%{y:,.0f}"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: $%{y:,.0f}<extra></extra>"
    else:
        bar_hover = (
            "<b>%{x|%b %d, %Y}</b><br>"
            "Volume: %{y:,.0f}"
            "<br>%{customdata[0]}"
            "<br>Percentile: %{customdata[1]:.0f}"
            "<br>Vs baseline: %{customdata[2]:.2f}x"
            "<br>1D move: %{customdata[3]:+.2f}%"
            "<extra></extra>"
        )
        baseline_hover = "<b>%{x|%b %d, %Y}</b><br>Baseline: %{y:,.0f}<extra></extra>"

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume_Display"],
            name="Participation",
            marker_color=bar_colors,
            customdata=bar_customdata,
            hovertemplate=bar_hover,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume_Baseline"],
            mode="lines",
            name="Baseline",
            line=dict(width=1.85, color=BLUE),
            hovertemplate=baseline_hover,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=740,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        bargap=0.06,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.012,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=11, color="#374151"),
        ),
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=holiday_values),
        ],
        showgrid=True,
        gridcolor="#f1f5f9",
        showline=False,
        zeroline=False,
        tickfont=dict(size=11, color="#6b7280"),
    )

    fig.update_yaxes(
        row=1,
        col=1,
        title_text="Price",
        showgrid=True,
        gridcolor="#eef2f7",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#6b7280"),
        title_font=dict(size=11, color="#6b7280"),
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text=vol_label,
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color="#6b7280"),
        title_font=dict(size=11, color="#6b7280"),
    )

    if "Dollar" in vol_label:
        fig.update_yaxes(row=2, col=1, tickformat="$.2s")
    elif "Turnover" in vol_label:
        fig.update_yaxes(row=2, col=1, ticksuffix="%")
    else:
        fig.update_yaxes(row=2, col=1, tickformat=".2s")

    return fig


# =============================================================================
# NATIVE STREAMLIT OUTPUTS
# =============================================================================


def render_current_read(
    latest: pd.Series,
    symbol: str,
    vol_label: str,
    data_source: str,
    latest_date: pd.Timestamp,
) -> None:
    latest_setup = str(latest.get("Setup", "Unavailable"))
    latest_close = float(latest.get("Close", np.nan))
    latest_ret = float(latest.get("Ret_1D", np.nan))
    latest_pctl = float(latest.get("Volume_Pctl", np.nan))
    latest_ratio = float(latest.get("Volume_Ratio", np.nan))
    latest_vol = float(latest.get("Volume_Display", np.nan))
    latest_base = float(latest.get("Volume_Baseline", np.nan))
    latest_rvol20 = float(latest.get("RVOL_20D", np.nan))

    vol_text = fmt_volume_value(latest_vol, vol_label)
    base_text = fmt_volume_value(latest_base, vol_label)

    st.caption(
        f"{symbol} | {latest_date:%Y-%m-%d} | {latest_setup} | "
        f"Volume percentile {fmt_pctl(latest_pctl)} ({vol_text}) | "
        f"{fmt_ratio(latest_ratio)} baseline (base {base_text}) | "
        f"Close {fmt_price(latest_close)} {fmt_pct(latest_ret)} | "
        f"RVOL 20D {fmt_ratio(latest_rvol20)} | {data_source}"
    )


def build_recent_events(
    df: pd.DataFrame, event_filter: str, max_rows: int
) -> pd.DataFrame:
    events = df[df["State"].isin(["Heavy", "Quiet"])].copy()

    if event_filter == "Heavy only":
        events = events[events["State"] == "Heavy"].copy()
    elif event_filter == "Quiet only":
        events = events[events["State"] == "Quiet"].copy()

    if events.empty:
        return pd.DataFrame()

    return events.tail(max_rows).iloc[::-1].copy()


def format_events_for_display(events: pd.DataFrame, vol_label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=events.index)

    out["Date"] = events.index.strftime("%Y-%m-%d")
    out["Setup"] = events["Setup"].astype(str)
    out["Close"] = events["Close"].map(fmt_price)
    out["1D"] = events["Ret_1D"].map(lambda x: fmt_pct(x))
    out["5D"] = events["Ret_5D"].map(lambda x: fmt_pct(x))
    out["20D"] = events["Ret_20D"].map(lambda x: fmt_pct(x))
    out["Volume"] = events["Volume_Display"].map(
        lambda x: fmt_volume_value(x, vol_label)
    )
    out["Vs Base"] = events["Volume_Ratio"].map(fmt_ratio)
    out["Pctl"] = events["Volume_Pctl"].map(fmt_pctl)
    out["Close Loc"] = events["Close_Location"].map(
        lambda x: fmt_pct(x * 100.0, digits=0, signed=False)
    )
    out["Fwd 5D"] = events["Fwd_5D"].map(lambda x: fmt_pct(x))
    out["Fwd 20D"] = events["Fwd_20D"].map(lambda x: fmt_pct(x))
    out["Max DD 20D"] = events["Max_DD_20D"].map(lambda x: fmt_pct(x))

    return out.reset_index(drop=True)


def build_setup_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized outcomes by setup using only rows with known futures."""
    realized = df.dropna(subset=["Setup", "Fwd_5D", "Fwd_20D", "Max_DD_20D"]).copy()
    realized = realized[~realized["Setup"].isin(["Unavailable", "Normal"])].copy()
    if realized.empty:
        return pd.DataFrame()

    grouped = realized.groupby("Setup", sort=False)
    out = grouped.agg(
        Observations=("Fwd_20D", "count"),
        **{
            "Avg 5D": ("Fwd_5D", "mean"),
            "Avg 20D": ("Fwd_20D", "mean"),
            "20D Hit Rate": ("Fwd_20D", lambda x: float((x > 0).mean() * 100.0)),
            "Median Max DD": ("Max_DD_20D", "median"),
        },
    ).reset_index()
    out = out[out["Observations"] >= 3].sort_values(
        ["Observations", "Avg 20D"], ascending=[False, False]
    )
    return out.reset_index(drop=True)


def style_events_table(display_df: pd.DataFrame):
    def color_return_text(value: str) -> str:
        if not isinstance(value, str):
            return ""

        if value.startswith("+"):
            return f"color: {PASTEL_GREEN}; font-weight: 700;"
        if value.startswith("-"):
            return f"color: {PASTEL_RED}; font-weight: 700;"

        return "color: #374151;"

    def color_setup_text(value: str) -> str:
        color = setup_color(value)

        if color == PASTEL_GREEN:
            return "color: #3f624e; font-weight: 700;"
        if color == PASTEL_RED:
            return "color: #875241; font-weight: 700;"
        if color == AMBER:
            return "color: #8a6a3e; font-weight: 700;"

        return "color: #374151; font-weight: 700;"

    styled = display_df.style

    return_cols = ["1D", "5D", "20D", "Fwd 5D", "Fwd 20D", "Max DD 20D"]
    return_cols = [c for c in return_cols if c in display_df.columns]

    styled = styled.map(color_return_text, subset=return_cols)

    if "Setup" in display_df.columns:
        styled = styled.map(color_setup_text, subset=["Setup"])

    styled = styled.set_properties(
        **{
            "font-size": "13px",
            "white-space": "nowrap",
        }
    )

    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-size", "12px"),
                    ("font-weight", "700"),
                    ("color", "#4b5563"),
                    ("background-color", "#f9fafb"),
                    ("border-bottom", "1px solid #e5e7eb"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #f1f5f9"),
                ],
            },
        ]
    )

    return styled


# =============================================================================
# SIDEBAR
# =============================================================================

if "vr_symbol" not in st.session_state:
    st.session_state["vr_symbol"] = "QQQ"

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Identify heavy and quiet participation regimes across ETFs and listed equities.

        **How to read it**
        - Heavy sessions show unusually high participation versus the selected percentile window.
        - Quiet sessions show unusually low participation.
        - Setup labels combine volume percentile, price direction, close location, and trend structure.
        - Forward columns show realized post-signal returns where enough future data exists.

        **Data source:** Yahoo Finance adjusted daily OHLCV.
        """
    )

    st.divider()
    st.header("Settings")

    st.markdown("**Quick ticker**")
    button_cols = st.columns(2)

    for i, sym in enumerate(DEFAULT_SYMBOLS):
        with button_cols[i % 2]:
            if st.button(sym, width="stretch"):
                st.session_state["vr_symbol"] = sym

    with st.expander("Signal window", expanded=True):
        percentile_window = st.slider(
            "Percentile window",
            min_value=60,
            max_value=252,
            value=126,
            step=21,
            format="%d trading days",
        )

        smooth_window = st.slider(
            "Baseline window",
            min_value=10,
            max_value=80,
            value=20,
            step=5,
            format="%d trading days",
        )

    with st.expander("Signal", expanded=True):
        high_cutoff = st.slider(
            "Heavy threshold",
            min_value=75,
            max_value=99,
            value=90,
            step=1,
            format="%d percentile",
        )

        low_cutoff = st.slider(
            "Quiet threshold",
            min_value=1,
            max_value=25,
            value=10,
            step=1,
            format="%d percentile",
        )

        show_price_mas = st.checkbox("Show 20D and 50D moving averages", value=True)

        use_incomplete_session = st.checkbox(
            "Use incomplete current session",
            value=False,
            help="If unchecked, the app excludes today's row while the NYSE session is still open.",
        )

    with st.expander("Table", expanded=False):
        event_filter = st.selectbox(
            "Recent extremes",
            options=["All extremes", "Heavy only", "Quiet only"],
            index=0,
        )

        max_event_rows = st.slider(
            "Rows",
            min_value=5,
            max_value=25,
            value=12,
            step=1,
        )


# =============================================================================
# TITLE
# =============================================================================

st.title(TITLE)
st.caption(SUBTITLE)

control_col1, control_col2, control_col3 = st.columns([2.0, 1.0, 1.0])

with control_col1:
    symbol_input = st.text_input("Ticker symbol", key="vr_symbol")
    symbol = normalize_symbol(symbol_input)

with control_col2:
    lookback_months = st.selectbox(
        "Visible history",
        options=list(range(6, 49, 3)),
        index=4,
        format_func=lambda months: f"{months} months",
    )

with control_col3:
    volume_mode = st.selectbox(
        "Volume mode",
        options=["Dollar volume", "Raw volume", "Turnover %"],
        index=0,
    )


# =============================================================================
# DATA PREP
# =============================================================================

if not symbol:
    st.warning("Enter a ticker to continue.")
    st.stop()

end_date = current_market_date()
visible_start = end_date - pd.DateOffset(months=lookback_months)

warmup_days = max(420, percentile_window * 3)
start_date = visible_start - pd.Timedelta(days=warmup_days)

with st.spinner(f"Loading {symbol}..."):
    try:
        raw_df, data_source = fetch_ohlcv(
            symbol=symbol,
            start_date_str=start_date.strftime("%Y-%m-%d"),
            end_date_str=end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}. Details: {e}")
        st.stop()

if raw_df.empty:
    st.warning("No data returned for this ticker.")
    st.stop()

raw_df = normalize_dt_index(raw_df)

if raw_df["Volume"].sum() <= 0:
    st.warning(
        "This symbol does not have usable exchange volume. Try an ETF or listed equity with reported volume."
    )
    st.stop()

incomplete_session_excluded = False
market_open = is_market_open_now()
latest_raw_date = raw_df.index.max()

if (
    market_open
    and latest_raw_date.date() == end_date.date()
    and not use_incomplete_session
    and len(raw_df) > 1
):
    raw_df = raw_df.loc[raw_df.index < latest_raw_date].copy()
    incomplete_session_excluded = True

shares_outstanding = None

if volume_mode == "Turnover %":
    with st.spinner("Fetching shares outstanding..."):
        shares_outstanding = fetch_shares_outstanding(symbol)

df_full, vol_label, turnover_fallback = compute_volume_framework(
    raw_df,
    volume_mode=volume_mode,
    shares_outstanding=shares_outstanding,
    percentile_window=percentile_window,
    smooth_window=smooth_window,
    high_cutoff=high_cutoff,
    low_cutoff=low_cutoff,
)

df = df_full.loc[df_full.index >= visible_start].copy()

if df.empty:
    st.warning("No usable data remained after processing.")
    st.stop()

latest = df.iloc[-1]
latest_date = df.index[-1]
setup_outcomes = build_setup_outcomes(df_full)

holiday_values = get_holiday_values(
    start_date_str=df.index.min().strftime("%Y-%m-%d"),
    end_date_str=df.index.max().strftime("%Y-%m-%d"),
)


# =============================================================================
# CURRENT READ
# =============================================================================

latest_setup = str(latest.get("Setup", "Unavailable"))
latest_state = str(latest.get("State", "Unavailable"))
latest_pctl = float(latest.get("Volume_Pctl", np.nan))
latest_ratio = float(latest.get("Volume_Ratio", np.nan))
latest_ret = float(latest.get("Ret_1D", np.nan))
latest_close_loc = float(latest.get("Close_Location", np.nan))

same_setup = setup_outcomes[setup_outcomes["Setup"] == latest_setup]
if same_setup.empty:
    outcome_value = "Insufficient sample"
    outcome_note = "Fewer than 3 realized historical matches"
else:
    setup_row = same_setup.iloc[0]
    outcome_value = fmt_pct(float(setup_row["Avg 20D"]), 1)
    outcome_note = (
        f"Avg 20D after {int(setup_row['Observations'])} matches · "
        f"{float(setup_row['20D Hit Rate']):.0f}% positive"
    )

setup_tone = "neutral"
latest_setup_color = setup_color(latest_setup, latest_ret)
if latest_setup_color == PASTEL_GREEN:
    setup_tone = "positive"
elif latest_setup_color == PASTEL_RED:
    setup_tone = "negative"

session_tone = "positive" if latest_ret > 0 else "negative" if latest_ret < 0 else "neutral"
outcome_tone = "positive" if outcome_value.startswith("+") else "negative" if outcome_value.startswith("-") else "neutral"

render_lens_header(symbol=symbol, volume_mode=vol_label)

render_signal_banner(
    "Sentiment",
    f"{latest_setup} · {latest_state} participation · {fmt_pctl(latest_pctl)}th percentile · "
    f"session {fmt_pct(latest_ret, 1)}.",
    tone=setup_tone,
)

render_signal_banner(
    "Now",
    f"{symbol} closed at {fmt_price(float(latest.get('Close', np.nan)))} on {latest_date:%b %d, %Y}. "
    f"Participation is {fmt_ratio(latest_ratio)} the prior {smooth_window}D median; "
    f"close location is {fmt_pct(latest_close_loc * 100, 0, False)}.",
    tone="neutral",
)

render_metric_grid(
    [
        (
            "Participation percentile",
            fmt_pctl(latest_pctl),
            f"vs prior {percentile_window} sessions",
            setup_tone,
        ),
        (
            "Relative volume",
            fmt_ratio(latest_ratio),
            f"vs prior {smooth_window}D median",
            setup_tone,
        ),
        (
            "Session move",
            fmt_pct(latest_ret, 1),
            f"close location {fmt_pct(latest_close_loc * 100, 0, False)}",
            session_tone,
        ),
        (
            "20D historical read",
            outcome_value,
            outcome_note,
            outcome_tone,
        ),
    ]
)

if incomplete_session_excluded:
    st.caption(
        "Current NYSE session appears open, so today's incomplete daily volume row was excluded from the signal."
    )

if turnover_fallback:
    st.caption(
        "Shares outstanding was unavailable or unreliable, so the app fell back to raw share volume."
    )


# =============================================================================
# CHART
# =============================================================================

render_volume_section(
    "Price and Participation Regime",
    f"Adjusted price with raw-price dollar volume where available · {lookback_months} visible months · thresholds {low_cutoff}/{high_cutoff} percentile.",
)

fig = build_chart(
    df=df,
    symbol=symbol,
    vol_label=vol_label,
    show_price_mas=show_price_mas,
    holiday_values=holiday_values,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displayModeBar": False,
        "scrollZoom": True,
        "responsive": True,
    },
)

st.caption(
    f"Data through {latest_date:%b %d, %Y} · Source: {data_source} · Mode: {vol_label} · "
    f"Percentile history: {percentile_window} sessions · Baseline: prior {smooth_window} sessions"
)


# =============================================================================
# RECENT EXTREMES TABLE
# =============================================================================

tab_events, tab_outcomes, tab_method = st.tabs(
    ["Recent Extremes", "Historical Outcomes", "Methodology"]
)

with tab_events:
    render_volume_section(
        "Recent Participation Extremes",
        "Heavy and quiet sessions in the active visible window, with realized forward outcomes when enough future sessions exist.",
    )
    events = build_recent_events(
        df=df,
        event_filter=event_filter,
        max_rows=max_event_rows,
    )
    if events.empty:
        st.info("No extreme sessions found in the visible window.")
    else:
        display_events = format_events_for_display(events, vol_label=vol_label)
        st.dataframe(
            style_events_table(display_events),
            width="stretch",
            hide_index=True,
            height=min(520, 38 + 32 * (len(display_events) + 1)),
        )

with tab_outcomes:
    render_volume_section(
        "Setup Outcome Matrix",
        "Descriptive event study across the full loaded history. Rows require at least three fully realized 20-session outcomes.",
    )
    if setup_outcomes.empty:
        st.info("Not enough fully realized setup observations are available.")
    else:
        outcome_display = setup_outcomes.copy()
        for column in ["Avg 5D", "Avg 20D", "20D Hit Rate", "Median Max DD"]:
            outcome_display[column] = pd.to_numeric(
                outcome_display[column], errors="coerce"
            )
        st.dataframe(
            outcome_display.style.format(
                {
                    "Avg 5D": "{:+.1f}%",
                    "Avg 20D": "{:+.1f}%",
                    "20D Hit Rate": "{:.0f}%",
                    "Median Max DD": "{:+.1f}%",
                },
                na_rep="N/A",
            ),
            width="stretch",
            hide_index=True,
        )

with tab_method:
    st.markdown(
        f"""
        **Participation score.** The current volume-to-baseline ratio is ranked against the prior
        {percentile_window} sessions; the current row is excluded from both the baseline and percentile history.
        This makes the live signal causal and reduces upward drift in dollar-volume levels.

        **Volume modes.** Dollar volume uses unadjusted close times reported volume where the source provides both
        raw and adjusted prices. Price returns use adjusted closes. Turnover falls back to raw shares when shares
        outstanding is unavailable.

        **Setup labels.** Labels combine participation percentile, one-day return, close location, ATR-normalized
        range, and 20D/50D trend state. Forward outcomes are descriptive and are not forecasts.
        """
    )

render_footer()

