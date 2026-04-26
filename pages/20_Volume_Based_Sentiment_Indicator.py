import datetime as dt
import html
import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Volume Regime Explorer",
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
    "QQQ", "SPY", "IWM", "DIA", "TLT",
    "GLD", "HYG", "SMH", "NVDA", "AMD",
    "META", "MSFT", "TSLA", "VST", "AXON",
]

US_MARKET_HOLIDAYS_STATIC = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
]

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"
DARK_TEXT = "#111827"
MID_TEXT = "#4b5563"
LIGHT_BORDER = "#e5e7eb"
AMBER = "#f59e0b"
BLUE = "#2563eb"


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.35rem;
            padding-bottom: 2.0rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 0.75rem 0.85rem;
            border-radius: 14px;
        }

        .adfm-title-row {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.25rem;
        }

        .adfm-title {
            font-size: 2.0rem;
            font-weight: 760;
            letter-spacing: -0.04em;
            color: #111827;
            margin: 0;
        }

        .adfm-caption {
            font-size: 0.95rem;
            color: #6b7280;
            margin-top: 0.25rem;
            margin-bottom: 1.0rem;
        }

        .adfm-card-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 10px;
            margin: 0.65rem 0 1.0rem 0;
        }

        .adfm-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.72rem 0.82rem;
            min-height: 76px;
        }

        .adfm-card-label {
            color: #6b7280;
            font-size: 0.74rem;
            font-weight: 650;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }

        .adfm-card-value {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 760;
            letter-spacing: -0.02em;
            white-space: nowrap;
        }

        .adfm-card-sub {
            color: #6b7280;
            font-size: 0.78rem;
            margin-top: 0.15rem;
            white-space: nowrap;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.26rem 0.50rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 720;
            line-height: 1.0;
            white-space: nowrap;
        }

        .table-wrap {
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            overflow: hidden;
            background: #ffffff;
            margin-top: 0.45rem;
        }

        .adfm-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
        }

        .adfm-table thead th {
            background: #f9fafb;
            color: #4b5563;
            font-weight: 760;
            text-align: right;
            padding: 0.70rem 0.75rem;
            border-bottom: 1px solid #e5e7eb;
            white-space: nowrap;
        }

        .adfm-table thead th.left {
            text-align: left;
        }

        .adfm-table tbody td {
            padding: 0.68rem 0.75rem;
            border-bottom: 1px solid #f1f5f9;
            color: #111827;
            text-align: right;
            white-space: nowrap;
        }

        .adfm-table tbody tr:last-child td {
            border-bottom: none;
        }

        .adfm-table tbody tr:hover td {
            background: #fafafa;
        }

        .adfm-table tbody td.left {
            text-align: left;
        }

        .num-pos {
            color: #15803d !important;
            font-weight: 720;
        }

        .num-neg {
            color: #b91c1c !important;
            font-weight: 720;
        }

        .num-neutral {
            color: #374151 !important;
            font-weight: 650;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        @media (max-width: 1100px) {
            .adfm-card-grid {
                grid-template-columns: repeat(3, minmax(120px, 1fr));
            }
        }

        @media (max-width: 700px) {
            .adfm-card-grid {
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }
        }
    </style>
    """,
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


def value_class(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "num-neutral"
    if x > 0:
        return "num-pos"
    if x < 0:
        return "num-neg"
    return "num-neutral"


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
# MARKET CALENDAR HELPERS
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

        open_time = dt.time(9, 30)
        close_time = dt.time(16, 0)
        return open_time <= n.time() <= close_time


# =============================================================================
# LOCAL CACHE
# =============================================================================

def cache_path_for_symbol(symbol: str) -> Path:
    return LOCAL_CACHE_DIR / f"{safe_symbol_for_path(symbol)}.csv"


def load_local_cache(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
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

        df = df[needed].copy()
        df = df.loc[(df.index >= start_date.normalize()) & (df.index <= end_date.normalize())].copy()

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
        out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
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

    out = out[needed].copy()

    for col in needed:
        out[col] = safe_numeric(out[col])

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    out = out[(out["Close"] > 0) & (out["High"] > 0) & (out["Low"] > 0)].copy()
    out = out[out["Volume"] >= 0].copy()

    if out.empty:
        raise ValueError(f"{source_name} had no valid OHLCV rows after cleaning.")

    return out


def fetch_from_yahoo_chart_api(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
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

    closes = adjclose if adjclose is not None else quote.get("close")

    if not ts:
        raise ValueError("Yahoo chart API returned no timestamps.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(ts, unit="s", utc=True),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": closes,
            "Volume": quote.get("volume"),
        }
    )

    df = df.set_index("Date")
    return validate_ohlcv(df, "Yahoo Chart API")


def fetch_from_yfinance(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)

    return validate_ohlcv(df, "yfinance")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_ohlcv(symbol: str, start_date_str: str, end_date_str: str) -> Tuple[pd.DataFrame, str]:
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
        cache_mtime = dt.datetime.fromtimestamp(cache_path_for_symbol(symbol).stat().st_mtime)
        source = f"Local cache, saved {cache_mtime.strftime('%Y-%m-%d %H:%M')}"
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


def setup_color(setup: str, ret_1d: Optional[float] = None, close_location: Optional[float] = None) -> str:
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


def setup_badge_html(setup: str) -> str:
    color = setup_color(setup)
    safe = html.escape(str(setup))

    if color == PASTEL_GREEN:
        bg = "rgba(82,183,136,0.14)"
        fg = "#166534"
        border = "rgba(82,183,136,0.32)"
    elif color == PASTEL_RED:
        bg = "rgba(232,93,93,0.14)"
        fg = "#991b1b"
        border = "rgba(232,93,93,0.32)"
    elif color == AMBER:
        bg = "rgba(245,158,11,0.15)"
        fg = "#92400e"
        border = "rgba(245,158,11,0.34)"
    else:
        bg = "rgba(139,148,158,0.14)"
        fg = "#374151"
        border = "rgba(139,148,158,0.28)"

    return (
        f"<span class='badge' "
        f"style='background:{bg}; color:{fg}; border:1px solid {border};'>"
        f"{safe}</span>"
    )


def compute_forward_outcomes(df: pd.DataFrame, horizons: Tuple[int, ...] = (5, 10, 20, 63)) -> pd.DataFrame:
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

        forward_low = low_arr[i + 1: i + 21]
        forward_high = high_arr[i + 1: i + 21]

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

    out["Dollar_Volume"] = out["Close"] * out["Volume"]

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

    out["Volume_Baseline"] = out["Volume_Display"].rolling(
        smooth_window,
        min_periods=min_smooth,
    ).median()

    out["Volume_Ratio"] = out["Volume_Display"] / out["Volume_Baseline"].replace(0, np.nan)

    out["RVOL_20D"] = out["Volume_Display"] / out["Volume_Display"].rolling(
        20,
        min_periods=10,
    ).median().replace(0, np.nan)

    out["RVOL_60D"] = out["Volume_Display"] / out["Volume_Display"].rolling(
        60,
        min_periods=30,
    ).median().replace(0, np.nan)

    out["Volume_Pctl"] = out["Volume_Display"].rolling(
        percentile_window,
        min_periods=max(40, percentile_window // 3),
    ).apply(compute_percentile_of_last, raw=False)

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

    out["State"] = out["Volume_Pctl"].apply(lambda x: classify_state(x, high_cutoff, low_cutoff))
    out["Setup"] = out.apply(lambda row: classify_setup(row, high_cutoff, low_cutoff), axis=1)

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
                return "rgba(82,183,136,0.72)"
            if ret < 0 or close_loc < 0.45:
                return "rgba(232,93,93,0.72)"
        return "rgba(245,158,11,0.68)"

    if state == "Quiet":
        return "rgba(139,148,158,0.26)"

    return "rgba(139,148,158,0.14)"


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
            line=dict(width=2.25, color="#111827"),
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
                line=dict(width=1.15, color="rgba(37,99,235,0.70)"),
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
            setup_color(row["Setup"], row.get("Ret_1D", np.nan), row.get("Close_Location", np.nan))
            for _, row in events.iterrows()
        ]

        marker_sizes = np.where(events["State"].eq("Heavy"), 8.5, 6.5)

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
                    line=dict(width=1.0, color="white"),
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
        height=760,
        margin=dict(l=12, r=12, t=10, b=10),
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
# TABLE RENDERING
# =============================================================================

def build_recent_events(df: pd.DataFrame, event_filter: str, max_rows: int) -> pd.DataFrame:
    events = df[df["State"].isin(["Heavy", "Quiet"])].copy()

    if event_filter == "Heavy only":
        events = events[events["State"] == "Heavy"].copy()
    elif event_filter == "Quiet only":
        events = events[events["State"] == "Quiet"].copy()

    if events.empty:
        return pd.DataFrame()

    return events.tail(max_rows).iloc[::-1].copy()


def render_events_table(events: pd.DataFrame, vol_label: str) -> None:
    if events.empty:
        st.info("No extreme sessions found in the visible window.")
        return

    rows_html = []

    for idx, row in events.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        setup = row.get("Setup", "Unavailable")

        ret_1d = row.get("Ret_1D", np.nan)
        ret_5d = row.get("Ret_5D", np.nan)
        ret_20d = row.get("Ret_20D", np.nan)

        fwd_5d = row.get("Fwd_5D", np.nan)
        fwd_20d = row.get("Fwd_20D", np.nan)
        max_dd_20d = row.get("Max_DD_20D", np.nan)

        vol_ratio = row.get("Volume_Ratio", np.nan)
        vol_pctl = row.get("Volume_Pctl", np.nan)
        close_loc = row.get("Close_Location", np.nan)

        vol_display = row.get("Volume_Display", np.nan)
        volume_text = fmt_volume_value(vol_display, vol_label)

        row_html = f"""
            <tr>
                <td class="left">{html.escape(date_str)}</td>
                <td class="left">{setup_badge_html(setup)}</td>
                <td>{fmt_price(row.get("Close", np.nan))}</td>
                <td class="{value_class(ret_1d)}">{fmt_pct(ret_1d)}</td>
                <td class="{value_class(ret_5d)}">{fmt_pct(ret_5d)}</td>
                <td class="{value_class(ret_20d)}">{fmt_pct(ret_20d)}</td>
                <td>{html.escape(volume_text)}</td>
                <td>{fmt_ratio(vol_ratio)}</td>
                <td>{fmt_pctl(vol_pctl)}</td>
                <td>{fmt_pct(close_loc * 100.0, digits=0, signed=False) if np.isfinite(close_loc) else "N/A"}</td>
                <td class="{value_class(fwd_5d)}">{fmt_pct(fwd_5d)}</td>
                <td class="{value_class(fwd_20d)}">{fmt_pct(fwd_20d)}</td>
                <td class="{value_class(max_dd_20d)}">{fmt_pct(max_dd_20d)}</td>
            </tr>
        """
        rows_html.append(row_html)

    table_html = f"""
        <div class="table-wrap">
            <table class="adfm-table">
                <thead>
                    <tr>
                        <th class="left">Date</th>
                        <th class="left">Setup</th>
                        <th>Close</th>
                        <th>1D</th>
                        <th>5D</th>
                        <th>20D</th>
                        <th>Volume</th>
                        <th>Vs Base</th>
                        <th>Pctl</th>
                        <th>Close Loc</th>
                        <th>Fwd 5D</th>
                        <th>Fwd 20D</th>
                        <th>Max DD 20D</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)


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

    cards = [
        {
            "label": "Ticker",
            "value": symbol,
            "sub": latest_date.strftime("%Y-%m-%d"),
        },
        {
            "label": "Setup",
            "value_html": setup_badge_html(latest_setup),
            "sub": "Latest tape label",
        },
        {
            "label": "Volume Percentile",
            "value": fmt_pctl(latest_pctl),
            "sub": f"{vol_text}",
        },
        {
            "label": "Vs Baseline",
            "value": fmt_ratio(latest_ratio),
            "sub": f"Baseline {base_text}",
        },
        {
            "label": "Close",
            "value": fmt_price(latest_close),
            "sub": f"1D {fmt_pct(latest_ret)}",
        },
        {
            "label": "RVOL 20D",
            "value": fmt_ratio(latest_rvol20),
            "sub": html.escape(data_source),
        },
    ]

    card_html = ['<div class="adfm-card-grid">']

    for card in cards:
        label = html.escape(card["label"])
        sub = html.escape(card.get("sub", ""))

        if "value_html" in card:
            value = card["value_html"]
        else:
            value = html.escape(str(card.get("value", "N/A")))

        card_html.append(
            f"""
            <div class="adfm-card">
                <div class="adfm-card-label">{label}</div>
                <div class="adfm-card-value">{value}</div>
                <div class="adfm-card-sub">{sub}</div>
            </div>
            """
        )

    card_html.append("</div>")
    st.markdown("".join(card_html), unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

if "vr_symbol" not in st.session_state:
    st.session_state["vr_symbol"] = "QQQ"

with st.sidebar:
    st.markdown("### Ticker")

    button_cols = st.columns(2)

    for i, sym in enumerate(DEFAULT_SYMBOLS):
        with button_cols[i % 2]:
            if st.button(sym, use_container_width=True):
                st.session_state["vr_symbol"] = sym

    symbol_input = st.text_input("Ticker", key="vr_symbol")
    symbol = normalize_symbol(symbol_input)

    st.markdown("---")
    st.markdown("### Window")

    lookback_months = st.slider(
        "Visible history",
        min_value=6,
        max_value=48,
        value=18,
        step=3,
        format="%d months",
    )

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

    st.markdown("---")
    st.markdown("### Signal")

    volume_mode = st.selectbox(
        "Volume mode",
        options=["Dollar volume", "Raw volume", "Turnover %"],
        index=0,
    )

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

    st.markdown("---")
    st.markdown("### Table")

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

st.markdown(
    """
    <div class="adfm-title-row">
        <div>
            <h1 class="adfm-title">Volume Regime Explorer</h1>
            <div class="adfm-caption">
                Tape participation, relative volume, and forward outcomes around extreme sessions.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
    st.warning("This symbol does not have usable exchange volume. Try an ETF or listed equity with reported volume.")
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

holiday_values = get_holiday_values(
    start_date_str=df.index.min().strftime("%Y-%m-%d"),
    end_date_str=df.index.max().strftime("%Y-%m-%d"),
)


# =============================================================================
# CURRENT READ
# =============================================================================

render_current_read(
    latest=latest,
    symbol=symbol,
    vol_label=vol_label,
    data_source=data_source,
    latest_date=latest_date,
)

if incomplete_session_excluded:
    st.caption(
        "Current NYSE session appears to be open, so today's incomplete daily volume row was excluded from the signal."
    )

if turnover_fallback:
    st.caption(
        "Shares outstanding was unavailable or unreliable, so the app fell back to raw share volume."
    )


# =============================================================================
# MAIN CHART
# =============================================================================

fig = build_chart(
    df=df,
    symbol=symbol,
    vol_label=vol_label,
    show_price_mas=show_price_mas,
    holiday_values=holiday_values,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": True,
        "responsive": True,
    },
)

st.caption(
    f"Source: {data_source}. Mode: {vol_label}. Percentile window: {percentile_window} trading days. "
    f"Baseline: {smooth_window} trading days. Forward columns use realized returns where enough future data exists."
)


# =============================================================================
# RECENT EXTREMES TABLE
# =============================================================================

st.markdown("### Recent extreme setups")

events = build_recent_events(
    df=df,
    event_filter=event_filter,
    max_rows=max_event_rows,
)

render_events_table(events, vol_label=vol_label)

st.caption("© 2026 AD Fund Management LP")
