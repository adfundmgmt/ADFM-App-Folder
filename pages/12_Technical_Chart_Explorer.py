import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from dataclasses import dataclass
from plotly.subplots import make_subplots


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="ADFM Chart Tool",
    layout="wide",
)


# ============================================================
# Constants
# ============================================================

PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
CAP_MAX_ROWS = 200_000

COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "ma8": "#00A6FB",
    "ma20": "#2962FF",
    "ma50": "#7E57C2",
    "ma100": "#C49A00",
    "ma200": "#111111",
    "bb": "#9E9E9E",
    "bb_fill": "rgba(158,158,158,0.10)",
    "grid": "rgba(120,120,120,0.18)",
    "text": "#222222",
    "volume_up": "rgba(38,166,154,0.55)",
    "volume_down": "rgba(239,83,80,0.55)",
    "volume_neutral": "rgba(150,150,150,0.45)",
    "rsi": "#5E35B1",
    "macd": "#1E88E5",
    "signal": "#FB8C00",
    "hist_up": "rgba(38,166,154,0.65)",
    "hist_down": "rgba(239,83,80,0.65)",
    "range": "rgba(80,80,80,0.60)",
    "last": "#1565C0",
    "zigzag": "rgba(30,30,30,0.65)",
}


# ============================================================
# Settings
# ============================================================

@dataclass(frozen=True)
class ChartSettings:
    ticker: str
    period: str
    interval: str
    auto_adjust: bool

    show_ma8: bool
    show_ma20: bool
    show_ma50: bool
    show_ma100: bool
    show_ma200: bool

    show_bbands: bool
    show_last_price: bool
    show_range_levels: bool

    show_volume: bool
    show_rsi: bool
    show_macd: bool

    show_structure: bool
    pivot_window: int
    min_swing_pct: float
    max_pivots: int
    structure_lookback_pivots: int
    structure_mode: str


def read_settings() -> ChartSettings:
    st.sidebar.header("About This Tool")
    st.sidebar.markdown(
        """
        **Purpose:** Clean technical chart workspace for trend, momentum, volatility, and structure context.

        **Default setup**
        - Candlesticks
        - MA8, MA20, MA50, MA100, MA200
        - Bollinger Bands
        - Volume when available
        - RSI and MACD

        **Data source**
        - Yahoo Finance via `yfinance`
        """
    )

    ticker = st.sidebar.text_input("Ticker", "^SPX").upper().strip()

    period = st.sidebar.selectbox(
        "Period",
        PERIOD_OPTIONS,
        index=3,
    )

    interval = st.sidebar.selectbox(
        "Interval",
        INTERVAL_OPTIONS,
        index=0,
    )

    auto_adjust = st.sidebar.checkbox("Use adjusted prices", value=False)

    st.sidebar.subheader("Moving Averages")
    show_ma8 = st.sidebar.checkbox("Show MA 8", value=True)
    show_ma20 = st.sidebar.checkbox("Show MA 20", value=True)
    show_ma50 = st.sidebar.checkbox("Show MA 50", value=True)
    show_ma100 = st.sidebar.checkbox("Show MA 100", value=True)
    show_ma200 = st.sidebar.checkbox("Show MA 200", value=True)

    st.sidebar.subheader("Price Overlays")
    show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_last_price = st.sidebar.checkbox("Show last price line", value=False)
    show_range_levels = st.sidebar.checkbox("Show visible-range high / low", value=False)

    st.sidebar.subheader("Indicators")
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    show_macd = st.sidebar.checkbox("Show MACD", value=True)

    st.sidebar.subheader("Structure Overlay")
    show_structure = st.sidebar.checkbox("Show ZigZag / Elliott heuristic", value=False)
    pivot_window = st.sidebar.slider("Pivot sensitivity", 2, 20, 8)
    min_swing_pct = st.sidebar.slider("Minimum swing filter (%)", 0.0, 10.0, 2.0, 0.25)
    max_pivots = st.sidebar.slider("Maximum pivots to draw", 20, 200, 80)
    structure_lookback_pivots = st.sidebar.slider("Lookback pivots for auto-count", 20, 200, 80)
    structure_mode = st.sidebar.selectbox(
        "Structure mode",
        ["ZigZag only", "Impulse only", "Impulse + ABC"],
        index=0,
    )

    return ChartSettings(
        ticker=ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        show_ma8=show_ma8,
        show_ma20=show_ma20,
        show_ma50=show_ma50,
        show_ma100=show_ma100,
        show_ma200=show_ma200,
        show_bbands=show_bbands,
        show_last_price=show_last_price,
        show_range_levels=show_range_levels,
        show_volume=show_volume,
        show_rsi=show_rsi,
        show_macd=show_macd,
        show_structure=show_structure,
        pivot_window=pivot_window,
        min_swing_pct=min_swing_pct,
        max_pivots=max_pivots,
        structure_lookback_pivots=structure_lookback_pivots,
        structure_mode=structure_mode,
    )


# ============================================================
# Dates
# ============================================================

def start_date_from_period(period: str) -> pd.Timestamp | None:
    today = pd.Timestamp.today().normalize()

    if period == "max":
        return None

    if period.endswith("mo"):
        months = int(period[:-2])
        return today - pd.DateOffset(months=months)

    if period.endswith("y"):
        years = int(period[:-1])
        return today - pd.DateOffset(years=years)

    return None


def warmup_start_date(period: str, interval: str) -> pd.Timestamp | None:
    start = start_date_from_period(period)

    if start is None:
        return None

    if interval == "1d":
        return start - pd.DateOffset(years=3)

    if interval == "1wk":
        return start - pd.DateOffset(years=6)

    return start - pd.DateOffset(years=15)


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

    level_0 = list(df.columns.get_level_values(0))
    level_1 = list(df.columns.get_level_values(1))

    required = set(REQUIRED_PRICE_COLUMNS)

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
    if df.empty:
        return pd.DataFrame()

    out = flatten_yfinance_columns(df).copy()

    if out.index.tz is not None:
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
    period: str,
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
    period: str,
    fetch_start: str | None,
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
        fallback = ticker_obj.history(
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )
        cleaned = clean_price_data(fallback)

    return cleaned


def fetch_history(settings: ChartSettings) -> tuple[pd.DataFrame, str | None]:
    fetch_start = warmup_start_date(settings.period, settings.interval)
    fetch_start_str = fetch_start.strftime("%Y-%m-%d") if fetch_start is not None else None

    try:
        df = fetch_with_yfinance_download(
            ticker=settings.ticker,
            interval=settings.interval,
            auto_adjust=settings.auto_adjust,
            period=settings.period,
            fetch_start=fetch_start_str,
        )

        if not df.empty:
            return df, None

    except Exception as exc:
        download_error = f"{type(exc).__name__}: {exc}"
    else:
        download_error = "Yahoo download returned no data."

    try:
        df = fetch_with_ticker_history(
            ticker=settings.ticker,
            interval=settings.interval,
            auto_adjust=settings.auto_adjust,
            period=settings.period,
            fetch_start=fetch_start_str,
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in [8, 20, 50, 100, 200]:
        out[f"MA{window}"] = out["Close"].rolling(
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

    return out


# ============================================================
# Structure Overlay
# ============================================================

def is_pivot_high(values: np.ndarray, i: int, window: int) -> bool:
    left = values[i - window : i]
    right = values[i + 1 : i + window + 1]

    if len(left) < window or len(right) < window:
        return False

    return values[i] >= left.max() and values[i] >= right.max()


def is_pivot_low(values: np.ndarray, i: int, window: int) -> bool:
    left = values[i - window : i]
    right = values[i + 1 : i + window + 1]

    if len(left) < window or len(right) < window:
        return False

    return values[i] <= left.min() and values[i] <= right.min()


def compute_pivots(price_df: pd.DataFrame, window: int) -> list[dict]:
    if price_df.empty:
        return []

    highs = price_df["High"].astype(float).values
    lows = price_df["Low"].astype(float).values
    closes = price_df["Close"].astype(float).values
    index = price_df.index

    pivots = []

    for i in range(window, len(price_df) - window):
        high_pivot = is_pivot_high(highs, i, window)
        low_pivot = is_pivot_low(lows, i, window)

        if high_pivot and low_pivot:
            prior_close = closes[i - 1] if i > 0 else closes[i]
            high_distance = abs(highs[i] - prior_close)
            low_distance = abs(lows[i] - prior_close)

            if high_distance >= low_distance:
                pivots.append({"i": i, "ts": index[i], "px": float(highs[i]), "type": "H"})
            else:
                pivots.append({"i": i, "ts": index[i], "px": float(lows[i]), "type": "L"})

        elif high_pivot:
            pivots.append({"i": i, "ts": index[i], "px": float(highs[i]), "type": "H"})

        elif low_pivot:
            pivots.append({"i": i, "ts": index[i], "px": float(lows[i]), "type": "L"})

    return sorted(pivots, key=lambda x: x["i"])


def compress_pivots(
    pivots: list[dict],
    min_swing_pct: float,
    max_pivots: int,
) -> list[dict]:
    if not pivots:
        return []

    merged = [pivots[0].copy()]

    for pivot in pivots[1:]:
        last = merged[-1]

        if pivot["type"] == last["type"]:
            if pivot["type"] == "H" and pivot["px"] >= last["px"]:
                merged[-1] = pivot.copy()
            elif pivot["type"] == "L" and pivot["px"] <= last["px"]:
                merged[-1] = pivot.copy()
        else:
            merged.append(pivot.copy())

    if len(merged) <= 2:
        return merged[-max_pivots:]

    filtered = [merged[0].copy()]
    min_fraction = min_swing_pct / 100.0

    for pivot in merged[1:]:
        last = filtered[-1]
        move = abs(pivot["px"] - last["px"])
        base = max(abs(last["px"]), 1e-9)

        if move / base >= min_fraction:
            filtered.append(pivot.copy())
        else:
            if pivot["type"] == last["type"]:
                if pivot["type"] == "H" and pivot["px"] > last["px"]:
                    filtered[-1] = pivot.copy()
                elif pivot["type"] == "L" and pivot["px"] < last["px"]:
                    filtered[-1] = pivot.copy()

    return filtered[-max_pivots:]


def impulse_score(sequence: list[dict]) -> float:
    if len(sequence) != 6:
        return -1e9

    pattern = "".join(p["type"] for p in sequence)

    is_up = pattern == "LHLHLH"
    is_down = pattern == "HLHLHL"

    if not is_up and not is_down:
        return -1e9

    px = np.array([p["px"] for p in sequence], dtype=float)

    wave_1 = abs(px[1] - px[0])
    wave_3 = abs(px[3] - px[2])
    wave_5 = abs(px[5] - px[4])

    if min(wave_1, wave_3, wave_5) <= 0:
        return -1e9

    retrace_2 = abs(px[2] - px[1]) / max(wave_1, 1e-9)
    retrace_4 = abs(px[4] - px[3]) / max(wave_3, 1e-9)

    score = 0.0
    score += 2.0 if wave_3 >= wave_1 else -1.0
    score += 1.5 if wave_3 >= wave_5 else 0.0
    score += 2.0 if wave_3 >= min(wave_1, wave_5) else -3.0

    def band_score(value: float, low: float, high: float, weight: float) -> float:
        if low <= value <= high:
            return 2.0 * weight

        if low - 0.15 <= value <= high + 0.15:
            return 1.0 * weight

        return -1.0 * weight

    score += band_score(retrace_2, 0.35, 0.80, 1.0)
    score += band_score(retrace_4, 0.20, 0.60, 0.8)

    if is_up:
        score += 2.0 if px[3] > px[1] else -2.0
        score += 2.0 if px[5] > px[3] else -1.0
        score += 0.5 if px[4] > px[1] else -0.5
    else:
        score += 2.0 if px[3] < px[1] else -2.0
        score += 2.0 if px[5] < px[3] else -1.0
        score += 0.5 if px[4] < px[1] else -0.5

    score += sequence[-1]["i"] / 100_000.0

    return score


def find_best_impulse(
    pivots: list[dict],
    lookback: int,
) -> tuple[list[dict] | None, float]:
    if len(pivots) < 6:
        return None, -1e9

    search_area = pivots[-lookback:] if len(pivots) > lookback else pivots

    best_sequence = None
    best_score = -1e9

    for i in range(len(search_area) - 5):
        sequence = search_area[i : i + 6]
        score = impulse_score(sequence)

        if score > best_score:
            best_sequence = sequence
            best_score = score

    return best_sequence, best_score


def find_abc_after_impulse(
    pivots: list[dict],
    impulse: list[dict] | None,
) -> list[dict] | None:
    if impulse is None or len(impulse) != 6:
        return None

    end_index = impulse[-1]["i"]
    end_type = impulse[-1]["type"]

    position = None

    for i, pivot in enumerate(pivots):
        if pivot["i"] == end_index and pivot["type"] == end_type:
            position = i
            break

    if position is None:
        return None

    tail = pivots[position + 1 :]

    if len(tail) < 3:
        return None

    abc = []
    last_type = end_type

    for pivot in tail:
        if pivot["type"] != last_type:
            abc.append(pivot)
            last_type = pivot["type"]

        if len(abc) == 3:
            break

    if len(abc) < 3:
        return None

    p5 = impulse[-1]["px"]
    a = abc[0]["px"]
    b = abc[1]["px"]
    c = abc[2]["px"]

    direction_a = np.sign(a - p5)

    if direction_a == 0:
        return None

    if np.sign(b - a) == direction_a:
        return None

    if np.sign(c - b) != direction_a:
        return None

    return abc


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
        return row_count, [1.0], 640

    if row_count == 2:
        return row_count, [0.80, 0.20], 760

    if row_count == 3:
        return row_count, [0.72, 0.14, 0.14], 860

    return row_count, [0.66, 0.12, 0.11, 0.11], 940


def add_last_price_line(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close

    change_pct = ((last_close / prev_close) - 1.0) * 100 if prev_close != 0 else 0.0

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
        annotation_text=f"Range High {visible_high:,.2f}",
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
        annotation_text=f"Range Low {visible_low:,.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )


def add_structure_overlay(
    fig: go.Figure,
    df: pd.DataFrame,
    settings: ChartSettings,
    row: int,
) -> None:
    structure_df = df[["High", "Low", "Close"]].dropna()

    if len(structure_df) < settings.pivot_window * 2 + 5:
        st.sidebar.warning("Not enough visible bars for the structure overlay.")
        return

    raw_pivots = compute_pivots(structure_df, settings.pivot_window)
    pivots = compress_pivots(
        raw_pivots,
        settings.min_swing_pct,
        settings.max_pivots,
    )

    if not pivots:
        st.sidebar.warning("No structure pivots detected with current settings.")
        return

    impulse = None
    impulse_fit_score = -1e9
    abc = None

    if settings.structure_mode in ["Impulse only", "Impulse + ABC"]:
        impulse, impulse_fit_score = find_best_impulse(
            pivots,
            settings.structure_lookback_pivots,
        )

    if settings.structure_mode == "Impulse + ABC" and impulse:
        abc = find_abc_after_impulse(pivots, impulse)

    fig.add_trace(
        go.Scatter(
            x=[p["ts"] for p in pivots],
            y=[p["px"] for p in pivots],
            mode="lines",
            line=dict(color=COLORS["zigzag"], width=1.6),
            name="ZigZag",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=row,
        col=1,
    )

    if impulse:
        for pivot, label in zip(impulse, ["0", "1", "2", "3", "4", "5"]):
            fig.add_trace(
                go.Scatter(
                    x=[pivot["ts"]],
                    y=[pivot["px"]],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color="rgba(20,20,20,0.90)",
                        symbol="circle",
                    ),
                    text=[label],
                    textposition="top center",
                    name=f"Wave {label}",
                    hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    if abc:
        for pivot, label in zip(abc, ["A", "B", "C"]):
            fig.add_trace(
                go.Scatter(
                    x=[pivot["ts"]],
                    y=[pivot["px"]],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        color="rgba(20,20,20,0.90)",
                        symbol="square",
                    ),
                    text=[label],
                    textposition="top center",
                    name=f"Wave {label}",
                    hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    with st.sidebar.expander("Structure diagnostics", expanded=False):
        st.write(f"Pivot count drawn: {len(pivots)}")

        if impulse:
            st.write(f"Impulse fit score: {impulse_fit_score:.2f}")
            st.write(f"Impulse pattern: {''.join(p['type'] for p in impulse)}")
        elif settings.structure_mode in ["Impulse only", "Impulse + ABC"]:
            st.write("No valid impulse found.")

        if settings.structure_mode == "Impulse + ABC":
            st.write("ABC found after impulse." if abc else "ABC not found after impulse.")

        st.caption("Heuristic only. Pivots are based on visible high and low data.")


def add_price_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    settings: ChartSettings,
    row: int,
) -> None:
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

    ma_config = [
        ("MA8", "MA 8", COLORS["ma8"], settings.show_ma8),
        ("MA20", "MA 20", COLORS["ma20"], settings.show_ma20),
        ("MA50", "MA 50", COLORS["ma50"], settings.show_ma50),
        ("MA100", "MA 100", COLORS["ma100"], settings.show_ma100),
        ("MA200", "MA 200", COLORS["ma200"], settings.show_ma200),
    ]

    for column, label, color, enabled in ma_config:
        if enabled and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode="lines",
                    line=dict(color=color, width=1.55),
                    name=label,
                    hovertemplate=f"{label}: " + "%{y:.2f}<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=1,
            )

    if settings.show_bbands:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_UPPER"],
                mode="lines",
                line=dict(color=COLORS["bb"], width=1.0, dash="dot"),
                name="Bollinger Bands",
                legendgroup="bbands",
                hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
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
                name="BB Lower",
                legendgroup="bbands",
                hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
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
                line=dict(color="rgba(100,100,100,0.70)", width=1.0, dash="dot"),
                name="BB Mid",
                legendgroup="bbands",
                hovertemplate="BB Mid: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    if settings.show_structure:
        add_structure_overlay(fig, df, settings, row)

    if settings.show_last_price:
        add_last_price_line(fig, df, row)

    if settings.show_range_levels:
        add_range_levels(fig, df, row)


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
            line=dict(color=COLORS["rsi"], width=1.6),
            name="RSI 14",
            hovertemplate="RSI 14: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(120,120,120,0.55)",
        line_width=1,
        row=row,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="rgba(120,120,120,0.55)",
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
            name="MACD Histogram",
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
            line=dict(color=COLORS["macd"], width=1.5),
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
            line=dict(color=COLORS["signal"], width=1.3),
            name="Signal",
            hovertemplate="Signal: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_color="rgba(120,120,120,0.25)",
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

    fig.update_yaxes(
        title_text="Price",
        nticks=10,
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
            nticks=7,
            row=row_map["rsi"],
            col=1,
        )

    if "macd" in row_map:
        fig.update_yaxes(
            title_text="MACD",
            nticks=7,
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

    legend_enabled = any(
        [
            settings.show_ma8,
            settings.show_ma20,
            settings.show_ma50,
            settings.show_ma100,
            settings.show_ma200,
            settings.show_bbands,
            settings.show_structure,
        ]
    )

    fig.update_layout(
        height=fig_height,
        title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=18, b=10),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"],
        ),
        showlegend=legend_enabled,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12, color="#444444"),
            title_text="",
        ),
        bargap=0.08,
        xaxis_rangeslider_visible=False,
    )

    return fig


# ============================================================
# App
# ============================================================

settings = read_settings()

if not settings.ticker:
    st.error("Enter a ticker.")
    st.stop()

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

header_suffix = "Adjusted ADFM Chart Tool" if settings.auto_adjust else "ADFM Chart Tool"
last_bar = format_last_bar(display_df.index[-1])

st.title(f"{settings.ticker} | {header_suffix}")
st.caption(f"Interval: {settings.interval} | Period: {settings.period} | Last bar: {last_bar}")

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
        ],
    },
)

st.caption("© 2026 AD Fund Management LP")
