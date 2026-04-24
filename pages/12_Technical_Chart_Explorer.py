import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# ============================================================
# Page Config
# ============================================================

st.set_page_config(page_title="ADFM Chart Tool", layout="wide")


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.55rem;
            padding-bottom: 0.75rem;
            max-width: 1650px;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.10);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================

COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "ma8": "#6EC6FF",
    "ma20": "#2962FF",
    "ma50": "#7E57C2",
    "ma100": "#C49A00",
    "ma200": "#111111",
    "bb": "#9E9E9E",
    "grid": "rgba(120,120,120,0.18)",
    "text": "#222222",
    "volume_up": "rgba(38,166,154,0.55)",
    "volume_down": "rgba(239,83,80,0.55)",
    "rsi": "#5E35B1",
    "macd": "#1E88E5",
    "signal": "#FB8C00",
    "hist_up": "rgba(38,166,154,0.65)",
    "hist_down": "rgba(239,83,80,0.65)",
    "range": "rgba(80,80,80,0.55)",
    "last": "#1565C0",
    "elliott": "rgba(30,30,30,0.65)",
}

PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}
CAP_MAX_ROWS = 200_000


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    **Purpose:** Technical chart workspace with trend, momentum, volatility, and optional wave-structure diagnostics.

    **What this tab shows**
    - Candlestick chart with moving averages and volatility bands.
    - RSI and MACD for momentum confirmation.
    - Optional ZigZag / Elliott heuristic for structure context.

    **Data source**
    - Yahoo Finance via `yfinance`.
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
show_ma8 = st.sidebar.checkbox("Show MA 8", value=False)
show_ma20 = st.sidebar.checkbox("Show MA 20", value=True)
show_ma50 = st.sidebar.checkbox("Show MA 50", value=True)
show_ma100 = st.sidebar.checkbox("Show MA 100", value=False)
show_ma200 = st.sidebar.checkbox("Show MA 200", value=True)

st.sidebar.subheader("Price Overlays")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=True)
show_last_price = st.sidebar.checkbox("Show last price line", value=False)
show_range_levels = st.sidebar.checkbox("Show visible-range high / low", value=False)

st.sidebar.subheader("Indicators")
show_volume_input = st.sidebar.checkbox("Show Volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)
show_macd = st.sidebar.checkbox("Show MACD (12, 26, 9)", value=True)

st.sidebar.subheader("ZigZag / Elliott Heuristic")
show_elliott = st.sidebar.checkbox("Show structure overlay", value=False)
pivot_window = st.sidebar.slider("Pivot sensitivity (bars left/right)", 2, 20, 8)
min_swing_pct = st.sidebar.slider("Min swing filter (%)", 0.0, 10.0, 2.0, 0.25)
max_pivots = st.sidebar.slider("Max pivots to draw", 20, 200, 80)
elliott_lookback_pivots = st.sidebar.slider("Lookback pivots for auto-count", 20, 200, 80)
elliott_mode = st.sidebar.selectbox(
    "Auto-count mode",
    ["Impulse + ABC (best fit)", "Impulse only (best fit)", "ZigZag only (no labels)"],
    index=0,
)


# ============================================================
# Helpers
# ============================================================

def start_date_from_period(p: str) -> pd.Timestamp | None:
    today = pd.Timestamp.today().normalize()

    if p == "max":
        return None

    if p.endswith("mo"):
        return today - pd.DateOffset(months=int(p[:-2]))

    if p.endswith("y"):
        return today - pd.DateOffset(years=int(p[:-1]))

    return None


def warmup_start_from_period(p: str, interval_choice: str) -> pd.Timestamp | None:
    start_dt = start_date_from_period(p)

    if start_dt is None:
        return None

    if interval_choice == "1d":
        warmup = pd.DateOffset(years=3)
    elif interval_choice == "1wk":
        warmup = pd.DateOffset(years=6)
    else:
        warmup = pd.DateOffset(years=15)

    return start_dt - warmup


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    for col in REQUIRED_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])

    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    if len(out) > CAP_MAX_ROWS:
        out = out.tail(CAP_MAX_ROWS)

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(
    tkr: str,
    intrvl: str,
    adjusted: bool,
    period_choice: str,
    fetch_start: str | None,
) -> tuple[pd.DataFrame, str | None]:
    try:
        ticker_obj = yf.Ticker(tkr)

        if fetch_start is None:
            df = ticker_obj.history(
                period="max",
                interval=intrvl,
                auto_adjust=adjusted,
                actions=False,
            )
        else:
            df = ticker_obj.history(
                start=fetch_start,
                interval=intrvl,
                auto_adjust=adjusted,
                actions=False,
            )

        df = clean_ohlcv(df)

        if df.empty:
            fallback_df = ticker_obj.history(
                period=period_choice,
                interval=intrvl,
                auto_adjust=adjusted,
                actions=False,
            )
            fallback_df = clean_ohlcv(fallback_df)

            if fallback_df.empty:
                return pd.DataFrame(), "No data returned from Yahoo Finance."

            return fallback_df, None

        return df, None

    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


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


def compute_bbands(
    close: pd.Series,
    window: int = 20,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)

    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()

    upper = ma + mult * std
    lower = ma - mult * std

    return ma, upper, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for window in (8, 20, 50, 100, 200):
        out[f"MA{window}"] = out["Close"].rolling(window=window, min_periods=window).mean()

    out["RSI14"] = compute_rsi(out["Close"], 14)

    macd, signal, hist = compute_macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd
    out["Signal"] = signal
    out["Hist"] = hist

    bb_ma, bb_upper, bb_lower = compute_bbands(out["Close"], 20, 2.0)
    out["BB_MA"] = bb_ma
    out["BB_UPPER"] = bb_upper
    out["BB_LOWER"] = bb_lower

    return out


def has_usable_volume(df: pd.DataFrame) -> bool:
    if "Volume" not in df.columns:
        return False

    vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()

    if vol.empty:
        return False

    if (vol > 0).sum() < max(5, int(len(vol) * 0.10)):
        return False

    return True


def volume_nticks(series: pd.Series) -> int:
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return 4

    vmax = float(clean.max())

    if vmax >= 1e10:
        return 7

    if vmax >= 1e9:
        return 6

    return 5


def build_rangebreaks(index: pd.DatetimeIndex, intrvl: str) -> list[dict]:
    if intrvl != "1d" or len(index) == 0:
        return []

    normalized = pd.DatetimeIndex(index).normalize().unique().sort_values()

    if len(normalized) == 0:
        return []

    full_bdays = pd.date_range(start=normalized.min(), end=normalized.max(), freq="B")
    missing_bdays = full_bdays.difference(normalized)

    rangebreaks = [dict(bounds=["sat", "mon"])]

    if len(missing_bdays) > 0:
        rangebreaks.append(dict(values=missing_bdays.strftime("%Y-%m-%d").tolist()))

    return rangebreaks


def is_pivot_high(high_values: np.ndarray, i: int, w: int) -> bool:
    left = high_values[i - w : i]
    right = high_values[i + 1 : i + w + 1]

    if len(left) < w or len(right) < w:
        return False

    return (high_values[i] >= left.max()) and (high_values[i] >= right.max())


def is_pivot_low(low_values: np.ndarray, i: int, w: int) -> bool:
    left = low_values[i - w : i]
    right = low_values[i + 1 : i + w + 1]

    if len(left) < w or len(right) < w:
        return False

    return (low_values[i] <= left.min()) and (low_values[i] <= right.min())


def compute_pivots(price_df: pd.DataFrame, w: int) -> list[dict]:
    if price_df.empty or not {"High", "Low"}.issubset(price_df.columns):
        return []

    highs = price_df["High"].astype(float).values
    lows = price_df["Low"].astype(float).values
    idx = price_df.index

    pivots: list[dict] = []

    for i in range(w, len(price_df) - w):
        high_pivot = is_pivot_high(highs, i, w)
        low_pivot = is_pivot_low(lows, i, w)

        if high_pivot and low_pivot:
            prior_close = float(price_df["Close"].iloc[i - 1]) if i > 0 else float(price_df["Close"].iloc[i])
            high_move = abs(float(highs[i]) - prior_close)
            low_move = abs(float(lows[i]) - prior_close)

            if high_move >= low_move:
                pivots.append({"i": i, "ts": idx[i], "px": float(highs[i]), "type": "H"})
            else:
                pivots.append({"i": i, "ts": idx[i], "px": float(lows[i]), "type": "L"})

        elif high_pivot:
            pivots.append({"i": i, "ts": idx[i], "px": float(highs[i]), "type": "H"})

        elif low_pivot:
            pivots.append({"i": i, "ts": idx[i], "px": float(lows[i]), "type": "L"})

    pivots.sort(key=lambda item: item["i"])

    return pivots


def compress_pivots(
    pivots: list[dict],
    min_pct: float,
    max_points: int,
) -> list[dict]:
    if not pivots:
        return []

    merged: list[dict] = [pivots[0].copy()]

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
        return merged[-max_points:]

    filtered: list[dict] = [merged[0]]
    min_frac = min_pct / 100.0

    for pivot in merged[1:]:
        last = filtered[-1]
        move = abs(pivot["px"] - last["px"])
        denom = max(abs(last["px"]), 1e-9)

        if (move / denom) >= min_frac:
            filtered.append(pivot.copy())
        else:
            if pivot["type"] == last["type"]:
                if pivot["type"] == "H" and pivot["px"] > last["px"]:
                    filtered[-1] = pivot.copy()
                elif pivot["type"] == "L" and pivot["px"] < last["px"]:
                    filtered[-1] = pivot.copy()

    return filtered[-max_points:]


def impulse_score(seq: list[dict]) -> float:
    if len(seq) != 6:
        return -1e9

    types = "".join(p["type"] for p in seq)
    is_up = types == "LHLHLH"
    is_down = types == "HLHLHL"

    if not (is_up or is_down):
        return -1e9

    px = np.array([p["px"] for p in seq], dtype=float)

    w1 = abs(px[1] - px[0])
    w3 = abs(px[3] - px[2])
    w5 = abs(px[5] - px[4])

    if min(w1, w3, w5) <= 0:
        return -1e9

    retr2 = abs(px[2] - px[1]) / max(w1, 1e-9)
    retr4 = abs(px[4] - px[3]) / max(w3, 1e-9)

    score = 0.0

    score += 2.0 if w3 >= w1 else -1.0
    score += 1.5 if w3 >= w5 else 0.0
    score += 2.0 if w3 >= min(w1, w5) else -3.0

    def band_bonus(x: float, lo: float, hi: float, weight: float = 1.0) -> float:
        if lo <= x <= hi:
            return 2.0 * weight

        if (lo - 0.15) <= x <= (hi + 0.15):
            return 1.0 * weight

        return -1.0 * weight

    score += band_bonus(retr2, 0.35, 0.80, 1.0)
    score += band_bonus(retr4, 0.20, 0.60, 0.8)

    if is_up:
        score += 2.0 if px[3] > px[1] else -2.0
        score += 2.0 if px[5] > px[3] else -1.0
        score += 0.5 if px[4] > px[1] else -0.5
    else:
        score += 2.0 if px[3] < px[1] else -2.0
        score += 2.0 if px[5] < px[3] else -1.0
        score += 0.5 if px[4] < px[1] else -0.5

    score += seq[-1]["i"] / 100_000.0

    return score


def find_best_impulse(
    pivots: list[dict],
    lookback: int,
) -> tuple[list[dict] | None, float]:
    if len(pivots) < 6:
        return None, -1e9

    subset = pivots[-lookback:] if len(pivots) > lookback else pivots

    best_seq = None
    best_score = -1e9

    for j in range(len(subset) - 5):
        seq = subset[j : j + 6]
        score = impulse_score(seq)

        if score > best_score:
            best_seq = seq
            best_score = score

    return best_seq, best_score


def try_abc_after_impulse(
    pivots: list[dict],
    impulse: list[dict] | None,
) -> list[dict] | None:
    if impulse is None or len(impulse) != 6:
        return None

    end_i = impulse[-1]["i"]
    pos = None

    for k, pivot in enumerate(pivots):
        if pivot["i"] == end_i and pivot["type"] == impulse[-1]["type"]:
            pos = k
            break

    if pos is None:
        return None

    tail = pivots[pos + 1 :]

    if len(tail) < 3:
        return None

    abc: list[dict] = []
    last_type = impulse[-1]["type"]

    for pivot in tail:
        if pivot["type"] != last_type:
            abc.append(pivot)
            last_type = pivot["type"]

        if len(abc) == 3:
            break

    if len(abc) < 3:
        return None

    p5 = impulse[-1]["px"]
    a, b, c = abc[0]["px"], abc[1]["px"], abc[2]["px"]

    dir_a = np.sign(a - p5)

    if dir_a == 0:
        return None

    if np.sign(b - a) == dir_a:
        return None

    if np.sign(c - b) != dir_a:
        return None

    return abc


def add_horizontal_price_line(
    fig: go.Figure,
    y: float,
    label: str,
    color: str,
    row: int = 1,
) -> None:
    fig.add_hline(
        y=y,
        line_width=1,
        line_dash="dot",
        line_color=color,
        row=row,
        col=1,
        annotation_text=label,
        annotation_position="top right",
        annotation_font=dict(color=color, size=11),
    )


def add_structure_overlay(
    fig: go.Figure,
    pivots: list[dict],
    impulse: list[dict] | None,
    abc: list[dict] | None,
    mode: str,
) -> None:
    if not pivots:
        return

    fig.add_trace(
        go.Scatter(
            x=[p["ts"] for p in pivots],
            y=[p["px"] for p in pivots],
            mode="lines",
            line=dict(width=1.6, color=COLORS["elliott"]),
            name="ZigZag",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if mode == "ZigZag only (no labels)":
        return

    if impulse:
        for pivot, label in zip(impulse, ["0", "1", "2", "3", "4", "5"]):
            fig.add_trace(
                go.Scatter(
                    x=[pivot["ts"]],
                    y=[pivot["px"]],
                    mode="markers+text",
                    marker=dict(size=10, symbol="circle", color="rgba(20,20,20,0.9)"),
                    text=[label],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if abc:
        for pivot, label in zip(abc, ["A", "B", "C"]):
            fig.add_trace(
                go.Scatter(
                    x=[pivot["ts"]],
                    y=[pivot["px"]],
                    mode="markers+text",
                    marker=dict(size=9, symbol="square", color="rgba(20,20,20,0.9)"),
                    text=[label],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )


def format_last_update(index_value: pd.Timestamp) -> str:
    try:
        return pd.Timestamp(index_value).strftime("%b %d, %Y")
    except Exception:
        return ""


def build_manual_legend(
    ma_items: list[tuple[str, str]],
    show_bbands_flag: bool,
    show_last_flag: bool,
    show_range_flag: bool,
) -> str:
    items = ma_items.copy()

    if show_bbands_flag:
        items.append(("Bollinger", COLORS["bb"]))

    if show_last_flag:
        items.append(("Last Price", COLORS["last"]))

    if show_range_flag:
        items.append(("Range", COLORS["range"]))

    if not items:
        return ""

    html = """
    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:18px; margin:2px 0 8px 2px;">
    """

    for label, color in items:
        html += f"""
        <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#444444; line-height:1;">
            <span style="display:inline-block; width:30px; height:0; border-top:3px solid {color};"></span>
            <span>{label}</span>
        </div>
        """

    html += "</div>"

    return html


# ============================================================
# Data Fetch
# ============================================================

if not ticker:
    st.error("Enter a ticker.")
    st.stop()

fetch_start_dt = warmup_start_from_period(period, interval)
fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d") if fetch_start_dt is not None else None

df_raw, fetch_error = get_history(
    tkr=ticker,
    intrvl=interval,
    adjusted=auto_adjust,
    period_choice=period,
    fetch_start=fetch_start_str,
)

if df_raw.empty:
    st.error(f"No data available for {ticker}.")
    if fetch_error:
        st.caption(fetch_error)
    st.stop()

if not REQUIRED_COLS.issubset(set(df_raw.columns)):
    missing = sorted(REQUIRED_COLS.difference(set(df_raw.columns)))
    st.error(f"Returned data is missing required columns: {', '.join(missing)}")
    st.stop()

if interval == "1d":
    df_raw = df_raw[df_raw.index.weekday < 5].copy()

df_ind = add_indicators(df_raw)

start_dt = start_date_from_period(period)

if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()

    if df_display.empty:
        df_display = df_ind.tail(300).copy()
else:
    df_display = df_ind.copy()

if df_display.empty:
    st.error("No display data available after filtering.")
    st.stop()

usable_volume = has_usable_volume(df_display)
show_volume = show_volume_input and usable_volume

if show_volume_input and not usable_volume:
    st.sidebar.info("Volume panel hidden because this symbol does not have usable volume data.")

rangebreaks = build_rangebreaks(df_display.index, interval)


# ============================================================
# Header
# ============================================================

header_suffix = "Adjusted ADFM Chart Tool" if auto_adjust else "ADFM Chart Tool"
last_update = format_last_update(df_display.index[-1])

st.markdown(
    f"""
    <div style="margin-bottom: 4px;">
        <h2 style="margin: 0; padding: 0; font-size: 32px; font-weight: 700; color: #222222;">
            {ticker} | {header_suffix}
        </h2>
        <div style="margin-top: 3px; color: #777777; font-size: 13px;">
            Interval: {interval} | Period: {period} | Last bar: {last_update}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Legend
# ============================================================

ma_config = [
    (8, COLORS["ma8"], show_ma8),
    (20, COLORS["ma20"], show_ma20),
    (50, COLORS["ma50"], show_ma50),
    (100, COLORS["ma100"], show_ma100),
    (200, COLORS["ma200"], show_ma200),
]

active_ma_items = [(f"MA {window}", color) for window, color, enabled in ma_config if enabled]
legend_html = build_manual_legend(
    ma_items=active_ma_items,
    show_bbands_flag=show_bbands,
    show_last_flag=show_last_price,
    show_range_flag=show_range_levels,
)

if legend_html:
    st.markdown(legend_html, unsafe_allow_html=True)


# ============================================================
# Panel Setup
# ============================================================

panel_flags = {
    "volume": show_volume,
    "rsi": show_rsi,
    "macd": show_macd,
}

active_panels = [name for name, enabled in panel_flags.items() if enabled]
row_count = 1 + len(active_panels)

if row_count == 1:
    row_heights = [1.0]
elif row_count == 2:
    row_heights = [0.80, 0.20]
elif row_count == 3:
    row_heights = [0.72, 0.14, 0.14]
else:
    row_heights = [0.66, 0.12, 0.11, 0.11]

height_map = {
    1: 620,
    2: 740,
    3: 840,
    4: 920,
}

fig_height = height_map.get(row_count, 920)

specs = [[{"type": "xy"}] for _ in range(row_count)]

fig = make_subplots(
    rows=row_count,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.012,
    row_heights=row_heights,
    specs=specs,
)

row_map = {"price": 1}
next_row = 2

for panel in active_panels:
    row_map[panel] = next_row
    next_row += 1


# ============================================================
# Price Panel
# ============================================================

fig.add_trace(
    go.Candlestick(
        x=df_display.index,
        open=df_display["Open"],
        high=df_display["High"],
        low=df_display["Low"],
        close=df_display["Close"],
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
    row=row_map["price"],
    col=1,
)

for window, color, enabled in ma_config:
    column = f"MA{window}"

    if enabled and column in df_display.columns:
        fig.add_trace(
            go.Scatter(
                x=df_display.index,
                y=df_display[column],
                mode="lines",
                line=dict(color=color, width=1.6),
                name=f"MA {window}",
                hovertemplate=f"MA {window}: " + "%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=row_map["price"],
            col=1,
        )

if show_bbands:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_UPPER"],
            mode="lines",
            line=dict(width=1.0, color=COLORS["bb"], dash="dot"),
            name="BB Upper",
            hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["price"],
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_LOWER"],
            mode="lines",
            line=dict(width=1.0, color=COLORS["bb"], dash="dot"),
            fill="tonexty",
            fillcolor="rgba(158,158,158,0.10)",
            name="BB Lower",
            hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["price"],
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_MA"],
            mode="lines",
            line=dict(width=1.0, color="rgba(100,100,100,0.7)", dash="dot"),
            name="BB Mid",
            hovertemplate="BB Mid: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["price"],
        col=1,
    )

if show_elliott:
    price_for_pivots = df_display[["High", "Low", "Close"]].dropna()

    if len(price_for_pivots) >= (pivot_window * 2 + 5):
        pivots_raw = compute_pivots(price_for_pivots, pivot_window)
        pivots = compress_pivots(pivots_raw, min_swing_pct, max_pivots)
        impulse, score = find_best_impulse(pivots, elliott_lookback_pivots)

        abc = None

        if elliott_mode == "Impulse + ABC (best fit)" and impulse:
            abc = try_abc_after_impulse(pivots, impulse)

        add_structure_overlay(fig, pivots, impulse, abc, elliott_mode)

        with st.sidebar.expander("Structure diagnostics", expanded=False):
            st.write(f"Pivot count drawn: {len(pivots)}")

            if impulse:
                st.write(f"Best impulse score: {score:.2f}")
                st.write(f"Impulse pivot pattern: {''.join(p['type'] for p in impulse)}")
            else:
                st.write("No valid impulse found in lookback.")

            if abc:
                st.write("ABC found after impulse.")
            elif elliott_mode == "Impulse + ABC (best fit)" and impulse:
                st.write("ABC not found with current settings.")

            st.caption("Heuristic only. Pivots are detected from visible high and low data.")
    else:
        st.sidebar.warning("Not enough visible bars for the structure overlay with this pivot sensitivity.")

last_close = float(df_display["Close"].iloc[-1])
prev_close = float(df_display["Close"].iloc[-2]) if len(df_display) >= 2 else last_close
chg_pct = ((last_close / prev_close) - 1.0) * 100 if prev_close != 0 else 0.0

if show_last_price:
    add_horizontal_price_line(
        fig=fig,
        y=last_close,
        label=f"Last {last_close:,.2f} ({chg_pct:+.2f}%)",
        color=COLORS["last"],
        row=row_map["price"],
    )

if show_range_levels:
    visible_high = float(df_display["High"].max())
    visible_low = float(df_display["Low"].min())

    fig.add_hline(
        y=visible_high,
        line_width=1,
        line_dash="dot",
        line_color=COLORS["range"],
        row=row_map["price"],
        col=1,
        annotation_text=f"Range High {visible_high:,.2f}",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )

    fig.add_hline(
        y=visible_low,
        line_width=1,
        line_dash="dot",
        line_color=COLORS["range"],
        row=row_map["price"],
        col=1,
        annotation_text=f"Range Low {visible_low:,.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )


# ============================================================
# Volume Panel
# ============================================================

if show_volume:
    closes = df_display["Close"].values
    vol_colors = []

    for i in range(len(df_display)):
        if i == 0:
            vol_colors.append("rgba(160,160,160,0.45)")
        else:
            vol_colors.append(
                COLORS["volume_up"] if closes[i] >= closes[i - 1] else COLORS["volume_down"]
            )

    fig.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display["Volume"],
            marker_color=vol_colors,
            name="Volume",
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["volume"],
        col=1,
    )


# ============================================================
# RSI Panel
# ============================================================

if show_rsi:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["RSI14"],
            mode="lines",
            line=dict(width=1.6, color=COLORS["rsi"]),
            name="RSI 14",
            hovertemplate="RSI 14: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["rsi"],
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="rgba(120,120,120,0.55)",
        row=row_map["rsi"],
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="rgba(120,120,120,0.55)",
        row=row_map["rsi"],
        col=1,
    )

    fig.update_yaxes(range=[0, 100], nticks=7, row=row_map["rsi"], col=1)


# ============================================================
# MACD Panel
# ============================================================

if show_macd:
    hist_colors = [
        COLORS["hist_up"] if value >= 0 else COLORS["hist_down"]
        for value in df_display["Hist"].fillna(0.0)
    ]

    fig.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display["Hist"],
            marker_color=hist_colors,
            name="MACD Hist",
            hovertemplate="Hist: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["macd"],
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["MACD"],
            mode="lines",
            line=dict(width=1.5, color=COLORS["macd"]),
            name="MACD",
            hovertemplate="MACD: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["macd"],
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["Signal"],
            mode="lines",
            line=dict(width=1.3, color=COLORS["signal"]),
            name="Signal",
            hovertemplate="Signal: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["macd"],
        col=1,
    )

    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(120,120,120,0.25)",
        row=row_map["macd"],
        col=1,
    )


# ============================================================
# Axes and Layout
# ============================================================

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

fig.update_yaxes(nticks=10, title_text="Price", row=row_map["price"], col=1)

if show_volume:
    fig.update_yaxes(
        nticks=volume_nticks(df_display["Volume"]),
        title_text="Vol",
        row=row_map["volume"],
        col=1,
    )

if show_rsi:
    fig.update_yaxes(title_text="RSI", row=row_map["rsi"], col=1)

if show_macd:
    fig.update_yaxes(nticks=7, title_text="MACD", row=row_map["macd"], col=1)

fig.update_xaxes(
    type="date",
    showgrid=True,
    gridcolor=COLORS["grid"],
    gridwidth=1,
    showline=False,
    rangeslider_visible=False,
    rangebreaks=rangebreaks,
    tickformat="%b '%y" if interval in ["1wk", "1mo"] else "%b %d\n%Y",
)

fig.update_layout(
    height=fig_height,
    title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=10, b=10),
    font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
    showlegend=False,
    legend_title_text="",
    bargap=0.08,
    xaxis_rangeslider_visible=False,
)


# ============================================================
# Render
# ============================================================

st.plotly_chart(
    fig,
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

st.markdown(
    """
    <div style="margin-top: 2px; color: #7a7a7a; font-size: 13px;">
        © 2026 AD Fund Management LP
    </div>
    """,
    unsafe_allow_html=True,
)
