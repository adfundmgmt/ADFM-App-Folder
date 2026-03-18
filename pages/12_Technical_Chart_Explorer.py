import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="ADFM Chart Tool", layout="wide")

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.6rem;
        max-width: 1600px;
    }
    .metric-card {
        background: #fafafa;
        border: 1px solid #e9e9e9;
        border-radius: 12px;
        padding: 12px 14px;
        height: 100%;
    }
    .metric-label {
        font-size: 12px;
        color: #666666;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #111111;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #666666;
        margin-top: 4px;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .small-note {
        font-size: 12px;
        color: #6d6d6d;
    }
    .header-wrap {
        margin-bottom: 0.35rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# PRESETS
# =========================================================
PRESETS: Dict[str, Dict] = {
    "Trend": {
        "period": "1y",
        "interval": "1d",
        "show_ma8": False,
        "show_ma20": True,
        "show_ma50": True,
        "show_ma100": True,
        "show_ma200": True,
        "show_bbands": False,
        "show_last_price": True,
        "show_range_levels": True,
        "show_breakout_levels": True,
        "show_prev_levels": True,
        "show_anchored_vwap": True,
        "show_volume": True,
        "show_rsi": True,
        "show_macd": False,
        "show_rel_strength": True,
        "show_elliott": False,
        "benchmark": "SPY",
    },
    "Swing": {
        "period": "6mo",
        "interval": "1d",
        "show_ma8": True,
        "show_ma20": True,
        "show_ma50": True,
        "show_ma100": False,
        "show_ma200": True,
        "show_bbands": True,
        "show_last_price": True,
        "show_range_levels": True,
        "show_breakout_levels": True,
        "show_prev_levels": True,
        "show_anchored_vwap": False,
        "show_volume": True,
        "show_rsi": True,
        "show_macd": True,
        "show_rel_strength": True,
        "show_elliott": False,
        "benchmark": "QQQ",
    },
    "Macro": {
        "period": "5y",
        "interval": "1wk",
        "show_ma8": False,
        "show_ma20": True,
        "show_ma50": True,
        "show_ma100": True,
        "show_ma200": False,
        "show_bbands": False,
        "show_last_price": True,
        "show_range_levels": True,
        "show_breakout_levels": False,
        "show_prev_levels": False,
        "show_anchored_vwap": False,
        "show_volume": True,
        "show_rsi": True,
        "show_macd": False,
        "show_rel_strength": True,
        "show_elliott": False,
        "benchmark": "SPY",
    },
    "Clean": {
        "period": "1y",
        "interval": "1d",
        "show_ma8": False,
        "show_ma20": True,
        "show_ma50": False,
        "show_ma100": False,
        "show_ma200": True,
        "show_bbands": False,
        "show_last_price": True,
        "show_range_levels": False,
        "show_breakout_levels": False,
        "show_prev_levels": False,
        "show_anchored_vwap": False,
        "show_volume": False,
        "show_rsi": False,
        "show_macd": False,
        "show_rel_strength": False,
        "show_elliott": False,
        "benchmark": "SPY",
    },
    "Custom": {},
}

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
A tighter internal chart workstation built for setup recognition rather than just plotting candles.

It combines price structure, multi-timeframe trend, momentum, relative strength, volume context, and actionable levels such as prior highs/lows, breakout bands, and anchored VWAP.

The goal is to answer a few questions quickly: where price sits versus trend, whether momentum confirms, whether volume is sponsoring the move, and what levels matter next.
"""
)

preset = st.sidebar.selectbox(
    "Workspace Preset",
    ["Trend", "Swing", "Macro", "Clean", "Custom"],
    index=0,
)

default_cfg = PRESETS.get(preset, PRESETS["Trend"])

ticker = st.sidebar.text_input("Ticker", "^SPX").upper().strip()

period = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"],
    index=["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"].index(
        default_cfg.get("period", "1y")
    ),
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=["1d", "1wk", "1mo"].index(default_cfg.get("interval", "1d")),
)

auto_adjust = st.sidebar.checkbox("Use adjusted prices", value=False)

benchmark = st.sidebar.text_input(
    "Benchmark for Relative Strength",
    value=default_cfg.get("benchmark", "SPY"),
).upper().strip()

st.sidebar.subheader("Overlays")
show_ma8 = st.sidebar.checkbox("Show MA 8", value=default_cfg.get("show_ma8", True))
show_ma20 = st.sidebar.checkbox("Show MA 20", value=default_cfg.get("show_ma20", True))
show_ma50 = st.sidebar.checkbox("Show MA 50", value=default_cfg.get("show_ma50", True))
show_ma100 = st.sidebar.checkbox("Show MA 100", value=default_cfg.get("show_ma100", True))
show_ma200 = st.sidebar.checkbox("Show MA 200", value=default_cfg.get("show_ma200", True))
show_bbands = st.sidebar.checkbox(
    "Show Bollinger Bands (20, 2.0)", value=default_cfg.get("show_bbands", False)
)
show_last_price = st.sidebar.checkbox(
    "Show last price line", value=default_cfg.get("show_last_price", True)
)
show_range_levels = st.sidebar.checkbox(
    "Show visible-range high / low", value=default_cfg.get("show_range_levels", False)
)
show_breakout_levels = st.sidebar.checkbox(
    "Show 20-bar breakout / breakdown", value=default_cfg.get("show_breakout_levels", True)
)
show_prev_levels = st.sidebar.checkbox(
    "Show prior day / week / month levels", value=default_cfg.get("show_prev_levels", True)
)
show_anchored_vwap = st.sidebar.checkbox(
    "Show anchored VWAP (visible window start)", value=default_cfg.get("show_anchored_vwap", False)
)

st.sidebar.subheader("Indicators")
show_volume = st.sidebar.checkbox("Show Volume", value=default_cfg.get("show_volume", True))
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=default_cfg.get("show_rsi", True))
show_macd = st.sidebar.checkbox("Show MACD (12, 26, 9)", value=default_cfg.get("show_macd", False))
show_rel_strength = st.sidebar.checkbox(
    "Show Relative Strength vs Benchmark", value=default_cfg.get("show_rel_strength", True)
)

st.sidebar.subheader("Experimental Overlay")
show_elliott = st.sidebar.checkbox(
    "Show Elliott wave overlay", value=default_cfg.get("show_elliott", False)
)
pivot_window = st.sidebar.slider("Pivot sensitivity (bars left/right)", 2, 20, 8)
min_swing_pct = st.sidebar.slider("Min swing filter (%)", 0.0, 10.0, 2.0, 0.25)
max_pivots = st.sidebar.slider("Max pivots to draw", 20, 200, 80)
elliott_lookback_pivots = st.sidebar.slider("Lookback pivots for auto-count", 20, 200, 80)
elliott_mode = st.sidebar.selectbox(
    "Auto-count mode",
    ["Impulse + ABC (best fit)", "Impulse only (best fit)", "ZigZag only (no labels)"],
    index=0,
)

# =========================================================
# HELPERS
# =========================================================
def start_date_from_period(p: str) -> Optional[pd.Timestamp]:
    today = pd.Timestamp.today().normalize()
    if p == "max":
        return None
    if p.endswith("mo"):
        return today - pd.DateOffset(months=int(p[:-2]))
    if p.endswith("y"):
        return today - pd.DateOffset(years=int(p[:-1]))
    return None


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.2f}%"


def safe_div(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def distance_pct(x: float, ref: float) -> float:
    if ref == 0 or pd.isna(ref):
        return np.nan
    return (x / ref - 1.0) * 100.0


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(tkr: str, intrvl: str, adjusted: bool) -> pd.DataFrame:
    last_err = None
    for _ in range(3):
        try:
            df = yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=adjusted)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def build_indicator_frame(
    tkr: str,
    intrvl: str,
    adjusted: bool,
) -> pd.DataFrame:
    df = fetch_history(tkr, intrvl, adjusted)
    if df.empty:
        return df

    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if intrvl == "1d":
        df = df[df.index.weekday < 5]

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for w in (8, 20, 50, 100, 200):
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    df["RSI14"] = compute_rsi(df["Close"], 14)

    macd, signal, hist = compute_macd(df["Close"], 12, 26, 9)
    df["MACD"] = macd
    df["Signal"] = signal
    df["Hist"] = hist

    bb_ma, bb_upper, bb_lower = compute_bbands(df["Close"], 20, 2.0)
    df["BB_MA"] = bb_ma
    df["BB_UPPER"] = bb_upper
    df["BB_LOWER"] = bb_lower

    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RVOL20"] = df["Volume"] / df["VOL20"].replace(0, np.nan)
    df["RET1D"] = df["Close"].pct_change()
    df["RV20"] = df["RET1D"].rolling(20).std() * np.sqrt(252) * 100
    df["HH20"] = df["High"].rolling(20).max()
    df["LL20"] = df["Low"].rolling(20).min()
    df["HH63"] = df["High"].rolling(63).max()
    df["LL63"] = df["Low"].rolling(63).min()
    return df


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + mult * std
    lower = ma - mult * std
    return ma, upper, lower


def compute_anchored_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = (typical * df["Volume"]).cumsum()
    vv = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vv


def compute_relative_strength(asset: pd.Series, bench: pd.Series) -> pd.Series:
    aligned = pd.concat([asset, bench], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    rs = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    return rs / rs.iloc[0] * 100.0


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Open"] = df["Open"].resample("W-FRI").first()
    out["High"] = df["High"].resample("W-FRI").max()
    out["Low"] = df["Low"].resample("W-FRI").min()
    out["Close"] = df["Close"].resample("W-FRI").last()
    out["Volume"] = df["Volume"].resample("W-FRI").sum()
    out = out.dropna(how="any")
    for w in (21, 50, 100):
        out[f"WMA{w}"] = out["Close"].rolling(w).mean()
    return out


def prior_period_levels(df_daily: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {
        "prior_day_high": np.nan,
        "prior_day_low": np.nan,
        "prior_week_high": np.nan,
        "prior_week_low": np.nan,
        "prior_month_high": np.nan,
        "prior_month_low": np.nan,
    }
    if len(df_daily) < 3:
        return out

    daily = df_daily.copy()
    daily["day"] = daily.index.normalize()

    day_groups = daily.groupby("day").agg({"High": "max", "Low": "min"})
    if len(day_groups) >= 2:
        out["prior_day_high"] = float(day_groups["High"].iloc[-2])
        out["prior_day_low"] = float(day_groups["Low"].iloc[-2])

    weekly = daily.resample("W-FRI").agg({"High": "max", "Low": "min"}).dropna()
    if len(weekly) >= 2:
        out["prior_week_high"] = float(weekly["High"].iloc[-2])
        out["prior_week_low"] = float(weekly["Low"].iloc[-2])

    monthly = daily.resample("M").agg({"High": "max", "Low": "min"}).dropna()
    if len(monthly) >= 2:
        out["prior_month_high"] = float(monthly["High"].iloc[-2])
        out["prior_month_low"] = float(monthly["Low"].iloc[-2])

    return out


def classify_setup(row: pd.Series) -> Tuple[str, str]:
    close = float(row["Close"])
    ma20 = float(row.get("MA20", np.nan))
    ma50 = float(row.get("MA50", np.nan))
    ma200 = float(row.get("MA200", np.nan))
    rsi = float(row.get("RSI14", np.nan))
    hist = float(row.get("Hist", np.nan))
    hh20 = float(row.get("HH20", np.nan))
    ll20 = float(row.get("LL20", np.nan))
    rvol = float(row.get("RVOL20", np.nan))

    above20 = close > ma20 if not pd.isna(ma20) else False
    above50 = close > ma50 if not pd.isna(ma50) else False
    above200 = close > ma200 if not pd.isna(ma200) else False

    if above20 and above50 and above200 and rsi >= 55 and hist >= 0:
        if not pd.isna(hh20) and close >= hh20 * 0.99:
            return "Uptrend near breakout", "Trend intact, pressing upper range."
        if rsi >= 70:
            return "Uptrend, extended", "Strong tape, but near-term stretched."
        return "Uptrend", "Trend and momentum aligned."

    if above50 and above200 and 45 <= rsi < 55:
        return "Base building", "Constructive structure, but momentum still neutral."

    if above200 and close < ma20 and rsi < 50:
        return "Pullback in larger uptrend", "Longer trend intact, near-term momentum weaker."

    if (not above200) and rsi < 45 and hist < 0:
        if not pd.isna(ll20) and close <= ll20 * 1.01:
            return "Downtrend near breakdown", "Weak tape, pressure near lower range."
        return "Downtrend", "Trend and momentum deteriorating."

    if 45 <= rsi <= 55 and (pd.isna(rvol) or rvol < 1.2):
        return "Range / compression", "Low-energy structure, waiting for expansion."

    return "Mixed / transitional", "Signals are not cleanly aligned."


def compute_breadth_style_summary(last_row: pd.Series, weekly_row: Optional[pd.Series]) -> Dict[str, str]:
    close = float(last_row["Close"])
    out = {}

    for w in [20, 50, 100, 200]:
        col = f"MA{w}"
        out[f"ma{w}"] = format_pct(distance_pct(close, float(last_row[col]))) if col in last_row else "n/a"

    if weekly_row is not None and not weekly_row.empty:
        wclose = float(weekly_row["Close"])
        for w in [21, 50, 100]:
            col = f"WMA{w}"
            val = weekly_row.get(col, np.nan)
            out[f"wma{w}"] = format_pct(distance_pct(wclose, float(val))) if not pd.isna(val) else "n/a"
    else:
        out["wma21"] = "n/a"
        out["wma50"] = "n/a"
        out["wma100"] = "n/a"

    return out


def _is_pivot_high(values: np.ndarray, i: int, w: int) -> bool:
    left = values[i - w : i]
    right = values[i + 1 : i + w + 1]
    if len(left) < w or len(right) < w:
        return False
    return (values[i] >= left.max()) and (values[i] >= right.max())


def _is_pivot_low(values: np.ndarray, i: int, w: int) -> bool:
    left = values[i - w : i]
    right = values[i + 1 : i + w + 1]
    if len(left) < w or len(right) < w:
        return False
    return (values[i] <= left.min()) and (values[i] <= right.min())


def compute_pivots(close: pd.Series, w: int) -> List[dict]:
    values = close.values.astype(float)
    idx = close.index
    pivots: List[dict] = []
    for i in range(w, len(values) - w):
        if _is_pivot_high(values, i, w):
            pivots.append({"i": i, "ts": idx[i], "px": float(values[i]), "type": "H"})
        elif _is_pivot_low(values, i, w):
            pivots.append({"i": i, "ts": idx[i], "px": float(values[i]), "type": "L"})
    pivots.sort(key=lambda x: x["i"])
    return pivots


def compress_pivots(pivots: List[dict], min_pct: float, max_points: int) -> List[dict]:
    if not pivots:
        return []

    merged: List[dict] = [pivots[0].copy()]
    for p in pivots[1:]:
        last = merged[-1]
        if p["type"] == last["type"]:
            if p["type"] == "H" and p["px"] >= last["px"]:
                merged[-1] = p.copy()
            elif p["type"] == "L" and p["px"] <= last["px"]:
                merged[-1] = p.copy()
        else:
            merged.append(p.copy())

    if len(merged) <= 2:
        return merged[-max_points:]

    filtered: List[dict] = [merged[0]]
    min_frac = min_pct / 100.0
    for p in merged[1:]:
        last = filtered[-1]
        move = abs(p["px"] - last["px"])
        denom = max(abs(last["px"]), 1e-9)
        if (move / denom) >= min_frac:
            filtered.append(p)
        else:
            if p["type"] == last["type"]:
                if p["type"] == "H" and p["px"] > last["px"]:
                    filtered[-1] = p
                elif p["type"] == "L" and p["px"] < last["px"]:
                    filtered[-1] = p

    return filtered[-max_points:]


def _impulse_score(seq: List[dict]) -> float:
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

    score += seq[-1]["i"] / 100000.0
    return score


def find_best_impulse(pivots: List[dict], lookback: int) -> Tuple[Optional[List[dict]], float]:
    if len(pivots) < 6:
        return None, -1e9

    subset = pivots[-lookback:] if len(pivots) > lookback else pivots
    best_seq = None
    best_score = -1e9

    for j in range(len(subset) - 5):
        seq = subset[j : j + 6]
        score = _impulse_score(seq)
        if score > best_score:
            best_score = score
            best_seq = seq

    return best_seq, best_score


def try_abc_after_impulse(pivots: List[dict], impulse: List[dict]) -> Optional[List[dict]]:
    if impulse is None or len(impulse) != 6:
        return None

    end_i = impulse[-1]["i"]
    pos = None
    for k, p in enumerate(pivots):
        if p["i"] == end_i and p["type"] == impulse[-1]["type"]:
            pos = k
            break

    if pos is None:
        return None

    tail = pivots[pos + 1 :]
    if len(tail) < 3:
        return None

    abc: List[dict] = []
    last_type = impulse[-1]["type"]
    for p in tail:
        if p["type"] != last_type:
            abc.append(p)
            last_type = p["type"]
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


def add_elliott_overlay(
    fig: go.Figure,
    pivots: List[dict],
    impulse: Optional[List[dict]],
    abc: Optional[List[dict]],
    mode: str,
):
    if not pivots:
        return

    fig.add_trace(
        go.Scatter(
            x=[p["ts"] for p in pivots],
            y=[p["px"] for p in pivots],
            mode="lines",
            line=dict(width=1.4, color="rgba(30,30,30,0.60)"),
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
        for p, lab in zip(impulse, ["0", "1", "2", "3", "4", "5"]):
            fig.add_trace(
                go.Scatter(
                    x=[p["ts"]],
                    y=[p["px"]],
                    mode="markers+text",
                    marker=dict(size=9, symbol="circle", color="rgba(20,20,20,0.9)"),
                    text=[lab],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"{lab}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if abc:
        for p, lab in zip(abc, ["A", "B", "C"]):
            fig.add_trace(
                go.Scatter(
                    x=[p["ts"]],
                    y=[p["px"]],
                    mode="markers+text",
                    marker=dict(size=8, symbol="square", color="rgba(20,20,20,0.9)"),
                    text=[lab],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=f"{lab}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )


def add_hline(fig: go.Figure, y: float, label: str, color: str, row: int = 1, dash: str = "dot"):
    if pd.isna(y):
        return
    fig.add_hline(
        y=y,
        line_width=1,
        line_dash=dash,
        line_color=color,
        row=row,
        col=1,
        annotation_text=label,
        annotation_position="top right",
        annotation_font=dict(color=color, size=10),
    )


def build_rangebreaks(index: pd.DatetimeIndex, intrvl: str) -> List[dict]:
    if intrvl != "1d" or len(index) == 0:
        return []

    normalized = pd.DatetimeIndex(index).normalize().unique().sort_values()
    full_bdays = pd.date_range(start=normalized.min(), end=normalized.max(), freq="B")
    missing_bdays = full_bdays.difference(normalized)

    rangebreaks = [dict(bounds=["sat", "mon"])]
    if len(missing_bdays) > 0:
        rangebreaks.append(dict(values=missing_bdays.strftime("%Y-%m-%d").tolist()))
    return rangebreaks


def volume_nticks(series: pd.Series) -> int:
    vmax = float(series.max()) if len(series) else 0.0
    if vmax >= 1e10:
        return 7
    if vmax >= 1e9:
        return 6
    return 5


# =========================================================
# DATA LOAD
# =========================================================
df_full = build_indicator_frame(ticker, interval, auto_adjust)
if df_full.empty:
    st.error("No data returned. Check the ticker or data source.")
    st.stop()

start_dt = start_date_from_period(period)
buffer_years = 10

if start_dt is not None:
    warmup_start = start_dt - pd.DateOffset(years=buffer_years)
    df_ind = df_full[df_full.index >= warmup_start].copy()
else:
    df_ind = df_full.copy()

if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()
    if df_display.empty:
        df_display = df_ind.tail(300).copy()
else:
    df_display = df_ind.copy()

if df_display.empty:
    st.error("No display data available after filtering.")
    st.stop()

# Daily frame for prior-day/week/month levels and weekly context
df_daily_full = build_indicator_frame(ticker, "1d", auto_adjust)
weekly_context = None
if not df_daily_full.empty:
    weekly_context = resample_to_weekly(df_daily_full)
else:
    weekly_context = pd.DataFrame()

bench_rel = pd.Series(dtype=float)
if show_rel_strength and benchmark:
    bench_df = build_indicator_frame(benchmark, interval, auto_adjust)
    if not bench_df.empty:
        bench_rel = compute_relative_strength(df_display["Close"], bench_df["Close"])

rangebreaks = build_rangebreaks(df_display.index, interval)

# =========================================================
# HEADER + SUMMARY STRIP
# =========================================================
last = df_display.iloc[-1]
prev = df_display.iloc[-2] if len(df_display) >= 2 else last

last_close = float(last["Close"])
prev_close = float(prev["Close"])
day_chg_pct = (last_close / prev_close - 1.0) * 100.0 if prev_close != 0 else np.nan

setup_label, setup_desc = classify_setup(last)

weekly_last = None
if weekly_context is not None and not weekly_context.empty:
    weekly_context = weekly_context[weekly_context.index <= df_display.index.max()]
    if not weekly_context.empty:
        weekly_last = weekly_context.iloc[-1]

dist_map = compute_breadth_style_summary(last, weekly_last)

ytd_high = float(df_full["High"].tail(252).max()) if len(df_full) >= 20 else float(df_full["High"].max())
ytd_low = float(df_full["Low"].tail(252).min()) if len(df_full) >= 20 else float(df_full["Low"].min())

with st.container():
    st.markdown(
        f"""
        <div class="header-wrap">
            <h2 style="margin:0; padding:0; font-size:30px; font-weight:700; color:#1a1a1a;">
                {ticker} | ADFM Technical Workstation
            </h2>
            <div style="margin-top:4px; color:#666666; font-size:13px;">
                {setup_label} · {setup_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

mcols = st.columns(7)

metrics_payload = [
    ("Last", format_price(last_close), format_pct(day_chg_pct)),
    ("Setup", setup_label, setup_desc),
    ("RSI 14", format_price(float(last["RSI14"])), "Momentum state"),
    ("RVOL 20", format_price(float(last["RVOL20"])), "Today vs 20-bar avg"),
    ("Realized Vol 20", format_price(float(last["RV20"])) + "%" if not pd.isna(last["RV20"]) else "n/a", "Annualized"),
    ("52W High Dist", format_pct(distance_pct(last_close, ytd_high)), f"High {format_price(ytd_high)}"),
    ("52W Low Dist", format_pct(distance_pct(last_close, ytd_low)), f"Low {format_price(ytd_low)}"),
]

for col, (lab, val, sub) in zip(mcols, metrics_payload):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{lab}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# CONTEXT BLOCKS
# =========================================================
c1, c2, c3 = st.columns([1.35, 1.35, 1.3])

with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Trend context**")
    st.write(
        f"Price vs MA20 {dist_map['ma20']}, vs MA50 {dist_map['ma50']}, vs MA100 {dist_map['ma100']}, vs MA200 {dist_map['ma200']}."
    )
    st.write(
        f"Weekly context sits at vs WMA21 {dist_map['wma21']}, vs WMA50 {dist_map['wma50']}, vs WMA100 {dist_map['wma100']}."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Actionable levels**")
    breakout_hi = float(df_display["HH20"].iloc[-2]) if len(df_display) >= 22 else np.nan
    breakdown_lo = float(df_display["LL20"].iloc[-2]) if len(df_display) >= 22 else np.nan
    st.write(
        f"20-bar breakout {format_price(breakout_hi)} and 20-bar breakdown {format_price(breakdown_lo)}."
    )
    if not df_daily_full.empty:
        levels = prior_period_levels(df_daily_full)
        st.write(
            f"Prior day H/L {format_price(levels['prior_day_high'])} / {format_price(levels['prior_day_low'])}, "
            f"prior week H/L {format_price(levels['prior_week_high'])} / {format_price(levels['prior_week_low'])}, "
            f"prior month H/L {format_price(levels['prior_month_high'])} / {format_price(levels['prior_month_low'])}."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Market structure read**")
    macd_state = "positive" if float(last["Hist"]) >= 0 else "negative"
    rsi_state = (
        "overbought" if float(last["RSI14"]) >= 70 else
        "oversold" if float(last["RSI14"]) <= 30 else
        "neutral / constructive" if float(last["RSI14"]) >= 50 else
        "neutral / weak"
    )
    st.write(
        f"MACD histogram is {macd_state}. RSI regime reads {rsi_state}. "
        f"Relative volume is {format_price(float(last['RVOL20']))}x."
    )
    if show_rel_strength and not bench_rel.empty:
        rs_trend = float(bench_rel.iloc[-1] / bench_rel.iloc[max(0, len(bench_rel) - 21)] - 1.0) * 100 if len(bench_rel) > 20 else np.nan
        st.write(
            f"Relative strength versus {benchmark} is at {format_price(float(bench_rel.iloc[-1]))} rebased, "
            f"with 1-month move {format_pct(rs_trend)}."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PANEL SETUP
# =========================================================
panel_flags = {
    "volume": show_volume,
    "rsi": show_rsi,
    "macd": show_macd,
    "rel": show_rel_strength and not bench_rel.empty,
}
active_panels = [k for k, v in panel_flags.items() if v]

row_count = 1 + len(active_panels)

if row_count == 1:
    row_heights = [1.0]
elif row_count == 2:
    row_heights = [0.78, 0.22]
elif row_count == 3:
    row_heights = [0.66, 0.17, 0.17]
elif row_count == 4:
    row_heights = [0.58, 0.14, 0.14, 0.14]
else:
    row_heights = [0.52, 0.12, 0.12, 0.12, 0.12]

height_map = {1: 680, 2: 820, 3: 940, 4: 1060, 5: 1180}
fig_height = height_map.get(row_count, 1180)

specs = [[{"type": "candlestick"}]]
for panel in active_panels:
    if panel == "volume":
        specs.append([{"type": "bar"}])
    else:
        specs.append([{"type": "scatter"}])

fig = make_subplots(
    rows=row_count,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.018,
    row_heights=row_heights,
    specs=specs,
)

row_map = {"price": 1}
current_row = 2
for panel in active_panels:
    row_map[panel] = current_row
    current_row += 1

# =========================================================
# STYLING
# =========================================================
COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "ma8": "#6EC6FF",
    "ma20": "#2962FF",
    "ma50": "#7E57C2",
    "ma100": "#C49A00",
    "ma200": "#111111",
    "bb": "#9E9E9E",
    "grid": "rgba(120,120,120,0.14)",
    "text": "#222222",
    "volume_up": "rgba(38,166,154,0.55)",
    "volume_down": "rgba(239,83,80,0.55)",
    "volume_ma": "#455A64",
    "rsi": "#5E35B1",
    "macd": "#1E88E5",
    "signal": "#FB8C00",
    "hist_up": "rgba(38,166,154,0.65)",
    "hist_down": "rgba(239,83,80,0.65)",
    "range": "rgba(80,80,80,0.55)",
    "last": "#1565C0",
    "breakout": "#2E7D32",
    "breakdown": "#C62828",
    "avwap": "#00897B",
    "prior_day": "#546E7A",
    "prior_week": "#8E24AA",
    "prior_month": "#5D4037",
    "rel": "#3949AB",
}

# =========================================================
# PRICE PANEL
# =========================================================
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
        showlegend=False,
        name="Price",
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

ma_config = [
    (8, COLORS["ma8"], show_ma8),
    (20, COLORS["ma20"], show_ma20),
    (50, COLORS["ma50"], show_ma50),
    (100, COLORS["ma100"], show_ma100),
    (200, COLORS["ma200"], show_ma200),
]

for w, color, enabled in ma_config:
    col = f"MA{w}"
    if enabled and col in df_display.columns:
        fig.add_trace(
            go.Scatter(
                x=df_display.index,
                y=df_display[col],
                mode="lines",
                line=dict(color=color, width=1.6),
                name=f"MA {w}",
                hovertemplate=f"MA {w}: " + "%{y:.2f}<extra></extra>",
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
            hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="BB Upper",
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
            hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="BB Lower",
        ),
        row=row_map["price"],
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_MA"],
            mode="lines",
            line=dict(width=1.0, color="rgba(100,100,100,0.65)", dash="dot"),
            hovertemplate="BB Mid: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="BB Mid",
        ),
        row=row_map["price"],
        col=1,
    )

if show_anchored_vwap:
    avwap = compute_anchored_vwap(df_display)
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=avwap,
            mode="lines",
            line=dict(width=1.7, color=COLORS["avwap"]),
            hovertemplate="Anchored VWAP: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="Anchored VWAP",
        ),
        row=row_map["price"],
        col=1,
    )

if show_last_price:
    add_hline(
        fig,
        y=last_close,
        label=f"Last {format_price(last_close)} ({format_pct(day_chg_pct)})",
        color=COLORS["last"],
        row=row_map["price"],
        dash="dot",
    )

if show_range_levels:
    add_hline(
        fig,
        y=float(df_display["High"].max()),
        label=f"Visible High {format_price(float(df_display['High'].max()))}",
        color=COLORS["range"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=float(df_display["Low"].min()),
        label=f"Visible Low {format_price(float(df_display['Low'].min()))}",
        color=COLORS["range"],
        row=row_map["price"],
        dash="dot",
    )

if show_breakout_levels and len(df_display) >= 22:
    breakout_level = float(df_display["HH20"].iloc[-2])
    breakdown_level = float(df_display["LL20"].iloc[-2])
    add_hline(
        fig,
        y=breakout_level,
        label=f"20-bar Breakout {format_price(breakout_level)}",
        color=COLORS["breakout"],
        row=row_map["price"],
        dash="dash",
    )
    add_hline(
        fig,
        y=breakdown_level,
        label=f"20-bar Breakdown {format_price(breakdown_level)}",
        color=COLORS["breakdown"],
        row=row_map["price"],
        dash="dash",
    )

if show_prev_levels and not df_daily_full.empty and interval == "1d":
    levels = prior_period_levels(df_daily_full)
    add_hline(
        fig,
        y=levels["prior_day_high"],
        label=f"PDH {format_price(levels['prior_day_high'])}",
        color=COLORS["prior_day"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=levels["prior_day_low"],
        label=f"PDL {format_price(levels['prior_day_low'])}",
        color=COLORS["prior_day"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=levels["prior_week_high"],
        label=f"PWH {format_price(levels['prior_week_high'])}",
        color=COLORS["prior_week"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=levels["prior_week_low"],
        label=f"PWL {format_price(levels['prior_week_low'])}",
        color=COLORS["prior_week"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=levels["prior_month_high"],
        label=f"PMH {format_price(levels['prior_month_high'])}",
        color=COLORS["prior_month"],
        row=row_map["price"],
        dash="dot",
    )
    add_hline(
        fig,
        y=levels["prior_month_low"],
        label=f"PML {format_price(levels['prior_month_low'])}",
        color=COLORS["prior_month"],
        row=row_map["price"],
        dash="dot",
    )

if show_elliott:
    close_for_pivots = df_display["Close"].dropna()
    if len(close_for_pivots) >= (pivot_window * 2 + 5):
        pivots_raw = compute_pivots(close_for_pivots, pivot_window)
        pivots = compress_pivots(pivots_raw, min_swing_pct, max_pivots)
        impulse, score = find_best_impulse(pivots, elliott_lookback_pivots)
        abc = None
        if elliott_mode == "Impulse + ABC (best fit)" and impulse:
            abc = try_abc_after_impulse(pivots, impulse)

        add_elliott_overlay(fig, pivots, impulse, abc, elliott_mode)

        with st.sidebar.expander("Elliott diagnostics", expanded=False):
            st.write("Experimental heuristic overlay.")
            st.write(f"Pivot count drawn: {len(pivots)}")
            st.write(
                f"Best impulse score: {score:.2f}" if impulse else "No valid impulse found in lookback."
            )
            if impulse:
                st.write(f"Impulse pivot pattern: {''.join(p['type'] for p in impulse)}")
            if abc:
                st.write("ABC found after impulse.")
            elif elliott_mode == "Impulse + ABC (best fit)" and impulse:
                st.write("ABC not found with current settings.")
    else:
        st.sidebar.warning("Not enough visible bars for the Elliott overlay with this pivot sensitivity.")

# =========================================================
# VOLUME PANEL
# =========================================================
if show_volume:
    vol_colors = []
    closes = df_display["Close"].values
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
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            showlegend=False,
            name="Volume",
        ),
        row=row_map["volume"],
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["VOL20"],
            mode="lines",
            line=dict(width=1.4, color=COLORS["volume_ma"]),
            hovertemplate="Vol MA20: %{y:,.0f}<extra></extra>",
            showlegend=False,
            name="Vol MA20",
        ),
        row=row_map["volume"],
        col=1,
    )

# =========================================================
# RSI PANEL
# =========================================================
if show_rsi:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["RSI14"],
            mode="lines",
            line=dict(width=1.6, color=COLORS["rsi"]),
            hovertemplate="RSI 14: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="RSI 14",
        ),
        row=row_map["rsi"],
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(120,120,120,0.55)", row=row_map["rsi"], col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(120,120,120,0.30)", row=row_map["rsi"], col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(120,120,120,0.55)", row=row_map["rsi"], col=1)
    fig.update_yaxes(range=[0, 100], nticks=7, row=row_map["rsi"], col=1)

# =========================================================
# MACD PANEL
# =========================================================
if show_macd:
    hist_colors = [
        COLORS["hist_up"] if h >= 0 else COLORS["hist_down"]
        for h in df_display["Hist"].fillna(0.0)
    ]
    fig.add_trace(
        go.Bar(
            x=df_display.index,
            y=df_display["Hist"],
            marker_color=hist_colors,
            hovertemplate="Hist: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="Hist",
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
            hovertemplate="MACD: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="MACD",
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
            hovertemplate="Signal: %{y:.2f}<extra></extra>",
            showlegend=False,
            name="Signal",
        ),
        row=row_map["macd"],
        col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(120,120,120,0.25)", row=row_map["macd"], col=1)

# =========================================================
# RELATIVE STRENGTH PANEL
# =========================================================
if show_rel_strength and not bench_rel.empty:
    fig.add_trace(
        go.Scatter(
            x=bench_rel.index,
            y=bench_rel.values,
            mode="lines",
            line=dict(width=1.8, color=COLORS["rel"]),
            hovertemplate=f"RS vs {benchmark}: " + "%{y:.2f}<extra></extra>",
            showlegend=False,
            name=f"RS vs {benchmark}",
        ),
        row=row_map["rel"],
        col=1,
    )
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(120,120,120,0.35)", row=row_map["rel"], col=1)

# =========================================================
# AXES / LAYOUT
# =========================================================
for r in range(1, row_count + 1):
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        fixedrange=False,
        automargin=True,
        row=r,
        col=1,
    )

fig.update_yaxes(nticks=10, row=row_map["price"], col=1)

if show_volume:
    fig.update_yaxes(nticks=volume_nticks(df_display["Volume"]), row=row_map["volume"], col=1)

if show_macd:
    fig.update_yaxes(nticks=7, row=row_map["macd"], col=1)

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
    margin=dict(l=40, r=18, t=18, b=10),
    font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=1.01,
        x=0.0,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.84)",
        borderwidth=0,
        font=dict(size=11),
        traceorder="normal",
        itemclick=False,
        itemdoubleclick=False,
    ),
    bargap=0.08,
)

# Clean legend entries and create intentional legend
for trace in fig.data:
    trace.showlegend = False

legend_specs = []
if show_ma8:
    legend_specs.append(("MA 8", COLORS["ma8"]))
if show_ma20:
    legend_specs.append(("MA 20", COLORS["ma20"]))
if show_ma50:
    legend_specs.append(("MA 50", COLORS["ma50"]))
if show_ma100:
    legend_specs.append(("MA 100", COLORS["ma100"]))
if show_ma200:
    legend_specs.append(("MA 200", COLORS["ma200"]))
if show_anchored_vwap:
    legend_specs.append(("Anchored VWAP", COLORS["avwap"]))
if show_rel_strength and not bench_rel.empty:
    legend_specs.append((f"RS vs {benchmark}", COLORS["rel"]))

for label, color in legend_specs:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=color, width=3),
            name=label,
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

fig.update_yaxes(title_text="Price", row=row_map["price"], col=1)
if show_volume:
    fig.update_yaxes(title_text="Vol", row=row_map["volume"], col=1)
if show_rsi:
    fig.update_yaxes(title_text="RSI", row=row_map["rsi"], col=1)
if show_macd:
    fig.update_yaxes(title_text="MACD", row=row_map["macd"], col=1)
if show_rel_strength and not bench_rel.empty:
    fig.update_yaxes(title_text="RS", row=row_map["rel"], col=1)

# =========================================================
# RENDER
# =========================================================
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# =========================================================
# FOOTER DIAGNOSTICS
# =========================================================
with st.expander("Diagnostics and notes", expanded=False):
    st.markdown(
        f"""
- **Ticker:** {ticker}
- **Preset:** {preset}
- **Period / Interval:** {period} / {interval}
- **Last close:** {format_price(last_close)}
- **20-bar breakout / breakdown:** {format_price(float(df_display["HH20"].iloc[-2]) if len(df_display) >= 22 else np.nan)} / {format_price(float(df_display["LL20"].iloc[-2]) if len(df_display) >= 22 else np.nan)}
- **Relative benchmark:** {benchmark if benchmark else "n/a"}
- **Data source:** Yahoo Finance via `yfinance`
- **Experimental note:** Elliott overlay is heuristic only and should be treated as secondary to price, trend, levels, volume, and relative strength.
"""
    )

st.markdown(
    """
    <div style="margin-top: 2px; color: #7a7a7a; font-size: 13px;">
        © 2026 AD Fund Management LP
    </div>
    """,
    unsafe_allow_html=True,
)
