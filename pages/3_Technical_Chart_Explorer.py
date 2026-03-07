import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Technical Chart", layout="wide")

# --------------------------- Sidebar ---------------------------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Clean technical charting with a tighter, more TradingView-like layout.

    What it does
    • Candlesticks with 8 / 20 / 50 / 200 day moving averages
    • Volume, RSI (14), and MACD (12, 26, 9)
    • Optional Bollinger Bands (20, 2.0)
    • Optional Elliott-style pivot overlay
    • Session-aware x-axis gap removal for weekends and market-closed weekdays
    • Separate volume-by-price panel below the main chart
    """
)

ticker = st.sidebar.text_input("Ticker", "^SPX").upper().strip()

period = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"],
    index=3,
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=0,
)

auto_adjust = st.sidebar.checkbox("Use adjusted prices", value=False)

st.sidebar.subheader("Overlays")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=False)
show_last_price = st.sidebar.checkbox("Show last price line", value=True)
show_range_levels = st.sidebar.checkbox("Show visible-range high / low", value=True)
show_volume_profile = st.sidebar.checkbox("Show volume-by-price panel", value=True)

st.sidebar.subheader("Elliott Overlay (Heuristic)")
show_elliott = st.sidebar.checkbox("Show Elliott wave overlay", value=False)
pivot_window = st.sidebar.slider("Pivot sensitivity (bars left/right)", 2, 20, 8)
min_swing_pct = st.sidebar.slider("Min swing filter (%)", 0.0, 10.0, 2.0, 0.25)
max_pivots = st.sidebar.slider("Max pivots to draw", 20, 200, 80)
elliott_lookback_pivots = st.sidebar.slider("Lookback pivots for auto-count", 20, 200, 80)
elliott_mode = st.sidebar.selectbox(
    "Auto-count mode",
    ["Impulse + ABC (best fit)", "Impulse only (best fit)", "ZigZag only (no labels)"],
    index=0,
)

# --------------------------- Helpers ---------------------------
def start_date_from_period(p: str) -> pd.Timestamp | None:
    today = pd.Timestamp.today().normalize()
    if p == "max":
        return None
    if p.endswith("mo"):
        return today - pd.DateOffset(months=int(p[:-2]))
    if p.endswith("y"):
        return today - pd.DateOffset(years=int(p[:-1]))
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_full_history(tkr: str, intrvl: str, adjusted: bool) -> pd.DataFrame:
    try:
        df = yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=adjusted)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
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


def compute_missing_market_days(idx: pd.DatetimeIndex) -> list[str]:
    """
    For daily charts, remove weekdays where the market was closed
    by detecting missing weekdays in the visible range.
    """
    if len(idx) == 0:
        return []

    start = idx.min().normalize()
    end = idx.max().normalize()
    all_weekdays = pd.date_range(start=start, end=end, freq="B")
    actual_days = pd.DatetimeIndex(pd.Series(idx.normalize().unique()).sort_values())
    missing = all_weekdays.difference(actual_days)
    return [d.strftime("%Y-%m-%d") for d in missing]


def build_rangebreaks(idx: pd.DatetimeIndex, intrvl: str) -> list[dict]:
    if intrvl != "1d":
        return []

    missing_market_days = compute_missing_market_days(idx)
    breaks = [dict(bounds=["sat", "mon"])]

    if missing_market_days:
        breaks.append(dict(values=missing_market_days))

    return breaks


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


def compute_pivots(close: pd.Series, w: int) -> list[dict]:
    values = close.values.astype(float)
    idx = close.index
    pivots: list[dict] = []
    for i in range(w, len(values) - w):
        if _is_pivot_high(values, i, w):
            pivots.append({"i": i, "ts": idx[i], "px": float(values[i]), "type": "H"})
        elif _is_pivot_low(values, i, w):
            pivots.append({"i": i, "ts": idx[i], "px": float(values[i]), "type": "L"})
    pivots.sort(key=lambda x: x["i"])
    return pivots


def compress_pivots(pivots: list[dict], min_pct: float, max_points: int) -> list[dict]:
    if not pivots:
        return []

    merged: list[dict] = [pivots[0].copy()]
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

    filtered: list[dict] = [merged[0]]
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


def _impulse_score(seq: list[dict]) -> float:
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


def find_best_impulse(pivots: list[dict], lookback: int) -> tuple[list[dict] | None, float]:
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


def try_abc_after_impulse(pivots: list[dict], impulse: list[dict]) -> list[dict] | None:
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

    abc: list[dict] = []
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
    pivots: list[dict],
    impulse: list[dict] | None,
    abc: list[dict] | None,
    mode: str,
):
    if not pivots:
        return

    x_line = [p["ts"] for p in pivots]
    y_line = [p["px"] for p in pivots]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(width=1.5, color="rgba(35,35,35,0.65)"),
            name="ZigZag",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    if mode == "ZigZag only (no labels)":
        return

    def _add_label(p: dict, text: str, symbol: str = "circle", size: int = 9):
        fig.add_trace(
            go.Scatter(
                x=[p["ts"]],
                y=[p["px"]],
                mode="markers+text",
                marker=dict(size=size, symbol=symbol, color="rgba(20,20,20,0.92)"),
                text=[text],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"{text}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if impulse:
        for p, lab in zip(impulse, ["0", "1", "2", "3", "4", "5"]):
            _add_label(p, lab, symbol="circle", size=10)

    if mode == "Impulse only (best fit)":
        return

    if abc:
        for p, lab in zip(abc, ["A", "B", "C"]):
            _add_label(p, lab, symbol="square", size=9)


def add_horizontal_price_line(fig: go.Figure, y: float, label: str, color: str, row: int = 1):
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


def build_volume_profile(df: pd.DataFrame, bins: int | None = None) -> pd.DataFrame:
    """
    Simple visible-range volume-by-price model.
    Volume is allocated into dynamic close-based bins.
    """
    if df.empty:
        return pd.DataFrame(columns=["price_mid", "volume"])

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())

    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return pd.DataFrame(columns=["price_mid", "volume"])

    n = len(df)
    if bins is None:
        if n <= 60:
            bins = 18
        elif n <= 130:
            bins = 24
        elif n <= 260:
            bins = 30
        else:
            bins = 36

    edges = np.linspace(price_min, price_max, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    close_vals = df["Close"].values.astype(float)
    vol_vals = df["Volume"].fillna(0).values.astype(float)

    bucket = np.clip(np.digitize(close_vals, edges) - 1, 0, bins - 1)
    vol_profile = np.zeros(bins, dtype=float)
    np.add.at(vol_profile, bucket, vol_vals)

    out = pd.DataFrame({"price_mid": mids, "volume": vol_profile})
    out = out[out["volume"] > 0].copy()
    return out


# --------------------------- Data Fetch ---------------------------
df_full = get_full_history(ticker, interval, auto_adjust)

if df_full.empty:
    st.error("No data returned. Check the symbol or data source.")
    st.stop()

required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(df_full.columns)):
    st.error("Returned data is missing one or more required OHLCV columns.")
    st.stop()

df_full = df_full.copy()

if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

df_full = df_full.sort_index()
df_full = df_full[~df_full.index.duplicated(keep="last")]

CAP_MAX_ROWS = 200000
if len(df_full) > CAP_MAX_ROWS:
    df_full = df_full.tail(CAP_MAX_ROWS)

# --------------------------- Windowing ---------------------------
start_dt = start_date_from_period(period)
buffer_years = 10

if start_dt is not None:
    warmup_start = start_dt - pd.DateOffset(years=buffer_years)
    df_ind = df_full[df_full.index >= warmup_start].copy()
else:
    df_ind = df_full.copy()

# --------------------------- Indicators ---------------------------
for w in (8, 20, 50, 200):
    df_ind[f"MA{w}"] = df_ind["Close"].rolling(w).mean()

df_ind["RSI14"] = compute_rsi(df_ind["Close"], 14)

macd, signal, hist = compute_macd(df_ind["Close"], 12, 26, 9)
df_ind["MACD"] = macd
df_ind["Signal"] = signal
df_ind["Hist"] = hist

bb_ma, bb_upper, bb_lower = compute_bbands(df_ind["Close"], 20, 2.0)
df_ind["BB_MA"] = bb_ma
df_ind["BB_UPPER"] = bb_upper
df_ind["BB_LOWER"] = bb_lower

# --------------------------- Display Slice ---------------------------
if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()
    if df_display.empty:
        df_display = df_ind.tail(300).copy()
else:
    df_display = df_ind.copy()

if df_display.empty:
    st.error("No display data available after filtering.")
    st.stop()

rangebreaks = build_rangebreaks(df_display.index, interval)

# --------------------------- Styling ---------------------------
COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "ma8": "#00ACC1",
    "ma20": "#2962FF",
    "ma50": "#7E57C2",
    "ma200": "#FB8C00",
    "bb": "#9E9E9E",
    "grid": "rgba(120,120,120,0.11)",
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
    "vpoc": "#C62828",
    "vpbar": "rgba(41,98,255,0.75)",
}

# --------------------------- Main Figure ---------------------------
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.025,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]],
)

# Price
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
    row=1,
    col=1,
)

# Bollinger Bands
if show_bbands:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_UPPER"],
            mode="lines",
            line=dict(width=1, color=COLORS["bb"]),
            name="BB Upper",
            hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_LOWER"],
            mode="lines",
            line=dict(width=1, color=COLORS["bb"]),
            fill="tonexty",
            fillcolor="rgba(158,158,158,0.10)",
            name="BB Lower",
            hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display["BB_MA"],
            mode="lines",
            line=dict(width=1, color="rgba(100,100,100,0.7)", dash="dot"),
            name="BB Mid",
            hovertemplate="BB Mid: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

# Moving averages
for w, color in [(8, COLORS["ma8"]), (20, COLORS["ma20"]), (50, COLORS["ma50"]), (200, COLORS["ma200"])]:
    fig.add_trace(
        go.Scatter(
            x=df_display.index,
            y=df_display[f"MA{w}"],
            mode="lines",
            line=dict(color=color, width=1.5),
            name=f"MA {w}",
            hovertemplate=f"MA {w}: " + "%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

# Elliott
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
            st.write(f"Pivot count drawn: {len(pivots)}")
            st.write(
                f"Best impulse score: {score:.2f}"
                if impulse
                else "No valid impulse found in lookback."
            )
            if impulse:
                st.write(f"Impulse pivot pattern: {''.join(p['type'] for p in impulse)}")
            if abc:
                st.write("ABC found after impulse.")
            elif elliott_mode == "Impulse + ABC (best fit)" and impulse:
                st.write("ABC not found with current settings.")
    else:
        st.sidebar.warning("Not enough visible bars for the Elliott overlay with this pivot sensitivity.")

# Price reference lines
last_close = float(df_display["Close"].iloc[-1])
prev_close = float(df_display["Close"].iloc[-2]) if len(df_display) >= 2 else last_close
chg_pct = ((last_close / prev_close) - 1.0) * 100 if prev_close != 0 else 0.0

if show_last_price:
    add_horizontal_price_line(
        fig,
        y=last_close,
        label=f"Last {last_close:,.2f} ({chg_pct:+.2f}%)",
        color=COLORS["last"],
        row=1,
    )

if show_range_levels:
    visible_high = float(df_display["High"].max())
    visible_low = float(df_display["Low"].min())

    fig.add_hline(
        y=visible_high,
        line_width=1,
        line_dash="dot",
        line_color=COLORS["range"],
        row=1,
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
        row=1,
        col=1,
        annotation_text=f"Range Low {visible_low:,.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color=COLORS["range"], size=10),
    )

# Volume
vol_colors = []
closes = df_display["Close"].values
for i in range(len(df_display)):
    if i == 0:
        vol_colors.append("rgba(160,160,160,0.45)")
    else:
        vol_colors.append(COLORS["volume_up"] if closes[i] >= closes[i - 1] else COLORS["volume_down"])

fig.add_trace(
    go.Bar(
        x=df_display.index,
        y=df_display["Volume"],
        marker_color=vol_colors,
        name="Volume",
        hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ),
    row=2,
    col=1,
)

# RSI
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
    row=3,
    col=1,
)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(120,120,120,0.55)", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(120,120,120,0.55)", row=3, col=1)
fig.update_yaxes(range=[0, 100], row=3, col=1)

# MACD
hist_colors = [COLORS["hist_up"] if h >= 0 else COLORS["hist_down"] for h in df_display["Hist"].fillna(0.0)]
fig.add_trace(
    go.Bar(
        x=df_display.index,
        y=df_display["Hist"],
        marker_color=hist_colors,
        name="MACD Hist",
        hovertemplate="Hist: %{y:.2f}<extra></extra>",
        showlegend=False,
    ),
    row=4,
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
    row=4,
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
    row=4,
    col=1,
)
fig.add_hline(y=0, line_dash="solid", line_color="rgba(120,120,120,0.25)", row=4, col=1)

# Axes
for r in range(1, 5):
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=False,
        fixedrange=False,
        row=r,
        col=1,
    )

fig.update_xaxes(
    type="date",
    showgrid=True,
    gridcolor=COLORS["grid"],
    showline=False,
    rangeslider_visible=False,
    rangebreaks=rangebreaks,
    tickformat="%b '%y" if interval in ["1wk", "1mo"] else "%b %d\n%Y",
)

fig.update_layout(
    height=1050,
    title=dict(
        text=f"{ticker} | Technical Chart",
        x=0.01,
        xanchor="left",
        y=0.985,
        font=dict(size=22, color=COLORS["text"]),
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=95, b=25),
    font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
    legend=dict(
        orientation="h",
        y=1.055,
        x=0.01,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.82)",
        borderwidth=0,
        font=dict(size=11),
        traceorder="normal",
        itemclick="toggleothers",
        itemdoubleclick="toggle",
    ),
    bargap=0.05,
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Vol", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# --------------------------- Volume-by-Price ---------------------------
if show_volume_profile:
    vp = build_volume_profile(df_display)

    if not vp.empty:
        vpoc_idx = vp["volume"].idxmax()
        vpoc_price = float(vp.loc[vpoc_idx, "price_mid"])
        vpoc_volume = float(vp.loc[vpoc_idx, "volume"])

        vp_fig = go.Figure()

        vp_fig.add_trace(
            go.Bar(
                x=vp["price_mid"],
                y=vp["volume"],
                marker_color=COLORS["vpbar"],
                name="Volume by Price",
                hovertemplate="Price: %{x:,.2f}<br>Volume: %{y:,.0f}<extra></extra>",
                showlegend=False,
            )
        )

        vp_fig.add_vline(
            x=vpoc_price,
            line_width=1.3,
            line_dash="dot",
            line_color=COLORS["vpoc"],
            annotation_text=f"VPOC {vpoc_price:,.2f}",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["vpoc"], size=11),
        )

        vp_fig.update_xaxes(
            title_text="Price Level",
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            showline=False,
        )
        vp_fig.update_yaxes(
            title_text="Volume",
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            showline=False,
        )

        vp_fig.update_layout(
            height=280,
            title=dict(
                text="Volume by Price",
                x=0.01,
                xanchor="left",
                font=dict(size=18, color=COLORS["text"]),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=40, r=20, t=45, b=30),
            font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
            bargap=0.08,
        )

        st.plotly_chart(vp_fig, use_container_width=True, config={"displaylogo": False})

st.caption("© 2026 AD Fund Management LP")
