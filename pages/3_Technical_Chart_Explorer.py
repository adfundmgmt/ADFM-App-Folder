import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ADFM Chart Tool", layout="wide")

# --------------------------- Sidebar ---------------------------
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    A technical chart built in Streamlit with Plotly.

    What it does
    • Real datetime x-axis for better spacing, zooming, and hover behavior  
    • Candlesticks with cleaner visual hierarchy  
    • 8 / 20 / 50 / 100 / 200 day moving averages  
    • Volume bars with muted up / down coloring  
    • Optional RSI (14) and MACD (12, 26, 9)  
    • Optional Bollinger Bands (20, 2.0)  
    • Optional Elliott-style pivot overlay using a heuristic swing model  

    Use the controls below to change ticker, period, interval, and overlays.
    """,
    unsafe_allow_html=True,
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
show_ma8 = st.sidebar.checkbox("Show MA 8", value=True)
show_ma100 = st.sidebar.checkbox("Show MA 100", value=True)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=False)
show_last_price = st.sidebar.checkbox("Show last price line", value=False)
show_range_levels = st.sidebar.checkbox("Show visible-range high / low", value=False)

st.sidebar.subheader("Indicators")
show_volume = st.sidebar.checkbox("Show Volume", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)
show_macd = st.sidebar.checkbox("Show MACD (12, 26, 9)", value=True)

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
        months = int(p[:-2])
        return today - pd.DateOffset(months=months)
    if p.endswith("y"):
        years = int(p[:-1])
        return today - pd.DateOffset(years=years)
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
            line=dict(width=1.6, color="rgba(30,30,30,0.65)"),
            name="ZigZag",
            hoverinfo="skip",
            showlegend=True,
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
                marker=dict(size=size, symbol=symbol, color="rgba(20,20,20,0.9)"),
                text=[text],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"{text}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )


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


def volume_nticks(series: pd.Series) -> int:
    vmax = float(series.max()) if len(series) else 0.0
    if vmax >= 1e10:
        return 7
    if vmax >= 1e9:
        return 6
    return 5


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
for w in (8, 20, 50, 100, 200):
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

# --------------------------- Header ---------------------------
header_suffix = "Adjusted ADFM Technical Chart" if auto_adjust else "ADFM Technical Chart"
st.markdown(
    f"""
    <div style="margin-bottom: 10px;">
        <h2 style="margin: 0; padding: 0; font-size: 32px; font-weight: 700; color: #222222;">
            {ticker} | {header_suffix}
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Panel Setup ---------------------------
panel_flags = {
    "volume": show_volume,
    "rsi": show_rsi,
    "macd": show_macd,
}
active_panels = [k for k, v in panel_flags.items() if v]

row_count = 1 + len(active_panels)

if row_count == 1:
    row_heights = [1.0]
elif row_count == 2:
    row_heights = [0.78, 0.22]
elif row_count == 3:
    row_heights = [0.68, 0.14, 0.18]
else:
    row_heights = [0.60, 0.14, 0.13, 0.13]

height_map = {1: 650, 2: 780, 3: 930, 4: 1080}
fig_height = height_map.get(row_count, 1080)

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
    vertical_spacing=0.02,
    row_heights=row_heights,
    specs=specs,
)

row_map = {"price": 1}
current_row = 2
for panel in active_panels:
    row_map[panel] = current_row
    current_row += 1

# --------------------------- Styling ---------------------------
COLORS = {
    "up": "#26A69A",
    "down": "#EF5350",
    "ma20": "#2962FF",
    "ma50": "#7E57C2",
    "ma200": "#FF9800",
    "ma8": "#00ACC1",
    "ma100": "#8D6E63",
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
}

# --------------------------- Price Panel ---------------------------
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

ma_config = [
    (8, COLORS["ma8"], show_ma8),
    (20, COLORS["ma20"], True),
    (50, COLORS["ma50"], True),
    (100, COLORS["ma100"], show_ma100),
    (200, COLORS["ma200"], True),
]

for w, color, enabled in ma_config:
    if enabled and f"MA{w}" in df_display.columns:
        fig.add_trace(
            go.Scatter(
                x=df_display.index,
                y=df_display[f"MA{w}"],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=f"MA {w}",
                hovertemplate=f"MA {w}: " + "%{y:.2f}<extra></extra>",
                showlegend=True,
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
            line=dict(width=1, color=COLORS["bb"]),
            name="BB Upper",
            hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=row_map["price"],
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
            showlegend=True,
        ),
        row=row_map["price"],
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
            showlegend=True,
        ),
        row=row_map["price"],
        col=1,
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

        if impulse:
            for p, lab in zip(impulse, ["0", "1", "2", "3", "4", "5"]):
                fig.add_trace(
                    go.Scatter(
                        x=[p["ts"]],
                        y=[p["px"]],
                        mode="markers+text",
                        marker=dict(size=10, symbol="circle", color="rgba(20,20,20,0.9)"),
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
                        marker=dict(size=9, symbol="square", color="rgba(20,20,20,0.9)"),
                        text=[lab],
                        textposition="top center",
                        showlegend=False,
                        hovertemplate=f"{lab}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        with st.sidebar.expander("Elliott diagnostics", expanded=False):
            st.write(f"Pivot count drawn: {len(pivots)}")
            st.write(
                f"Best impulse score: {score:.2f}"
                if impulse
                else "No valid impulse found in lookback."
            )
            if impulse:
                types = "".join(p["type"] for p in impulse)
                st.write(f"Impulse pivot pattern: {types}")
            if abc:
                st.write("ABC found after impulse.")
            elif elliott_mode == "Impulse + ABC (best fit)" and impulse:
                st.write("ABC not found with current settings.")
    else:
        st.sidebar.warning("Not enough visible bars for the Elliott overlay with this pivot sensitivity.")

last_close = float(df_display["Close"].iloc[-1])
prev_close = float(df_display["Close"].iloc[-2]) if len(df_display) >= 2 else last_close
chg_pct = ((last_close / prev_close) - 1.0) * 100 if prev_close != 0 else 0.0

if show_last_price:
    add_horizontal_price_line(
        fig,
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

# --------------------------- Volume Panel ---------------------------
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
            name="Volume",
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=row_map["volume"],
        col=1,
    )

# --------------------------- RSI Panel ---------------------------
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

# --------------------------- MACD Panel ---------------------------
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

# --------------------------- Axes / Layout ---------------------------
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
    rangebreaks=[dict(bounds=["sat", "mon"])] if interval == "1d" else [],
    tickformat="%b '%y" if interval in ["1wk", "1mo"] else "%b %d\n%Y",
)

fig.update_layout(
    height=fig_height,
    title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=20, b=10),
    font=dict(family="Arial, sans-serif", size=12, color=COLORS["text"]),
    legend=dict(
        orientation="h",
        y=1.02,
        x=0.0,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.80)",
        borderwidth=0,
        font=dict(size=11),
    ),
    bargap=0.05,
)

fig.update_yaxes(title_text="Price", row=row_map["price"], col=1)
if show_volume:
    fig.update_yaxes(title_text="Vol", row=row_map["volume"], col=1)
if show_rsi:
    fig.update_yaxes(title_text="RSI", row=row_map["rsi"], col=1)
if show_macd:
    fig.update_yaxes(title_text="MACD", row=row_map["macd"], col=1)

# --------------------------- Render ---------------------------
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.markdown(
    """
    <div style="margin-top: 2px; color: #7a7a7a; font-size: 13px;">
        © 2026 AD Fund Management LP
    </div>
    """,
    unsafe_allow_html=True,
)
