import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
    Quickly inspect price action and key technical indicators for any stock, ETF, or index.

    • Interactive OHLC candlesticks using Yahoo Finance data
    • 8 / 20 / 50 / 100 / 200 day moving averages with full history continuity
    • Volume bars color coded by up / down days
    • RSI (14 day) with Wilder style smoothing
    • MACD (12, 26, 9) with histogram for momentum inflections
    • Optional Bollinger Bands (20, 2.0) to frame short term ranges
    • Optional Elliott-style wave overlay (heuristic) built from swing pivots

    Use the controls below to adjust the time window, sampling interval, and overlays.
    """,
    unsafe_allow_html=True,
)

ticker = st.sidebar.text_input("Ticker", "^SPX").upper()
period = st.sidebar.selectbox(
    "Period", ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"], index=3
)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

show_bbands = st.sidebar.checkbox("Show Bollinger Bands (20, 2.0)", value=True)

# Elliott overlay controls
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

# ── Helpers ────────────────────────────────────────────────────────────────
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

@st.cache_data(ttl=3600)
def get_full_history(tkr: str, intrvl: str) -> pd.DataFrame:
    try:
        return yf.Ticker(tkr).history(period="max", interval=intrvl, auto_adjust=False)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

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
    """
    Returns a list of pivot dicts: {"i": int, "ts": timestamp, "px": float, "type": "H"|"L"}.
    Pivot is a fractal-style local extremum with w bars on both sides.
    """
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

def compress_pivots(
    pivots: list[dict],
    min_pct: float,
    max_points: int,
) -> list[dict]:
    """
    1) Merge consecutive same-type pivots, keep the most extreme.
    2) Enforce a minimum percent move between pivots.
    3) Cap the number of pivots to the most recent max_points.
    """
    if not pivots:
        return []

    # Step 1: merge consecutive same-type, keep extreme
    merged: list[dict] = [pivots[0].copy()]
    for p in pivots[1:]:
        last = merged[-1]
        if p["type"] == last["type"]:
            if p["type"] == "H":
                if p["px"] >= last["px"]:
                    merged[-1] = p.copy()
            else:
                if p["px"] <= last["px"]:
                    merged[-1] = p.copy()
        else:
            merged.append(p.copy())

    if len(merged) <= 2:
        return merged[-max_points:]

    # Step 2: min swing filter
    filtered: list[dict] = [merged[0]]
    min_frac = min_pct / 100.0
    for p in merged[1:]:
        last = filtered[-1]
        move = abs(p["px"] - last["px"])
        denom = max(abs(last["px"]), 1e-9)
        if (move / denom) >= min_frac:
            filtered.append(p)
        else:
            # If we skip due to threshold, we still want the most extreme continuation in same direction
            # Replace last pivot if p extends it (helps avoid "stair-step noise")
            if p["type"] == last["type"]:
                if p["type"] == "H" and p["px"] > last["px"]:
                    filtered[-1] = p
                if p["type"] == "L" and p["px"] < last["px"]:
                    filtered[-1] = p

    # Step 3: cap
    return filtered[-max_points:]

def _impulse_score(seq: list[dict]) -> float:
    """
    Score a 6-pivot impulse candidate (0..5). Higher is better.
    Works for both up and down depending on pivot types.
    Relaxed Elliott constraints, aiming for a usable map.
    """
    if len(seq) != 6:
        return -1e9

    # Determine direction from types
    types = "".join(p["type"] for p in seq)
    is_up = types in ("LHLHLH",)
    is_down = types in ("HLHLHL",)
    if not (is_up or is_down):
        return -1e9

    px = np.array([p["px"] for p in seq], dtype=float)

    # wave lengths (absolute)
    w1 = abs(px[1] - px[0])
    w3 = abs(px[3] - px[2])
    w5 = abs(px[5] - px[4])
    if min(w1, w3, w5) <= 0:
        return -1e9

    # Retrace ratios
    # For up: 2 retraces 1, 4 retraces 3.
    # For down: same math on absolute lengths.
    retr2 = abs(px[2] - px[1]) / max(w1, 1e-9)
    retr4 = abs(px[4] - px[3]) / max(w3, 1e-9)

    # Basic rules / preferences
    score = 0.0

    # Prefer wave 3 to be strong
    score += 2.0 if w3 >= w1 else -1.0
    score += 1.5 if w3 >= w5 else 0.0

    # Prefer wave 3 not to be the shortest (classic rule)
    score += 2.0 if (w3 >= min(w1, w5)) else -3.0

    # Prefer reasonable retracements
    def band_bonus(x: float, lo: float, hi: float, w: float = 1.0) -> float:
        if lo <= x <= hi:
            return 2.0 * w
        if (lo - 0.15) <= x <= (hi + 0.15):
            return 1.0 * w
        return -1.0 * w

    score += band_bonus(retr2, 0.35, 0.80, 1.0)
    score += band_bonus(retr4, 0.20, 0.60, 0.8)

    # Prefer progression (impulse makes new extreme at 3 and 5)
    if is_up:
        score += 2.0 if (px[3] > px[1]) else -2.0
        score += 2.0 if (px[5] > px[3]) else -1.0
        # Overlap check (relaxed)
        score += 0.5 if (px[4] > px[1]) else -0.5
    else:
        score += 2.0 if (px[3] < px[1]) else -2.0
        score += 2.0 if (px[5] < px[3]) else -1.0
        score += 0.5 if (px[4] < px[1]) else -0.5

    # Prefer later pivots (recent structure)
    score += seq[-1]["i"] / 100000.0

    return score

def find_best_impulse(pivots: list[dict], lookback: int) -> tuple[list[dict] | None, float]:
    """
    Search over the most recent 'lookback' pivots for best 6-pivot impulse candidate.
    """
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
    """
    Given impulse seq (6 pivots), attempt to take the next 4 alternating pivots as A-B-C.
    Returns [A,B,C] pivots if plausible, else None.
    """
    if impulse is None or len(impulse) != 6:
        return None

    # find where impulse ends in pivots list
    end_i = impulse[-1]["i"]
    # find the pivot with same index
    pos = None
    for k, p in enumerate(pivots):
        if p["i"] == end_i and p["type"] == impulse[-1]["type"]:
            pos = k
    if pos is None:
        return None

    tail = pivots[pos + 1 :]
    if len(tail) < 3:
        return None

    # Build A,B,C using next alternating pivots
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

    # Plausibility: A should move away from 5, B should retrace A, C should extend A in same direction as A.
    p5 = impulse[-1]["px"]
    a, b, c = abc[0]["px"], abc[1]["px"], abc[2]["px"]
    dir_a = np.sign(a - p5)
    if dir_a == 0:
        return None
    # B should pull back against A
    if np.sign(b - a) == dir_a:
        return None
    # C should go in direction of A from B
    if np.sign(c - b) != dir_a:
        return None

    return abc

def add_elliott_overlay(
    fig: go.Figure,
    df_disp: pd.DataFrame,
    pivots: list[dict],
    impulse: list[dict] | None,
    abc: list[dict] | None,
    mode: str,
):
    """
    Draw ZigZag line and optional labels on row=1 price panel.
    """
    if not pivots:
        return

    # Map timestamps to DateStr for plotly
    ts_to_datestr = {ts: ds for ts, ds in zip(df_disp.index, df_disp["DateStr"])}

    # ZigZag line
    x_line = []
    y_line = []
    for p in pivots:
        ds = ts_to_datestr.get(p["ts"])
        if ds is None:
            continue
        x_line.append(ds)
        y_line.append(p["px"])

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(width=2, color="black"),
            name="ZigZag",
            opacity=0.65,
        ),
        row=1,
        col=1,
    )

    if mode == "ZigZag only (no labels)":
        return

    def _add_label(p: dict, text: str, symbol: str = "circle", size: int = 10):
        ds = ts_to_datestr.get(p["ts"])
        if ds is None:
            return
        fig.add_trace(
            go.Scatter(
                x=[ds],
                y=[p["px"]],
                mode="markers+text",
                marker=dict(size=size, symbol=symbol, color="black"),
                text=[text],
                textposition="top center",
                name=text,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if impulse:
        labels = ["0", "1", "2", "3", "4", "5"]
        for p, lab in zip(impulse, labels):
            _add_label(p, lab, symbol="circle", size=12)

    if mode == "Impulse only (best fit)":
        return

    if abc:
        for p, lab in zip(abc, ["A", "B", "C"]):
            _add_label(p, lab, symbol="square", size=11)

# ── Fetch full history ────────────────────────────────────────────────────
df_full = get_full_history(ticker, interval)
if df_full.empty:
    st.error("No data returned. Check symbol or internet.")
    st.stop()

if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)
if interval == "1d":
    df_full = df_full[df_full.index.weekday < 5]

CAP_MAX_ROWS = 200000
if len(df_full) > CAP_MAX_ROWS:
    df_full = df_full.tail(CAP_MAX_ROWS)

# ── Date-based windowing with warmup ──────────────────────────────────────
start_dt = start_date_from_period(period)
BUFFER_YEARS = 50
if start_dt is not None:
    warmup_start = start_dt - pd.DateOffset(years=BUFFER_YEARS)
    df_ind = df_full[df_full.index >= warmup_start].copy()
else:
    df_ind = df_full.copy()

# ── Indicators ───────────────────────────────────────────────────────────
for w in (8, 20, 50, 100, 200):
    df_ind[f"MA{w}"] = df_ind["Close"].rolling(w).mean()

delta = df_ind["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
rs = avg_gain / avg_loss
df_ind["RSI14"] = 100 - (100 / (1 + rs))

df_ind["EMA12"] = df_ind["Close"].ewm(span=12, adjust=False).mean()
df_ind["EMA26"] = df_ind["Close"].ewm(span=26, adjust=False).mean()
df_ind["MACD"] = df_ind["EMA12"] - df_ind["EMA26"]
df_ind["Signal"] = df_ind["MACD"].ewm(span=9, adjust=False).mean()
df_ind["Hist"] = df_ind["MACD"] - df_ind["Signal"]

bb_window, bb_mult = 20, 2.0
df_ind["BB_MA"] = df_ind["Close"].rolling(bb_window).mean()
df_ind["BB_STD"] = df_ind["Close"].rolling(bb_window).std()
df_ind["BB_UPPER"] = df_ind["BB_MA"] + bb_mult * df_ind["BB_STD"]
df_ind["BB_LOWER"] = df_ind["BB_MA"] - bb_mult * df_ind["BB_STD"]

# ── Final slice ──────────────────────────────────────────────────────────
if start_dt is not None:
    df_display = df_ind[df_ind.index >= start_dt].copy()
    if df_display.empty:
        df_display = df_ind.tail(300).copy()
else:
    df_display = df_ind.copy()

# ── Plotly Interactive Plot ──────────────────────────────────────────────
df_display["DateStr"] = df_display.index.strftime("%Y-%m-%d")
available_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) >= w]
missing_mas = [w for w in (8, 20, 50, 100, 200) if len(df_display) < w]
if missing_mas:
    st.warning(
        "Some MAs may appear choppy due to limited visible data: "
        + ", ".join(f"MA{w}" for w in missing_mas)
    )

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.60, 0.14, 0.13, 0.13],
    vertical_spacing=0.04,
    specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}]],
)

# Candlesticks
fig.add_trace(
    go.Candlestick(
        x=df_display["DateStr"],
        open=df_display["Open"],
        high=df_display["High"],
        low=df_display["Low"],
        close=df_display["Close"],
        increasing_line_color="#43A047",
        decreasing_line_color="#E53935",
        name="Price",
    ),
    row=1,
    col=1,
)

# Bollinger Bands
if show_bbands:
    fig.add_trace(
        go.Scatter(
            x=df_display["DateStr"],
            y=df_display["BB_UPPER"],
            mode="lines",
            line=dict(width=1, color="#cfc61f"),
            name="BB Upper",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display["DateStr"],
            y=df_display["BB_LOWER"],
            mode="lines",
            line=dict(width=1, color="#cfc61f"),
            fill="tonexty",
            fillcolor="rgba(250,245,147,0.15)",
            name="BB Lower",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_display["DateStr"],
            y=df_display["BB_MA"],
            mode="lines",
            line=dict(width=1, color="#faf593", dash="dot"),
            name="BB Mid",
        ),
        row=1,
        col=1,
    )

# Moving Averages
for w, color in zip(available_mas, ("#7CB342", "#8E44AD", "#1565C0", "#F39C12", "#607D8B")):
    fig.add_trace(
        go.Scatter(
            x=df_display["DateStr"],
            y=df_display[f"MA{w}"],
            mode="lines",
            line=dict(color=color, width=1.3),
            name=f"MA{w}",
        ),
        row=1,
        col=1,
    )
fig.update_yaxes(title_text="Price", row=1, col=1)

# Elliott overlay
if show_elliott:
    close_for_pivots = df_display["Close"].dropna()
    if len(close_for_pivots) >= (pivot_window * 2 + 5):
        pivots_raw = compute_pivots(close_for_pivots, pivot_window)
        pivots = compress_pivots(pivots_raw, min_swing_pct, max_pivots)

        impulse, score = find_best_impulse(pivots, elliott_lookback_pivots)
        abc = None
        if elliott_mode == "Impulse + ABC (best fit)" and impulse:
            abc = try_abc_after_impulse(pivots, impulse)

        add_elliott_overlay(fig, df_display, pivots, impulse, abc, elliott_mode)

        with st.sidebar.expander("Elliott diagnostics", expanded=False):
            st.write(f"Pivot count drawn: {len(pivots)}")
            st.write(f"Best impulse score: {score:.2f}" if impulse else "No valid impulse found in lookback.")
            if impulse:
                types = "".join(p["type"] for p in impulse)
                st.write(f"Impulse pivot pattern: {types}")
            if abc:
                st.write("ABC found after impulse.")
            elif elliott_mode == "Impulse + ABC (best fit)" and impulse:
                st.write("ABC not found with the current pivot settings. Try lowering min swing % or pivot sensitivity.")
    else:
        st.sidebar.warning("Not enough data in the visible window for the Elliott overlay with this pivot sensitivity.")

# Volume
vol_colors = [
    "#B0BEC5"
    if i == 0
    else ("#43A047" if df_display["Close"].iat[i] > df_display["Close"].iat[i - 1] else "#E53935")
    for i in range(len(df_display))
]
fig.add_trace(
    go.Bar(
        x=df_display["DateStr"],
        y=df_display["Volume"],
        width=0.4,
        marker_color=vol_colors,
        opacity=0.7,
        name="Volume",
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="Volume", row=2, col=1)

# RSI
fig.add_trace(
    go.Scatter(
        x=df_display["DateStr"],
        y=df_display["RSI14"],
        mode="lines",
        line=dict(width=1.5, color="purple"),
        name="RSI (14)",
    ),
    row=3,
    col=1,
)
fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="gray", row=3, col=1)

# MACD
hist_colors = ["#43A047" if h > 0 else "#E53935" for h in df_display["Hist"]]
fig.add_trace(
    go.Bar(
        x=df_display["DateStr"],
        y=df_display["Hist"],
        marker_color=hist_colors,
        opacity=0.6,
        width=0.4,
        name="MACD Hist",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_display["DateStr"],
        y=df_display["MACD"],
        mode="lines",
        line=dict(width=1.5, color="blue"),
        name="MACD",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_display["DateStr"],
        y=df_display["Signal"],
        mode="lines",
        line=dict(width=1.2, color="orange"),
        name="Signal",
    ),
    row=4,
    col=1,
)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# Month ticks
month_starts = df_display.index.to_series().groupby(df_display.index.to_period("M")).first()
tickvals = month_starts.dt.strftime("%Y-%m-%d").tolist()
ticktext = month_starts.dt.strftime("%b-%y").tolist()
fig.update_xaxes(
    type="category",
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickangle=-45,
    showgrid=True,
    gridwidth=0.5,
    gridcolor="#e1e5ed",
)

fig.update_layout(
    height=950,
    title=dict(
        text=f"{ticker} - OHLC + RSI & MACD"
        + (" + Bollinger Bands" if show_bbands else "")
        + (" + Elliott Overlay" if show_elliott else ""),
        x=0.5,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(l=60, r=20, t=60, b=40),
    xaxis=dict(rangeslider_visible=False, showline=True, linewidth=1, linecolor="black"),
    legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    font=dict(family="Arial, sans-serif", size=12),
)

st.plotly_chart(fig, use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
