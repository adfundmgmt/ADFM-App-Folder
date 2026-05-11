import html
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Rate of Change Regime Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


TIMEFRAME_MAP = {
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
    "Max": "max",
}

ROC_PERIODS = {
    "5D": 5,
    "10D": 10,
    "20D": 20,
    "63D": 63,
    "126D": 126,
}

SMA_WINDOWS = [21, 50, 100, 200]

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"

SMA_COLORS = {
    "SMA_21": "#4c78a8",
    "SMA_50": "#f58518",
    "SMA_100": "#9c6ade",
    "SMA_200": "#2f4858",
}


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3.0rem;
            padding-bottom: 3.0rem;
            max-width: 1500px;
        }

        section[data-testid="stSidebar"] {
            background-color: #f3f6fa;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #2d3748;
            font-weight: 750;
            letter-spacing: -0.01em;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li {
            color: #3f4a5a;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .tool-divider {
            margin-top: 1.25rem;
            margin-bottom: 1.25rem;
            border-top: 1px solid #d8dee8;
        }

        .hero-title {
            font-size: 2.65rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #2d3142;
            margin-bottom: 0.25rem;
        }

        .hero-caption {
            color: #8b949e;
            font-size: 1.0rem;
            margin-bottom: 1.8rem;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stToggle"] label,
        div[data-testid="stSlider"] label {
            color: #536171;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        div[data-baseweb="input"] {
            border-radius: 12px;
            background-color: #f5f7fb;
            border: 1px solid #e0e6ef;
        }

        div[data-baseweb="select"] > div {
            border-radius: 12px;
            background-color: #f5f7fb;
            border-color: #e0e6ef;
        }

        .chart-wrap {
            margin-top: 1.1rem;
        }

        .commentary-box {
            margin-top: 1.2rem;
            padding: 1.15rem 1.25rem;
            border: 1px solid #dbe3ef;
            border-radius: 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        .commentary-kicker {
            color: #6b7788;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .commentary-title {
            color: #2d3142;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.45rem;
        }

        .commentary-box p {
            color: #344054;
            font-size: 0.98rem;
            line-height: 1.6;
            margin-bottom: 0.65rem;
        }

        .commentary-box p:last-child {
            margin-bottom: 0;
        }

        .inline-stat {
            font-weight: 800;
            color: #202633;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(
    ticker: str,
    period: str,
    interval: str = "1d",
    retries: int = 3,
) -> pd.DataFrame:
    last_err = None

    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.rename(columns=str.title)

            keep_cols = [
                c
                for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                if c in data.columns
            ]
            data = data[keep_cols].dropna(how="all")

            if data.empty:
                raise ValueError(f"No data returned for {ticker}")

            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            return data

        except Exception as err:
            last_err = err
            time.sleep(1.2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {ticker} after {retries} retries: {last_err}")


def clean_number(value: float, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value) or np.isinf(value):
            return default
    except TypeError:
        return default
    return float(value)


def pct_text(value: float, decimals: int = 2, signed: bool = True) -> str:
    if value is None or pd.isna(value) or np.isinf(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}%}"


def ratio_text(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{decimals}f}x"


def compute_features(df: pd.DataFrame, roc_n: int, spy_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    px = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]
    px = px.astype(float)

    for win in SMA_WINDOWS:
        out[f"SMA_{win}"] = px.rolling(win, min_periods=1).mean()

    out["ROC"] = px.pct_change(roc_n)
    out["ROC_Slope"] = out["ROC"].diff(5)

    out["Second_Derivative"] = out["ROC"].diff()
    out["Second_Derivative_Slope"] = out["Second_Derivative"].diff(3)

    prev_sd = out["Second_Derivative"].shift(1)
    out["Pos_Inflect"] = (prev_sd <= 0) & (out["Second_Derivative"] > 0)
    out["Neg_Inflect"] = (prev_sd >= 0) & (out["Second_Derivative"] < 0)

    if {"High", "Low", "Close"}.issubset(out.columns):
        tr = pd.concat(
            [
                out["High"] - out["Low"],
                (out["High"] - out["Close"].shift(1)).abs(),
                (out["Low"] - out["Close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["ATR_14"] = tr.rolling(14, min_periods=1).mean()
        out["ATR_Pct"] = out["ATR_14"] / px.replace(0, np.nan)
    else:
        out["ATR_14"] = np.nan
        out["ATR_Pct"] = np.nan

    out["Vol_20"] = px.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252)
    out["Vol_60"] = px.pct_change().rolling(60, min_periods=20).std() * np.sqrt(252)
    out["Vol_Regime"] = np.where(out["Vol_20"] > out["Vol_60"], "Expansion", "Compression")

    if "Volume" in out.columns:
        vol_fast = out["Volume"].rolling(10, min_periods=3).mean()
        vol_slow = out["Volume"].rolling(50, min_periods=10).mean()
        out["Volume_Accel"] = vol_fast / vol_slow.replace(0, np.nan)
    else:
        out["Volume_Accel"] = np.nan

    spy_px = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
    spy_px = spy_px.astype(float).reindex(out.index).ffill().bfill()

    if spy_px.notna().sum() > 0:
        px_base = px / px.dropna().iloc[0]
        spy_base = spy_px / spy_px.dropna().iloc[0]
        out["Rel_Strength_SPY"] = px_base / spy_base
    else:
        out["Rel_Strength_SPY"] = np.nan

    out["Breadth_Proxy"] = (
        (px > out["SMA_50"]).astype(float)
        + (px > out["SMA_200"]).astype(float)
    ) / 2

    return out


def classify_regime(row: pd.Series) -> str:
    close = clean_number(row.get("Close"), np.nan)
    sma_50 = clean_number(row.get("SMA_50"), np.nan)
    sma_200 = clean_number(row.get("SMA_200"), np.nan)
    roc = clean_number(row.get("ROC"), 0.0)
    roc_slope = clean_number(row.get("ROC_Slope"), 0.0)
    second_derivative = clean_number(row.get("Second_Derivative"), 0.0)
    second_derivative_slope = clean_number(row.get("Second_Derivative_Slope"), 0.0)

    if pd.isna(close) or pd.isna(sma_50) or pd.isna(sma_200):
        return "Insufficient Trend Data"

    bull = close > sma_50 > sma_200
    bear = close < sma_50 < sma_200
    roc_up = roc_slope > 0
    sd_up = second_derivative_slope > 0

    if bull and roc_up and sd_up:
        return "Early Bull Acceleration"
    if bull and second_derivative >= 0:
        return "Mature Bull"
    if bull and second_derivative < 0 and roc > 0:
        return "Momentum Exhaustion"
    if bear and second_derivative < 0:
        return "Early Bear Acceleration"
    if bear and second_derivative >= 0:
        return "Capitulation Stabilization"
    if close < sma_200 and sd_up:
        return "Reflexive Recovery"

    return "Transition"


def score_snapshot(row: pd.Series) -> Dict[str, float]:
    roc = clean_number(row.get("ROC"), 0.0)
    roc_slope = clean_number(row.get("ROC_Slope"), 0.0)
    second_derivative = clean_number(row.get("Second_Derivative"), 0.0)
    second_derivative_slope = clean_number(row.get("Second_Derivative_Slope"), 0.0)
    close = clean_number(row.get("Close"), np.nan)
    sma_50 = clean_number(row.get("SMA_50"), np.nan)
    sma_200 = clean_number(row.get("SMA_200"), np.nan)
    atr_pct = clean_number(row.get("ATR_Pct"), 0.0)
    volume_accel = clean_number(row.get("Volume_Accel"), 1.0)

    momentum = 50 + 30 * np.tanh(roc * 8) + 20 * np.tanh(roc_slope * 10)
    acceleration = (
        50
        + 40 * np.tanh(second_derivative * 15)
        + 10 * np.tanh(second_derivative_slope * 20)
    )

    trend = 0.0
    if not pd.isna(close) and not pd.isna(sma_50) and not pd.isna(sma_200):
        trend += 35 if close > sma_200 else 10
        trend += 25 if close > sma_50 else 5
        trend += 20 if sma_50 > sma_200 else 0
        trend += 20 * np.clip((close / sma_50 - 1) * 10, -1, 1)

    risk = 50.0
    if row.get("Vol_Regime") == "Expansion":
        risk += 20
    risk += 20 * np.clip(atr_pct * 30, 0, 1)
    risk -= 10 * np.clip(volume_accel - 1, -1, 1)

    return {
        "momentum": float(np.clip(momentum, 0, 100)),
        "acceleration": float(np.clip(acceleration, 0, 100)),
        "trend_strength": float(np.clip(trend, 0, 100)),
        "risk_regime": float(np.clip(risk, 0, 100)),
    }


def institutional_signal(row: pd.Series, regime: str) -> Tuple[str, float]:
    roc = clean_number(row.get("ROC"), 0.0)
    second_derivative = clean_number(row.get("Second_Derivative"), 0.0)

    if second_derivative > 0 and roc < 0:
        return "Momentum is improving beneath the price surface while the trailing return is still negative.", 0.78

    if second_derivative < 0 and roc > 0:
        return "Acceleration is fading while the trailing return remains positive, which raises the odds of near-term exhaustion.", 0.80

    if regime in ["Early Bull Acceleration", "Reflexive Recovery"]:
        return "Acceleration is confirming trend repair and the tape is shifting toward a more constructive regime.", 0.74

    if regime == "Early Bear Acceleration":
        return "Downside acceleration is building and the tape is becoming less forgiving.", 0.76

    if regime == "Momentum Exhaustion":
        return "The trend is still intact, but the rate-of-change impulse is losing force.", 0.72

    if regime == "Capitulation Stabilization":
        return "The trend remains damaged, but acceleration is stabilizing from a weak base.", 0.68

    return "The setup is transitional and needs confirmation from price, moving averages, and volatility.", 0.58


def risk_label(score: float) -> str:
    if score > 65:
        return "elevated"
    if score > 45:
        return "moderate"
    return "contained"


def directional_bias(regime: str, scores: Dict[str, float]) -> str:
    if regime in ["Early Bull Acceleration", "Mature Bull", "Reflexive Recovery"] and scores["trend_strength"] >= 55:
        return "constructive"
    if regime in ["Momentum Exhaustion", "Early Bear Acceleration"] or scores["risk_regime"] > 70:
        return "cautious"
    return "balanced"


def build_commentary_html(
    ticker: str,
    latest: pd.Series,
    regime: str,
    signal: str,
    confidence: float,
    scores: Dict[str, float],
    roc_label: str,
) -> str:
    close = clean_number(latest.get("Close"), np.nan)
    sma_50 = clean_number(latest.get("SMA_50"), np.nan)
    sma_200 = clean_number(latest.get("SMA_200"), np.nan)
    volume_accel = clean_number(latest.get("Volume_Accel"), np.nan)
    rel_strength = clean_number(latest.get("Rel_Strength_SPY"), np.nan)
    atr_pct = clean_number(latest.get("ATR_Pct"), np.nan)
    vix = clean_number(latest.get("VIX"), np.nan)

    price_vs_50 = close / sma_50 - 1 if not pd.isna(close) and not pd.isna(sma_50) and sma_50 != 0 else np.nan
    price_vs_200 = close / sma_200 - 1 if not pd.isna(close) and not pd.isna(sma_200) and sma_200 != 0 else np.nan

    if bool(latest.get("Pos_Inflect", False)):
        inflection = "The latest bar printed a positive acceleration inflection."
    elif bool(latest.get("Neg_Inflect", False)):
        inflection = "The latest bar printed a negative acceleration inflection."
    else:
        inflection = "The latest bar did not print a fresh acceleration inflection."

    bias = directional_bias(regime, scores)

    escaped_ticker = html.escape(ticker)
    escaped_regime = html.escape(regime)
    escaped_signal = html.escape(signal)
    escaped_bias = html.escape(bias)

    latest_date = latest.name.strftime("%Y-%m-%d") if hasattr(latest.name, "strftime") else str(latest.name)

    return f"""
    <div class="commentary-box">
        <div class="commentary-kicker">Current Market Interpretation</div>
        <div class="commentary-title">{escaped_ticker} rate-of-change read as of {latest_date}</div>

        <p>
            <span class="inline-stat">{escaped_ticker}</span> is currently classified as
            <span class="inline-stat">{escaped_regime}</span>. The {roc_label} rate of change is
            <span class="inline-stat">{pct_text(latest.get("ROC"))}</span>, the latest acceleration reading is
            <span class="inline-stat">{pct_text(latest.get("Second_Derivative"))}</span>, and the model confidence is
            <span class="inline-stat">{confidence:.0%}</span>.
        </p>

        <p>
            {escaped_signal} {inflection} Price is
            <span class="inline-stat">{pct_text(price_vs_50)}</span> versus the 50-day SMA and
            <span class="inline-stat">{pct_text(price_vs_200)}</span> versus the 200-day SMA, which puts the current bias at
            <span class="inline-stat">{escaped_bias}</span> rather than requiring a forced directional call.
        </p>

        <p>
            The risk backdrop is <span class="inline-stat">{risk_label(scores["risk_regime"])}</span>. Relative strength versus SPY is
            <span class="inline-stat">{clean_number(rel_strength, np.nan):.3f}</span>, volume acceleration is
            <span class="inline-stat">{ratio_text(volume_accel)}</span>, ATR is
            <span class="inline-stat">{pct_text(atr_pct, signed=False)}</span> of price, and the VIX proxy is
            <span class="inline-stat">{clean_number(vix, np.nan):.2f}</span>. For this signal to improve, acceleration needs to keep rising while price holds the major SMAs. For it to break, acceleration should roll over while price loses the 50-day SMA, especially if volatility expands at the same time.
        </p>
    </div>
    """


with st.sidebar:
    st.markdown("## About This Tool")
    st.markdown(
        """
        **Purpose:** Rate-of-change regime explorer for identifying when price momentum is accelerating, fading, or transitioning.

        **What this tab shows**

        - Price trend with 21, 50, 100, and 200-day simple moving averages.
        - Rolling rate of change over the selected lookback window.
        - Second-derivative acceleration to flag turning points in momentum.
        - Positive and negative inflection markers when acceleration crosses the zero line.
        - One interpretation box tying trend, momentum, volatility, relative strength, and volume together.

        **Data source**

        - Yahoo Finance daily OHLCV history via `yfinance`.
        - SPY used as the relative-strength benchmark.
        - VIX used as a volatility proxy.
        """
    )

    st.markdown('<div class="tool-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Display Controls")
    show_inflections = st.toggle("Show inflection markers", value=True)
    compact_weekends = st.toggle("Compress weekends", value=True)


st.markdown('<div class="hero-title">Rate of Change Regime Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-caption">Track price trend, rate of change, and acceleration through a clean regime lens.</div>',
    unsafe_allow_html=True,
)

input_col_1, input_col_2, input_col_3, input_col_4 = st.columns([1.55, 1.0, 1.0, 1.0])

with input_col_1:
    ticker = st.text_input("Ticker symbol", value="^SPX").upper().strip()

with input_col_2:
    timeframe = st.selectbox("Analysis window", list(TIMEFRAME_MAP.keys()), index=3)

with input_col_3:
    roc_label = st.selectbox("ROC period", list(ROC_PERIODS.keys()), index=2)

with input_col_4:
    chart_view = st.selectbox("Chart view", ["Candlestick", "Line"], index=0)


if not ticker:
    st.warning("Enter a ticker symbol.")
    st.stop()


try:
    df = fetch_history(ticker, TIMEFRAME_MAP[timeframe])
    spy = fetch_history("SPY", TIMEFRAME_MAP[timeframe])
    vix = fetch_history("^VIX", TIMEFRAME_MAP[timeframe])
except Exception as e:
    st.error(str(e))
    st.stop()


if len(df) < 60:
    st.warning("The selected window has fewer than 60 observations, so derivative readings may be unstable.")


feat = compute_features(df, ROC_PERIODS[roc_label], spy)

vix_close = vix["Adj Close"] if "Adj Close" in vix.columns else vix["Close"]
feat["VIX"] = vix_close.reindex(feat.index).ffill().bfill()

feat = feat.dropna(subset=["ROC", "Second_Derivative"], how="any")

if feat.empty:
    st.warning("Data became empty after indicator calculations. Try a longer analysis window.")
    st.stop()


latest = feat.iloc[-1]
regime = classify_regime(latest)
scores = score_snapshot(latest)
signal, confidence = institutional_signal(latest, regime)


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    row_heights=[0.52, 0.24, 0.24],
)

if chart_view == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=feat.index,
            open=feat["Open"],
            high=feat["High"],
            low=feat["Low"],
            close=feat["Close"],
            name="Price",
            increasing_line_color=PASTEL_GREEN,
            decreasing_line_color=PASTEL_RED,
            increasing_fillcolor="rgba(82, 183, 136, 0.60)",
            decreasing_fillcolor="rgba(232, 93, 93, 0.60)",
        ),
        row=1,
        col=1,
    )
else:
    fig.add_trace(
        go.Scatter(
            x=feat.index,
            y=feat["Close"],
            mode="lines",
            name="Price",
            line=dict(color="#111827", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


for ma in ["SMA_21", "SMA_50", "SMA_100", "SMA_200"]:
    fig.add_trace(
        go.Scatter(
            x=feat.index,
            y=feat[ma],
            mode="lines",
            name=ma.replace("_", " "),
            line=dict(color=SMA_COLORS[ma], width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


fig.add_trace(
    go.Scatter(
        x=feat.index,
        y=feat["ROC"],
        mode="lines",
        name=f"ROC {roc_label}",
        line=dict(color="#4c78a8", width=2.0),
        hovertemplate="%{x|%Y-%m-%d}<br>ROC: %{y:.2%}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color=PASTEL_GREY,
    row=2,
    col=1,
)


bar_colors = np.where(feat["Second_Derivative"] >= 0, PASTEL_GREEN, PASTEL_RED)

fig.add_trace(
    go.Bar(
        x=feat.index,
        y=feat["Second_Derivative"],
        marker_color=bar_colors,
        name="Acceleration",
        hovertemplate="%{x|%Y-%m-%d}<br>Acceleration: %{y:.2%}<extra></extra>",
    ),
    row=3,
    col=1,
)

fig.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color=PASTEL_GREY,
    row=3,
    col=1,
)


if show_inflections:
    pos_marks = feat[feat["Pos_Inflect"]]
    neg_marks = feat[feat["Neg_Inflect"]]

    fig.add_trace(
        go.Scatter(
            x=pos_marks.index,
            y=pos_marks["Second_Derivative"],
            mode="markers",
            name="Positive inflection",
            marker=dict(color=PASTEL_GREEN, size=8, symbol="triangle-up", line=dict(width=0.8, color="#ffffff")),
            hovertemplate="%{x|%Y-%m-%d}<br>Positive acceleration inflection<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=neg_marks.index,
            y=neg_marks["Second_Derivative"],
            mode="markers",
            name="Negative inflection",
            marker=dict(color=PASTEL_RED, size=8, symbol="triangle-down", line=dict(width=0.8, color="#ffffff")),
            hovertemplate="%{x|%Y-%m-%d}<br>Negative acceleration inflection<extra></extra>",
        ),
        row=3,
        col=1,
    )


fig.update_layout(
    height=860,
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=40, r=30, t=80, b=70),
    title=dict(
        text=f"{ticker} | Price, Rate of Change, and Acceleration",
        x=0.02,
        xanchor="left",
        y=0.97,
        font=dict(size=24, color="#111827", family="Arial, sans-serif"),
    ),
    font=dict(color="#334155", family="Arial, sans-serif"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.10,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0)",
    ),
    xaxis_rangeslider_visible=False,
)

fig.update_yaxes(
    title_text="Price",
    row=1,
    col=1,
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)

fig.update_yaxes(
    title_text=f"ROC {roc_label}",
    row=2,
    col=1,
    tickformat=".1%",
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)

fig.update_yaxes(
    title_text="Acceleration",
    row=3,
    col=1,
    tickformat=".2%",
    gridcolor="rgba(226, 232, 240, 0.75)",
    zeroline=False,
)

fig.update_xaxes(
    gridcolor="rgba(226, 232, 240, 0.55)",
    showline=True,
    linewidth=1,
    linecolor="#d7dde8",
)

if compact_weekends:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    build_commentary_html(
        ticker=ticker,
        latest=latest,
        regime=regime,
        signal=signal,
        confidence=confidence,
        scores=scores,
        roc_label=roc_label,
    ),
    unsafe_allow_html=True,
)
