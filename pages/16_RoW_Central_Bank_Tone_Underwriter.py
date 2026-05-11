import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Rate of Change Regime Dashboard", layout="wide")

TIMEFRAME_MAP = {
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "3Y": "3y",
    "5Y": "5y",
    "Max": "max",
}

ROC_PERIODS = {"5D": 5, "10D": 10, "20D": 20, "63D": 63, "126D": 126}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str, interval: str = "1d", retries: int = 3) -> pd.DataFrame:
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
            data = data[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in data.columns]]
            data = data.dropna(how="all")
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            return data
        except Exception as err:
            last_err = err
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {ticker} after {retries} retries: {last_err}")


def compute_features(df: pd.DataFrame, roc_n: int, spy_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]

    for win in [21, 50, 100, 200]:
        out[f"SMA_{win}"] = px.rolling(win, min_periods=1).mean()
    out["ROC"] = px.pct_change(roc_n)
    out["ROC_Slope"] = out["ROC"].diff(5)
    out["Second_Derivative"] = out["ROC"].diff()
    out["Second_Derivative_Slope"] = out["Second_Derivative"].diff(3)

    prev_sd = out["Second_Derivative"].shift(1)
    out["Pos_Inflect"] = (prev_sd <= 0) & (out["Second_Derivative"] > 0)
    out["Neg_Inflect"] = (prev_sd >= 0) & (out["Second_Derivative"] < 0)

    tr = pd.concat(
        [
            (out["High"] - out["Low"]),
            (out["High"] - out["Close"].shift(1)).abs(),
            (out["Low"] - out["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = tr.rolling(14, min_periods=1).mean()
    out["ATR_Pct"] = out["ATR_14"] / px.replace(0, np.nan)

    out["Vol_20"] = px.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252)
    out["Vol_60"] = px.pct_change().rolling(60, min_periods=20).std() * np.sqrt(252)
    out["Vol_Regime"] = np.where(out["Vol_20"] > out["Vol_60"], "Expansion", "Compression")

    out["Volume_Accel"] = out["Volume"].rolling(10, min_periods=3).mean() / out["Volume"].rolling(50, min_periods=10).mean()

    spy_px = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
    rel = (px / px.iloc[0]) / (spy_px.reindex(out.index).ffill() / spy_px.reindex(out.index).ffill().iloc[0])
    out["Rel_Strength_SPY"] = rel

    out["Breadth_Proxy"] = ((px > out["SMA_50"]).astype(float) + (px > out["SMA_200"]).astype(float)) / 2

    return out


def classify_regime(row: pd.Series) -> str:
    bull = row["Close"] > row["SMA_50"] > row["SMA_200"]
    bear = row["Close"] < row["SMA_50"] < row["SMA_200"]
    roc_up = row["ROC_Slope"] > 0
    sd_up = row["Second_Derivative_Slope"] > 0

    if bull and roc_up and sd_up:
        return "Early Bull Acceleration"
    if bull and row["Second_Derivative"] >= 0:
        return "Mature Bull"
    if bull and (row["Second_Derivative"] < 0) and (row["ROC"] > 0):
        return "Momentum Exhaustion"
    if bear and (row["Second_Derivative"] < 0):
        return "Early Bear Acceleration"
    if bear and (row["Second_Derivative"] >= 0):
        return "Capitulation Stabilization"
    if (row["Close"] < row["SMA_200"]) and sd_up:
        return "Reflexive Recovery"
    return "Transition"


def score_snapshot(row: pd.Series) -> Dict[str, float]:
    momentum = 50 + 30 * np.tanh((row["ROC"] or 0) * 8) + 20 * np.tanh((row["ROC_Slope"] or 0) * 10)
    accel = 50 + 40 * np.tanh((row["Second_Derivative"] or 0) * 15) + 10 * np.tanh((row["Second_Derivative_Slope"] or 0) * 20)

    trend = 0
    trend += 35 if row["Close"] > row["SMA_200"] else 10
    trend += 25 if row["Close"] > row["SMA_50"] else 5
    trend += 20 if row["SMA_50"] > row["SMA_200"] else 0
    trend += 20 * np.clip((row["Close"] / row["SMA_50"] - 1) * 10, -1, 1)

    risk = 50
    if row["Vol_Regime"] == "Expansion":
        risk += 20
    risk += 20 * np.clip((row["ATR_Pct"] or 0) * 30, 0, 1)
    risk -= 10 * np.clip((row["Volume_Accel"] or 1) - 1, -1, 1)

    return {
        "momentum": float(np.clip(momentum, 0, 100)),
        "acceleration": float(np.clip(accel, 0, 100)),
        "trend_strength": float(np.clip(trend, 0, 100)),
        "risk_regime": float(np.clip(risk, 0, 100)),
    }


def institutional_signal(row: pd.Series, regime: str) -> Tuple[str, float]:
    if row["Second_Derivative"] > 0 and row["ROC"] < 0:
        return "Momentum improving beneath the surface", 0.78
    if row["Second_Derivative"] < 0 and row["ROC"] > 0:
        return "Acceleration rolling over despite price strength", 0.80
    if regime in ["Early Bull Acceleration", "Reflexive Recovery"]:
        return "Potential early-cycle setup developing", 0.74
    if regime == "Early Bear Acceleration":
        return "Reflexive breakdown risk increasing", 0.76
    if regime == "Momentum Exhaustion":
        return "Trend exhaustion likely", 0.72
    return "Balanced transition regime", 0.58




with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Identify market inflection points by tracking momentum (ROC) and acceleration/deceleration (second derivative).

        **What this tab shows**
        - Price trend with 21/50/100/200-day SMAs.
        - Rolling ROC and second-derivative behavior.
        - Positive/negative inflection markers and regime transitions.
        - Summary scores for momentum, acceleration, trend strength, and risk.

        **Data source**
        - Daily OHLCV history from Yahoo Finance (`yfinance`) with retry + cache handling.
        """
    )
    st.divider()
    st.header("Controls")
    ticker = st.text_input("Ticker", value="QQQ").upper().strip()
    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=3)
    roc_label = st.selectbox("ROC Period", list(ROC_PERIODS.keys()), index=2)
    show_candle = st.toggle("Candlestick view", value=True)

try:
    df = fetch_history(ticker, TIMEFRAME_MAP[timeframe])
    spy = fetch_history("SPY", TIMEFRAME_MAP[timeframe])
    vix = fetch_history("^VIX", TIMEFRAME_MAP[timeframe])
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("Rate of Change Regime Dashboard")
st.caption("Institutional market inflection monitor using momentum and acceleration regime logic.")

if len(df) < 60:
    st.warning("Not enough data points for robust derivative analysis.")

feat = compute_features(df, ROC_PERIODS[roc_label], spy)
feat["VIX"] = (vix["Adj Close"] if "Adj Close" in vix.columns else vix["Close"]).reindex(feat.index).ffill()
feat = feat.dropna(subset=["ROC", "Second_Derivative"], how="any")
if feat.empty:
    st.warning("Data became empty after indicator calculations.")
    st.stop()

latest = feat.iloc[-1]
regime = classify_regime(latest)
scores = score_snapshot(latest)
signal, confidence = institutional_signal(latest, regime)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Current Regime", regime)
m2.metric("Momentum Score", f"{scores['momentum']:.1f}/100")
m3.metric("Acceleration Score", f"{scores['acceleration']:.1f}/100")
m4.metric("Risk Regime", "High" if scores["risk_regime"] > 65 else "Moderate" if scores["risk_regime"] > 45 else "Low")
m5.metric("Trend Strength", f"{scores['trend_strength']:.1f}/100")
m6.metric("Signal Confidence", f"{confidence*100:.0f}%")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.25, 0.25])
if show_candle:
    fig.add_trace(go.Candlestick(x=feat.index, open=feat["Open"], high=feat["High"], low=feat["Low"], close=feat["Close"], name="Price"), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=feat.index, y=feat["Close"], mode="lines", name="Price", line=dict(color="#85c1ff", width=2)), row=1, col=1)

for ma, color in [("SMA_21", "#f1c40f"), ("SMA_50", "#e67e22"), ("SMA_100", "#9b59b6"), ("SMA_200", "#ecf0f1")]:
    fig.add_trace(go.Scatter(x=feat.index, y=feat[ma], mode="lines", name=ma, line=dict(color=color, width=1.3)), row=1, col=1)

fig.add_trace(go.Scatter(x=feat.index, y=feat["ROC"], mode="lines", name=f"ROC {roc_label}", line=dict(color="#5dade2", width=2)), row=2, col=1)
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

colors = np.where(feat["Second_Derivative"] >= 0, "#2ecc71", "#e74c3c")
fig.add_trace(go.Bar(x=feat.index, y=feat["Second_Derivative"], marker_color=colors, name="Second Derivative"), row=3, col=1)
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)

pos_marks = feat[feat["Pos_Inflect"]]
neg_marks = feat[feat["Neg_Inflect"]]
fig.add_trace(go.Scatter(x=pos_marks.index, y=pos_marks["Second_Derivative"], mode="markers", name="Positive inflection", marker=dict(color="#7DFF9B", size=8, symbol="triangle-up")), row=3, col=1)
fig.add_trace(go.Scatter(x=neg_marks.index, y=neg_marks["Second_Derivative"], mode="markers", name="Negative inflection", marker=dict(color="#FF7A7A", size=8, symbol="triangle-down")), row=3, col=1)

fig.update_layout(
    height=950,
    template="plotly",
    title=f"{ticker} | Price, ROC, and Second Derivative",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, x=0.01),
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Institutional Overlays")
    st.write(f"**Relative Strength vs SPY:** {latest['Rel_Strength_SPY']:.3f}")
    st.write(f"**Volume Acceleration (10/50):** {latest['Volume_Accel']:.2f}x")
    st.write(f"**ATR % of Price:** {latest['ATR_Pct']*100:.2f}%")
    st.write(f"**VIX Proxy:** {latest['VIX']:.2f}")
    st.write(f"**Volatility Regime:** {latest['Vol_Regime']}")
    st.write(f"**Breadth Proxy:** {latest['Breadth_Proxy']*100:.0f}%")

with right:
    st.subheader("Macro Signal Desk")
    st.info(signal)
    st.markdown("### Current Market Interpretation")
    st.write(
        f"Regime reads **{regime}** with momentum score **{scores['momentum']:.1f}** and acceleration score **{scores['acceleration']:.1f}**. "
        f"Risk profile is **{'elevated' if scores['risk_regime'] > 65 else 'contained'}**, while trend strength sits at **{scores['trend_strength']:.1f}**."
    )

    phase = "trend continuation"
    if regime in ["Early Bull Acceleration", "Reflexive Recovery", "Capitulation Stabilization"]:
        phase = "early accumulation"
    elif regime in ["Momentum Exhaustion", "Early Bear Acceleration"]:
        phase = "late-cycle exhaustion"
    st.markdown("### Phase Assessment")
    st.success(f"Conditions currently resemble **{phase}**.")
