# app.py
# Streamlit: Hedge Timing Dashboard for SPY / QQQ
# Data sources: Yahoo Finance (via yfinance) + optional Finviz snapshot (light scrape)
#
# Run:
#   pip install streamlit yfinance pandas numpy plotly requests
#   streamlit run app.py

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Hedge Timing: SPY / QQQ", layout="wide")

# ----------------------------- Style (tight + clean) -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 1.2rem; }
div[data-testid="stMetric"] { background: #0b1220; border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 12px 10px 12px; border-radius: 14px; }
div[data-testid="stMetric"] > label { opacity: 0.85; }
hr { margin: 0.8rem 0 0.6rem 0; opacity: 0.25; }
.small { opacity: 0.85; font-size: 0.92rem; }
.callout { border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(255,255,255,0.10); }
.good { background: rgba(16, 185, 129, 0.10); }
.mid  { background: rgba(245, 158, 11, 0.10); }
.bad  { background: rgba(239, 68, 68, 0.10); }
.kv { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.kv > div { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  padding: 8px 10px; border-radius: 12px; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Defaults -----------------------------
RISK_BASKET = [
    "SPY",
    "QQQ",
    "^VIX",
    "^VIX9D",  # may be missing sometimes, code handles gracefully
    "TLT",
    "HYG",
    "LQD",
    "RSP",
    "IWM",
    "XLY",
    "XLP",
    "XLF",
    "GLD",
    "USO",
    "UUP",  # USD proxy
    "^TNX",  # 10Y yield index
]

PRICE_TICKERS = ["SPY", "QQQ", "TLT", "HYG", "LQD", "RSP", "IWM", "XLY", "XLP", "XLF", "GLD", "USO", "UUP"]
INDEX_TICKERS = ["^VIX", "^VIX9D", "^TNX"]

# ----------------------------- Helpers -----------------------------
def _today() -> date:
    return date.today()

def _start_date(lookback_years: int) -> date:
    # pad to reduce missing MA windows
    return _today() - timedelta(days=int(lookback_years * 365.25) + 60)

@st.cache_data(ttl=900, show_spinner=False)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # yfinance returns multi-index columns when multiple tickers
    return df

def extract_close(df_raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    if isinstance(df_raw.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df_raw.columns:
                closes[t] = df_raw[(t, "Close")]
            elif (t, "Adj Close") in df_raw.columns:
                closes[t] = df_raw[(t, "Adj Close")]
        out = pd.DataFrame(closes)
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    else:
        # single ticker
        if "Close" in df_raw.columns:
            out = df_raw[["Close"]].rename(columns={"Close": tickers[0]})
        elif "Adj Close" in df_raw.columns:
            out = df_raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            return pd.DataFrame()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

def last_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x*100:.2f}%"

def num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"

def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()

def rolling_vol(ret: pd.Series, w: int = 20) -> pd.Series:
    return ret.rolling(w).std() * np.sqrt(252.0)

def drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return px / peak - 1.0

def slope(s: pd.Series, w: int = 20) -> float:
    s = s.dropna()
    if len(s) < w:
        return float("nan")
    y = s.iloc[-w:].values.astype(float)
    x = np.arange(w, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    b = np.polyfit(x, y, 1)[0]
    return float(b)

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

# Optional: lightweight Finviz snapshot (best effort)
@st.cache_data(ttl=3600, show_spinner=False)
def finviz_snapshot(ticker: str) -> Dict[str, str]:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        # Finviz quote page usually contains a 2-column key-value grid table
        # Find the table that looks like [key, value, key, value, ...]
        best = None
        for t in tables:
            if t.shape[1] >= 4 and t.shape[0] >= 8:
                best = t
                break
        if best is None:
            return {}
        kv = {}
        # Flatten rows: [k1, v1, k2, v2, ...]
        for _, row in best.iterrows():
            vals = [str(x) for x in row.values.tolist()]
            for i in range(0, len(vals) - 1, 2):
                k = vals[i].strip()
                v = vals[i + 1].strip()
                if k and k.lower() != "nan" and v.lower() != "nan":
                    kv[k] = v
        return kv
    except Exception:
        return {}

@dataclass
class HedgeSignal:
    score: int
    stance: str
    tag: str
    reasons: List[str]
    invalidation: List[str]
    preferred_hedge: str

def build_hedge_signal(
    px: pd.DataFrame,
    idx: pd.DataFrame,
    risk_mode: str,
) -> HedgeSignal:
    # Weights by risk mode: conservative hedges sooner; aggressive waits for alignment
    if risk_mode == "Conservative":
        w_trend_200 = 28
        w_trend_50 = 18
        w_vol = 16
        w_term = 10
        w_credit = 16
        w_breadth = 12
        w_defensive = 8
    elif risk_mode == "Aggressive":
        w_trend_200 = 30
        w_trend_50 = 14
        w_vol = 14
        w_term = 10
        w_credit = 18
        w_breadth = 14
        w_defensive = 6
    else:
        # Balanced
        w_trend_200 = 30
        w_trend_50 = 16
        w_vol = 15
        w_term = 10
        w_credit = 17
        w_breadth = 12
        w_defensive = 6

    spy = px["SPY"].dropna()
    qqq = px["QQQ"].dropna()

    def _score_for_symbol(sym: str) -> Tuple[int, List[str], List[str]]:
        s = px[sym].dropna()
        ret = s.pct_change()

        ma21 = rolling_ma(s, 21)
        ma50 = rolling_ma(s, 50)
        ma200 = rolling_ma(s, 200)

        dd = drawdown(s)
        rv20 = rolling_vol(ret, 20)

        score = 0
        reasons = []
        invalid = []

        # Trend (primary)
        if len(ma200.dropna()) and last_valid(s) < last_valid(ma200):
            score += w_trend_200
            reasons.append(f"{sym} below 200d MA (trend risk-on is broken).")
            invalid.append(f"Cover hedge if {sym} closes back above its 200d MA for 3 sessions.")
        if len(ma50.dropna()) and last_valid(s) < last_valid(ma50):
            score += w_trend_50
            reasons.append(f"{sym} below 50d MA (medium-term momentum weak).")
            invalid.append(f"Reduce hedge if {sym} reclaims the 50d MA with breadth improving.")

        # Vol regime
        vix = idx.get("^VIX", pd.Series(dtype=float)).dropna()
        if len(vix):
            vix_last = last_valid(vix)
            if vix_last >= 20:
                score += int(w_vol * 0.7)
                reasons.append(f"VIX elevated ({vix_last:.2f}).")
                invalid.append("De-escalate hedge if VIX holds below 18 for a week.")
            if vix_last >= 25:
                score += int(w_vol * 0.4)

        # Front-end stress: VIX9D / VIX
        vix9 = idx.get("^VIX9D", pd.Series(dtype=float)).dropna()
        if len(vix) and len(vix9):
            ratio = vix9 / vix
            ratio_last = last_valid(ratio)
            if ratio_last >= 1.0:
                score += w_term
                reasons.append(f"Near-term stress up (VIX9D/VIX = {ratio_last:.2f}).")
                invalid.append("Ease hedges if VIX9D/VIX falls below 0.90 and stays there.")

        # Credit risk: HYG/LQD
        if "HYG" in px and "LQD" in px:
            hyg_lqd = safe_ratio(px["HYG"], px["LQD"]).dropna()
            if len(hyg_lqd) > 220:
                hl_ma100 = rolling_ma(hyg_lqd, 100)
                hl_ma200 = rolling_ma(hyg_lqd, 200)
                if last_valid(hyg_lqd) < last_valid(hl_ma200):
                    score += w_credit
                    reasons.append("Credit risk-off (HYG/LQD below 200d MA).")
                    invalid.append("Ease hedges if HYG/LQD reclaims its 200d MA.")
                elif last_valid(hyg_lqd) < last_valid(hl_ma100):
                    score += int(w_credit * 0.6)
                    reasons.append("Credit softening (HYG/LQD below 100d MA).")

        # Breadth proxies: RSP/SPY and IWM/SPY
        if "RSP" in px:
            rsp_spy = safe_ratio(px["RSP"], px["SPY"]).dropna()
            if len(rsp_spy) > 220:
                rs_ma200 = rolling_ma(rsp_spy, 200)
                if last_valid(rsp_spy) < last_valid(rs_ma200):
                    score += w_breadth
                    reasons.append("Breadth weak (RSP underperforming SPY on 200d basis).")
                    invalid.append("Ease hedges if RSP/SPY turns up and reclaims its 200d MA.")
        if "IWM" in px:
            iwm_spy = safe_ratio(px["IWM"], px["SPY"]).dropna()
            if len(iwm_spy) > 220:
                rs_ma200 = rolling_ma(iwm_spy, 200)
                if last_valid(iwm_spy) < last_valid(rs_ma200):
                    score += int(w_breadth * 0.4)
                    reasons.append("Small caps lagging (IWM/SPY below 200d MA).")

        # Defensive tape: XLY/XLP
        if "XLY" in px and "XLP" in px:
            xly_xlp = safe_ratio(px["XLY"], px["XLP"]).dropna()
            if len(xly_xlp) > 220:
                xx_ma200 = rolling_ma(xly_xlp, 200)
                if last_valid(xly_xlp) < last_valid(xx_ma200):
                    score += w_defensive
                    reasons.append("Consumer discretionary lagging staples (risk-off behavior).")

        # Cap score
        score = int(max(0, min(100, score)))

        # Keep invalidation readable
        invalid = list(dict.fromkeys(invalid))[:4]
        reasons = list(dict.fromkeys(reasons))[:6]
        return score, reasons, invalid

    # Decide which hedge fits better today based on relative weakness (QQQ/SPY trend)
    qqq_spy = safe_ratio(px["QQQ"], px["SPY"]).dropna()
    pref = "SPY"
    if len(qqq_spy) > 220:
        rs_ma200 = rolling_ma(qqq_spy, 200)
        if last_valid(qqq_spy) < last_valid(rs_ma200) and slope(qqq_spy, 30) < 0:
            pref = "QQQ"
        else:
            pref = "SPY"

    score_spy, reasons_spy, invalid_spy = _score_for_symbol("SPY")
    score_qqq, reasons_qqq, invalid_qqq = _score_for_symbol("QQQ")

    # Use the preferred hedge symbol's score for stance, but show both
    chosen_score = score_qqq if pref == "QQQ" else score_spy
    chosen_reasons = reasons_qqq if pref == "QQQ" else reasons_spy
    chosen_invalid = invalid_qqq if pref == "QQQ" else invalid_spy

    if chosen_score >= 65:
        stance = "HEDGE ON"
        tag = "bad"
    elif chosen_score >= 45:
        stance = "HEDGE BIAS"
        tag = "mid"
    else:
        stance = "HEDGE OFF"
        tag = "good"

    return HedgeSignal(
        score=chosen_score,
        stance=stance,
        tag=tag,
        reasons=chosen_reasons,
        invalidation=chosen_invalid,
        preferred_hedge=pref,
    )

def plot_price_with_mas(px: pd.Series, title: str) -> go.Figure:
    s = px.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Price", mode="lines"))
    for w in [21, 50, 200]:
        ma = s.rolling(w).mean()
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"MA{w}", mode="lines"))
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def plot_indicator_lines(series_map: Dict[str, pd.Series], title: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for name, s in series_map.items():
        s = s.dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines"))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def plot_corr_heatmap(rets: pd.DataFrame, title: str) -> go.Figure:
    r = rets.dropna().copy()
    if r.shape[0] < 40:
        corr = r.corr()
    else:
        corr = r.iloc[-60:].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            hovertemplate="Corr %{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Hedge Timing")
    st.caption("Actionable read on whether to short SPY / QQQ right now.")

    lookback = st.slider("Lookback (years)", 1, 10, 5)
    risk_mode = st.selectbox("Decision style", ["Balanced", "Conservative", "Aggressive"], index=0)
    focus = st.selectbox("Focus", ["Both", "SPY", "QQQ"], index=0)

    st.divider()
    show_finviz = st.toggle("Show Finviz snapshot (best effort)", value=False)
    show_heatmap = st.toggle("Show correlation heatmap", value=True)

# ----------------------------- Load data -----------------------------
start = _start_date(lookback)

raw = yf_download(RISK_BASKET, start)
close_all = extract_close(raw, RISK_BASKET)

# Split price vs indices for readability
px = close_all[[c for c in close_all.columns if c in PRICE_TICKERS]].copy()
idx = close_all[[c for c in close_all.columns if c in INDEX_TICKERS]].copy()

# Ensure core symbols exist
missing = [t for t in ["SPY", "QQQ"] if t not in px.columns or px[t].dropna().empty]
if missing:
    st.error(f"Missing critical data for: {', '.join(missing)}. Try again later (Yahoo sometimes hiccups).")
    st.stop()

# Returns for correlation panel
rets = px[["SPY", "QQQ"] + [t for t in ["TLT", "HYG", "LQD", "GLD", "USO", "UUP"] if t in px.columns]].pct_change()

# ----------------------------- Build signals -----------------------------
signal = build_hedge_signal(px=px, idx=idx, risk_mode=risk_mode)

spy_last = last_valid(px["SPY"])
qqq_last = last_valid(px["QQQ"])
spy_dd = last_valid(drawdown(px["SPY"]))
qqq_dd = last_valid(drawdown(px["QQQ"]))
vix_last = last_valid(idx["^VIX"]) if "^VIX" in idx.columns else float("nan")
tlt_last = last_valid(px["TLT"]) if "TLT" in px.columns else float("nan")

# ----------------------------- Header + decision box -----------------------------
st.title("Hedge Timing Dashboard")
st.caption("Objective: decide whether shorting SPY / QQQ is justified by trend + stress + credit + breadth proxies.")

callout_cls = signal.tag
st.markdown(
    f"""
<div class="callout {callout_cls}">
  <div style="display:flex; justify-content:space-between; align-items:baseline; gap:18px;">
    <div style="font-size:1.25rem; font-weight:700;">{signal.stance}</div>
    <div style="opacity:0.9;">Hedge score: <b>{signal.score}/100</b> | Preferred hedge: <b>{signal.preferred_hedge}</b></div>
  </div>

  <div class="kv">
    <div>SPY: <b>{num(spy_last,2)}</b> | DD: <b>{pct(spy_dd)}</b></div>
    <div>QQQ: <b>{num(qqq_last,2)}</b> | DD: <b>{pct(qqq_dd)}</b></div>
    <div>VIX: <b>{num(vix_last,2)}</b></div>
    <div>TLT: <b>{num(tlt_last,2)}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("Why this stance")
    if signal.reasons:
        st.markdown("\n".join([f"- {r}" for r in signal.reasons]))
    else:
        st.markdown("- NA (insufficient data for some indicators).")

with c2:
    st.subheader("What flips it")
    if signal.invalidation:
        st.markdown("\n".join([f"- {x}" for x in signal.invalidation]))
    else:
        st.markdown("- NA")

st.divider()

# ----------------------------- Key charts -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    if focus in ["Both", "SPY"]:
        st.plotly_chart(plot_price_with_mas(px["SPY"], "SPY price and key moving averages"), use_container_width=True)
    if focus in ["Both", "QQQ"]:
        st.plotly_chart(plot_price_with_mas(px["QQQ"], "QQQ price and key moving averages"), use_container_width=True)

with right:
    # Stress + credit + breadth proxies
    series_map = {}
    if "^VIX" in idx.columns:
        series_map["VIX"] = idx["^VIX"]
    if "^VIX9D" in idx.columns and "^VIX" in idx.columns:
        series_map["VIX9D/VIX"] = (idx["^VIX9D"] / idx["^VIX"]).replace([np.inf, -np.inf], np.nan)

    if "HYG" in px.columns and "LQD" in px.columns:
        series_map["HYG/LQD"] = safe_ratio(px["HYG"], px["LQD"])
    if "RSP" in px.columns and "SPY" in px.columns:
        series_map["RSP/SPY"] = safe_ratio(px["RSP"], px["SPY"])
    if "XLY" in px.columns and "XLP" in px.columns:
        series_map["XLY/XLP"] = safe_ratio(px["XLY"], px["XLP"])

    st.plotly_chart(plot_indicator_lines(series_map, "Stress, credit, breadth proxies (higher is better)"), use_container_width=True)

st.divider()

# ----------------------------- Correlations -----------------------------
if show_heatmap:
    st.subheader("Correlations (last ~60 trading days)")
    st.caption("This helps you sanity-check whether your hedge is likely to behave as expected in the current tape.")
    st.plotly_chart(plot_corr_heatmap(rets, "Rolling correlation heatmap"), use_container_width=True)

# ----------------------------- Optional: Finviz snapshot -----------------------------
if show_finviz:
    st.divider()
    st.subheader("Finviz snapshot (best effort)")
    a, b = st.columns([1, 1], gap="large")

    def show_kv(title: str, ticker: str):
        kv = finviz_snapshot(ticker)
        if not kv:
            st.warning(f"Could not fetch Finviz for {ticker} (blocked or rate-limited).")
            return
        keys = ["ETF", "Index", "Perf Week", "Perf Month", "Perf YTD", "Volatility", "Avg Volume", "RSI (14)"]
        shown = {k: kv.get(k) for k in keys if k in kv}
        if not shown:
            # fallback: show a compact selection of whatever exists
            keep = list(kv.keys())[:18]
            shown = {k: kv[k] for k in keep}
        st.markdown(f"**{title} ({ticker})**")
        st.write(pd.DataFrame(list(shown.items()), columns=["Metric", "Value"]))

    with a:
        show_kv("SPY", "SPY")
    with b:
        show_kv("QQQ", "QQQ")

# ----------------------------- Final action rubric -----------------------------
st.divider()
st.subheader("How to use this (tight rubric)")
st.markdown(
    """
- Treat the score as a permission slip. Above 65, you are allowed to carry an active index short. Between 45 and 65, you hedge smaller and demand confirmation (trend breaks plus credit or breadth weakness). Below 45, the tape is usually paying you to stay patient.
- If Preferred hedge = QQQ, the Nasdaq is the stress center (QQQ/SPY relative strength is rolling over). If Preferred hedge = SPY, the hedge is broader market beta.
- Your invalidation is mechanical. If the flip conditions hit, you stop debating and you reduce or cover.
""".strip()
)

st.caption("This is a decision aid, not investment advice. Data can be missing or stale on any given day; the app degrades gracefully when feeds fail.")
