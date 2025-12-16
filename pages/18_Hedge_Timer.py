# app.py
# Hedge Timer (SPY / QQQ)
# Clean, fixed UI. No toggles, no sliders. About This Tool lives in the sidebar.
# Signals: credit + breadth + vol term structure + RSI/MACD (daily/weekly/monthly) + trend confirmation.
# Threshold is calibrated using data since 2020-01-01 (computed each run, then held static on-page).
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# ============================== Page + Style ==============================
st.set_page_config(page_title="Hedge Timer", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.1rem; max-width: 1420px; }
h1,h2,h3 { letter-spacing: -0.2px; }
.small { opacity: 0.85; font-size: 0.92rem; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 16px;
  padding: 14px 16px;
}
.row { display:flex; justify-content:space-between; gap:14px; align-items:baseline; flex-wrap:wrap; }
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10);
  font-weight: 800; font-size: 0.92rem;
  background: rgba(0,0,0,0.03);
}
.b_good { background: rgba(16,185,129,0.16); border-color: rgba(16,185,129,0.25); }
.b_mid  { background: rgba(245,158,11,0.16); border-color: rgba(245,158,11,0.25); }
.b_bad  { background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.25); }

.kv { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
.kv > div {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.09);
  padding: 7px 10px; border-radius: 12px; font-size: 0.92rem;
}
hr { opacity: 0.20; margin: 0.9rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown("## Hedge Timer")
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown(
        """
Daily metrics: credit and breadth ratios (HYG/LQD, RSP/SPY, XLY/XLP), volatility stress and term structure (VIX, VIX9D, VIX3M, VVIX when available), multi-timeframe RSI and MACD (daily, weekly, monthly), and trend confirmation on the target index.

Decision output is a composite score (0 to 100). Thresholds are calibrated on the full sample since 2020-01-01 to balance early warning against over-trading.

Assumptions: close-to-close data, weekly is Friday close, monthly is month-end close, no transaction costs or borrow costs, no execution model. If Yahoo misses VIX9D, VIX3M, or VVIX, the model renormalizes around the remaining layers.
""".strip()
    )


# ============================== Data Universe ==============================
TICKERS = [
    "SPY",
    "QQQ",
    "RSP",
    "IWM",
    "XLY",
    "XLP",
    "HYG",
    "LQD",
    "TLT",
    "^VIX",
    "^VIX9D",   # sometimes missing
    "^VIX3M",   # sometimes missing
    "^VVIX",    # sometimes missing
]

CALIBRATION_START = "2020-01-01"
HORIZON_DAYS = 20
LEAD_LOOKBACK = 40
CHART_WINDOW_DAYS = 720


# ============================== Helpers ==============================
def _today() -> date:
    return date.today()

def _start_date() -> date:
    # Enough history for MA200 and multi-timeframe indicators
    return _today() - timedelta(days=int(9 * 365.25) + 120)

@st.cache_data(ttl=900, show_spinner=False)
def yf_download(tickers: List[str], start: date) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

def extract_close(df_raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    if isinstance(df_raw.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if (t, "Close") in df_raw.columns:
                out[t] = df_raw[(t, "Close")]
            elif (t, "Adj Close") in df_raw.columns:
                out[t] = df_raw[(t, "Adj Close")]
        df = pd.DataFrame(out)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    if "Close" in df_raw.columns:
        df = df_raw[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Adj Close" in df_raw.columns:
        df = df_raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def last_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x*100:.2f}%"

def fmt_num(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"

def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def drawdown(px: pd.Series) -> pd.Series:
    s = px.dropna()
    peak = s.cummax()
    return s / peak - 1.0

def rolling_ma(s: pd.Series, w: int) -> pd.Series:
    # min_periods=1 keeps MAs visible across the whole chart window
    return s.rolling(w, min_periods=1).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m - sig

def resample_last(s: pd.Series, rule: str) -> pd.Series:
    return s.dropna().resample(rule).last()

def mtf_to_daily(daily_index: pd.DatetimeIndex, mtf_series: pd.Series) -> pd.Series:
    return mtf_series.reindex(daily_index, method="ffill")

def forward_min_return(px: pd.Series, h: int) -> pd.Series:
    s = px.dropna()
    fut_min = s[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return fut_min / s - 1.0

def find_drawdown_episodes(px: pd.Series, threshold: float = -0.08, recovery: float = -0.02) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    s = px.dropna()
    if len(s) < 260:
        return []
    dd = drawdown(s)

    episodes = []
    in_ep = False
    start = None

    for ts, v in dd.items():
        if (not in_ep) and (v <= threshold):
            in_ep = True
            start = ts
        if in_ep and (v >= recovery):
            seg = dd.loc[start:ts]
            trough = seg.idxmin()
            depth = float(seg.min())
            episodes.append((start, trough, depth))
            in_ep = False
            start = None

    if in_ep and start is not None:
        seg = dd.loc[start:]
        trough = seg.idxmin()
        depth = float(seg.min())
        episodes.append((start, trough, depth))

    return sorted(episodes, key=lambda x: x[2])  # deepest first


# ============================== Signal Model ==============================
@dataclass
class Component:
    key: str
    label: str
    weight: int

COMPONENTS: List[Component] = [
    Component("credit_risk", "Credit risk-off", 18),
    Component("breadth_risk", "Breadth weak", 14),
    Component("defensive_tape", "Defensive tape", 10),
    Component("vol_stress", "Vol stress", 18),
    Component("mtf_momentum", "RSI/MACD rollover (D/W/M)", 22),
    Component("trend_confirm", "Trend confirms", 18),
]

def compute_components(df: pd.DataFrame, target: str) -> Tuple[Dict[str, pd.Series], int]:
    idx = df.index

    spy = df.get("SPY", pd.Series(index=idx, dtype=float))
    rsp = df.get("RSP", pd.Series(index=idx, dtype=float))
    xly = df.get("XLY", pd.Series(index=idx, dtype=float))
    xlp = df.get("XLP", pd.Series(index=idx, dtype=float))
    hyg = df.get("HYG", pd.Series(index=idx, dtype=float))
    lqd = df.get("LQD", pd.Series(index=idx, dtype=float))

    vix = df.get("^VIX", pd.Series(index=idx, dtype=float))
    vix9 = df.get("^VIX9D", pd.Series(index=idx, dtype=float))
    vix3m = df.get("^VIX3M", pd.Series(index=idx, dtype=float))
    vvix = df.get("^VVIX", pd.Series(index=idx, dtype=float))

    tgt = df.get(target, pd.Series(index=idx, dtype=float))

    # Credit
    credit = safe_ratio(hyg, lqd)
    credit_ma200 = rolling_ma(credit, 200)
    credit_ma20 = rolling_ma(credit, 20)
    credit_rollover = (credit < credit_ma200) & (credit_ma20 < rolling_ma(credit_ma200, 20))

    # Breadth
    rsp_spy = safe_ratio(rsp, spy)
    rsp_ma200 = rolling_ma(rsp_spy, 200)
    rsp_ma20 = rolling_ma(rsp_spy, 20)
    breadth_rollover = (rsp_spy < rsp_ma200) & (rsp_ma20 < rolling_ma(rsp_ma200, 20))

    # Defensive tape
    xly_xlp = safe_ratio(xly, xlp)
    xlyxlp_ma200 = rolling_ma(xly_xlp, 200)
    defensive = xly_xlp < xlyxlp_ma200

    # Vol stress
    vix_ma50 = rolling_ma(vix, 50)
    vol_1 = (vix > vix_ma50) & (vix.diff(10) > 0)
    vol_2 = (safe_ratio(vix9, vix) >= 1.00) if vix9.notna().sum() > 50 else pd.Series(False, index=idx)
    vol_3 = (safe_ratio(vix, vix3m) >= 1.00) if vix3m.notna().sum() > 50 else pd.Series(False, index=idx)
    vol_4 = (vvix >= vvix.rolling(252, min_periods=126).quantile(0.70)) if vvix.notna().sum() > 100 else pd.Series(False, index=idx)
    vol_stress = (vol_1 | vol_2 | vol_3) & (vol_4 | (vix >= 18))

    # RSI/MACD multi-timeframe
    rsi_d = rsi(tgt, 14)
    macd_d = macd_hist(tgt)

    w = resample_last(tgt, "W-FRI")
    m = resample_last(tgt, "M")

    rsi_w = mtf_to_daily(idx, rsi(w, 14))
    rsi_m = mtf_to_daily(idx, rsi(m, 14))
    macd_w = mtf_to_daily(idx, macd_hist(w))
    macd_m = mtf_to_daily(idx, macd_hist(m))

    rsi_d_roll = ((rsi_d.shift(1) >= 70) & (rsi_d < 70)) | ((rsi_d >= 68) & (rsi_d.diff(5) < 0))
    rsi_w_roll = ((rsi_w.shift(1) >= 65) & (rsi_w < 65)) | ((rsi_w >= 62) & (rsi_w.diff(3) < 0))
    rsi_m_roll = ((rsi_m.shift(1) >= 60) & (rsi_m < 60)) | ((rsi_m >= 58) & (rsi_m.diff(2) < 0))

    macd_d_bear = (macd_d < 0) & (macd_d.shift(5) > macd_d)
    macd_w_bear = (macd_w < 0) & (macd_w.shift(3) > macd_w)
    macd_m_bear = (macd_m < 0) & (macd_m.shift(2) > macd_m)

    rsi_votes = (rsi_d_roll.astype(int) + rsi_w_roll.astype(int) + rsi_m_roll.astype(int))
    macd_votes = (macd_d_bear.astype(int) + macd_w_bear.astype(int) + macd_m_bear.astype(int))
    mtf_momentum = (rsi_votes >= 2) | (macd_votes >= 2) | ((rsi_votes >= 1) & (macd_votes >= 1))

    # Trend confirmation
    ma50 = rolling_ma(tgt, 50)
    ma200 = rolling_ma(tgt, 200)
    trend_confirm = (tgt < ma50) | (tgt < ma200)

    cond = {
        "credit_risk": credit_rollover,
        "breadth_risk": breadth_rollover,
        "defensive_tape": defensive,
        "vol_stress": vol_stress,
        "mtf_momentum": mtf_momentum,
        "trend_confirm": trend_confirm,
    }

    denom = 0
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is not None and s.notna().sum() > 200:
            denom += c.weight

    return cond, max(denom, 1)

def compute_score(df: pd.DataFrame, target: str) -> pd.Series:
    cond, denom = compute_components(df, target)
    score = pd.Series(0.0, index=df.index)
    for c in COMPONENTS:
        s = cond.get(c.key)
        if s is None or s.notna().sum() <= 200:
            continue
        score = score.add(s.astype(float) * c.weight, fill_value=0.0)
    score = (score / denom) * 100.0
    return score.clip(0, 100)

def calibrate_threshold(score_spy: pd.Series, score_qqq: pd.Series, px_spy: pd.Series, px_qqq: pd.Series) -> int:
    s0 = pd.to_datetime(CALIBRATION_START)
    best_t, best_obj = 65, -1e9

    for t in range(50, 81):
        objs = []
        for score, px in [(score_spy, px_spy), (score_qqq, px_qqq)]:
            sc = score.loc[score.index >= s0].dropna()
            pr = px.loc[px.index >= s0].dropna()
            idx = sc.index.intersection(pr.index)
            sc = sc.reindex(idx)
            pr = pr.reindex(idx)
            if len(sc) < 800:
                continue

            sig = sc >= t
            rate = float(sig.mean())

            fwd_min = forward_min_return(pr, HORIZON_DAYS).reindex(idx)
            hit = fwd_min[sig].dropna()
            if hit.shape[0] < 20:
                continue

            avg_worst = float(hit.mean())  # negative
            obj = (-avg_worst * 100.0) - (rate * 100.0 * 0.9)

            if rate < 0.03:
                obj -= 2.5
            if rate > 0.22:
                obj -= 3.5

            objs.append(obj)

        if not objs:
            continue
        obj_avg = float(np.mean(objs))
        if obj_avg > best_obj:
            best_obj = obj_avg
            best_t = t

    return int(best_t)

def stance_from_score(x: float, t_short: int) -> Tuple[str, str]:
    t_bias = max(40, t_short - 12)
    if x >= t_short:
        return "SHORT ALLOWED", "b_bad"
    if x >= t_bias:
        return "HEDGE BIAS", "b_mid"
    return "STAND DOWN", "b_good"

def pick_target_today(df: pd.DataFrame) -> str:
    rs = safe_ratio(df["QQQ"], df["SPY"]).dropna()
    if len(rs) < 260:
        return "SPY"
    rs_ma200 = rolling_ma(rs, 200)
    rs_ma20 = rolling_ma(rs, 20)
    if (last_valid(rs) < last_valid(rs_ma200)) and (last_valid(rs_ma20) < last_valid(rolling_ma(rs_ma200, 20))):
        return "QQQ"
    return "SPY"

def fig_main(px: pd.Series, score: pd.Series, t_short: int, episodes: List[Tuple[pd.Timestamp, pd.Timestamp, float]], title: str) -> go.Figure:
    s = px.dropna()
    sc = score.reindex(s.index).dropna()
    idx = s.index.intersection(sc.index)
    s = s.reindex(idx)
    sc = sc.reindex(idx)

    if len(s) > CHART_WINDOW_DAYS:
        s = s.iloc[-CHART_WINDOW_DAYS:]
        sc = sc.reindex(s.index)

    ma50 = rolling_ma(s, 50)
    ma200 = rolling_ma(s, 200)
    t_bias = max(40, t_short - 12)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.70, 0.30]
    )

    fig.add_trace(go.Scatter(x=s.index, y=s, name="Price", mode="lines", line=dict(width=2.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50", mode="lines", line=dict(width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA200", mode="lines", line=dict(width=1.3)), row=1, col=1)

    sig_idx = sc.index[sc >= t_short]
    if len(sig_idx):
        fig.add_trace(
            go.Scatter(
                x=sig_idx, y=s.reindex(sig_idx), name="Short signal",
                mode="markers", marker=dict(symbol="triangle-down", size=10),
                hovertemplate="Short allowed<br>%{x|%Y-%m-%d}<extra></extra>",
            ),
            row=1, col=1,
        )

    for start, trough, depth in episodes[:10]:
        x0, x1 = start, trough
        if x1 < s.index.min() or x0 > s.index.max():
            continue
        fig.add_vrect(x0=max(x0, s.index.min()), x1=min(x1, s.index.max()), opacity=0.10, line_width=0)

    fig.add_trace(go.Scatter(x=sc.index, y=sc, name="Score", mode="lines", line=dict(width=2.0)), row=2, col=1)
    fig.add_hline(y=t_short, line_width=1, opacity=0.55, row=2, col=1)
    fig.add_hline(y=t_bias, line_width=1, opacity=0.35, row=2, col=1)

    fig.update_layout(
        title=title,
        height=640,
        margin=dict(l=10, r=10, t=55, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1, range=[0, 100], title_text="Score (0-100)")
    return fig

def lead_before_episode(score: pd.Series, start_ts: pd.Timestamp, t_short: int, lookback: int) -> int:
    s = score.dropna()
    if len(s) == 0:
        return -1
    if start_ts not in s.index:
        loc = s.index.get_indexer([start_ts], method="nearest")[0]
        start_ts = s.index[loc]
    loc = s.index.get_loc(start_ts)
    lo = max(0, loc - lookback)
    window = s.iloc[lo:loc + 1]
    hits = window[window >= t_short]
    if hits.empty:
        return -1
    first = hits.index[0]
    return int(loc - s.index.get_loc(first))

def forward_stats(score: pd.Series, px: pd.Series, t_short: int) -> Dict[str, float]:
    idx = score.dropna().index.intersection(px.dropna().index)
    sc = score.reindex(idx)
    pr = px.reindex(idx)

    sig = sc >= t_short
    fwd_min = forward_min_return(pr, HORIZON_DAYS).reindex(idx)
    hit = fwd_min[sig].dropna()
    miss = fwd_min[~sig].dropna()

    def q(x: pd.Series, p: float) -> float:
        return float(np.nanquantile(x.values, p)) if len(x) else float("nan")

    return {
        "signal_rate": float(sig.mean()),
        "signals": float(sig.sum()),
        "avg_worst_signal": float(hit.mean()) if len(hit) else float("nan"),
        "med_worst_signal": q(hit, 0.50),
        "avg_worst_nosig": float(miss.mean()) if len(miss) else float("nan"),
        "med_worst_nosig": q(miss, 0.50),
    }


# ============================== App ==============================
st.title("Hedge Timer")
st.caption("Decision slip for shorting SPY or QQQ based on early-warning stress plus multi-timeframe momentum and trend confirmation.")

start = _start_date()
raw = yf_download(TICKERS, start)
df = extract_close(raw, TICKERS)

if df.empty or "SPY" not in df.columns or "QQQ" not in df.columns:
    st.error("Yahoo feed failed for SPY/QQQ. Retry later.")
    st.stop()

score_spy = compute_score(df, "SPY")
score_qqq = compute_score(df, "QQQ")
t_short = calibrate_threshold(score_spy, score_qqq, df["SPY"], df["QQQ"])
t_bias = max(40, t_short - 12)

target = pick_target_today(df)

spy_last = last_valid(df["SPY"])
qqq_last = last_valid(df["QQQ"])
vix_last = last_valid(df["^VIX"]) if "^VIX" in df.columns else float("nan")

dd_spy = last_valid(drawdown(df["SPY"]))
dd_qqq = last_valid(drawdown(df["QQQ"]))

score_today_spy = float(last_valid(score_spy))
score_today_qqq = float(last_valid(score_qqq))

stance_spy, badge_spy = stance_from_score(score_today_spy, t_short)
stance_qqq, badge_qqq = stance_from_score(score_today_qqq, t_short)

stance_target = stance_qqq if target == "QQQ" else stance_spy
badge_target = badge_qqq if target == "QQQ" else badge_spy
score_target = score_today_qqq if target == "QQQ" else score_today_spy

st.markdown(
    f"""
<div class="card">
  <div class="row">
    <div>
      <span class="badge {badge_target}">{stance_target}</span>
      <span class="small" style="margin-left:10px;">Decision target: <b>{target}</b></span>
    </div>
    <div class="small">Score: <b>{int(round(score_target))}/100</b> | Short threshold: <b>{t_short}</b> | Hedge bias: <b>{t_bias}</b></div>
  </div>

  <div class="kv">
    <div>SPY: <b>{fmt_num(spy_last,2)}</b> | DD: <b>{fmt_pct(dd_spy)}</b> | Stance: <b>{stance_spy}</b></div>
    <div>QQQ: <b>{fmt_num(qqq_last,2)}</b> | DD: <b>{fmt_pct(dd_qqq)}</b> | Stance: <b>{stance_qqq}</b></div>
    <div>VIX: <b>{fmt_num(vix_last,2)}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("How to execute")
    if stance_target == "SHORT ALLOWED":
        st.write(
            f"Run the short in {target}. Cover discipline: if the score falls below {t_bias} and stays there for 3 sessions, start reducing. If it falls below {t_bias - 5} quickly, cover fast."
        )
    elif stance_target == "HEDGE BIAS":
        st.write(
            f"Keep it smaller. If you need protection, use a partial short or options. Wait for score to clear {t_short} for full permission."
        )
    else:
        st.write("Stand down. If you hedge, keep it light and tactical.")
with c2:
    st.subheader("What flips it")
    st.write(
        f"Above {t_short}: short allowed. Between {t_bias} and {t_short}: hedge bias. Below {t_bias}: stand down."
    )

st.markdown("<hr/>", unsafe_allow_html=True)

episodes_target = find_drawdown_episodes(df[target], threshold=-0.08, recovery=-0.02)

st.plotly_chart(
    fig_main(
        px=df[target],
        score=score_qqq if target == "QQQ" else score_spy,
        t_short=t_short,
        episodes=episodes_target,
        title=f"{target}: price, moving averages, short signals, and score",
    ),
    use_container_width=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Sanity check since 2020")

left, right = st.columns([1, 1], gap="large")

with left:
    stats_spy = forward_stats(score_spy[score_spy.index >= CALIBRATION_START], df["SPY"][df.index >= CALIBRATION_START], t_short)
    stats_qqq = forward_stats(score_qqq[score_qqq.index >= CALIBRATION_START], df["QQQ"][df.index >= CALIBRATION_START], t_short)

    st.markdown(
        f"""
<div class="card">
  <div class="small"><b>SPY</b> | signal rate: {stats_spy["signal_rate"]*100:.1f}% | signals: {int(stats_spy["signals"])}</div>
  <div class="small" style="margin-top:6px;">Avg worst next {HORIZON_DAYS}d after signal: <b>{stats_spy["avg_worst_signal"]*100:.2f}%</b> | when no signal: <b>{stats_spy["avg_worst_nosig"]*100:.2f}%</b></div>
  <div class="small" style="margin-top:6px;">Median worst next {HORIZON_DAYS}d after signal: <b>{stats_spy["med_worst_signal"]*100:.2f}%</b> | when no signal: <b>{stats_spy["med_worst_nosig"]*100:.2f}%</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="card" style="margin-top:10px;">
  <div class="small"><b>QQQ</b> | signal rate: {stats_qqq["signal_rate"]*100:.1f}% | signals: {int(stats_qqq["signals"])}</div>
  <div class="small" style="margin-top:6px;">Avg worst next {HORIZON_DAYS}d after signal: <b>{stats_qqq["avg_worst_signal"]*100:.2f}%</b> | when no signal: <b>{stats_qqq["avg_worst_nosig"]*100:.2f}%</b></div>
  <div class="small" style="margin-top:6px;">Median worst next {HORIZON_DAYS}d after signal: <b>{stats_qqq["med_worst_signal"]*100:.2f}%</b> | when no signal: <b>{stats_qqq["med_worst_nosig"]*100:.2f}%</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.write(f"Largest drawdowns and whether the score crossed {t_short} within the prior {LEAD_LOOKBACK} sessions.")

    def summarize_eps(name: str, eps: List[Tuple[pd.Timestamp, pd.Timestamp, float]], score: pd.Series) -> pd.DataFrame:
        rows = []
        for start_ts, trough_ts, depth in eps[:8]:
            lead = lead_before_episode(score, start_ts, t_short, LEAD_LOOKBACK)
            rows.append(
                {
                    "Index": name,
                    "Start": start_ts.date().isoformat(),
                    "Trough": trough_ts.date().isoformat(),
                    "Depth": f"{depth*100:.2f}%",
                    "Lead (sessions)": lead if lead >= 0 else "No",
                }
            )
        return pd.DataFrame(rows)

    tbl = pd.concat(
        [
            summarize_eps("SPY", find_drawdown_episodes(df["SPY"]), score_spy),
            summarize_eps("QQQ", find_drawdown_episodes(df["QQQ"]), score_qqq),
        ],
        ignore_index=True,
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)
