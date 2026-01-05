# app.py
# Hedge Timer
# Static (image) charts, no toggles, no sliders.
# Sidebar: About This Tool + Sanity check since 2020 (forward risk stats).
#
# Goal update:
# - Avoid "short at the bottom" by adding (1) early-stage gating, (2) oversold block for NEW signals.
# - Calibrate threshold to maximize recall of 10%+ drawdowns since 2020 across ^SPX and ^NDX.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

plt.rcParams["figure.dpi"] = 200


# ============================== Constants ==============================
# Yahoo uses ^GSPC for S&P 500. We display it as ^SPX.
SPX_TICKER = "^GSPC"
NDX_TICKER = "^NDX"
SPX_LABEL = "^SPX"
NDX_LABEL = "^NDX"

CALIBRATION_START = "2020-01-01"
DISPLAY_SESSIONS = 252  # ~12 months of trading sessions
HORIZON_DAYS = 20
LEAD_LOOKBACK = 40

# Drawdown definition for "major selloffs"
DD_MAJOR = -0.10

# Prevent "short at the bottom" by only allowing NEW signals while drawdown-from-recent-high is still early.
# -12% is intentionally loose so fast breaks still qualify early; beyond that, we do not allow fresh shorts.
EARLY_STAGE_DD63 = -0.12

# Oversold block for NEW signals (do not initiate new shorts in washed-out tape)
RSI_OVERSOLD = 30.0
RSI_SOFT_OVERSOLD = 35.0

# Core inputs (targets + risk layers). Targets are indices. Ratios use liquid ETFs.
TICKERS = [
    SPX_TICKER,
    NDX_TICKER,
    "SPY",
    "RSP",
    "XLY",
    "XLP",
    "HYG",
    "LQD",
    "TLT",
    "^VIX",
    "^VIX9D",
    "^VIX3M",
    "^VVIX",
]


# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        f"""
Daily metrics: credit and breadth ratios (HYG/LQD, RSP/SPY, XLY/XLP), volatility stress and term structure (VIX, VIX9D, VIX3M, VVIX when available), multi-timeframe RSI and MACD (daily, weekly, monthly), short-term trend breaks, and longer trend confirmation on the target index.

Decision output is a composite score (0 to 100). Thresholds are calibrated on the full sample since {CALIBRATION_START} with an explicit objective: maximize recall of 10%+ drawdowns while penalizing over-trading and late signals.

Execution constraint: the model only allows NEW short signals in an "early-stage" window (drawdown from 63-session high above {EARLY_STAGE_DD63*100:.0f}%) and blocks NEW shorts in oversold/bounce conditions (RSI14 < {RSI_OVERSOLD:.0f} or RSI < {RSI_SOFT_OVERSOLD:.0f} and rising).

Assumptions: close-to-close data, weekly is Friday close, monthly is month-end close, no transaction costs or borrow costs, no execution model. If Yahoo misses VIX9D, VIX3M, or VVIX, the model de-weights those layers inside the vol block.
""".strip()
    )
    st.markdown("---")
    st.markdown("### Sanity check since 2020")
    sanity_box = st.empty()


# ============================== Helpers ==============================
def _today() -> date:
    return date.today()


def _start_date() -> date:
    # enough history to compute MA200 and multi-timeframe indicators cleanly
    return _today() - timedelta(days=int(10 * 365.25) + 180)


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

    # single ticker fallback
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


def rolling_ma(s: pd.Series, w: int, minp: int = 1) -> pd.Series:
    return s.rolling(w, min_periods=minp).mean()


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


def drawdown(px: pd.Series) -> pd.Series:
    s = px.dropna()
    peak = s.cummax()
    return s / peak - 1.0


def dd_from_rolling_high(px: pd.Series, window: int) -> pd.Series:
    s = px.dropna()
    hi = s.rolling(window, min_periods=max(20, window // 4)).max()
    out = (s / hi) - 1.0
    return out.reindex(px.index)


def forward_min_return(px: pd.Series, h: int) -> pd.Series:
    s = px.dropna()
    fut_min = s[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return fut_min / s - 1.0


def find_drawdown_episodes(
    px: pd.Series,
    threshold: float,
    recovery: float = -0.02,
    start_after: str | None = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """
    Episodes:
      start: first date DD <= threshold
      end: first date after start where DD >= recovery (or last date)
      trough: min DD date between start and end
      depth: min DD between start and end
    """
    s = px.dropna()
    if start_after is not None:
        s = s.loc[s.index >= pd.Timestamp(start_after)]
    if len(s) < 260:
        return []

    dd = drawdown(s)

    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
    in_ep = False
    start = None

    for ts, v in dd.items():
        if (not in_ep) and (v <= threshold):
            in_ep = True
            start = ts
            continue

        if in_ep and (v >= recovery) and start is not None:
            seg = dd.loc[start:ts]
            trough = seg.idxmin()
            depth = float(seg.min())
            end = ts
            episodes.append((start, end, trough, depth))
            in_ep = False
            start = None

    if in_ep and start is not None:
        seg = dd.loc[start:]
        trough = seg.idxmin()
        depth = float(seg.min())
        end = seg.index[-1]
        episodes.append((start, end, trough, depth))

    return sorted(episodes, key=lambda x: x[3])  # deepest first


# ============================== Signal Model ==============================
@dataclass
class Component:
    key: str
    label: str
    weight: int


COMPONENTS: List[Component] = [
    Component("credit_risk", "Credit risk-off", 16),
    Component("breadth_risk", "Breadth weak", 12),
    Component("defensive_tape", "Defensive tape", 8),
    Component("vol_stress", "Vol stress", 18),
    Component("mtf_momentum", "RSI/MACD rollover (D/W/M)", 24),
    Component("short_term_break", "Short-term break", 8),
    Component("trend_confirm", "Trend confirms", 14),
]


def compute_components_and_meta(
    df: pd.DataFrame, target_ticker: str
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], int]:
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

    tgt = df.get(target_ticker, pd.Series(index=idx, dtype=float))

    # Credit: HYG/LQD below long MA and deteriorating
    credit = safe_ratio(hyg, lqd)
    credit_ma200 = rolling_ma(credit, 200)
    credit_rollover = (credit < credit_ma200) & (credit.diff(30) < 0)

    # Breadth: RSP/SPY below long MA and deteriorating
    rsp_spy = safe_ratio(rsp, spy)
    rsp_ma200 = rolling_ma(rsp_spy, 200)
    breadth_rollover = (rsp_spy < rsp_ma200) & (rsp_spy.diff(30) < 0)

    # Defensive tape: XLY/XLP below long MA and deteriorating
    xly_xlp = safe_ratio(xly, xlp)
    xlyxlp_ma200 = rolling_ma(xly_xlp, 200)
    defensive = (xly_xlp < xlyxlp_ma200) & (xly_xlp.diff(30) < 0)

    # Vol stress: level, term structure, and VVIX tail
    vix_ma50 = rolling_ma(vix, 50)
    vol_level = (vix > vix_ma50) & (vix.diff(10) > 0)

    has_vix9 = vix9.notna().sum() > 50
    has_vix3m = vix3m.notna().sum() > 50
    has_vvix = vvix.notna().sum() > 100

    vol_term_front = (safe_ratio(vix9, vix) >= 1.00) if has_vix9 else pd.Series(False, index=idx)
    vol_term_back = (safe_ratio(vix, vix3m) >= 1.00) if has_vix3m else pd.Series(False, index=idx)
    vvix_tail = (vvix >= vvix.rolling(252, min_periods=126).quantile(0.70)) if has_vvix else pd.Series(False, index=idx)

    vol_stress = (vol_level | vol_term_front | vol_term_back) & (vvix_tail | (vix >= 18))

    # Multi-timeframe RSI/MACD
    rsi_d = rsi(tgt, 14)
    macd_d = macd_hist(tgt)

    w = resample_last(tgt, "W-FRI")
    m = resample_last(tgt, "M")

    rsi_w = mtf_to_daily(idx, rsi(w, 14))
    rsi_m = mtf_to_daily(idx, rsi(m, 14))
    macd_w = mtf_to_daily(idx, macd_hist(w))
    macd_m = mtf_to_daily(idx, macd_hist(m))

    # We want rollover-from-strength, not "already nuked".
    # Daily: prefer crossing down from >= 60 or a clear loss of momentum while still > 45.
    rsi_d_roll = ((rsi_d.shift(1) >= 60) & (rsi_d < 60)) | (
        (rsi_d >= 45)
        & (rsi_d.diff(5) < 0)
        & (rsi_d < rsi_d.rolling(10, min_periods=5).max() - 2)
    )
    rsi_w_roll = ((rsi_w.shift(1) >= 58) & (rsi_w < 58)) | ((rsi_w >= 48) & (rsi_w.diff(3) < 0))
    rsi_m_roll = ((rsi_m.shift(1) >= 55) & (rsi_m < 55)) | ((rsi_m >= 50) & (rsi_m.diff(2) < 0))

    # MACD: look for histogram turning down, especially from positive to negative or steep deterioration.
    macd_d_bear = ((macd_d.shift(1) > 0) & (macd_d < 0)) | ((macd_d < macd_d.shift(3)) & (macd_d.diff(3) < 0))
    macd_w_bear = ((macd_w.shift(1) > 0) & (macd_w < 0)) | ((macd_w < macd_w.shift(2)) & (macd_w.diff(2) < 0))
    macd_m_bear = ((macd_m.shift(1) > 0) & (macd_m < 0)) | ((macd_m < macd_m.shift(2)) & (macd_m.diff(2) < 0))

    rsi_votes = (
        rsi_d_roll.fillna(False).astype(int)
        + rsi_w_roll.fillna(False).astype(int)
        + rsi_m_roll.fillna(False).astype(int)
    )
    macd_votes = (
        macd_d_bear.fillna(False).astype(int)
        + macd_w_bear.fillna(False).astype(int)
        + macd_m_bear.fillna(False).astype(int)
    )

    # Momentum block: require at least one timeframe rolling from strength, and a second confirm.
    mtf_momentum = (rsi_votes >= 2) | (macd_votes >= 2) | ((rsi_votes >= 1) & (macd_votes >= 1))

    # Short-term break: fast structure break
    ema9 = ema(tgt, 9)
    ema21 = ema(tgt, 21)
    short_term_break = ((tgt < ema21) & (ema21.diff(10) < 0)) | ((tgt < ema9) & (ema9 < ema21))

    # Trend confirmation: longer trend and slope
    ma50 = rolling_ma(tgt, 50)
    ma200 = rolling_ma(tgt, 200)
    trend_confirm = ((tgt < ma50) & (ma50.diff(20) < 0)) | (tgt < ma200)

    cond = {
        "credit_risk": credit_rollover,
        "breadth_risk": breadth_rollover,
        "defensive_tape": defensive,
        "vol_stress": vol_stress,
        "mtf_momentum": mtf_momentum,
        "short_term_break": short_term_break,
        "trend_confirm": trend_confirm,
    }

    # Meta for gating / reporting
    dd63 = dd_from_rolling_high(tgt, 63)
    oversold_block = (rsi_d < RSI_OVERSOLD) | ((rsi_d < RSI_SOFT_OVERSOLD) & (rsi_d.diff(5) > 0))
    early_stage = dd63 > EARLY_STAGE_DD63

    meta = {
        "rsi_d": rsi_d,
        "dd63": dd63,
        "oversold_block": oversold_block,
        "early_stage": early_stage,
        "ma50": ma50,
        "ma200": ma200,
    }

    # Fill gaps as False to keep every trading session in the score series.
    cond = {k: v.reindex(idx).fillna(False).astype(bool) for k, v in cond.items()}
    meta = {k: v.reindex(idx) for k, v in meta.items()}

    denom = sum(c.weight for c in COMPONENTS)
    return cond, meta, max(denom, 1)


def compute_score_and_meta(df: pd.DataFrame, target_ticker: str) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    cond, meta, denom = compute_components_and_meta(df, target_ticker)
    score = pd.Series(0.0, index=df.index)
    for c in COMPONENTS:
        score = score.add(cond[c.key].astype(float) * c.weight, fill_value=0.0)
    score = (score / denom) * 100.0
    return score.clip(0, 100).fillna(0.0), meta


def stance_from_score(x: float, t_short: int) -> Tuple[str, str]:
    t_bias = max(40, t_short - 12)
    if x >= t_short:
        return "SHORT ALLOWED", "b_bad"
    if x >= t_bias:
        return "HEDGE BIAS", "b_mid"
    return "STAND DOWN", "b_good"


def pick_target_today(df: pd.DataFrame) -> str:
    spx = df[SPX_TICKER].dropna()
    ndx = df[NDX_TICKER].dropna()
    idx = spx.index.intersection(ndx.index)
    if len(idx) < 260:
        return SPX_TICKER

    rs = (ndx.reindex(idx) / spx.reindex(idx)).dropna()
    rs_ma200 = rolling_ma(rs, 200)
    rs_ma20 = rolling_ma(rs, 20)

    # If NDX is in persistent relative downtrend, hedge NDX. Else hedge SPX.
    if (last_valid(rs) < last_valid(rs_ma200)) and (last_valid(rs_ma20) < last_valid(rolling_ma(rs_ma200, 20))):
        return NDX_TICKER
    return SPX_TICKER


def signal_series(score: pd.Series, meta: Dict[str, pd.Series], t_short: int) -> pd.Series:
    """
    Full "permission" series (can be True deep into a selloff).
    New-signal gating (early-stage + not oversold) is applied when we plot/score 'signal_on'.
    """
    idx = score.index
    s = (score >= t_short).reindex(idx).fillna(False)
    return s.astype(bool)


def signal_onset(score: pd.Series, meta: Dict[str, pd.Series], t_short: int) -> pd.Series:
    """
    NEW signal only if:
      - score >= threshold
      - early_stage True (dd from 63-session high > -12%)
      - oversold_block False
    Once signaled, it can remain "allowed" even if tape falls further; we only care about onset markers.
    """
    idx = score.index
    base = signal_series(score, meta, t_short)

    early = meta.get("early_stage", pd.Series(True, index=idx)).reindex(idx).fillna(True).astype(bool)
    oversold = meta.get("oversold_block", pd.Series(False, index=idx)).reindex(idx).fillna(False).astype(bool)

    new_allowed = base & early & (~oversold)
    on = new_allowed & (~new_allowed.shift(1).fillna(False))
    return on.astype(bool)


def forward_stats(score: pd.Series, px: pd.Series, meta: Dict[str, pd.Series], t_short: int) -> Dict[str, float]:
    idx = score.index.intersection(px.index)
    sc = score.reindex(idx)
    pr = px.reindex(idx)

    sig_on = signal_onset(sc, {k: v.reindex(idx) for k, v in meta.items()}, t_short)
    fwd_min = forward_min_return(pr, HORIZON_DAYS).reindex(idx)

    hit = fwd_min[sig_on].dropna()
    miss = fwd_min[~sig_on].dropna()

    def q(x: pd.Series, p: float) -> float:
        return float(np.nanquantile(x.values, p)) if len(x) else float("nan")

    return {
        "signal_rate": float(sig_on.mean()),
        "signals": float(sig_on.sum()),
        "avg_worst_signal": float(hit.mean()) if len(hit) else float("nan"),
        "med_worst_signal": q(hit, 0.50),
        "avg_worst_nosig": float(miss.mean()) if len(miss) else float("nan"),
        "med_worst_nosig": q(miss, 0.50),
    }


def lead_before_episode_onset(sig_on: pd.Series, start_ts: pd.Timestamp, lookback: int) -> int:
    s = sig_on.dropna()
    if len(s) == 0:
        return -1

    if start_ts not in s.index:
        loc = s.index.get_indexer([start_ts], method="nearest")[0]
        start_ts = s.index[loc]

    loc = s.index.get_loc(start_ts)
    lo = max(0, loc - lookback)
    window = s.iloc[lo : loc + 1]

    hits = window[window]
    if hits.empty:
        return -1

    first = hits.index[0]
    return int(loc - s.index.get_loc(first))


def episode_coverage_obj(
    px: pd.Series,
    sig_on: pd.Series,
    threshold: float,
    start_after: str,
    lookback: int,
) -> Tuple[float, int]:
    eps = find_drawdown_episodes(px, threshold=threshold, recovery=-0.02, start_after=start_after)
    if not eps:
        return 0.0, 0
    covered = 0
    for start_ts, _, _, _ in eps:
        lead = lead_before_episode_onset(sig_on, start_ts, lookback)
        if lead >= 0:
            covered += 1
    return covered / max(len(eps), 1), len(eps)


def calibrate_threshold(
    score_spx: pd.Series,
    meta_spx: Dict[str, pd.Series],
    px_spx: pd.Series,
    score_ndx: pd.Series,
    meta_ndx: Dict[str, pd.Series],
    px_ndx: pd.Series,
) -> int:
    s0 = pd.to_datetime(CALIBRATION_START)

    best_t, best_obj = 68, -1e18

    for t in range(55, 86):
        sc_a = score_spx.loc[score_spx.index >= s0].dropna()
        pr_a = px_spx.loc[px_spx.index >= s0].dropna()
        idx_a = sc_a.index.intersection(pr_a.index)
        sc_a = sc_a.reindex(idx_a)
        pr_a = pr_a.reindex(idx_a)
        meta_a = {k: v.reindex(idx_a) for k, v in meta_spx.items()}

        sc_b = score_ndx.loc[score_ndx.index >= s0].dropna()
        pr_b = px_ndx.loc[px_ndx.index >= s0].dropna()
        idx_b = sc_b.index.intersection(pr_b.index)
        sc_b = sc_b.reindex(idx_b)
        pr_b = pr_b.reindex(idx_b)
        meta_b = {k: v.reindex(idx_b) for k, v in meta_ndx.items()}

        if len(sc_a) < 800 or len(sc_b) < 800:
            continue

        sig_on_a = signal_onset(sc_a, meta_a, t)
        sig_on_b = signal_onset(sc_b, meta_b, t)

        # Coverage of 10%+ drawdowns
        cov_a, n_a = episode_coverage_obj(pr_a, sig_on_a, DD_MAJOR, CALIBRATION_START, LEAD_LOOKBACK)
        cov_b, n_b = episode_coverage_obj(pr_b, sig_on_b, DD_MAJOR, CALIBRATION_START, LEAD_LOOKBACK)
        cov = 0.5 * (cov_a + cov_b)

        # Forward risk usefulness (worst next N sessions AFTER a new signal)
        fwd_a = forward_min_return(pr_a, HORIZON_DAYS).reindex(idx_a)
        fwd_b = forward_min_return(pr_b, HORIZON_DAYS).reindex(idx_b)
        hit_a = fwd_a[sig_on_a].dropna()
        hit_b = fwd_b[sig_on_b].dropna()
        if hit_a.shape[0] < 12 or hit_b.shape[0] < 12:
            continue

        avg_worst = 0.5 * (float(hit_a.mean()) + float(hit_b.mean()))  # negative is good (warned before pain)

        # Penalties: over-trading and "late"
        rate = 0.5 * (float(sig_on_a.mean()) + float(sig_on_b.mean()))
        dd63_a = meta_a.get("dd63", pd.Series(index=idx_a, dtype=float))
        dd63_b = meta_b.get("dd63", pd.Series(index=idx_b, dtype=float))
        late_a = float((dd63_a[sig_on_a] <= EARLY_STAGE_DD63).mean()) if sig_on_a.sum() else 0.0
        late_b = float((dd63_b[sig_on_b] <= EARLY_STAGE_DD63).mean()) if sig_on_b.sum() else 0.0
        late = 0.5 * (late_a + late_b)

        # Objective:
        # - Primary: maximize coverage of 10%+ drawdowns
        # - Secondary: reward negative avg_worst (signals that precede further downside)
        # - Penalize signal rate and any late onsets
        obj = (
            cov * 1000.0
            + (-avg_worst * 100.0) * 3.0
            - rate * 300.0
            - late * 400.0
        )

        # Keep rate in a sane band; hard penalties if wildly off
        if rate < 0.02:
            obj -= 120.0
        if rate > 0.20:
            obj -= 180.0

        # Prefer solutions with non-trivial episode counts present
        if (n_a + n_b) < 6:
            obj -= 80.0

        if obj > best_obj:
            best_obj = obj
            best_t = t

    return int(best_t)


# ============================== Static Chart (rolling 12 months, no calendar gaps) ==============================
def _display_name(ticker: str) -> str:
    if ticker == SPX_TICKER:
        return SPX_LABEL
    if ticker == NDX_TICKER:
        return NDX_LABEL
    return ticker


def _apply_subtle_grid(ax: plt.Axes, y_only: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="y" if y_only else "both",
        linestyle="-",
        linewidth=0.7,
        alpha=0.16,
    )
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_alpha(0.30)
    ax.spines["bottom"].set_alpha(0.30)


def plot_price_and_score_image(
    price: pd.Series,
    score: pd.Series,
    meta: Dict[str, pd.Series],
    t_short: int,
    title_prefix: str,
) -> plt.Figure:
    dfp = pd.DataFrame({"price": price, "score": score}).copy()
    dfp = dfp.dropna(subset=["price"])
    if dfp.empty:
        fig = plt.figure(figsize=(13, 6))
        plt.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    # Rolling 12 months by sessions
    if len(dfp) > DISPLAY_SESSIONS:
        dfp = dfp.iloc[-DISPLAY_SESSIONS:].copy()

    x = np.arange(len(dfp))
    idx = dfp.index

    ma50 = meta.get("ma50", rolling_ma(price, 50)).reindex(idx)
    ma200 = meta.get("ma200", rolling_ma(price, 200)).reindex(idx)

    t_bias = max(40, t_short - 12)

    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.25], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Price + MAs (explicit colors to avoid default blue dominance)
    price_c = "#111827"   # near-black
    ma50_c = "#6B7280"    # mid gray
    ma200_c = "#9CA3AF"   # light gray

    ax1.plot(x, dfp["price"].values, linewidth=2.3, color=price_c, label="Price")
    ax1.plot(x, ma50.values, linewidth=1.5, color=ma50_c, label="MA50")
    ax1.plot(x, ma200.values, linewidth=1.5, color=ma200_c, label="MA200")

    pmin = float(np.nanmin(dfp["price"].values))
    pmax = float(np.nanmax(dfp["price"].values))
    pad = (pmax - pmin) * 0.04 if pmax > pmin else 1.0
    ax1.set_ylim(pmin - pad, pmax + pad)
    ax1.set_xlim(-0.5, len(dfp) - 0.5)

    # NEW short signal onsets (early-stage + not oversold) in RED
    sig_on = signal_onset(score.reindex(idx), {k: v.reindex(idx) for k, v in meta.items()}, t_short)
    if sig_on.any():
        ax1.scatter(
            x[sig_on.values],
            dfp["price"].values[sig_on.values],
            marker="v",
            s=72,
            color="#DC2626",
            edgecolors="white",
            linewidths=0.8,
            label="Short signal (new)",
            zorder=6,
        )

    _apply_subtle_grid(ax1, y_only=False)
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Score panel (explicit styling; no blue dashed artifacts)
    score_c = "#374151"
    t_short_c = "#111827"
    t_bias_c = "#9CA3AF"

    ax2.plot(x, dfp["score"].fillna(0.0).values, linewidth=1.9, color=score_c, label="Score")
    ax2.axhline(t_short, linewidth=1.1, color=t_short_c, alpha=0.70)
    ax2.axhline(t_bias, linewidth=1.0, color=t_bias_c, alpha=0.55)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Score (0-100)")
    _apply_subtle_grid(ax2, y_only=True)
    ax2.set_xlim(-0.5, len(dfp) - 0.5)

    # Month ticks
    months = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="MS")
    tick_pos, tick_lbl = [], []
    for d in months:
        loc = idx.get_indexer([d], method="nearest")[0]
        if 0 <= loc < len(idx):
            if not tick_pos or loc - tick_pos[-1] >= 18:
                tick_pos.append(int(loc))
                tick_lbl.append(d.strftime("%b %Y"))
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl, rotation=0, ha="center")

    fig.suptitle(f"{title_prefix} (last 12 months, no calendar gaps)", fontsize=14, fontweight="bold", y=0.985)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    return fig


def plot_episode_table_image(table_df: pd.DataFrame, title: str) -> plt.Figure:
    dfp = table_df.copy()
    if dfp.empty:
        fig = plt.figure(figsize=(13, 2.2))
        plt.text(0.5, 0.5, "No episodes found", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(13.6, 5.2))
    ax.axis("off")

    # Keep title close to the table (reduce top padding)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)

    cell_text = dfp.values.tolist()
    col_labels = dfp.columns.tolist()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.02, 1.0, 0.90],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.18)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor((0, 0, 0, 0.08))
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor((0.96, 0.96, 0.96, 1.0))

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.965])
    return fig


# ============================== App ==============================
st.title("Hedge Timer")
st.caption("Decision slip for shorting ^SPX or ^NDX based on early-warning stress plus multi-timeframe momentum and trend confirmation, with explicit early-stage gating to avoid bottom-shorting.")

start = _start_date()
raw = yf_download(TICKERS, start)
df0 = extract_close(raw, TICKERS)

if df0.empty or SPX_TICKER not in df0.columns or NDX_TICKER not in df0.columns:
    st.error("Yahoo feed failed for ^SPX/^NDX. Retry later.")
    st.stop()

# Align everything to the index trading calendar to avoid missing-session gaps from ancillary series.
base_idx = df0[SPX_TICKER].dropna().index.intersection(df0[NDX_TICKER].dropna().index)
df = df0.reindex(base_idx).ffill()

score_spx, meta_spx = compute_score_and_meta(df, SPX_TICKER)
score_ndx, meta_ndx = compute_score_and_meta(df, NDX_TICKER)

t_short = calibrate_threshold(score_spx, meta_spx, df[SPX_TICKER], score_ndx, meta_ndx, df[NDX_TICKER])
t_bias = max(40, t_short - 12)

target = pick_target_today(df)
target_label = _display_name(target)

spx_last = last_valid(df[SPX_TICKER])
ndx_last = last_valid(df[NDX_TICKER])
vix_last = last_valid(df["^VIX"]) if "^VIX" in df.columns else float("nan")

dd_spx = last_valid(drawdown(df[SPX_TICKER]))
dd_ndx = last_valid(drawdown(df[NDX_TICKER]))

score_today_spx = float(last_valid(score_spx))
score_today_ndx = float(last_valid(score_ndx))

stance_spx, badge_spx = stance_from_score(score_today_spx, t_short)
stance_ndx, badge_ndx = stance_from_score(score_today_ndx, t_short)

stance_target = stance_ndx if target == NDX_TICKER else stance_spx
badge_target = badge_ndx if target == NDX_TICKER else badge_spx
score_target = score_today_ndx if target == NDX_TICKER else score_today_spx

# Sidebar sanity check block (below About This Tool)
stats_spx = forward_stats(
    score_spx[score_spx.index >= CALIBRATION_START],
    df[SPX_TICKER][df.index >= CALIBRATION_START],
    {k: v[v.index >= CALIBRATION_START] for k, v in meta_spx.items()},
    t_short,
)
stats_ndx = forward_stats(
    score_ndx[score_ndx.index >= CALIBRATION_START],
    df[NDX_TICKER][df.index >= CALIBRATION_START],
    {k: v[v.index >= CALIBRATION_START] for k, v in meta_ndx.items()},
    t_short,
)

sanity_box.markdown(
    f"""
Forward risk stats (next {HORIZON_DAYS} sessions). Signals are NEW-signal onsets only (early-stage gated, oversold blocked).

**{SPX_LABEL}** | signal rate: {stats_spx["signal_rate"]*100:.1f}% | signals: {int(stats_spx["signals"])}  
Avg worst next {HORIZON_DAYS}d after signal: {stats_spx["avg_worst_signal"]*100:.2f}% | when no signal: {stats_spx["avg_worst_nosig"]*100:.2f}%  
Median worst next {HORIZON_DAYS}d after signal: {stats_spx["med_worst_signal"]*100:.2f}% | when no signal: {stats_spx["med_worst_nosig"]*100:.2f}%  

**{NDX_LABEL}** | signal rate: {stats_ndx["signal_rate"]*100:.1f}% | signals: {int(stats_ndx["signals"])}  
Avg worst next {HORIZON_DAYS}d after signal: {stats_ndx["avg_worst_signal"]*100:.2f}% | when no signal: {stats_ndx["avg_worst_nosig"]*100:.2f}%  
Median worst next {HORIZON_DAYS}d after signal: {stats_ndx["med_worst_signal"]*100:.2f}% | when no signal: {stats_ndx["med_worst_nosig"]*100:.2f}%
""".strip()
)

# Extra transparency for the gating today on the decision target
meta_target = meta_ndx if target == NDX_TICKER else meta_spx
dd63_today = float(last_valid(meta_target["dd63"]))
rsi_today = float(last_valid(meta_target["rsi_d"]))
early_today = bool(last_valid(meta_target["early_stage"].astype(float)) > 0.5) if "early_stage" in meta_target else True
oversold_today = bool(last_valid(meta_target["oversold_block"].astype(float)) > 0.5) if "oversold_block" in meta_target else False

st.markdown(
    f"""
<div class="card">
  <div class="row">
    <div>
      <span class="badge {badge_target}">{stance_target}</span>
      <span class="small" style="margin-left:10px;">Decision target: <b>{target_label}</b></span>
    </div>
    <div class="small">Score: <b>{int(round(score_target))}/100</b> | Short threshold: <b>{t_short}</b> | Hedge bias: <b>{t_bias}</b></div>
  </div>

  <div class="kv">
    <div>{SPX_LABEL}: <b>{fmt_num(spx_last,2)}</b> | DD: <b>{fmt_pct(dd_spx)}</b> | Stance: <b>{stance_spx}</b></div>
    <div>{NDX_LABEL}: <b>{fmt_num(ndx_last,2)}</b> | DD: <b>{fmt_pct(dd_ndx)}</b> | Stance: <b>{stance_ndx}</b></div>
    <div>VIX: <b>{fmt_num(vix_last,2)}</b></div>
    <div>Target RSI14 (D): <b>{fmt_num(rsi_today,1)}</b> | DD from 63d high: <b>{fmt_pct(dd63_today)}</b> | Early-stage: <b>{'Yes' if early_today else 'No'}</b> | Oversold block: <b>{'Yes' if oversold_today else 'No'}</b></div>
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
            f"Run the short in {target_label}. Cover discipline: if the score falls below {t_bias} and stays there for 3 sessions, start reducing. If it falls below {t_bias - 5} quickly, cover fast."
        )
    elif stance_target == "HEDGE BIAS":
        st.write(
            f"Keep it smaller. If you need protection, use a partial short or options. Wait for score to clear {t_short} for full permission."
        )
    else:
        st.write("Stand down. If you hedge, keep it light and tactical.")
with c2:
    st.subheader("What flips it")
    st.write(f"Above {t_short}: short allowed. Between {t_bias} and {t_short}: hedge bias. Below {t_bias}: stand down.")

st.markdown("<hr/>", unsafe_allow_html=True)

score_target_series = score_ndx if target == NDX_TICKER else score_spx
meta_target_series = meta_ndx if target == NDX_TICKER else meta_spx

fig = plot_price_and_score_image(
    price=df[target],
    score=score_target_series,
    meta=meta_target_series,
    t_short=t_short,
    title_prefix=f"{target_label}: price, moving averages, short signals, and score",
)
st.pyplot(fig, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Did it warn before major selloffs?")
st.write(f"We look at 10%+ drawdowns and ask if a NEW short signal fired within the prior {LEAD_LOOKBACK} sessions. New signals are gated (early-stage) and oversold-blocked.")


def summarize_eps(name_label: str, px: pd.Series, score: pd.Series, meta: Dict[str, pd.Series]) -> pd.DataFrame:
    eps = find_drawdown_episodes(px, threshold=DD_MAJOR, recovery=-0.02, start_after=CALIBRATION_START)
    sig_on = signal_onset(score, meta, t_short)
    rows = []
    for start_ts, end_ts, trough_ts, depth in eps[:12]:
        lead = lead_before_episode_onset(sig_on, start_ts, LEAD_LOOKBACK)
        rows.append(
            {
                "Index": name_label,
                "Start": start_ts.date().isoformat(),
                "End": end_ts.date().isoformat(),
                "Trough": trough_ts.date().isoformat(),
                "Depth": f"{depth*100:.2f}%",
                "Lead (sessions)": lead if lead >= 0 else "No",
            }
        )
    return pd.DataFrame(rows)


tbl = pd.concat(
    [
        summarize_eps(SPX_LABEL, df[SPX_TICKER], score_spx, meta_spx),
        summarize_eps(NDX_LABEL, df[NDX_TICKER], score_ndx, meta_ndx),
    ],
    ignore_index=True,
)

fig_tbl = plot_episode_table_image(
    tbl,
    title="Largest drawdowns (10%+) and whether a NEW short signal fired beforehand",
)
st.pyplot(fig_tbl, use_container_width=True)

st.caption("© 2026 AD Fund Management LP")

