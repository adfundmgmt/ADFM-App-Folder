from __future__ import annotations
from dataclasses import dataclass
from datetime import date,timedelta
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
SPX_TICKER = "^GSPC"
NDX_TICKER = "^NDX"
SPX_LABEL = "^SPX"
NDX_LABEL = "^NDX"
CALIBRATION_START = "2020-01-01"
DISPLAY_SESSIONS_DEFAULT = 252
HORIZON_DAYS = 20
LEAD_LOOKBACK = 40
DD_MAJOR = -0.10
EARLY_STAGE_DD63 = -0.12
RSI_OVERSOLD = 30.0
RSI_SOFT_OVERSOLD = 35.0
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
LOOKBACK_OPTIONS = [1, 2, 3, 5, 10]
def _today() -> date:
    return date.today()

def _start_date() -> date:
    # enough history to compute MA200 and multi-timeframe indicators cleanly
    return _today() - timedelta(days=int(10 * 365.25) + 180)

def sessions_for_years(years: int) -> int:
    return int(round(252 * years))

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
    s = s.dropna().copy()
    if s.empty:
        return s

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    # pandas now expects explicit end-of-period aliases like ME/QE/YE
    freq_map = {
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "A": "YE",
        "BM": "BME",
        "BQ": "BQE",
        "BY": "BYE",
    }
    rule = freq_map.get(rule, rule)

    return s.resample(rule).last()

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

    return sorted(episodes, key=lambda x: x[3])

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
    vvix_tail = (
        vvix >= vvix.rolling(252, min_periods=126).quantile(0.70)
        if has_vvix
        else pd.Series(False, index=idx)
    )

    vol_stress = (vol_level | vol_term_front | vol_term_back) & (vvix_tail | (vix >= 18))

    # Multi-timeframe RSI/MACD
    rsi_d = rsi(tgt, 14)
    macd_d = macd_hist(tgt)

    w = resample_last(tgt, "W-FRI")
    m = resample_last(tgt, "ME")

    rsi_w = mtf_to_daily(idx, rsi(w, 14))
    rsi_m = mtf_to_daily(idx, rsi(m, 14))
    macd_w = mtf_to_daily(idx, macd_hist(w))
    macd_m = mtf_to_daily(idx, macd_hist(m))

    # We want rollover-from-strength, not "already nuked".
    rsi_d_roll = ((rsi_d.shift(1) >= 60) & (rsi_d < 60)) | (
        (rsi_d >= 45)
        & (rsi_d.diff(5) < 0)
        & (rsi_d < rsi_d.rolling(10, min_periods=5).max() - 2)
    )
    rsi_w_roll = ((rsi_w.shift(1) >= 58) & (rsi_w < 58)) | ((rsi_w >= 48) & (rsi_w.diff(3) < 0))
    rsi_m_roll = ((rsi_m.shift(1) >= 55) & (rsi_m < 55)) | ((rsi_m >= 50) & (rsi_m.diff(2) < 0))

    macd_d_bear = ((macd_d.shift(1) > 0) & (macd_d < 0)) | (
        (macd_d < macd_d.shift(3)) & (macd_d.diff(3) < 0)
    )
    macd_w_bear = ((macd_w.shift(1) > 0) & (macd_w < 0)) | (
        (macd_w < macd_w.shift(2)) & (macd_w.diff(2) < 0)
    )
    macd_m_bear = ((macd_m.shift(1) > 0) & (macd_m < 0)) | (
        (macd_m < macd_m.shift(2)) & (macd_m.diff(2) < 0)
    )

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

    if (last_valid(rs) < last_valid(rs_ma200)) and (
        last_valid(rs_ma20) < last_valid(rolling_ma(rs_ma200, 20))
    ):
        return NDX_TICKER
    return SPX_TICKER

def signal_series(score: pd.Series, meta: Dict[str, pd.Series], t_short: int) -> pd.Series:
    idx = score.index
    s = (score >= t_short).reindex(idx).fillna(False)
    return s.astype(bool)

def signal_onset(score: pd.Series, meta: Dict[str, pd.Series], t_short: int) -> pd.Series:
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

        cov_a, n_a = episode_coverage_obj(pr_a, sig_on_a, DD_MAJOR, CALIBRATION_START, LEAD_LOOKBACK)
        cov_b, n_b = episode_coverage_obj(pr_b, sig_on_b, DD_MAJOR, CALIBRATION_START, LEAD_LOOKBACK)
        cov = 0.5 * (cov_a + cov_b)

        fwd_a = forward_min_return(pr_a, HORIZON_DAYS).reindex(idx_a)
        fwd_b = forward_min_return(pr_b, HORIZON_DAYS).reindex(idx_b)
        hit_a = fwd_a[sig_on_a].dropna()
        hit_b = fwd_b[sig_on_b].dropna()
        if hit_a.shape[0] < 12 or hit_b.shape[0] < 12:
            continue

        avg_worst = 0.5 * (float(hit_a.mean()) + float(hit_b.mean()))

        rate = 0.5 * (float(sig_on_a.mean()) + float(sig_on_b.mean()))
        dd63_a = meta_a.get("dd63", pd.Series(index=idx_a, dtype=float))
        dd63_b = meta_b.get("dd63", pd.Series(index=idx_b, dtype=float))
        late_a = float((dd63_a[sig_on_a] <= EARLY_STAGE_DD63).mean()) if sig_on_a.sum() else 0.0
        late_b = float((dd63_b[sig_on_b] <= EARLY_STAGE_DD63).mean()) if sig_on_b.sum() else 0.0
        late = 0.5 * (late_a + late_b)

        obj = (
            cov * 1000.0
            + (-avg_worst * 100.0) * 3.0
            - rate * 300.0
            - late * 400.0
        )

        if rate < 0.02:
            obj -= 120.0
        if rate > 0.20:
            obj -= 180.0

        if (n_a + n_b) < 6:
            obj -= 80.0

        if obj > best_obj:
            best_obj = obj
            best_t = t

    return int(best_t)

def _display_name(ticker: str) -> str:
    if ticker == SPX_TICKER:
        return SPX_LABEL
    if ticker == NDX_TICKER:
        return NDX_LABEL
    return ticker

def tick_rule_for_years(years: int) -> str:
    if years <= 1:
        return "MS"
    if years == 2:
        return "QS"
    if years == 3:
        return "4MS"
    if years == 5:
        return "2QS"
    return "YS"

def tick_label_for_years(d: pd.Timestamp, years: int) -> str:
    if years <= 1:
        return d.strftime("%b %Y")
    if years <= 3:
        return d.strftime("%b %Y")
    return d.strftime("%Y")

def chart_style_for_years(years: int) -> Dict[str, float]:
    if years <= 1:
        return {
            "fig_w": 13.6,
            "fig_h": 7.4,
            "price_lw": 2.3,
            "ma_lw": 1.7,
            "score_lw": 2.2,
            "marker_s": 72,
            "marker_lw": 0.8,
            "title_fs": 14,
            "legend_fs": 9,
            "label_fs": 10,
            "xtick_fs": 9,
        }
    if years <= 2:
        return {
            "fig_w": 13.8,
            "fig_h": 7.5,
            "price_lw": 2.1,
            "ma_lw": 1.55,
            "score_lw": 2.0,
            "marker_s": 58,
            "marker_lw": 0.75,
            "title_fs": 14,
            "legend_fs": 9,
            "label_fs": 10,
            "xtick_fs": 9,
        }
    if years <= 3:
        return {
            "fig_w": 14.0,
            "fig_h": 7.6,
            "price_lw": 1.9,
            "ma_lw": 1.45,
            "score_lw": 1.9,
            "marker_s": 50,
            "marker_lw": 0.70,
            "title_fs": 14,
            "legend_fs": 9,
            "label_fs": 10,
            "xtick_fs": 8,
        }
    if years <= 5:
        return {
            "fig_w": 14.2,
            "fig_h": 7.8,
            "price_lw": 1.75,
            "ma_lw": 1.30,
            "score_lw": 1.75,
            "marker_s": 38,
            "marker_lw": 0.65,
            "title_fs": 14,
            "legend_fs": 8.5,
            "label_fs": 10,
            "xtick_fs": 8,
        }
    return {
        "fig_w": 14.4,
        "fig_h": 8.0,
        "price_lw": 1.55,
        "ma_lw": 1.15,
        "score_lw": 1.55,
        "marker_s": 30,
        "marker_lw": 0.60,
        "title_fs": 14,
        "legend_fs": 8.5,
        "label_fs": 10,
        "xtick_fs": 8,
    }

def summarize_eps(name_label: str, px: pd.Series, score: pd.Series, meta: Dict[str, pd.Series],t_short:int) -> pd.DataFrame:
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


def compute_hedge(df0):
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

    meta_target = meta_ndx if target == NDX_TICKER else meta_spx
    dd63_today = float(last_valid(meta_target["dd63"]))
    rsi_today = float(last_valid(meta_target["rsi_d"]))
    early_today = bool(last_valid(meta_target["early_stage"].astype(float)) > 0.5) if "early_stage" in meta_target else True
    oversold_today = bool(last_valid(meta_target["oversold_block"].astype(float)) > 0.5) if "oversold_block" in meta_target else False

    return locals()
