"""Pure signal, calibration, and audit logic for the Hedge Timer page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

SPX_TICKER = "^GSPC"
NDX_TICKER = "^NDX"
SPX_LABEL = "^SPX"
NDX_LABEL = "^NDX"

CALIBRATION_START = "2020-01-01"
DD_MAJOR = -0.10
LEAD_LOOKBACK = 40
EARLY_STAGE_DD63 = -0.12
RSI_OVERSOLD = 30.0
RSI_SOFT_OVERSOLD = 35.0
CONFIRM_THRESHOLD = 50.0
HORIZON_DAYS = 20

SECTOR_TICKERS = (
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLB",
    "XLRE",
    "XLK",
    "XLU",
)

TICKERS = (
    SPX_TICKER,
    NDX_TICKER,
    "SPY",
    "RSP",
    "IWM",
    "HYG",
    "LQD",
    "^VIX",
    "^VIX9D",
    "^VIX3M",
    "^VVIX",
    *SECTOR_TICKERS,
)


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    weight: int


WATCH_COMPONENTS = (
    Component("credit_risk", "Credit deterioration", 14),
    Component("breadth_ratio", "Equal-weight breadth", 10),
    Component("sector_breadth", "Sector participation", 14),
    Component("smallcap_breadth", "Small-cap breadth", 8),
    Component("defensive_rotation", "Defensive rotation", 8),
    Component("vol_stress", "Volatility stress", 12),
    Component("vol_acceleration", "Volatility acceleration", 10),
    Component("drawdown_velocity", "Drawdown velocity", 12),
    Component("momentum_rollover", "Daily / weekly momentum", 12),
)

CONFIRM_COMPONENTS = (
    Component("short_term_break", "Short-term structure break", 35),
    Component("realized_vol_expansion", "Realized-vol expansion", 25),
    Component("trend_confirm", "Trend confirmation", 40),
)


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def rolling_ma(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minp = window if min_periods is None else min_periods
    return s.rolling(window, min_periods=minp).mean()


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
    macd = ema(s, fast) - ema(s, slow)
    sig = ema(macd, signal)
    return macd - sig


def resample_last(s: pd.Series, rule: str) -> pd.Series:
    s = s.dropna().copy()
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample(rule).last()


def to_daily(daily_index: pd.DatetimeIndex, slower: pd.Series) -> pd.Series:
    return slower.reindex(daily_index, method="ffill")


def drawdown(px: pd.Series) -> pd.Series:
    s = px.dropna()
    return s / s.cummax() - 1.0


def drawdown_from_rolling_high(px: pd.Series, window: int) -> pd.Series:
    high = px.rolling(window, min_periods=max(20, window // 4)).max()
    return px / high - 1.0


def realized_vol(px: pd.Series, window: int) -> pd.Series:
    return px.pct_change().rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252.0)


def forward_min_return(px: pd.Series, horizon: int) -> pd.Series:
    s = px.dropna()
    future_min = s[::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    return future_min / s - 1.0


def _series(df: pd.DataFrame, key: str) -> pd.Series:
    return df.get(key, pd.Series(index=df.index, dtype=float))


def _weighted_score(
    conditions: dict[str, pd.Series], components: Iterable[Component], index: pd.Index
) -> pd.Series:
    components = tuple(components)
    score = pd.Series(0.0, index=index)
    denominator = sum(item.weight for item in components)
    for item in components:
        score = score.add(
            conditions[item.key].reindex(index).fillna(False).astype(float) * item.weight,
            fill_value=0.0,
        )
    return ((score / max(denominator, 1)) * 100.0).clip(0.0, 100.0)


def compute_scores(
    df: pd.DataFrame, target_ticker: str
) -> tuple[pd.Series, pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    """Return watch score, confirmation score, meta series, and component conditions."""

    idx = df.index
    tgt = _series(df, target_ticker)
    spy = _series(df, "SPY")
    rsp = _series(df, "RSP")
    iwm = _series(df, "IWM")
    hyg = _series(df, "HYG")
    lqd = _series(df, "LQD")
    vix = _series(df, "^VIX")
    vix9 = _series(df, "^VIX9D")
    vix3m = _series(df, "^VIX3M")
    vvix = _series(df, "^VVIX")

    credit = safe_ratio(hyg, lqd)
    credit_ma100 = rolling_ma(credit, 100, 40)
    credit_risk = ((credit < credit_ma100) & (credit.pct_change(20) < 0)) | (
        credit.pct_change(10) <= -0.012
    )

    rsp_spy = safe_ratio(rsp, spy)
    breadth_ma100 = rolling_ma(rsp_spy, 100, 40)
    breadth_ratio = (rsp_spy < breadth_ma100) & (rsp_spy.pct_change(20) < 0)

    iwm_spy = safe_ratio(iwm, spy)
    smallcap_ma100 = rolling_ma(iwm_spy, 100, 40)
    smallcap_breadth = (iwm_spy < smallcap_ma100) & (iwm_spy.pct_change(20) < 0)

    sector_below50 = []
    for ticker in SECTOR_TICKERS:
        sector = _series(df, ticker)
        sector_ma50 = rolling_ma(sector, 50, 30)
        available = sector.notna() & sector_ma50.notna()
        sector_below50.append((sector < sector_ma50).astype(float).where(available))
    sector_breadth_share = pd.concat(sector_below50, axis=1).mean(axis=1, skipna=True)
    sector_breadth = (sector_breadth_share >= 0.55) & (
        (sector_breadth_share.diff(10) >= 0.18) | (sector_breadth_share >= 0.73)
    )

    cyclical = pd.concat(
        [_series(df, key).pct_change(20) for key in ("XLY", "XLI", "XLF", "XLB")],
        axis=1,
    ).mean(axis=1)
    defensive = pd.concat(
        [_series(df, key).pct_change(20) for key in ("XLP", "XLV", "XLU")],
        axis=1,
    ).mean(axis=1)
    defensive_rotation = (cyclical - defensive) <= -0.02

    vix_ma50 = rolling_ma(vix, 50, 25)
    vol_level = (vix > vix_ma50) & (vix.diff(10) > 0)
    vol_term_front = safe_ratio(vix9, vix) >= 1.0
    vol_term_back = safe_ratio(vix, vix3m) >= 1.0
    vvix_tail = vvix >= vvix.rolling(252, min_periods=100).quantile(0.70)
    vol_stress = (vol_level | vol_term_front | vol_term_back) & (vvix_tail | (vix >= 18))

    rv10 = realized_vol(tgt, 10)
    rv63 = realized_vol(tgt, 63)
    rv_ratio = safe_ratio(rv10, rv63)
    vol_acceleration = (
        (vix.pct_change(5) >= 0.25)
        | (safe_ratio(vix, rolling_ma(vix, 20, 10)) >= 1.15)
        | (rv_ratio >= 1.30)
    )

    ret5 = tgt.pct_change(5)
    ret10 = tgt.pct_change(10)
    dd20 = drawdown_from_rolling_high(tgt, 20)
    drawdown_velocity = (ret5 <= -0.03) | (ret10 <= -0.045) | (dd20 <= -0.05)

    rsi_d = rsi(tgt, 14)
    macd_d = macd_hist(tgt)
    weekly = resample_last(tgt, "W-FRI")
    rsi_w = to_daily(idx, rsi(weekly, 14))
    macd_w = to_daily(idx, macd_hist(weekly))

    rsi_d_roll = ((rsi_d.shift(1) >= 60) & (rsi_d < 60)) | (
        (rsi_d >= 43) & (rsi_d.diff(5) < 0) & (rsi_d < rsi_d.rolling(10, min_periods=5).max() - 2)
    )
    rsi_w_roll = ((rsi_w.shift(1) >= 58) & (rsi_w < 58)) | ((rsi_w >= 46) & (rsi_w.diff(3) < 0))
    macd_d_bear = ((macd_d.shift(1) > 0) & (macd_d < 0)) | (
        (macd_d < macd_d.shift(3)) & (macd_d.diff(3) < 0)
    )
    macd_w_bear = ((macd_w.shift(1) > 0) & (macd_w < 0)) | (
        (macd_w < macd_w.shift(2)) & (macd_w.diff(2) < 0)
    )
    momentum_rollover = (rsi_d_roll & (rsi_w_roll | macd_w_bear)) | (
        macd_d_bear & (rsi_w_roll | macd_w_bear)
    )

    ema9 = ema(tgt, 9)
    ema21 = ema(tgt, 21)
    short_term_break = ((tgt < ema21) & (ema21.diff(10) < 0)) | ((tgt < ema9) & (ema9 < ema21))
    realized_vol_expansion = rv_ratio >= 1.15
    ma50 = rolling_ma(tgt, 50, 30)
    ma200 = rolling_ma(tgt, 200, 100)
    trend_confirm = ((tgt < ma50) & (ma50.diff(20) < 0)) | (tgt < ma200)

    watch_conditions = {
        "credit_risk": credit_risk,
        "breadth_ratio": breadth_ratio,
        "sector_breadth": sector_breadth,
        "smallcap_breadth": smallcap_breadth,
        "defensive_rotation": defensive_rotation,
        "vol_stress": vol_stress,
        "vol_acceleration": vol_acceleration,
        "drawdown_velocity": drawdown_velocity,
        "momentum_rollover": momentum_rollover,
    }
    confirm_conditions = {
        "short_term_break": short_term_break,
        "realized_vol_expansion": realized_vol_expansion,
        "trend_confirm": trend_confirm,
    }

    watch_score = _weighted_score(watch_conditions, WATCH_COMPONENTS, idx)
    confirm_score = _weighted_score(confirm_conditions, CONFIRM_COMPONENTS, idx)

    dd63 = drawdown_from_rolling_high(tgt, 63)
    oversold_block = (rsi_d < RSI_OVERSOLD) | (
        (rsi_d < RSI_SOFT_OVERSOLD) & (rsi_d.diff(5) > 0)
    )
    early_stage = dd63 > EARLY_STAGE_DD63

    meta = {
        "rsi_d": rsi_d,
        "dd63": dd63,
        "oversold_block": oversold_block.fillna(False),
        "early_stage": early_stage.fillna(True),
        "ma50": ma50,
        "ma200": ma200,
        "sector_breadth_share": sector_breadth_share,
        "realized_vol_ratio": rv_ratio,
    }
    all_conditions = {**watch_conditions, **confirm_conditions}
    all_conditions = {
        key: value.reindex(idx).fillna(False).astype(bool)
        for key, value in all_conditions.items()
    }
    return watch_score.fillna(0.0), confirm_score.fillna(0.0), meta, all_conditions


def watch_signal(score: pd.Series, threshold: float) -> pd.Series:
    """Underlying warning state. Deliberately independent of oversold/late gates."""

    return (score >= threshold).reindex(score.index).fillna(False).astype(bool)


def onset(signal: pd.Series) -> pd.Series:
    signal = signal.fillna(False).astype(bool)
    return (signal & ~signal.shift(1, fill_value=False)).astype(bool)


def fresh_short_onset(
    score: pd.Series,
    meta: dict[str, pd.Series],
    threshold: float,
    confirmation: pd.Series | None = None,
) -> pd.Series:
    """New short-initiation signal after confirmation and anti-bottom-short gates."""

    idx = score.index
    base = watch_signal(score, threshold)
    if confirmation is not None:
        base = base & confirmation.reindex(idx).fillna(False).astype(bool)
    early = meta.get("early_stage", pd.Series(True, index=idx)).reindex(idx).fillna(True).astype(bool)
    oversold = meta.get("oversold_block", pd.Series(False, index=idx)).reindex(idx).fillna(False).astype(bool)
    return onset(base & early & ~oversold)


def find_drawdown_episodes(
    px: pd.Series,
    threshold: float = DD_MAJOR,
    recovery: float = -0.02,
    start_after: str = CALIBRATION_START,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """Find distinct threshold-crossing drawdown episodes from a running peak."""

    s = px.dropna().loc[lambda x: x.index >= pd.Timestamp(start_after)]
    if len(s) < 20:
        return []

    dd = drawdown(s)
    episodes: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
    start: pd.Timestamp | None = None

    for ts, value in dd.items():
        if start is None and value <= threshold:
            start = ts
            continue
        if start is not None and value >= recovery:
            segment = dd.loc[start:ts]
            episodes.append((start, ts, segment.idxmin(), float(segment.min())))
            start = None

    if start is not None:
        segment = dd.loc[start:]
        episodes.append((start, segment.index[-1], segment.idxmin(), float(segment.min())))

    return episodes


def _lead_for_episode(onsets: pd.Series, start_ts: pd.Timestamp, lookback: int) -> int:
    signal = onsets.fillna(False).astype(bool)
    if signal.empty:
        return -1
    loc = signal.index.get_indexer([start_ts], method="nearest")[0]
    if loc < 0:
        return -1
    lo = max(0, loc - lookback)
    hits = signal.iloc[lo : loc + 1]
    hits = hits[hits]
    if hits.empty:
        return -1
    first_loc = signal.index.get_loc(hits.index[0])
    return int(loc - first_loc)


def episode_audit(
    name: str,
    px: pd.Series,
    warning_onsets: pd.Series,
    threshold: float = DD_MAJOR,
    start_after: str = CALIBRATION_START,
    lookback: int = LEAD_LOOKBACK,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start, end, trough, depth in find_drawdown_episodes(
        px, threshold=threshold, start_after=start_after
    ):
        lead = _lead_for_episode(warning_onsets, start, lookback)
        window = warning_onsets.loc[:start].tail(lookback + 1)
        warning_dates = window[window.fillna(False)].index
        first_warning = warning_dates[0] if len(warning_dates) else pd.NaT
        rows.append(
            {
                "Index": name,
                "Start": start,
                "End": end,
                "Trough": trough,
                "Depth": depth,
                "First warning": first_warning,
                "Lead sessions": lead if lead >= 0 else np.nan,
                "Captured": lead >= 0,
            }
        )
    return pd.DataFrame(rows)


def _useful_warning_dates(
    onsets: pd.Series,
    px: pd.Series,
    lookback: int = LEAD_LOOKBACK,
) -> set[pd.Timestamp]:
    useful: set[pd.Timestamp] = set()
    for start, _, _, _ in find_drawdown_episodes(px):
        loc = onsets.index.get_indexer([start], method="nearest")[0]
        if loc < 0:
            continue
        lo = max(0, loc - lookback)
        window = onsets.iloc[lo : loc + 1]
        useful.update(window[window.fillna(False)].index)
    return useful


def _false_warning_rate(
    onsets: pd.Series,
    px: pd.Series,
    lookback: int = LEAD_LOOKBACK,
) -> float:
    onset_dates = onsets[onsets.fillna(False)].index
    if len(onset_dates) == 0:
        return 0.0
    useful = _useful_warning_dates(onsets, px, lookback)
    false_count = sum(date not in useful for date in onset_dates)
    return float(false_count / len(onset_dates))


def coverage_stats(
    px: pd.Series, warning_onsets: pd.Series, lookback: int = LEAD_LOOKBACK
) -> tuple[float, list[int]]:
    episodes = find_drawdown_episodes(px)
    if not episodes:
        return 1.0, []
    leads = [_lead_for_episode(warning_onsets, start, lookback) for start, _, _, _ in episodes]
    captured = [lead for lead in leads if lead >= 0]
    return len(captured) / len(episodes), captured


def warning_summary(
    px: pd.Series, warning_onsets: pd.Series, lookback: int = LEAD_LOOKBACK
) -> dict[str, float | int]:
    """Summarize drawdown recall, lead time, and false warning count."""

    episodes = find_drawdown_episodes(px)
    leads = [_lead_for_episode(warning_onsets, start, lookback) for start, _, _, _ in episodes]
    captured_leads = [lead for lead in leads if lead >= 0]
    onset_dates = warning_onsets[warning_onsets.fillna(False)].index
    useful = _useful_warning_dates(warning_onsets, px, lookback)
    false_warnings = sum(date not in useful for date in onset_dates)
    return {
        "episodes": len(episodes),
        "captured": len(captured_leads),
        "warnings": int(len(onset_dates)),
        "false_warnings": int(false_warnings),
        "median_lead": float(np.median(captured_leads)) if captured_leads else float("nan"),
    }


def select_full_recall_candidate(candidates: list[dict[str, float]]) -> dict[str, float]:
    """Choose lead time first among candidates that fully capture both indices."""

    if not candidates:
        raise ValueError("At least one calibration candidate is required")

    full_recall = [
        row
        for row in candidates
        if row["spx_coverage"] >= 1.0 and row["ndx_coverage"] >= 1.0
    ]
    if full_recall:
        return max(
            full_recall,
            key=lambda row: (
                row["median_lead"],
                -row["false_warning_rate"],
                row["threshold"],
            ),
        )

    return max(
        candidates,
        key=lambda row: (
            min(row["spx_coverage"], row["ndx_coverage"]),
            row["spx_coverage"] + row["ndx_coverage"],
            row["median_lead"],
            -row["false_warning_rate"],
        ),
    )


def calibrate_watch_threshold(
    score_spx: pd.Series,
    px_spx: pd.Series,
    score_ndx: pd.Series,
    px_ndx: pd.Series,
) -> dict[str, float]:
    """Calibrate warning threshold with full drawdown recall as the hard first constraint."""

    start = pd.Timestamp(CALIBRATION_START)
    score_spx = score_spx.loc[score_spx.index >= start]
    score_ndx = score_ndx.loc[score_ndx.index >= start]
    px_spx = px_spx.reindex(score_spx.index).dropna()
    px_ndx = px_ndx.reindex(score_ndx.index).dropna()

    candidates: list[dict[str, float]] = []
    for threshold in range(35, 86):
        on_spx = onset(watch_signal(score_spx.reindex(px_spx.index), threshold))
        on_ndx = onset(watch_signal(score_ndx.reindex(px_ndx.index), threshold))
        spx_coverage, spx_leads = coverage_stats(px_spx, on_spx)
        ndx_coverage, ndx_leads = coverage_stats(px_ndx, on_ndx)
        leads = spx_leads + ndx_leads
        median_lead = float(np.median(leads)) if leads else -1.0
        false_rate = 0.5 * (
            _false_warning_rate(on_spx, px_spx) + _false_warning_rate(on_ndx, px_ndx)
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "spx_coverage": float(spx_coverage),
                "ndx_coverage": float(ndx_coverage),
                "median_lead": median_lead,
                "false_warning_rate": float(false_rate),
            }
        )

    return select_full_recall_candidate(candidates)


def state_label(
    watch_score: float,
    confirm_score: float,
    threshold: float,
    early_stage: bool,
    oversold_block: bool,
) -> str:
    if watch_score < threshold:
        return "STAND DOWN"
    if confirm_score < CONFIRM_THRESHOLD:
        return "HEDGE WATCH"
    if not early_stage or oversold_block:
        return "HEDGE CONFIRMED"
    return "SHORT ALLOWED"


def forward_stats(score: pd.Series, px: pd.Series, threshold: float) -> dict[str, float]:
    idx = score.index.intersection(px.index)
    warning_onsets = onset(watch_signal(score.reindex(idx), threshold))
    fwd = forward_min_return(px.reindex(idx), HORIZON_DAYS)
    hit = fwd[warning_onsets].dropna()
    miss = fwd[~warning_onsets].dropna()
    return {
        "warnings": float(warning_onsets.sum()),
        "warning_rate": float(warning_onsets.mean()),
        "avg_worst_warning": float(hit.mean()) if len(hit) else float("nan"),
        "avg_worst_no_warning": float(miss.mean()) if len(miss) else float("nan"),
    }
