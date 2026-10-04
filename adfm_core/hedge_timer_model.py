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
EPISODE_PEAK_WINDOW = 63
LEAD_LOOKBACK = 20
EARLY_WARNING_LOSS = -0.03
EPISODE_REBOUND = 0.10
WARNING_OUTCOME_SESSIONS = 60
FROZEN_WATCH_THRESHOLD = 20.0
MODEL_FIT_END = "2026-10-02"
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


@dataclass(frozen=True)
class CalloutRules:
    """SPX-fitted event rules; identical defaults apply to NDX."""

    price_retreat: float = 0.0125
    volatility_jump: float = 0.05
    relative_volatility: float = 1.15
    credit_retreat: float = 0.01
    shock_loss: float = 0.0125
    shock_volatility: float = 0.15
    divergence_jump: float = 0.15
    recovery_sessions: int = 3
    minimum_spacing: int = 10


FROZEN_CALLOUT_RULES = CalloutRules()
CALLOUT_LEAD_LOOKBACK = 5


def debounce_callouts(
    candidate: pd.Series, recovery: pd.Series, valid: pd.Series, *,
    recovery_sessions: int = 3, minimum_spacing: int = 10,
) -> pd.DataFrame:
    """Latch an event until sustained recovery, rather than each score crossing.

    Unknown rows cannot emit an event or count towards consecutive recovery.
    minimum_spacing is a lower bound; time alone never rearms the signal.
    """
    idx = candidate.index
    valid = valid.reindex(idx).fillna(False).astype(bool)
    candidates = candidate.reindex(idx).fillna(False).astype(bool) & valid
    recovered = recovery.reindex(idx).fillna(False).astype(bool) & valid
    reset = recovered.rolling(recovery_sessions, min_periods=recovery_sessions).sum().eq(recovery_sessions)
    dots, latched = [], []
    active, last = False, -minimum_spacing
    for loc, (hit, healed, known) in enumerate(zip(candidates, reset, valid, strict=True)):
        if known and active and loc - last >= minimum_spacing and healed:
            active = False
        emit = bool(known and not active and hit)
        if emit:
            active, last = True, loc
        dots.append(emit)
        latched.append(active)
    return pd.DataFrame({"Callout": dots, "Latched": latched, "Armed": [not value for value in latched]}, index=idx)


def compute_callouts(
    df: pd.DataFrame, target_ticker: str, rules: CalloutRules = FROZEN_CALLOUT_RULES,
) -> pd.DataFrame:
    """Causal close-of-session hedge events, with no weighted trend score.

    Price break + two distinct risk groups, a price/volatility shock, or
    near-high price/volatility divergence. Outcome highs/lows and future
    drawdown labels are never used to generate or reset an event.
    """
    idx = df.index
    observed = df.reindex(columns=list(TICKERS)).apply(pd.to_numeric, errors="coerce")
    df = observed.where(np.isfinite(observed) & observed.gt(0))
    price, vix = _series(df, target_ticker), _series(df, "^VIX")
    retreat = safe_ratio(price, price.rolling(20).max()) - 1
    rsp_spy = safe_ratio(_series(df, "RSP"), _series(df, "SPY"))
    iwm_spy = safe_ratio(_series(df, "IWM"), _series(df, "SPY"))
    rsp_weak = (rsp_spy < rolling_ma(rsp_spy, 100, 40)) & (rsp_spy.pct_change(20, fill_method=None) < 0)
    iwm_weak = (iwm_spy < rolling_ma(iwm_spy, 100, 40)) & (iwm_spy.pct_change(20, fill_method=None) < 0)
    sector_rows = []
    for ticker in SECTOR_TICKERS:
        sector = _series(df, ticker)
        average = rolling_ma(sector, 50, 30)
        sector_rows.append((sector < average).astype(float).where(sector.notna() & average.notna()))
    sector_share = pd.concat(sector_rows, axis=1).mean(axis=1)
    sector_weak = (sector_share >= .55) & ((sector_share.diff(10) >= .18) | (sector_share >= .73))
    breadth = rsp_weak | iwm_weak | sector_weak

    ratio = safe_ratio(_series(df, "HYG"), _series(df, "LQD"))
    credit = (safe_ratio(ratio, ratio.rolling(10).max()) - 1 <= -rules.credit_retreat) & (ratio.pct_change(3, fill_method=None) < 0)
    vix_change = vix.pct_change(fill_method=None)
    vix_relative = safe_ratio(vix, vix.rolling(20).mean())
    volatility = (
        (vix_change >= rules.volatility_jump)
        | (vix.pct_change(3, fill_method=None) >= .10)
        | (vix_relative >= rules.relative_volatility)
        | (safe_ratio(_series(df, "^VIX9D"), vix) >= 1)
    )
    group_count = breadth.astype(int) + credit.astype(int) + volatility.astype(int)
    early = retreat >= EARLY_WARNING_LOSS - 1e-12
    price_break = (price < price.shift(1).rolling(3).min()) & (retreat <= -rules.price_retreat) & early
    shock = (price.pct_change(fill_method=None) <= -rules.shock_loss) & (vix_change >= rules.shock_volatility) & early
    divergence = (
        (retreat >= -.005) & (price.pct_change(5, fill_method=None) > 0) & breadth
        & (vix.pct_change(5, fill_method=None) >= rules.divergence_jump) & (vix_relative >= 1.05)
    )
    valid = df.notna().all(axis=1) & price.rolling(20).count().eq(20)
    candidate = ((price_break & (group_count >= 2)) | shock | divergence) & valid
    recovery = (price > ema(price, 10)) & (retreat >= -.01) & (vix.pct_change(3, fill_method=None) <= 0)
    result = debounce_callouts(candidate, recovery, valid,
                              recovery_sessions=rules.recovery_sessions, minimum_spacing=rules.minimum_spacing)
    result["Inputs valid"] = valid
    result["Price break"] = price_break & valid
    result["Breadth"] = breadth & valid
    result["Volatility"] = volatility & valid
    result["Credit proxy"] = credit & valid
    result["Shock"] = shock & valid
    result["Divergence"] = divergence & valid
    result["Recovery"] = recovery & valid
    result["Retreat"] = retreat
    result["Risk groups"] = group_count.where(valid)
    result["Trigger"] = pd.Series(
        np.select([shock & valid, divergence & valid, price_break & (group_count >= 2) & valid],
                  ["Price / volatility shock", "Near-high volatility divergence", "Price break + risk groups"], default=""),
        index=idx,
    ).where(result["Callout"], "")
    return result


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
    drawdown_velocity = (ret5 <= -0.02) | (ret10 <= -0.03) | (dd20 <= -0.015)

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


def _price_bars(px: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(px, pd.Series):
        return pd.DataFrame({"Close": px, "High": px, "Low": px}).dropna().sort_index()
    return px.loc[:, ["Close", "High", "Low"]].dropna().sort_index()


@dataclass(frozen=True)
class DrawdownEpisode:
    peak: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    trough: pd.Timestamp
    depth: float
    deadline: pd.Timestamp


def _drawdown_ledger(
    px: pd.Series | pd.DataFrame, threshold: float = DD_MAJOR,
    start_after: str = CALIBRATION_START,
) -> list[DrawdownEpisode]:
    """Price swings; a 10% close rebound from the trough rearms a local peak.

    Peak and trough prices stay fixed until a genuine reversal. Future prices
    label historical outcomes only; they are never inputs to the warning score.
    """
    bars = _price_bars(px)
    s, highs, lows = bars["Close"], bars["High"], bars["Low"]
    if s.empty:
        return []
    begin = int(s.index.searchsorted(pd.Timestamp(start_after)))
    if begin == len(s):
        return []
    warm = highs.iloc[max(0, begin - EPISODE_PEAK_WINDOW):begin + 1]
    peak = warm.index[-1 - int(np.argmax(warm.values[::-1]))]
    peak_price = float(highs.loc[peak])
    start = None
    trough = peak
    trough_price = peak_price
    deadline = s.index[max(0, int(s.index.get_loc(peak)) - 1)]
    breached = False
    result = []

    def record(end):
        result.append(DrawdownEpisode(
            peak, start, end, trough, trough_price / peak_price - 1, deadline
        ))

    for loc, (ts, price, high, low) in enumerate(zip(s.index[begin:], s.values[begin:], highs.values[begin:], lows.values[begin:], strict=True), start=begin):
        price, high, low = float(price), float(high), float(low)
        if start is None:
            if high >= peak_price:
                peak, peak_price, breached = ts, high, False
                deadline = s.index[max(0, loc - 1)]
            loss = low / peak_price - 1
            if not breached:
                if loss >= EARLY_WARNING_LOSS - 1e-12:
                    deadline = ts
                else:
                    breached = True
            if loss <= threshold + 1e-12:
                start, trough, trough_price = ts, ts, low
        else:
            if low < trough_price:
                trough, trough_price = ts, low
            if price / trough_price - 1 >= EPISODE_REBOUND - 1e-12:
                record(ts)
                peak, peak_price = ts, high
                breached = low / high - 1 < EARLY_WARNING_LOSS - 1e-12
                deadline = s.index[max(0, loc - 1)] if breached else ts
                start = None
    if start is not None:
        record(s.index[-1])
    return result


def find_drawdown_episodes(
    px: pd.Series | pd.DataFrame, threshold: float = DD_MAJOR, recovery: float = -0.02,
    start_after: str = CALIBRATION_START,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """Distinct 10% declines, separated by a 10% rebound from their trough.

    recovery is retained for compatibility; the ledger uses a fixed rebound
    rule to capture separate bear-market legs without expiring the original peak.
    """
    return [(e.start, e.end, e.trough, e.depth)
            for e in _drawdown_ledger(px, threshold, start_after)]


def _warning_window(px: pd.Series | pd.DataFrame, episode: DrawdownEpisode, lookback: int) -> pd.Index:
    idx = _price_bars(px).index
    peak_loc = int(idx.get_loc(episode.peak))
    return idx[max(0, peak_loc - lookback):int(idx.get_loc(episode.deadline)) + 1]


def episode_audit(
    name: str, px: pd.Series | pd.DataFrame, warning_onsets: pd.Series,
    threshold: float = DD_MAJOR, start_after: str = CALIBRATION_START,
    lookback: int = LEAD_LOOKBACK, warning_active: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    bars = _price_bars(px)
    s = bars["Close"]
    previous_end = None
    for e in _drawdown_ledger(bars, threshold, start_after):
        window = _warning_window(bars, e, lookback)
        if previous_end is not None:
            window = window[window >= previous_end]
        hits = warning_onsets.reindex(window).fillna(False).astype(bool)
        first = hits[hits].index[0] if hits.any() else pd.NaT
        # If the peak session breached 3%, only a warning known at the prior
        # close can qualify. An end-of-day alert would already be late.
        active_date = e.peak if e.deadline >= e.peak else s.index[max(0, int(s.index.get_loc(e.peak)) - 1)]
        ongoing = False
        active_at_peak = bool(warning_active.reindex([active_date]).fillna(False).iloc[0]) if warning_active is not None else False
        if pd.isna(first) and active_at_peak:
            first, ongoing = active_date, True
        late = warning_onsets.reindex(s.loc[e.peak:e.trough].index).fillna(False).astype(bool)
        late_dates = late[late].index
        first_any = first if pd.notna(first) else (late_dates[0] if len(late_dates) else pd.NaT)
        rows.append({
            "Index": name, "Peak": e.peak, "3% deadline": e.deadline,
            "Start": e.start, "End": e.end, "Trough": e.trough, "Depth": e.depth,
            "First warning": first_any,
            "Loss at warning": (0.0 if first_any < e.peak else float(s.loc[first_any] / bars.loc[e.peak, "High"] - 1)) if pd.notna(first_any) else np.nan,
            "Lead sessions": int(s.index.get_loc(e.start) - s.index.get_loc(first_any)) if pd.notna(first_any) else np.nan,
            "Captured": bool(pd.notna(first)),
            "Timing": (("Active at peak" if active_date == e.peak else "Active before peak") if ongoing else
                       "Before peak" if pd.notna(first) and first < e.peak else
                       "At peak" if pd.notna(first) and first == e.peak else
                       "Early decline" if pd.notna(first) else
                       "Late" if pd.notna(first_any) else "No warning"),
        })
        previous_end = e.end
    return pd.DataFrame(rows, columns=[
        "Index", "Peak", "3% deadline", "Start", "End", "Trough", "Depth",
        "First warning", "Loss at warning", "Lead sessions", "Captured", "Timing",
    ])


def _useful_warning_dates(onsets: pd.Series, px: pd.Series | pd.DataFrame, lookback: int = LEAD_LOOKBACK) -> set[pd.Timestamp]:
    useful = set()
    previous_end = None
    for e in _drawdown_ledger(px):
        window = _warning_window(px, e, lookback)
        if previous_end is not None:
            window = window[window >= previous_end]
        hits = onsets.reindex(window).fillna(False).astype(bool)
        useful.update(hits[hits].index)
        previous_end = e.end
    return useful


def coverage_stats(px: pd.Series | pd.DataFrame, warning_onsets: pd.Series, lookback: int = LEAD_LOOKBACK, warning_active: pd.Series | None = None) -> tuple[float, list[int]]:
    audit = episode_audit("", px, warning_onsets, lookback=lookback, warning_active=warning_active)
    if audit.empty:
        return float("nan"), []
    captured = audit.loc[audit["Captured"], "Lead sessions"].tolist()
    return len(captured) / len(audit), captured


def warning_summary(px: pd.Series | pd.DataFrame, warning_onsets: pd.Series, lookback: int = LEAD_LOOKBACK, warning_active: pd.Series | None = None) -> dict[str, float | int]:
    """Early recall; non-capturing onsets need 60 sessions to mature."""
    audit = episode_audit("", px, warning_onsets, lookback=lookback, warning_active=warning_active)
    onsets = warning_onsets.reindex(_price_bars(px).index).fillna(False).astype(bool)
    dates = onsets[onsets & (onsets.index >= pd.Timestamp(CALIBRATION_START))].index
    useful = _useful_warning_dates(onsets, px, lookback)
    pending_dates = set(onsets.index[-WARNING_OUTCOME_SESSIONS:])
    late_dates = set()
    for e in _drawdown_ledger(px):
        late_dates.update(onsets.loc[(onsets.index > e.deadline) & (onsets.index <= e.end)].index)
    late = sum(date not in useful and date in late_dates for date in dates)
    false = sum(date not in useful and date not in late_dates and date not in pending_dates for date in dates)
    pending = sum(date not in useful and date not in late_dates and date in pending_dates for date in dates)
    captured = audit.loc[audit["Captured"], "Lead sessions"]
    return {
        "episodes": len(audit), "captured": len(captured), "warnings": len(dates),
        "false_warnings": false, "pending_warnings": pending, "late_warnings": late,
        "median_lead": float(captured.median()) if len(captured) else float("nan"),
    }


def select_full_recall_candidate(candidates: list[dict[str, float]]) -> dict[str, float]:
    """SPX recall first, then least time in warning and false alarms. NDX is held out."""
    if not candidates:
        raise ValueError("At least one calibration candidate is required")
    return max(candidates, key=lambda row: (
        row["spx_coverage"] if np.isfinite(row["spx_coverage"]) else -1,
        -row.get("warning_time", 1.0), -row["false_warning_rate"], row["threshold"],
    ))


def calibrate_watch_threshold(
    score_spx: pd.Series, px_spx: pd.Series | pd.DataFrame,
    score_ndx: pd.Series | None = None, px_ndx: pd.Series | pd.DataFrame | None = None,
) -> dict[str, float]:
    """Fit SPX only. Optional NDX series supply diagnostics after selection."""
    index = _price_bars(px_spx).index.intersection(score_spx.index)
    prices = px_spx.reindex(index)
    score = score_spx.reindex(index)
    fit = index >= pd.Timestamp(CALIBRATION_START)
    candidates = []
    for threshold in range(1, 86):
        active = watch_signal(score, threshold)
        signals = onset(active)
        coverage, leads = coverage_stats(prices, signals, warning_active=active)
        summary = warning_summary(prices, signals, warning_active=active)
        mature = summary["warnings"] - summary["pending_warnings"]
        candidates.append({
            "threshold": float(threshold), "spx_coverage": float(coverage),
            "median_lead": float(np.median(leads)) if leads else float("nan"),
            "false_warning_rate": summary["false_warnings"] / mature if mature else 0.0,
            "warning_time": float(active.loc[fit].mean()) if fit.any() else float("nan"),
        })
    chosen = select_full_recall_candidate(candidates).copy()
    chosen["ndx_coverage"] = float("nan")
    if score_ndx is not None and px_ndx is not None:
        chosen["ndx_coverage"] = coverage_stats(
            px_ndx, onset(watch_signal(score_ndx, chosen["threshold"])),
            warning_active=watch_signal(score_ndx, chosen["threshold"])
        )[0]
    return chosen


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
