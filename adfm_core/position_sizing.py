"""Historical position-sizing analytics for the ADFM Conviction Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
HORIZON_TRADING_DAYS: Mapping[str, int] = {
    "1 month": 21,
    "3 months": 63,
    "1 year": 252,
    "5 years": 1260,
}


@dataclass(frozen=True)
class SizingResult:
    """Auditable position-size result and each binding cap."""

    conviction_ceiling: float
    suggested_size: float
    volatility_cap: float
    invalidation_cap: float
    event_cap: float
    tail_cap: float
    liquidity_cap: float
    volatility_factor: float
    binding_constraint: str


def normalize_direction(direction: str) -> str:
    value = str(direction or "long").strip().lower()
    if value not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'")
    return value


def directional_return(asset_return: pd.Series | np.ndarray, direction: str) -> pd.Series | np.ndarray:
    """Translate underlying returns into long- or short-notional returns."""
    sign = 1.0 if normalize_direction(direction) == "long" else -1.0
    return asset_return * sign


def conviction_ceiling(conviction: int) -> float:
    """Map conviction 1-5 to a 5%-25% gross-exposure ceiling."""
    level = int(conviction)
    if level < 1 or level > 5:
        raise ValueError("conviction must be between 1 and 5")
    return level * 0.05


def annualized_volatility(returns: pd.Series, window: int | None = None) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if window is not None:
        clean = clean.tail(int(window))
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_annualized_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce")
    return clean.rolling(window, min_periods=max(20, window // 2)).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def maximum_drawdown_from_returns(returns: pd.Series | np.ndarray) -> float:
    clean = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if clean.size == 0:
        return np.nan
    wealth = np.cumprod(1.0 + clean)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    return float(np.min(drawdowns))


def maximum_drawdown_from_prices(close: pd.Series) -> float:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    peaks = clean.cummax()
    return float((clean / peaks - 1.0).min())


def historical_windows(
    close: pd.Series,
    horizon_days: int,
    direction: str,
    *,
    step: int = 5,
) -> pd.DataFrame:
    """Build overlapping historical holding-period paths and path-risk statistics."""
    clean = pd.to_numeric(close, errors="coerce").dropna()
    horizon = int(horizon_days)
    if horizon < 1 or len(clean) <= horizon:
        return pd.DataFrame(
            columns=["start", "end", "return", "mfe", "mae", "max_drawdown"]
        )
    sign = 1.0 if normalize_direction(direction) == "long" else -1.0
    stride = max(1, int(step))
    rows: list[dict[str, object]] = []
    values = clean.to_numpy(dtype=float)
    dates = clean.index
    for start in range(0, len(clean) - horizon, stride):
        path = values[start : start + horizon + 1] / values[start]
        path_returns = sign * (path - 1.0)
        daily_asset = np.diff(path) / path[:-1]
        daily_directional = sign * daily_asset
        rows.append(
            {
                "start": dates[start],
                "end": dates[start + horizon],
                "return": float(path_returns[-1]),
                "mfe": float(np.max(path_returns)),
                "mae": float(np.min(path_returns)),
                "max_drawdown": maximum_drawdown_from_returns(daily_directional),
            }
        )
    return pd.DataFrame(rows)


def historical_tail_move(returns: pd.Series, direction: str, horizon_days: int = 5) -> float:
    """Estimate a severe but recurring adverse move from rolling historical returns."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) <= horizon_days:
        return np.nan
    compounded = (1.0 + clean).rolling(horizon_days).apply(np.prod, raw=True) - 1.0
    directional = pd.Series(directional_return(compounded, direction), index=compounded.index).dropna()
    if directional.empty:
        return np.nan
    return float(abs(directional.quantile(0.05)))


def expected_shortfall(returns: pd.Series, direction: str, quantile: float = 0.01) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    directional = pd.Series(directional_return(clean, direction), index=clean.index)
    if directional.empty:
        return np.nan
    threshold = directional.quantile(quantile)
    tail = directional.loc[directional.le(threshold)]
    return float(abs(tail.mean())) if not tail.empty else np.nan


def first_touch_statistics(
    ohlcv: pd.DataFrame,
    horizon_days: int,
    direction: str,
    target_move: float,
    stop_move: float,
    *,
    step: int = 5,
) -> dict[str, float | int]:
    """Evaluate whether equivalent percentage targets or stops were reached first."""
    required = {"High", "Low", "Close"}
    if ohlcv.empty or not required.issubset(ohlcv.columns):
        return {}
    frame = ohlcv.loc[:, ["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    horizon = int(horizon_days)
    target = float(target_move)
    stop = float(stop_move)
    if target <= 0 or stop <= 0 or len(frame) <= horizon:
        return {}
    direction_value = normalize_direction(direction)
    counts = {"target_first": 0, "stop_first": 0, "same_day": 0, "neither": 0}
    target_days: list[int] = []
    stop_days: list[int] = []
    stride = max(1, int(step))
    for start in range(0, len(frame) - horizon, stride):
        entry = float(frame["Close"].iloc[start])
        window = frame.iloc[start + 1 : start + horizon + 1]
        if direction_value == "long":
            target_level = entry * (1.0 + target)
            stop_level = entry * (1.0 - stop)
            target_hits = np.flatnonzero(window["High"].to_numpy() >= target_level)
            stop_hits = np.flatnonzero(window["Low"].to_numpy() <= stop_level)
        else:
            target_level = entry * (1.0 - target)
            stop_level = entry * (1.0 + stop)
            target_hits = np.flatnonzero(window["Low"].to_numpy() <= target_level)
            stop_hits = np.flatnonzero(window["High"].to_numpy() >= stop_level)
        target_day = int(target_hits[0] + 1) if target_hits.size else None
        stop_day = int(stop_hits[0] + 1) if stop_hits.size else None
        if target_day is None and stop_day is None:
            counts["neither"] += 1
        elif target_day is not None and stop_day is not None and target_day == stop_day:
            counts["same_day"] += 1
        elif stop_day is None or (target_day is not None and target_day < stop_day):
            counts["target_first"] += 1
            target_days.append(target_day)
        else:
            counts["stop_first"] += 1
            stop_days.append(stop_day)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        **counts,
        "sample_count": total,
        "target_first_rate": counts["target_first"] / total,
        "stop_first_rate": counts["stop_first"] / total,
        "same_day_rate": counts["same_day"] / total,
        "neither_rate": counts["neither"] / total,
        "median_target_days": float(np.median(target_days)) if target_days else np.nan,
        "median_stop_days": float(np.median(stop_days)) if stop_days else np.nan,
    }


def earnings_reaction_frame(
    ohlcv: pd.DataFrame,
    earnings_dates: Iterable[pd.Timestamp | str],
) -> pd.DataFrame:
    """Measure historical overnight and close-to-close earnings reactions."""
    required = {"Open", "Close"}
    if ohlcv.empty or not required.issubset(ohlcv.columns):
        return pd.DataFrame(columns=["date", "gap", "session", "close_to_close", "abs_move"])
    frame = ohlcv.loc[:, ["Open", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        return pd.DataFrame(columns=["date", "gap", "session", "close_to_close", "abs_move"])
    index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.copy()
    frame.index = index
    rows: list[dict[str, object]] = []
    for raw_date in earnings_dates:
        event = pd.Timestamp(raw_date)
        if event.tzinfo is not None:
            event = event.tz_convert(None)
        event = event.normalize()
        location = frame.index.searchsorted(event)
        if location <= 0 or location >= len(frame):
            continue
        event_date = frame.index[location]
        previous_close = float(frame["Close"].iloc[location - 1])
        event_open = float(frame["Open"].iloc[location])
        event_close = float(frame["Close"].iloc[location])
        if previous_close <= 0 or event_open <= 0:
            continue
        gap = event_open / previous_close - 1.0
        session = event_close / event_open - 1.0
        close_to_close = event_close / previous_close - 1.0
        rows.append(
            {
                "date": event_date,
                "gap": gap,
                "session": session,
                "close_to_close": close_to_close,
                "abs_move": abs(close_to_close),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "gap", "session", "close_to_close", "abs_move"])
    return pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")


def daily_gap_proxy(ohlcv: pd.DataFrame, quantile: float = 0.90) -> float:
    """Fallback event-risk proxy derived from the full overnight-gap history."""
    required = {"Open", "Close"}
    if ohlcv.empty or not required.issubset(ohlcv.columns):
        return np.nan
    opens = pd.to_numeric(ohlcv["Open"], errors="coerce")
    closes = pd.to_numeric(ohlcv["Close"], errors="coerce")
    gaps = (opens / closes.shift(1) - 1.0).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return float(gaps.quantile(quantile)) if not gaps.empty else np.nan


def _finite_cap(value: float, ceiling: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return ceiling
    return float(min(value, ceiling))


def calculate_sizing(
    *,
    conviction: int,
    max_nav_loss: float,
    stop_distance: float,
    current_volatility: float,
    historical_median_volatility: float,
    event_move: float,
    tail_move: float,
    portfolio_nav: float,
    median_dollar_volume: float,
    hold_through_earnings: bool,
    participation_rate: float = 0.10,
    liquidation_days: int = 3,
    round_to: float = 0.005,
) -> SizingResult:
    """Calculate the lowest of conviction, risk, event, tail, and liquidity caps."""
    ceiling = conviction_ceiling(conviction)
    risk_budget = max(float(max_nav_loss), 0.0)

    if np.isfinite(current_volatility) and current_volatility > 0 and np.isfinite(
        historical_median_volatility
    ) and historical_median_volatility > 0:
        volatility_factor = float(
            np.clip(historical_median_volatility / current_volatility, 0.50, 1.00)
        )
    else:
        volatility_factor = 1.0
    volatility_cap = ceiling * volatility_factor

    invalidation_cap = _finite_cap(
        risk_budget / float(stop_distance) if stop_distance > 0 else np.nan, ceiling
    )
    tail_cap = _finite_cap(
        risk_budget / float(tail_move) if tail_move > 0 else np.nan, ceiling
    )
    event_cap = ceiling
    if hold_through_earnings:
        event_cap = _finite_cap(
            risk_budget / float(event_move) if event_move > 0 else np.nan, ceiling
        )

    liquidity_cap = ceiling
    nav = float(portfolio_nav)
    adv = float(median_dollar_volume)
    if nav > 0 and adv > 0 and participation_rate > 0 and liquidation_days > 0:
        liquidity_cap = _finite_cap(
            adv * float(participation_rate) * int(liquidation_days) / nav,
            ceiling,
        )

    caps = {
        "Conviction ceiling": ceiling,
        "Volatility adjustment": volatility_cap,
        "Invalidation loss budget": invalidation_cap,
        "Earnings/event risk": event_cap,
        "Historical tail risk": tail_cap,
        "Liquidity": liquidity_cap,
    }
    binding_constraint = min(caps, key=caps.get)
    raw_suggested = float(min(caps.values()))
    increment = max(float(round_to), 0.0001)
    suggested = max(0.0, np.floor(raw_suggested / increment + 1e-12) * increment)
    return SizingResult(
        conviction_ceiling=ceiling,
        suggested_size=float(min(suggested, ceiling)),
        volatility_cap=float(volatility_cap),
        invalidation_cap=float(invalidation_cap),
        event_cap=float(event_cap),
        tail_cap=float(tail_cap),
        liquidity_cap=float(liquidity_cap),
        volatility_factor=float(volatility_factor),
        binding_constraint=binding_constraint,
    )


def bootstrap_portfolio_paths(
    returns: pd.Series,
    *,
    direction: str,
    position_size: float,
    horizon_days: int,
    n_paths: int = 1000,
    block_size: int = 5,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-bootstrap portfolio paths at a constant gross exposure."""
    clean = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    horizon = int(horizon_days)
    paths = int(n_paths)
    block = max(1, int(block_size))
    if clean.size < block or horizon < 1 or paths < 1:
        return np.empty((0, 0)), np.array([]), np.array([])
    sign = 1.0 if normalize_direction(direction) == "long" else -1.0
    rng = np.random.default_rng(seed)
    max_start = clean.size - block + 1
    blocks_needed = int(np.ceil(horizon / block))
    sampled = np.empty((paths, blocks_needed * block), dtype=float)
    for path_index in range(paths):
        starts = rng.integers(0, max_start, size=blocks_needed)
        sampled_blocks = [clean[start : start + block] for start in starts]
        sampled[path_index] = np.concatenate(sampled_blocks)
    directional = sampled[:, :horizon] * sign
    portfolio_daily = directional * float(position_size)
    wealth = np.cumprod(1.0 + portfolio_daily, axis=1)
    peaks = np.maximum.accumulate(wealth, axis=1)
    drawdowns = wealth / peaks - 1.0
    ending_returns = wealth[:, -1] - 1.0
    max_drawdowns = drawdowns.min(axis=1)
    return wealth, ending_returns, max_drawdowns
