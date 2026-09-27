"""Native transport for the Position Sizing Lab's historical decision inputs."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from adfm_engine.data.market import adjusted_ohlcv, fetch_daily_ohlcv, unique_tickers
from adfm_engine.position_sizing_math import (
    HORIZON_TRADING_DAYS, annualized_volatility, calculate_sizing,
    daily_gap_proxy, earnings_reaction_frame, expected_shortfall,
    first_touch_statistics, historical_tail_move, historical_windows,
    maximum_drawdown_from_prices, rolling_annualized_volatility,
)
from adfm_engine.serialization import records
from adfm_engine.services import DataUnavailable

BENCHMARKS = ("SPY", "QQQ", "TLT", "UUP", "USO", "^VIX")
LABELS = ("US equities", "Growth / duration", "Long duration", "US dollar", "Oil", "Equity volatility")
BLOCK_SIZE = 20


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _close(frames, symbol):
    frame = frames.get(symbol)
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=symbol)
    result = pd.to_numeric(adjusted_ohlcv(frame).get("Close"), errors="coerce").dropna()
    result.name = symbol
    return result


def _earnings_dates(symbol):
    try:
        data = yf.Ticker(symbol).get_earnings_dates(limit=48)
        if data is None or data.empty:
            return ()
        index = pd.DatetimeIndex(data.index)
        if index.tz is not None:
            index = index.tz_convert("America/New_York").tz_localize(None)
        return tuple(pd.Timestamp(value) for value in index)
    except Exception:
        return ()


def _directional_returns(close, direction):
    daily = pd.to_numeric(close, errors="coerce").pct_change(fill_method=None)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    daily = daily.loc[daily.gt(-1)]
    if direction == "Short":
        daily = -daily
    return daily


def _log_optimal(values, ceiling):
    returns = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(returns) < 20 or ceiling <= 0:
        return None
    candidates = np.linspace(0, ceiling, 501)
    scores = np.full(len(candidates), -np.inf)
    for index, fraction in enumerate(candidates):
        gross = 1 + fraction * returns
        if np.all(gross > 0):
            scores[index] = np.mean(np.log(gross))
    return _number(candidates[int(np.argmax(scores))]) if np.isfinite(scores).any() else None


def _session_index(size, session_no, mode, seed, target_sessions):
    if size <= 1:
        return 0
    if mode == "Chronological regime replay":
        start = seed % max(1, size - target_sessions + 1)
        return min(start + session_no, size - 1)
    block_no, offset = divmod(session_no, BLOCK_SIZE)
    maximum_start = max(1, size - BLOCK_SIZE + 1)
    rng = np.random.default_rng(seed + block_no * 7919)
    if mode == "Recent-regime weighted blocks":
        weights = np.linspace(1.0, 4.0, maximum_start)
        start = int(rng.choice(maximum_start, p=weights / weights.sum()))
    else:
        start = int(rng.integers(0, maximum_start))
    return min(start + offset, size - 1)


def _sensitivities(returns, frames, direction):
    sign = 1 if direction == "Long" else -1
    rows = []
    for symbol, label in zip(BENCHMARKS, LABELS):
        benchmark = _close(frames, symbol).pct_change(fill_method=None)
        aligned = pd.concat([returns.rename("asset"), benchmark.rename("benchmark")], axis=1, sort=False).dropna().tail(756)
        if len(aligned) < 40:
            continue
        variance = aligned["benchmark"].var(ddof=1)
        beta = aligned["asset"].cov(aligned["benchmark"]) / variance if variance > 0 else np.nan
        rows.append({"Exposure": label, "Proxy": symbol,
                     "Position correlation": _number(sign * aligned["asset"].corr(aligned["benchmark"])),
                     "Position beta": _number(sign * beta), "Observations": len(aligned)})
    return rows


def load_position_sizing(*, ticker="AAPL", direction="Long", conviction=3,
                         horizon_label="3 months", portfolio_nav=5_000_000,
                         max_loss_pct=1.25, hold_earnings=True, participation=10,
                         liquidation_days=3, entry=None, target=None, stop=None,
                         sampling_mode="Random historical blocks", simulation_position_pct=None,
                         starting_balance=None, seed=None, session_hour=None):
    frames, missing = fetch_daily_ohlcv(unique_tickers([ticker, *BENCHMARKS]), period="max")
    close = _close(frames, ticker)
    if len(close) < 63:
        raise DataUnavailable(f"No usable historical series was returned for {ticker}.", diagnostics=records(missing))
    ohlcv = adjusted_ohlcv(frames[ticker]).dropna(subset=["Open", "High", "Low", "Close"])
    latest = float(close.iloc[-1])
    entry = latest if entry is None else float(entry)
    target = entry * (1.15 if direction == "Long" else .85) if target is None else float(target)
    stop = entry * (.92 if direction == "Long" else 1.08) if stop is None else float(stop)
    target_move = target / entry - 1 if direction == "Long" else 1 - target / entry
    stop_distance = 1 - stop / entry if direction == "Long" else stop / entry - 1
    if stop_distance <= 0:
        raise ValueError("Invalidation must be below entry for a long, or above entry for a short.")
    horizon_days = HORIZON_TRADING_DAYS[horizon_label]
    returns = close.pct_change(fill_method=None).dropna()
    daily = _directional_returns(close, direction)
    rolling_vol = rolling_annualized_volatility(returns, 63)
    current_vol = annualized_volatility(returns, 63)
    median_vol = float(rolling_vol.dropna().median())
    tail_move = historical_tail_move(returns, direction.lower(), 5)
    daily_es = expected_shortfall(returns, direction.lower(), .01)
    median_dollar_volume = float((close.reindex(ohlcv.index) * pd.to_numeric(ohlcv["Volume"], errors="coerce")).dropna().tail(252).median())
    dates = _earnings_dates(ticker)
    events = earnings_reaction_frame(ohlcv, dates)
    if len(events) >= 4:
        event_move = float(events["abs_move"].quantile(.90))
        event_basis = f"90th-percentile earnings reaction across {len(events)} events"
    else:
        event_move = daily_gap_proxy(ohlcv, .90)
        event_basis = "90th-percentile overnight gap; earnings history unavailable"
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    future_earnings = [date for date in dates if date.normalize() >= today]
    next_earnings = min(future_earnings).date().isoformat() if future_earnings else None
    step = {21: 5, 63: 5, 252: 21, 1260: 63}[horizon_days]
    paths = historical_windows(close, horizon_days, direction.lower(), step=step)
    sizing = calculate_sizing(conviction=conviction, max_nav_loss=max_loss_pct / 100,
                              stop_distance=stop_distance, current_volatility=current_vol,
                              historical_median_volatility=median_vol, event_move=event_move,
                              tail_move=tail_move, portfolio_nav=portfolio_nav,
                              median_dollar_volume=median_dollar_volume, hold_through_earnings=hold_earnings,
                              participation_rate=participation / 100, liquidation_days=liquidation_days)
    ceiling_pct = sizing.conviction_ceiling * 100
    suggested_pct = sizing.suggested_size * 100
    fraction_pct = min(ceiling_pct, max(.5, round(suggested_pct * 2) / 2)) if simulation_position_pct is None else simulation_position_pct
    if fraction_pct < .5 or fraction_pct > ceiling_pct:
        raise ValueError("Simulation exposure must be between 0.5% and the conviction ceiling.")
    balance = portfolio_nav if starting_balance is None else starting_balance
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(f"{ticker}|{direction}|{horizon_label}|{conviction}".encode()).digest()[:4], "big") % 1_000_000
    session_rows = []
    current = float(balance)
    if not paths.empty and len(daily) >= horizon_days:
        for session in range(horizon_days):
            index = _session_index(len(daily), session, sampling_mode, seed, horizon_days)
            observed = float(daily.iloc[index]); nav_return = observed * fraction_pct / 100
            current = max(0.0, current * (1 + nav_return))
            session_rows.append({"session": session + 1, "date": daily.index[index].date().isoformat(),
                                 "ticker_return": observed, "nav_return": nav_return, "balance": current})
            if current <= 0:
                break
    caps = [
        ("Conviction ceiling", sizing.conviction_ceiling, f"Conviction {conviction} × 5%"),
        ("Volatility adjustment", sizing.volatility_cap, "Current 63D volatility versus full-history median"),
        ("Invalidation loss budget", sizing.invalidation_cap, "NAV loss budget divided by invalidation distance"),
        ("Earnings / event risk", sizing.event_cap, event_basis if hold_earnings else "Disabled"),
        ("Historical tail risk", sizing.tail_cap, "NAV loss budget divided by five-day adverse tail"),
        ("Liquidity", sizing.liquidity_cap, f"{participation}% of median dollar volume across {liquidation_days} days"),
    ]
    touch = first_touch_statistics(ohlcv, horizon_days, direction.lower(), max(target_move, 0), stop_distance, step=step)
    return {
        "source": "Yahoo Finance", "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "data_through": close.index[-1].date().isoformat(), "ticker": ticker,
        "history_sessions": len(close), "latest_price": latest,
        "trade": {"entry": entry, "target": target, "stop": stop, "target_move": target_move,
                  "stop_distance": stop_distance},
        "summary": {"conviction_ceiling": sizing.conviction_ceiling, "suggested_exposure": sizing.suggested_size,
                    "binding_constraint": sizing.binding_constraint, "position_notional": portfolio_nav * sizing.suggested_size,
                    "loss_at_invalidation": sizing.suggested_size * stop_distance,
                    "nav_impact_at_target": _number(sizing.suggested_size * target_move) if target_move > 0 else None,
                    "event_nav_loss": _number(sizing.suggested_size * event_move)},
        "caps": [{"Constraint": name, "Maximum position": value * 100,
                  "Reduction vs ceiling": (sizing.conviction_ceiling - value) * 100, "Method": method,
                  "Binding": name == sizing.binding_constraint or (name == "Earnings / event risk" and sizing.binding_constraint == "Earnings/event risk")}
                 for name, value, method in caps],
        "metrics": [{"Metric": key, "Value": _number(value)} for key, value in (
            ("Latest price", latest), ("21D realized volatility", annualized_volatility(returns, 21)),
            ("63D realized volatility", current_vol), ("252D realized volatility", annualized_volatility(returns, 252)),
            ("Historical median 63D volatility", median_vol),
            ("Maximum historical drawdown", maximum_drawdown_from_prices(close)),
            ("1% daily expected shortfall", daily_es), ("Five-day adverse tail", tail_move),
            ("Median daily dollar volume", median_dollar_volume))],
        "paths": records(paths), "touch": {key: _number(value) for key, value in touch.items()},
        "events": records(events.sort_values("date", ascending=False)), "event_move": _number(event_move),
        "event_basis": event_basis, "next_earnings": next_earnings,
        "sensitivities": _sensitivities(returns, frames, direction),
        "simulation": {"seed": seed, "mode": sampling_mode, "fraction_pct": fraction_pct,
                       "starting_balance": balance, "horizon_days": horizon_days, "sessions": session_rows,
                       "daily_hit_rate": _number(daily.gt(0).mean()),
                       "horizon_hit_rate": _number(paths["return"].gt(0).mean()) if not paths.empty else None,
                       "median_horizon_return": _number(paths["return"].median()) if not paths.empty else None,
                       "log_optimal": _log_optimal(daily, sizing.conviction_ceiling)},
        "diagnostics": records(missing), "warnings": (["Target is on the wrong side of entry; target-hit statistics are disabled."] if target_move <= 0 else [])
    }
