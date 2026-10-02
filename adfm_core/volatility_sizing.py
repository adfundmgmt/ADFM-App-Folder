"""Transparent inverse-volatility sizing, with a lagged historical baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureScale:
    target: float
    change: float
    uncapped: float
    multiplier: float
    loss_cap: float | None = None
    binding: str = "Volatility reference"


def scale_exposure(base: float, current: float, recent: float, baseline: float, ceiling: float,
                   *, loss_budget: float | None = None, stop_distance: float | None = None) -> ExposureScale:
    """Exposure magnitudes are NAV fractions; volatility is annualized."""
    values = np.asarray([base, current, recent, baseline, ceiling], dtype=float)
    if not np.isfinite(values).all() or min(base, current, ceiling) < 0 or min(recent, baseline) <= 0:
        raise ValueError("Sizing needs nonnegative exposures and finite positive volatility.")
    multiplier = baseline / recent
    uncapped = base * multiplier
    caps = {"Volatility reference": uncapped, "Exposure ceiling": ceiling}
    loss_cap = None
    if stop_distance is not None:
        if loss_budget is None or not np.isfinite([loss_budget, stop_distance]).all() or loss_budget < 0 or stop_distance <= 0:
            raise ValueError("Invalidation sizing needs a nonnegative NAV loss budget and positive stop distance.")
        loss_cap = loss_budget / stop_distance
        caps["Invalidation loss budget"] = loss_cap
    binding = min(caps, key=caps.get)
    target = caps[binding]
    return ExposureScale(target, target - current, uncapped, multiplier, loss_cap, binding)


def invalidation_distance(price: float, invalidation: float, direction: str) -> float | None:
    """Measure distance from the latest mark, not a historic entry price."""
    if direction.lower() not in {"long", "short"}:
        raise ValueError("Direction must be Long or Short.")
    if not np.isfinite([price, invalidation]).all() or price <= 0 or invalidation < 0:
        raise ValueError("Prices must be finite and nonnegative, with a positive current mark.")
    if invalidation == 0:
        return None
    distance = (1 - invalidation / price) if direction.lower() == "long" else (invalidation / price - 1)
    if distance <= 0:
        raise ValueError("Invalidation must be below the latest price for a long and above it for a short. Set it to 0 to disable the loss budget.")
    return distance


def volatility_history(close: pd.Series, window: int = 20) -> pd.DataFrame:
    """Recent rolling sigma against the median of 252 prior rolling observations.

    The baseline ends before the current return window. Every history row uses
    only observations available on that date; no backfilled baseline is used.
    """
    if window < 2:
        raise ValueError("Volatility window must contain at least two sessions.")
    prices = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices.loc[prices.gt(0)].sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    returns = prices.pct_change(fill_method=None)
    recent = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    baseline = recent.shift(window).rolling(252, min_periods=252).median()
    frame = pd.DataFrame({"recent": recent, "baseline": baseline})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame.loc[frame.recent.gt(1e-8) & frame.baseline.gt(1e-8)]


def volatility_context(close: pd.Series) -> pd.DataFrame:
    """10/20/60-session levels, rank versus 252 prior observations, and 20-session change."""
    prices = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices.loc[prices.gt(0)].sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    returns = prices.pct_change(fill_method=None)
    rows = []
    for window in (10, 20, 60):
        rolling = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
        recent = float(rolling.iloc[-1]) if len(rolling) else np.nan
        prior = rolling.shift(window).tail(252).dropna()
        baseline = float(prior.median()) if len(prior) == 252 else np.nan
        percentile = float(prior.le(recent).mean()*100) if len(prior) == 252 and np.isfinite(recent) else np.nan
        previous = float(rolling.iloc[-21]) if len(rolling) > 20 else np.nan
        change = recent / previous - 1 if previous > 1e-8 else np.nan
        rows.append({"window": window, "recent": recent, "baseline": baseline,
                     "percentile": percentile, "change": change})
    return pd.DataFrame(rows).set_index("window")


def downside_statistics(adjusted_frame: pd.DataFrame, direction: str) -> dict[str, float]:
    """Adverse moves of the entered instrument over the available adjusted-price history.

    Five-session price returns assume a fixed initial notional. Short returns
    negate that cumulative underlying return, rather than compounding negated
    daily returns. Tail averages use the worst ceil(5% of observations).
    """
    if direction.lower() not in {"long", "short"}:
        raise ValueError("Direction must be Long or Short.")
    sign = 1 if direction.lower() == "long" else -1
    frame = adjusted_frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    close = pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).where(lambda values: values > 0)
    daily = close.pct_change(fill_method=None).mul(sign).dropna()
    weekly = close.pct_change(periods=5, fill_method=None).mul(sign).dropna()
    stats = {"Average worst 5% day": np.nan, "Average worst 5% week": np.nan,
             "Worst adverse day": np.nan, "Worst adverse gap": np.nan}
    if len(daily) >= 100:
        stats["Average worst 5% day"] = max(0.0, -float(daily.nsmallest(int(np.ceil(len(daily)*.05))).mean()))
        stats["Worst adverse day"] = max(0.0, -float(daily.min()))
    if len(weekly) >= 100:
        stats["Average worst 5% week"] = max(0.0, -float(weekly.nsmallest(int(np.ceil(len(weekly)*.05))).mean()))
    if "Open" in frame:
        opens = pd.to_numeric(frame["Open"], errors="coerce").where(lambda values: values > 0)
        gaps = (opens / close.shift(1) - 1).replace([np.inf, -np.inf], np.nan).mul(sign).dropna()
        if len(gaps) >= 20:
            stats["Worst adverse gap"] = max(0.0, -float(gaps.min()))
    return stats


def comparison_table(current: float, result: ExposureScale, stats: dict[str, float], direction: str,
                     *, daily_sigma: float | None = None, stop_distance: float | None = None,
                     nav: float = 0) -> pd.DataFrame:
    """One auditable table: exposures, common shocks, and resulting position-only NAV moves."""
    sign = 1 if direction.lower() == "long" else -1
    sizes = {"Current": current, "Half size": current/2,
             "Vol reference": result.uncapped, "Permitted": result.target}
    rows = [{"Scenario": "Exposure / NAV", "Market move": np.nan,
             **{label: size*100 for label, size in sizes.items()}}]
    scenarios = {}
    if stop_distance is not None:
        scenarios["At invalidation"] = stop_distance
    if daily_sigma is not None:
        scenarios["2σ adverse day"] = daily_sigma * 2
    scenarios.update({"5% adverse move": .05, "10% adverse move": .10})
    scenarios.update(stats)
    for label, move in scenarios.items():
        rows.append({"Scenario": label, "Market move": -sign*move*100,
                     **{label: -size*move*100 for label, size in sizes.items()}})
    if nav > 0:
        rows.append({"Scenario": "Position notional (USD)", "Market move": np.nan,
                     **{label: size*nav for label, size in sizes.items()}})
    return pd.DataFrame(rows)
