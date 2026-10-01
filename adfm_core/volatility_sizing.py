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


def scale_exposure(base: float, current: float, recent: float, baseline: float, ceiling: float) -> ExposureScale:
    """Exposure magnitudes are NAV fractions; volatility is annualized."""
    values = np.asarray([base, current, recent, baseline, ceiling], dtype=float)
    if not np.isfinite(values).all() or min(base, current, ceiling) < 0 or min(recent, baseline) <= 0:
        raise ValueError("Sizing needs nonnegative exposures and finite positive volatility.")
    multiplier = baseline / recent
    uncapped = base * multiplier
    target = min(uncapped, ceiling)
    return ExposureScale(target, target - current, uncapped, multiplier)


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
