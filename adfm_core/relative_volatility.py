"""Causal realized-volatility helpers for the Relative Volatility Lab."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def annualized_realized_volatility(
    close: pd.Series,
    window: int = 21,
    periods_per_year: int = 252,
) -> pd.Series:
    """Return close-to-close log-return volatility in annualized percent units."""
    window = max(int(window), 2)
    clean = pd.to_numeric(close, errors="coerce").where(lambda values: values > 0)
    log_returns = np.log(clean).diff()
    realized = log_returns.rolling(window, min_periods=window).std(ddof=1)
    return realized.mul(math.sqrt(int(periods_per_year)) * 100.0).rename("rvol")


def rolling_zscore_previous(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Standardize each value against trailing observations, excluding itself."""
    window = max(int(window), 2)
    required = max(2, min(int(min_periods or window), window))
    clean = pd.to_numeric(series, errors="coerce")
    prior = clean.shift(1)
    mean = prior.rolling(window, min_periods=required).mean()
    std = prior.rolling(window, min_periods=required).std(ddof=1).replace(0, np.nan)
    return clean.sub(mean).div(std).replace([np.inf, -np.inf], np.nan).rename("zscore")


def prior_percentile_rank(series: pd.Series) -> float:
    """Rank the latest finite value against all earlier finite observations."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    current = float(clean.iloc[-1])
    history = clean.iloc[:-1].to_numpy(dtype=float)
    below = float(np.sum(history < current))
    tied = float(np.sum(history == current))
    return ((below + 0.5 * tied) / len(history)) * 100.0


def relative_volatility_frame(
    primary_close: pd.Series,
    comparison_close: pd.Series,
    *,
    rvol_window: int,
    normalization_window: int,
    normalization_min_periods: int | None = None,
) -> pd.DataFrame:
    """Build aligned RVOL, ratio, and causal z-score series for two assets."""
    primary_rvol = annualized_realized_volatility(primary_close, rvol_window)
    comparison_rvol = annualized_realized_volatility(comparison_close, rvol_window)
    frame = pd.concat(
        {
            "primary_rvol": primary_rvol,
            "comparison_rvol": comparison_rvol,
        },
        axis=1,
    )
    frame["rvol_ratio"] = frame["primary_rvol"].div(
        frame["comparison_rvol"].replace(0, np.nan)
    )
    frame["primary_zscore"] = rolling_zscore_previous(
        frame["primary_rvol"], normalization_window, normalization_min_periods
    )
    frame["comparison_zscore"] = rolling_zscore_previous(
        frame["comparison_rvol"], normalization_window, normalization_min_periods
    )
    return frame.replace([np.inf, -np.inf], np.nan)
