"""History-aware ranks and explicitly vintage-selected macro alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_context(series: pd.Series | None, years: int) -> tuple[float, str]:
    if series is None:
        return np.nan, "Unavailable"
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(clean) < 30:
        return np.nan, "Insufficient history"
    cutoff = clean.index[-1] - pd.DateOffset(years=years)
    window = clean.loc[clean.index >= cutoff]
    if len(window) < 30:
        return np.nan, "Insufficient history"
    span = (window.index[-1] - window.index[0]).days / 365.25
    complete = clean.index[0] <= cutoff + pd.Timedelta(days=7)
    label = f"{years}Y" if complete else f"{span:.2f}Y available"
    return float(window.le(window.iloc[-1]).mean()), label


def align_available_observations(series: pd.Series, dates: pd.DatetimeIndex, *, max_age_days: int = 7) -> pd.Series:
    """Align daily rates without extending old observations indefinitely."""
    clean = series.dropna().sort_index()
    if clean.empty:
        return pd.Series(np.nan, index=dates)
    left = pd.DataFrame({"date": dates}).sort_values("date")
    right = clean.rename("value").rename_axis("observed").reset_index()
    result = pd.merge_asof(left, right, left_on="date", right_on="observed", direction="backward", tolerance=pd.Timedelta(days=max_age_days))
    return result.set_index("date")["value"].reindex(dates)
