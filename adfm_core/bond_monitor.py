"""Date-aware rate snapshots for the bond monitor.

Yield levels are percentages. Changes are basis points, including for spreads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .global_macro import clean


DAILY_WINDOWS = {
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
}


def spread_series(long_leg: pd.Series, short_leg: pd.Series) -> pd.Series:
    """Subtract yields only on dates observed for both instruments."""
    paired = pd.concat([clean(long_leg), clean(short_leg)], axis=1, join="inner").dropna()
    if paired.empty:
        return pd.Series(dtype=float)
    return paired.iloc[:, 0] - paired.iloc[:, 1]


def _blank() -> dict:
    return {"Yield": np.nan, "1D": np.nan, "1W": np.nan, "1M": np.nan,
            "3M": np.nan, "YTD": np.nan, "Observation": "", "Status": "Unavailable"}


def daily_snapshot(series: pd.Series, today: pd.Timestamp) -> dict:
    result = _blank()
    now = pd.Timestamp(today).normalize()
    data = clean(series).loc[lambda x: x.index.normalize() <= now]
    if data.empty:
        return result
    end = data.index[-1]
    result["Yield"] = float(data.iloc[-1])
    result["Observation"] = end.strftime("%Y-%m-%d")
    if (now - end.normalize()).days > 7:
        result["Status"] = "Stale"
        return result
    result["Status"] = "Current"
    previous = data.iloc[:-1]
    if not previous.empty and (end - previous.index[-1]).days <= 4:
        result["1D"] = (data.iloc[-1] - previous.iloc[-1]) * 100
    for label, offset in DAILY_WINDOWS.items():
        anchor = end - offset
        baseline = data.loc[data.index <= anchor]
        if not baseline.empty and (anchor - baseline.index[-1]).days <= 4:
            result[label] = (data.iloc[-1] - baseline.iloc[-1]) * 100
    anchor = pd.Timestamp(end.year - 1, 12, 31)
    baseline = data.loc[data.index <= anchor]
    if not baseline.empty and (anchor - baseline.index[-1]).days <= 5:
        result["YTD"] = (data.iloc[-1] - baseline.iloc[-1]) * 100
    return result


def monthly_snapshot(series: pd.Series, today: pd.Timestamp) -> dict:
    result = _blank()
    current = pd.Timestamp(today).to_period("M")
    data = clean(series)
    data = pd.Series(data.to_numpy(), index=data.index.to_period("M"))
    data = data.loc[~data.index.duplicated(keep="last")]
    data = data.loc[data.index < current]
    if data.empty:
        return result
    end = data.index[-1]
    result["Yield"] = float(data.iloc[-1])
    result["Observation"] = str(end)
    if current.ordinal - end.ordinal > 4:
        result["Status"] = "Stale"
        return result
    result["Status"] = "Current"
    for label, steps in (("1M", 1), ("3M", 3), ("YTD", end.month)):
        baseline = end - steps
        if baseline in data.index:
            result[label] = (data.loc[end] - data.loc[baseline]) * 100
    return result
