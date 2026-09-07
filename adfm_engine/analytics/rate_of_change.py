"""Pure calculations behind the ADFM Rate of Change Regime Explorer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

SMA_WINDOWS = (21, 50, 100, 200)


def compute_features(frame: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    """Add trend, momentum, acceleration, and inflection fields to OHLCV data."""
    if "Close" not in frame:
        raise ValueError("Rate-of-change features require a Close column.")
    if roc_period <= 0:
        raise ValueError("roc_period must be positive.")
    out = frame.copy()
    price = out["Close"].astype(float)
    for window in SMA_WINDOWS:
        out[f"SMA_{window}"] = price.rolling(window, min_periods=1).mean()
    # Keep pandas' established percentage-change behavior used by the original
    # dashboard so this extraction does not alter the displayed calculations.
    out["ROC"] = price.pct_change(roc_period)
    out["ROC_Slope"] = out["ROC"].diff(5)
    out["Second_Derivative"] = out["ROC"].diff()
    out["Second_Derivative_Slope"] = out["Second_Derivative"].diff(3)
    previous_acceleration = out["Second_Derivative"].shift(1)
    out["Pos_Inflect"] = (previous_acceleration <= 0) & (out["Second_Derivative"] > 0)
    out["Neg_Inflect"] = (previous_acceleration >= 0) & (out["Second_Derivative"] < 0)
    return out


def padded_range(
    series_list: Sequence[pd.Series],
    pad_pct: float = 0.05,
    include_zero: bool = False,
) -> Optional[list[float]]:
    """Return a finite plotting range around one or more series."""
    values: list[pd.Series] = []
    for series in series_list:
        if series is None:
            continue
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not clean.empty:
            values.append(clean)
    if not values:
        return None
    combined = pd.concat(values)
    if include_zero:
        combined = pd.concat([combined, pd.Series([0.0])])
    minimum, maximum = float(combined.min()), float(combined.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    if minimum == maximum:
        base = abs(minimum) if minimum else 1.0
        pad = base * 0.05
        return [minimum - pad, maximum + pad]
    pad = (maximum - minimum) * pad_pct
    return [minimum - pad, maximum + pad]


def add_trading_session_axis(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compressed trading-session coordinates and stable date labels."""
    out = frame.copy()
    out["Session"] = np.arange(len(out))
    out["Date_Label"] = out.index.strftime("%Y-%m-%d")
    return out


def make_date_ticks(
    index: pd.DatetimeIndex,
    session_values: pd.Series,
    max_ticks: int = 10,
) -> tuple[list[int], list[str]]:
    """Create readable x-axis labels for a compressed trading-session axis."""
    if len(index) == 0:
        return [], []
    tick_count = min(max_ticks, max(2, len(index)))
    positions = np.unique(np.linspace(0, len(index) - 1, tick_count).round().astype(int))
    tickvals = session_values.iloc[positions].tolist()
    if len(index) > 900:
        ticktext = index[positions].strftime("%Y").tolist()
    elif len(index) > 260:
        ticktext = index[positions].strftime("%b %Y").tolist()
    else:
        ticktext = index[positions].strftime("%b %d").tolist()
    return tickvals, ticktext
