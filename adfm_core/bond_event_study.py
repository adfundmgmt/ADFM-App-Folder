"""Historical exhaustion studies on yield levels, never on inferred bond prices.

A top in yield is a potential bottom in the corresponding bond price. Every
signal uses observations available on its date; forward outcomes are basis-point
changes in the same yield series, with exact monthly periods when applicable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .global_macro import clean


PROFILES = ("Early Warning", "Confirmed Exhaustion", "Failed Breakout")
PRESETS = {
    "Early Warning": dict(change_pctile=90, trend_z=1.25, rsi=65, vol_pctile=60,
                          memory=5, reversal_components=0),
    "Confirmed Exhaustion": dict(change_pctile=90, trend_z=1.25, rsi=65, vol_pctile=60,
                                  memory=8, reversal_components=2),
    "Failed Breakout": dict(change_pctile=90, trend_z=1.25, rsi=65, vol_pctile=60,
                            memory=5, reversal_components=2),
}
HORIZONS = {"daily": {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126},
            "monthly": {"1M": 1, "3M": 3, "6M": 6, "12M": 12}}


def _percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    def rank(values):
        valid = values[np.isfinite(values)]
        return 100.0 * np.mean(valid <= values[-1]) if np.isfinite(values[-1]) and len(valid) else np.nan
    return series.rolling(window, min_periods=minimum).apply(rank, raw=True)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    total = up + down
    return (100 * up / total).where(total > 0, 50.0)


def signal_frame(rates: pd.Series, frequency: str, profile: str,
                 settings: dict | None = None) -> pd.DataFrame:
    if frequency not in HORIZONS or profile not in PROFILES:
        raise ValueError("Unsupported frequency or signal profile")
    setting = {**PRESETS[profile], **(settings or {})}
    yield_level = clean(rates)
    if frequency == "monthly" and not yield_level.empty:
        yield_level = yield_level.resample("MS").last()
    delta = yield_level.diff()
    short, medium, long = (5, 20, 200) if frequency == "daily" else (2, 6, 24)
    change = yield_level - yield_level.shift(medium)
    percentile = _percentile(change, 1260 if frequency == "daily" else 60,
                             252 if frequency == "daily" else 24)
    rolling_vol = delta.rolling(20 if frequency == "daily" else 6,
                                min_periods=15 if frequency == "daily" else 5).std(ddof=0)
    vol_pctile = _percentile(rolling_vol, 756 if frequency == "daily" else 60,
                             126 if frequency == "daily" else 24)
    average = yield_level.rolling(long, min_periods=int(long * .8)).mean()
    trend_z = (yield_level - average) / (rolling_vol * np.sqrt(medium)).replace(0, np.nan)
    rsi = _rsi(yield_level, 14 if frequency == "daily" else 6)
    ma_short = yield_level.rolling(short, min_periods=short).mean()
    prior_low = yield_level.shift(1).rolling(short, min_periods=short).min()
    reversal = ((yield_level < yield_level.shift(short)).astype(int)
                + (yield_level < ma_short).astype(int)
                + (yield_level < prior_low).astype(int))
    core = ((trend_z >= setting["trend_z"]).astype(int)
            + (rsi >= setting["rsi"]).astype(int)
            + (vol_pctile >= setting["vol_pctile"]).astype(int))
    watch = (core + (percentile >= setting["change_pctile"]).astype(int) >= 2) & yield_level.notna()
    setup = (percentile >= setting["change_pctile"]) & (core >= 2) & yield_level.notna()
    recent = setup.rolling(int(setting["memory"]), min_periods=1).max().astype(bool)
    if profile == "Early Warning":
        signal = setup
    elif profile == "Confirmed Exhaustion":
        signal = recent & (reversal >= int(setting["reversal_components"]))
    else:
        previous_high = yield_level.shift(1).rolling(long, min_periods=int(long * .8)).max()
        breakout = yield_level > previous_high
        pending = breakout.rolling(int(setting["memory"]), min_periods=1).max().astype(bool)
        signal = pending & (yield_level < previous_high) & (reversal >= int(setting["reversal_components"]))
    return pd.DataFrame({"Yield": yield_level, "ChangePctile": percentile, "TrendZ": trend_z,
                         "RSI": rsi, "VolPctile": vol_pctile, "ReversalScore": reversal,
                         "Watch": watch.fillna(False), "Setup": setup.fillna(False),
                         "Signal": signal.fillna(False)})


def event_dates(frame: pd.DataFrame, spacing: int) -> pd.DatetimeIndex:
    active = frame["Signal"].fillna(False).astype(bool)
    starts = np.flatnonzero((active & ~active.shift(1, fill_value=False)).to_numpy())
    kept = []
    for pos in starts:
        if not kept or pos - kept[-1] >= spacing:
            kept.append(int(pos))
    return pd.DatetimeIndex(frame.index[kept])


def event_summary(rates: pd.Series, events: pd.DatetimeIndex, frequency: str):
    if frequency not in HORIZONS:
        raise ValueError("Unsupported frequency")
    data = clean(rates)
    if frequency == "monthly" and not data.empty:
        data = data.resample("MS").last()
    positions = {date: index for index, date in enumerate(data.index)}
    event_positions = sorted({positions[date] for date in events if date in positions})
    history = []
    for pos in event_positions:
        row = {"Date": data.index[pos], "Yield (%)": float(data.iloc[pos])}
        for label, steps in HORIZONS[frequency].items():
            row[label] = _forward(data, pos, steps, frequency)
        history.append(row)
    history = pd.DataFrame(history, columns=["Date", "Yield (%)", *HORIZONS[frequency]])
    metrics = ("Signal median", "Baseline median", "Median edge", "% lower yield",
               "Baseline % lower", "Hit-rate lift", "Independent N")
    summary = pd.DataFrame(index=metrics, columns=list(HORIZONS[frequency]), dtype=float)
    for label, steps in HORIZONS[frequency].items():
        independent = []
        last = -10**9
        for pos in event_positions:
            if pos - last >= steps:
                result = _forward(data, pos, steps, frequency)
                if np.isfinite(result):
                    independent.append(result)
                    last = pos
        excluded = set(event_positions)
        baseline = []
        last = -10**9
        for pos in range(max(0, len(data) - steps)):
            if pos - last < steps or any(abs(pos - event) < steps for event in excluded):
                continue
            result = _forward(data, pos, steps, frequency)
            if np.isfinite(result):
                baseline.append(result)
                last = pos
        summary.loc["Independent N", label] = len(independent)
        if independent:
            summary.loc["Signal median", label] = np.median(independent)
            summary.loc["% lower yield", label] = 100 * np.mean(np.asarray(independent) < 0)
        if baseline:
            summary.loc["Baseline median", label] = np.median(baseline)
            summary.loc["Baseline % lower", label] = 100 * np.mean(np.asarray(baseline) < 0)
        if independent and baseline:
            summary.loc["Median edge", label] = np.median(independent) - np.median(baseline)
            summary.loc["Hit-rate lift", label] = (100 * np.mean(np.asarray(independent) < 0)
                                                   - 100 * np.mean(np.asarray(baseline) < 0))
    return summary, history


def _forward(data: pd.Series, pos: int, steps: int, frequency: str) -> float:
    end = pos + steps
    if end >= len(data) or pd.isna(data.iloc[pos]) or pd.isna(data.iloc[end]):
        return np.nan
    if frequency == "monthly" and data.index[end].to_period("M") != data.index[pos].to_period("M") + steps:
        return np.nan
    return float((data.iloc[end] - data.iloc[pos]) * 100)
