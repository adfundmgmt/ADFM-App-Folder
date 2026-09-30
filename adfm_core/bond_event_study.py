"""Historical exhaustion studies on yield levels, never on inferred bond prices.

A top in yield is a potential bottom in the corresponding bond price. Every
daily signal uses observations through its date; monthly studies use
a clearly labeled month-end availability proxy, not verified release dates; forward outcomes are basis-point
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
    # method=max exactly preserves the <= tie-inclusive empirical percentile.
    return series.rolling(window, min_periods=minimum).rank(method="max", pct=True) * 100


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
        yield_level = monthly_history(yield_level)
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
        # Each breakout owns its original level and expiry. Later highs create
        # independent pending setups without resetting older confirmation windows.
        active = []
        signals = []
        for pos, (level, prior) in enumerate(zip(yield_level, previous_high, strict=True)):
            active = [(start, anchor) for start, anchor in active
                      if pos - start <= int(setting["memory"])]
            confirmed = reversal.iloc[pos] >= int(setting["reversal_components"])
            failed = [(start, anchor) for start, anchor in active if level < anchor and confirmed]
            signals.append(bool(failed))
            active = [setup for setup in active if setup not in failed]
            if np.isfinite(prior) and level > prior:
                active.append((pos, float(prior)))
        signal = pd.Series(signals, index=yield_level.index)
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


def monthly_history(rates: pd.Series) -> pd.Series:
    """Month-end availability proxy; actual release dates/vintages are unknown.

    Reindexing retains missing calendar months so horizon and excursion windows
    cannot silently skip missing periods. This is retrospective revised data.
    """
    data = clean(rates)
    return data.resample("ME").last() if not data.empty else data


def regime_labels(data: pd.Series, frequency: str) -> pd.Series:
    """Trailing trend direction and volatility relative to its trailing history."""
    window = 20 if frequency == "daily" else 6
    mean = data.rolling(window, min_periods=window).mean()
    vol = data.diff().rolling(window, min_periods=window).std()
    reference = vol.rolling(window * 3, min_periods=window).median()
    labels = pd.Series("Unknown", index=data.index)
    valid = mean.notna() & reference.notna() & data.notna()
    labels.loc[valid] = (np.where(data.loc[valid] >= mean.loc[valid], "Rising", "Falling")
                         + np.where(vol.loc[valid] >= reference.loc[valid], "/high vol", "/low vol"))
    return labels


def _interval(data: pd.Series, pos: int, steps: int, frequency: str) -> np.ndarray:
    end = pos + steps
    if end >= len(data):
        return np.array([])
    path = data.iloc[pos:end + 1]
    if path.isna().any():
        return np.array([])
    if frequency == "monthly" and data.index[end].to_period("M") != data.index[pos].to_period("M") + steps:
        return np.array([])
    return (path.to_numpy(dtype=float) - float(path.iloc[0])) * 100


def _forward(data: pd.Series, pos: int, steps: int, frequency: str) -> float:
    path = _interval(data, pos, steps, frequency)
    return float(path[-1]) if len(path) else np.nan


def _edge_ci(signals: list, controls: list) -> tuple[float, float]:
    if len(signals) < 2 or len(controls) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(20260929)
    # Resample BOTH populations; treating the estimated control median as fixed
    # would understate uncertainty. Windows have already been de-overlapped.
    draws = (np.median(rng.choice(signals, (1000, len(signals))), axis=1)
             - np.median(rng.choice(controls, (1000, len(controls))), axis=1))
    return tuple(np.quantile(draws, [.025, .975]))


def event_summary(rates: pd.Series, events: pd.DatetimeIndex, frequency: str):
    if frequency not in HORIZONS:
        raise ValueError("Unsupported frequency")
    data = monthly_history(rates) if frequency == "monthly" else clean(rates)
    events = pd.DatetimeIndex(events)
    if frequency == "monthly":
        events = events.to_period("M").to_timestamp("M")
    positions = {date: index for index, date in enumerate(data.index)}
    event_positions = sorted({positions[date] for date in events if date in positions})
    split = int(len(data) * .7)
    regimes = regime_labels(data, frequency)
    columns = ["Date", "Yield (%)", "Regime", "Partition"]
    for label in HORIZONS[frequency]:
        columns += [label, f"{label} adverse (bp)", f"{label} favorable (bp)"]
    history = []
    for pos in event_positions:
        row = {"Date": data.index[pos], "Yield (%)": float(data.iloc[pos]),
               "Regime": regimes.iloc[pos], "Partition": "Train" if pos < split else "Holdout"}
        for label, steps in HORIZONS[frequency].items():
            path = _interval(data, pos, steps, frequency)
            row[label] = float(path[-1]) if len(path) else np.nan
            row[f"{label} adverse (bp)"] = float(path.max()) if len(path) else np.nan
            row[f"{label} favorable (bp)"] = float(path.min()) if len(path) else np.nan
        history.append(row)
    history = pd.DataFrame(history, columns=columns)
    metrics = ("Signal median", "Baseline median", "Median edge", "% lower yield",
               "Baseline % lower", "Hit-rate lift", "Independent N", "Control N",
               "Matched signal N", "Edge CI low", "Edge CI high", "Median adverse (bp)",
               "Median favorable (bp)", "Train N", "Holdout N", "Train edge",
               "Holdout edge", "Holdout CI low", "Holdout CI high", "Holdout control N")
    summary = pd.DataFrame(index=metrics, columns=list(HORIZONS[frequency]), dtype=float)
    summary.attrs.update(cautions={}, controls={}, matches={}, split_date=data.index[split].isoformat() if len(data) else None)
    levels = data.to_numpy(dtype=float)
    regime_values = regimes.to_numpy()
    for label, steps in HORIZONS[frequency].items():
        # Reverse rolling extrema give the full forward path in O(history).
        reverse = pd.Series(levels[::-1])
        forward_max = reverse.rolling(steps + 1, min_periods=steps + 1).max().to_numpy()[::-1]
        forward_min = reverse.rolling(steps + 1, min_periods=steps + 1).min().to_numpy()[::-1]
        outcomes = np.full(len(levels), np.nan)
        if steps < len(levels):
            outcomes[:-steps] = (levels[steps:] - levels[:-steps]) * 100
        eligible_outcome = np.isfinite(forward_max) & np.isfinite(forward_min)
        outcomes[~eligible_outcome] = np.nan
        adverse = (forward_max - levels) * 100
        favorable = (forward_min - levels) * 100
        independent = []
        last = -10**9
        for pos in event_positions:
            if pos - last >= steps and np.isfinite(outcomes[pos]):
                independent.append(pos)
                last = pos
        summary.loc["Independent N", label] = len(independent)
        values = outcomes[independent].tolist()
        if values:
            summary.loc["Signal median", label] = np.median(values)
            summary.loc["% lower yield", label] = 100 * np.mean(np.asarray(values) < 0)
            summary.loc["Median adverse (bp)", label] = np.median(adverse[independent])
            summary.loc["Median favorable (bp)", label] = np.median(favorable[independent])
        available = np.isfinite(outcomes) & (regime_values != "Unknown")
        # Block any control window overlapping ANY signal window; half-open
        # windows may share the boundary observation but no return interval.
        for event in event_positions:
            available[max(0, event - steps + 1):event + steps] = False
        positions_array = np.arange(len(data))
        groups = {}
        for regime in np.unique(regime_values):
            groups[(regime, "Train")] = positions_array[available & (regime_values == regime) & (positions_array + steps < split)]
            groups[(regime, "Holdout")] = positions_array[available & (regime_values == regime) & (positions_array >= split)]
        matches, used = [], []
        for event in independent:
            regime = regime_values[event]
            if regime == "Unknown" or event < split <= event + steps:
                continue
            partition = "Holdout" if event >= split else "Train"
            candidates = groups[(regime, partition)]
            candidates = candidates[available[candidates]]
            if len(candidates):
                control = int(candidates[np.argmin(np.abs(candidates - event))])
                used.append(control)
                matches.append((event, control))
                available[max(0, control - steps + 1):control + steps] = False
        matched_values = [outcomes[e] for e, _ in matches]
        baseline = [outcomes[c] for _, c in matches]
        summary.attrs["controls"][label] = sorted(used)
        summary.attrs["matches"][label] = matches
        summary.loc["Control N", label] = len(baseline)
        summary.loc["Matched signal N", label] = len(matched_values)
        if baseline:
            summary.loc["Baseline median", label] = np.median(baseline)
            summary.loc["Baseline % lower", label] = 100 * np.mean(np.asarray(baseline) < 0)
            # The edge uses matched signals, not unmatched/unknown regimes.
            summary.loc["Median edge", label] = np.median(matched_values) - np.median(baseline)
            summary.loc["Hit-rate lift", label] = 100 * (np.mean(np.asarray(matched_values) < 0) - np.mean(np.asarray(baseline) < 0))
            summary.loc[["Edge CI low", "Edge CI high"], label] = _edge_ci(matched_values, baseline)
        for partition, condition in (("Train", lambda p, steps=steps: p + steps < split), ("Holdout", lambda p: p >= split)):
            subset = [(e, c) for e, c in matches if condition(e)]
            n = sum(condition(p) for p in independent)
            summary.loc[f"{partition} N", label] = n
            if partition == "Holdout":
                summary.loc["Holdout control N", label] = len(subset)
            if subset:
                sv = [outcomes[e] for e, _ in subset]
                bv = [outcomes[c] for _, c in subset]
                summary.loc[f"{partition} edge", label] = np.median(sv) - np.median(bv)
                if partition == "Holdout":
                    summary.loc[["Holdout CI low", "Holdout CI high"], label] = _edge_ci(sv, bv)
        summary.attrs["cautions"][label] = ("small sample: fewer than 20 independent events or controls" if min(len(independent), len(baseline)) < 20 else "Descriptive retrospective study; regimes and parameters are not independently optimized")
    return summary, history
