"""Pure calculations for the ADFM PM command center."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data_registry import COCKPIT_PROXIES, ProxyDefinition
from .market_data import safe_divide

HORIZONS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}
SCORE_WEIGHTS = {"1W": 0.25, "1M": 0.45, "3M": 0.30}


@dataclass(frozen=True)
class CockpitSummary:
    """Top-level state derived from the current signal cross-section."""

    regime: str
    composite: float
    confidence: float
    breadth: float
    dispersion: float
    impulse: float
    as_of: str | None
    available_signals: int
    total_signals: int


def _clean_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def _constructive_proxy(prices: pd.DataFrame, definition: ProxyDefinition) -> pd.Series:
    if definition.numerator not in prices:
        return pd.Series(dtype=float)
    numerator = _clean_series(prices[definition.numerator])
    if definition.denominator:
        if definition.denominator not in prices:
            return pd.Series(dtype=float)
        denominator = _clean_series(prices[definition.denominator])
        aligned = pd.concat([numerator, denominator], axis=1, join="inner").dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        proxy = safe_divide(aligned.iloc[:, 0], aligned.iloc[:, 1])
    else:
        proxy = numerator
    if definition.direction < 0:
        proxy = safe_divide(pd.Series(1.0, index=proxy.index), proxy)
    return _clean_series(proxy)


def causal_percentile_score(
    series: pd.Series,
    periods: int,
    *,
    history_window: int = 504,
    min_history: int = 126,
) -> float:
    """Score the latest log change against prior changes only, from -1 to +1."""

    clean = _clean_series(series)
    if len(clean) <= periods:
        return np.nan
    changes = np.log(clean.where(clean > 0)).diff(periods).dropna()
    if changes.empty:
        return np.nan
    current = float(changes.iloc[-1])
    prior = changes.iloc[max(0, len(changes) - history_window - 1) : -1]
    if len(prior) < min_history:
        return np.nan
    less = float((prior < current).sum())
    equal = float((prior == current).sum())
    percentile = (less + 0.5 * equal) / len(prior)
    return float(np.clip(percentile * 2.0 - 1.0, -1.0, 1.0))


def _trend_score(series: pd.Series) -> float:
    clean = _clean_series(series)
    if len(clean) < 126:
        return np.nan
    latest = float(clean.iloc[-1])
    ma63 = float(clean.iloc[-63:].mean())
    ma126 = float(clean.iloc[-126:].mean())
    score = 0.5 * np.sign(latest - ma63) + 0.5 * np.sign(ma63 - ma126)
    return float(np.clip(score, -1.0, 1.0))


def build_signal_snapshot(
    prices: pd.DataFrame,
    definitions: Sequence[ProxyDefinition] = COCKPIT_PROXIES,
) -> pd.DataFrame:
    """Build the current cross-asset signal table from an unfilled close panel."""

    rows: list[dict[str, object]] = []
    for definition in definitions:
        proxy = _constructive_proxy(prices, definition)
        scores = {
            label: causal_percentile_score(proxy, periods)
            for label, periods in HORIZONS.items()
        }
        trend = _trend_score(proxy)
        weighted = [
            (scores[label], weight)
            for label, weight in SCORE_WEIGHTS.items()
            if np.isfinite(scores[label])
        ]
        if np.isfinite(trend):
            weighted.append((trend, 0.20))
        weight_sum = sum(weight for _, weight in weighted)
        composite = (
            sum(value * weight for value, weight in weighted) / weight_sum
            if weight_sum
            else np.nan
        )
        available = sum(np.isfinite(value) for value in scores.values())
        confidence = min(1.0, available / len(HORIZONS))
        if np.isfinite(trend):
            confidence = min(1.0, confidence + 0.10)
        rows.append(
            {
                "Signal": definition.label,
                "Key": definition.key,
                "Group": definition.group,
                **scores,
                "Trend": trend,
                "Composite": composite,
                "Impulse": (
                    scores["1W"] - scores["3M"]
                    if np.isfinite(scores["1W"]) and np.isfinite(scores["3M"])
                    else np.nan
                ),
                "Confidence": confidence,
                "Data Through": (
                    proxy.index.max().date().isoformat() if not proxy.empty else None
                ),
                "Description": definition.description,
            }
        )
    return pd.DataFrame(rows)


def _regime_label(score: float) -> str:
    if not np.isfinite(score):
        return "Unavailable"
    if score >= 0.40:
        return "Broad Risk-On"
    if score >= 0.12:
        return "Constructive"
    if score <= -0.40:
        return "Broad Risk-Off"
    if score <= -0.12:
        return "Defensive"
    return "Mixed / Transitional"


def summarize_snapshot(snapshot: pd.DataFrame) -> CockpitSummary:
    """Summarize cross-sectional direction, confidence, and disagreement."""

    if snapshot.empty or "Composite" not in snapshot:
        return CockpitSummary(
            regime="Unavailable",
            composite=np.nan,
            confidence=0.0,
            breadth=0.0,
            dispersion=np.nan,
            impulse=np.nan,
            as_of=None,
            available_signals=0,
            total_signals=len(snapshot),
        )
    scores = pd.to_numeric(snapshot["Composite"], errors="coerce").dropna()
    confidence = pd.to_numeric(snapshot["Confidence"], errors="coerce")
    impulse = pd.to_numeric(snapshot["Impulse"], errors="coerce").dropna()
    composite = float(scores.mean()) if not scores.empty else np.nan
    dates = pd.to_datetime(snapshot["Data Through"], errors="coerce").dropna()
    return CockpitSummary(
        regime=_regime_label(composite),
        composite=composite,
        confidence=float(confidence.mean()) if confidence.notna().any() else 0.0,
        breadth=float((scores > 0.10).mean()) if not scores.empty else 0.0,
        dispersion=float(scores.std(ddof=0)) if len(scores) > 1 else 0.0,
        impulse=float(impulse.mean()) if not impulse.empty else np.nan,
        as_of=dates.max().date().isoformat() if not dates.empty else None,
        available_signals=len(scores),
        total_signals=len(snapshot),
    )


def group_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signals by analytical sleeve without filling missing values."""

    if snapshot.empty:
        return pd.DataFrame(columns=["Group", "Composite", "Impulse", "Signals"])
    return (
        snapshot.groupby("Group", as_index=False)
        .agg(
            Composite=("Composite", "mean"),
            Impulse=("Impulse", "mean"),
            Signals=("Composite", "count"),
        )
        .sort_values("Composite", ascending=False)
        .reset_index(drop=True)
    )


def largest_changes(
    snapshot: pd.DataFrame, limit: int = 3
) -> Mapping[str, pd.DataFrame]:
    """Return the strongest improving and deteriorating near-term impulses."""

    frame = snapshot.dropna(subset=["Impulse"]).copy()
    return {
        "improving": frame.nlargest(limit, "Impulse"),
        "deteriorating": frame.nsmallest(limit, "Impulse"),
    }
