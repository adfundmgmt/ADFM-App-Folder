"""Reusable, causal scoring helpers for ADFM regime dashboards."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


def rolling_percentile_previous(
    series: pd.Series,
    window: int,
    min_periods: int,
    scale: float = 1.0,
) -> pd.Series:
    """Rank each observation against prior observations only.

    Excluding the current observation avoids a small but persistent look-ahead bias
    in live percentile signals. Ties receive half credit.
    """
    clean = pd.to_numeric(series, errors="coerce")
    window = max(int(window), 2)
    min_periods = max(1, min(int(min_periods), window))

    def _rank(values: np.ndarray) -> float:
        current = values[-1]
        history = pd.Series(values[:-1]).dropna().to_numpy(dtype=float)
        if not np.isfinite(current) or len(history) < min_periods:
            return np.nan
        below = float(np.sum(history < current))
        tied = float(np.sum(history == current))
        return ((below + 0.5 * tied) / len(history)) * float(scale)

    return clean.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)


def grouped_weighted_composite(
    scores: pd.DataFrame,
    specs: Iterable[Mapping[str, object]],
    group_weights: Mapping[str, float] | None = None,
    min_groups: int = 3,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build a two-stage composite so proxy-rich groups do not dominate.

    Component weights are normalized inside each group. Group scores are then
    combined using the supplied group weights (or equal weights by default).
    Returns group scores, composite, positive-group breadth, and group coverage.
    """
    spec_rows = [dict(spec) for spec in specs]
    groups: Dict[str, list[dict]] = {}
    for spec in spec_rows:
        name = str(spec.get("name", ""))
        group = str(spec.get("category", "Other"))
        if name and name in scores.columns:
            groups.setdefault(group, []).append(spec)

    def _inside_group(row: pd.Series, member_weights: pd.Series) -> float:
        valid = row.dropna()
        if valid.empty:
            return np.nan
        active = member_weights.loc[valid.index]
        total = float(active.sum())
        return float((valid * active).sum() / total) if total > 0 else np.nan

    group_frame = pd.DataFrame(index=scores.index)
    for group, members in groups.items():
        names = [str(member["name"]) for member in members]
        weights = pd.Series(
            {
                str(member["name"]): float(member.get("weight", 1.0))
                for member in members
            }
        )
        group_frame[group] = scores[names].apply(
            _inside_group,
            axis=1,
            member_weights=weights,
        )

    if group_frame.empty:
        empty = pd.Series(index=scores.index, dtype=float)
        return group_frame, empty, empty, empty

    declared = pd.Series(
        {
            group: float((group_weights or {}).get(group, 1.0))
            for group in group_frame.columns
        }
    )

    def _across_groups(row: pd.Series) -> float:
        valid = row.dropna()
        if len(valid) < max(1, int(min_groups)):
            return np.nan
        active = declared.loc[valid.index]
        total = float(active.sum())
        return float((valid * active).sum() / total) if total > 0 else np.nan

    composite = group_frame.apply(_across_groups, axis=1)
    valid_groups = group_frame.notna().sum(axis=1)
    breadth = (group_frame.gt(0).sum(axis=1) / valid_groups.replace(0, np.nan)) * 100.0
    coverage = (valid_groups / max(1, len(group_frame.columns))) * 100.0
    breadth = breadth.where(valid_groups >= max(1, int(min_groups)))
    coverage = coverage.where(valid_groups > 0)
    return group_frame, composite, breadth, coverage
