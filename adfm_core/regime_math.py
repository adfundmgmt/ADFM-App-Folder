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

    The reference sample excludes the current observation. Ties receive half
    credit. No future observations enter the calculation.
    """
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    window = max(int(window), 2)
    min_periods = max(1, min(int(min_periods), window))

    rolling = clean.rolling(window + 1, min_periods=min_periods + 1)
    # Average rank = prior values below + half prior ties + 1 (the current
    # observation). Subtracting 1 preserves the previous-only formula exactly.
    return (rolling.rank(method="average") - 1.0).div(rolling.count() - 1.0) * float(scale)


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

    group_frame = pd.DataFrame(index=scores.index)
    for group, members in groups.items():
        names = [str(member["name"]) for member in members]
        weights = pd.Series(
            {
                str(member["name"]): float(member.get("weight", 1.0))
                for member in members
            }
        )
        members_frame = scores[names]
        total = members_frame.notna().mul(weights).sum(axis=1)
        group_frame[group] = members_frame.mul(weights).sum(axis=1, min_count=1).div(
            total.where(total > 0)
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

    valid_groups = group_frame.notna().sum(axis=1)
    total = group_frame.notna().mul(declared).sum(axis=1)
    composite = group_frame.mul(declared).sum(axis=1, min_count=1).div(
        total.where(total > 0)
    ).where(valid_groups >= max(1, int(min_groups)))
    breadth = (group_frame.gt(0).sum(axis=1) / valid_groups.replace(0, np.nan)) * 100.0
    coverage = (valid_groups / max(1, len(group_frame.columns))) * 100.0
    breadth = breadth.where(valid_groups >= max(1, int(min_groups)))
    coverage = coverage.where(valid_groups > 0)
    return group_frame, composite, breadth, coverage
