"""Pure calculations for the equity leadership and rotation scanner.

The Streamlit page supplies relative-price series. This module converts those
series into stable cross-sectional scores that can be tested without loading
the page or making a market-data request.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

HORIZONS: tuple[tuple[str, int], ...] = (
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("6M", 126),
)

SCORE_WEIGHTS: Mapping[str, float] = {
    "1W": 0.20,
    "1M": 0.35,
    "3M": 0.30,
    "6M": 0.15,
}


def period_return(series: pd.Series, periods: int) -> float:
    """Return the trailing change over ``periods`` completed observations."""

    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= periods:
        return np.nan

    base = float(clean.iloc[-periods - 1])
    latest = float(clean.iloc[-1])
    if not np.isfinite(base) or base == 0 or not np.isfinite(latest):
        return np.nan
    return latest / base - 1.0


def centered_cross_sectional_rank(values: pd.Series) -> pd.Series:
    """Map valid cross-sectional ranks to a symmetric -100 to +100 scale."""

    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    ranked = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.empty:
        return ranked
    if len(valid) == 1:
        ranked.loc[valid.index] = 0.0
        return ranked

    ordinal = valid.rank(method="average")
    ranked.loc[valid.index] = ((ordinal - 1.0) / (len(valid) - 1.0) * 200.0) - 100.0
    return ranked


def classify_state(score: float, acceleration: float) -> str:
    """Classify leadership level and short-horizon momentum impulse."""

    if not np.isfinite(score) or not np.isfinite(acceleration):
        return "Unavailable"
    if score >= 0 and acceleration >= 0:
        return "Leading"
    if score >= 0 and acceleration < 0:
        return "Weakening"
    if score < 0 and acceleration >= 0:
        return "Improving"
    return "Lagging"


def build_leadership_frame(
    ratios: Mapping[str, pd.Series],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a ranked multi-horizon leadership table.

    ``metadata`` must be indexed by the same stable keys as ``ratios`` and may
    contain any descriptive columns needed by the presentation layer.
    """

    rows: list[dict[str, object]] = []
    for key, series in ratios.items():
        clean = (
            pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(clean) < 22:
            continue

        row: dict[str, object] = {
            "Key": key,
            "Last": float(clean.iloc[-1]),
            "As Of": pd.Timestamp(clean.index[-1]).date(),
        }
        for label, periods in HORIZONS:
            row[label] = period_return(clean, periods)

        ma50 = clean.rolling(50, min_periods=30).mean().iloc[-1]
        ma200 = clean.rolling(200, min_periods=100).mean().iloc[-1]
        row["vs 50D"] = float(clean.iloc[-1] / ma50 - 1.0) if np.isfinite(ma50) and ma50 else np.nan
        row["vs 200D"] = (
            float(clean.iloc[-1] / ma200 - 1.0) if np.isfinite(ma200) and ma200 else np.nan
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).set_index("Key")
    frame = metadata.join(frame, how="inner")

    rank_columns: list[str] = []
    for label, _ in HORIZONS:
        rank_name = f"{label} Rank"
        frame[rank_name] = centered_cross_sectional_rank(frame[label])
        rank_columns.append(rank_name)

    score = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    for label, weight in SCORE_WEIGHTS.items():
        rank_name = f"{label} Rank"
        valid = frame[rank_name].notna()
        score.loc[valid] += frame.loc[valid, rank_name] * weight
        available_weight.loc[valid] += weight
    frame["Leadership Score"] = score.div(available_weight.replace(0, np.nan))

    short_term = frame[["1W Rank", "1M Rank"]].mean(axis=1)
    medium_term = frame[["3M Rank", "6M Rank"]].mean(axis=1)
    frame["Acceleration"] = short_term - medium_term
    frame["State"] = [
        classify_state(score_value, acceleration)
        for score_value, acceleration in zip(
            frame["Leadership Score"], frame["Acceleration"], strict=True
        )
    ]

    above_50 = frame["vs 50D"] >= 0
    above_200 = frame["vs 200D"] >= 0
    frame["Trend"] = np.select(
        [above_50 & above_200, ~above_50 & ~above_200],
        ["Above 50D + 200D", "Below 50D + 200D"],
        default="Mixed",
    )
    frame = frame.sort_values(
        ["Leadership Score", "Acceleration"], ascending=[False, False]
    )
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    return frame


def summarize_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the scanner into a compact family-level read."""

    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby("Family", sort=False)
    summary = grouped.agg(
        **{
            "Leadership Score": ("Leadership Score", "mean"),
            "Acceleration": ("Acceleration", "mean"),
            "1M": ("1M", "mean"),
            "3M": ("3M", "mean"),
            "Relationships": ("Leadership Score", "size"),
        }
    )
    summary["Leading Share"] = grouped["Leadership Score"].apply(
        lambda values: float((values >= 0).mean())
    )
    summary = summary.sort_values("Leadership Score", ascending=False)
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
    return summary.reset_index()

