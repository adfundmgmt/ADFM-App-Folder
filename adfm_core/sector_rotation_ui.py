"""Presentation helpers for Sector Breadth and Rotation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

STATE_PALETTE = {
    "Leading": "#B9DFC4",
    "Improving": "#BFDDEC",
    "Weakening": "#F4DEAE",
    "Lagging": "#E9B9BD",
    "Neutral": "#DDE2E7",
}

STATE_EDGE = {
    "Leading": "#3A7D52",
    "Improving": "#4E86A0",
    "Weakening": "#A67A22",
    "Lagging": "#A44D55",
    "Neutral": "#7B8791",
}

POSITIVE_FILL = "#CFE8D8"
POSITIVE_FILL_STRONG = "#9FD0B0"
NEGATIVE_FILL = "#F1D1D3"
NEGATIVE_FILL_STRONG = "#E5A6AA"
NEUTRAL_FILL = "#F6F7F8"
BREADTH_FILL = "#D9EEE1"
BREADTH_FILL_STRONG = "#A8D4B6"


def display_name(ticker: str, industry: str) -> str:
    """Return a compact human-readable label for ETF or synthetic exposures."""
    ticker_text = str(ticker)
    if ticker_text.startswith("BASKET_"):
        return str(industry)
    return ticker_text


def robust_axis_range(
    values: Iterable[float],
    lower_quantile: float = 0.08,
    upper_quantile: float = 0.92,
    min_span: float = 0.06,
) -> tuple[float, float]:
    """Create a robust chart range so one extreme point cannot flatten the map."""
    clean = pd.Series(list(values), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (-0.05, 0.05)

    if len(clean) < 5:
        lo = float(clean.min())
        hi = float(clean.max())
    else:
        lo = float(clean.quantile(lower_quantile))
        hi = float(clean.quantile(upper_quantile))

    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = max(hi - lo, min_span)
    pad = max(span * 0.12, 0.006)
    return lo - pad, hi + pad


def clip_to_axis(values: pd.Series, axis_range: tuple[float, float]) -> pd.Series:
    """Clip plotted positions to a robust range while preserving true values for hover."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.clip(lower=axis_range[0], upper=axis_range[1])


def select_auto_labels(
    frame: pd.DataFrame,
    selected_ids: Sequence[str] | None = None,
    max_labels: int = 14,
) -> set[str]:
    """Label selected exposures plus the most visually informative outliers."""
    if frame.empty or max_labels <= 0:
        return set()

    selected = [item for item in (selected_ids or []) if item in set(frame["Id"])]
    keep: list[str] = list(dict.fromkeys(selected))[:max_labels]
    remaining = max_labels - len(keep)
    if remaining <= 0:
        return set(keep)

    work = frame[["Id", "Map X", "Map Y", "5D Speed"]].copy()
    score = pd.Series(0.0, index=work.index)
    for col in ("Map X", "Map Y", "5D Speed"):
        numeric = pd.to_numeric(work[col], errors="coerce")
        median = numeric.median()
        mad = (numeric - median).abs().median()
        if pd.notna(mad) and mad > 0:
            score = score.add((numeric - median).abs() / mad, fill_value=0.0)
        else:
            score = score.add(numeric.abs().rank(pct=True), fill_value=0.0)
    work["_score"] = score
    ranked = work[~work["Id"].isin(keep)].sort_values("_score", ascending=False)
    keep.extend(ranked["Id"].head(remaining).tolist())
    return set(keep)


def _signed_fill(value: object) -> str:
    if pd.isna(value):
        return f"background-color: {NEUTRAL_FILL}; color: #6B7280"
    numeric = float(value)
    if numeric > 0:
        fill = POSITIVE_FILL_STRONG if numeric >= 0.10 else POSITIVE_FILL
    elif numeric < 0:
        fill = NEGATIVE_FILL_STRONG if numeric <= -0.10 else NEGATIVE_FILL
    else:
        fill = NEUTRAL_FILL
    return f"background-color: {fill}; color: #111111"


def _rank_fill(value: object) -> str:
    if pd.isna(value):
        return f"background-color: {NEUTRAL_FILL}; color: #6B7280"
    numeric = float(value)
    if numeric > 0:
        fill = POSITIVE_FILL_STRONG if numeric >= 5 else POSITIVE_FILL
    elif numeric < 0:
        fill = NEGATIVE_FILL_STRONG if numeric <= -5 else NEGATIVE_FILL
    else:
        fill = NEUTRAL_FILL
    return f"background-color: {fill}; color: #111111"


def _breadth_fill(value: object) -> str:
    if pd.isna(value):
        return f"background-color: {NEUTRAL_FILL}; color: #6B7280"
    numeric = float(value)
    if numeric >= 75:
        fill = BREADTH_FILL_STRONG
    elif numeric >= 50:
        fill = BREADTH_FILL
    elif numeric <= 25:
        fill = NEGATIVE_FILL
    else:
        fill = NEUTRAL_FILL
    return f"background-color: {fill}; color: #111111"


def _state_fill(value: object) -> str:
    fill = STATE_PALETTE.get(str(value), NEUTRAL_FILL)
    return f"background-color: {fill}; color: #111111; font-weight: 600"


def style_rotation_table(frame: pd.DataFrame):
    """Return a compact heatmapped Styler matching the Public Equities table language."""
    display = frame.copy()
    if "ETF" in display.columns and "Industry" in display.columns:
        display["ETF"] = [
            display_name(ticker, industry)
            for ticker, industry in zip(display["ETF"], display["Industry"], strict=True)
        ]

    styler = display.style
    signed_candidates = (
        "1W Rel",
        "1M Rel",
        "3M Rel",
        "Weekly Rel Change",
        "1W Δ Rel",
        "1M Abs",
        "vs Parent 1M",
        "vs Parent",
        "Dist. 50D",
        "vs 50D",
        "52W Drawdown",
        "52W DD",
        "Breadth 1M Chg",
        "Breadth Δ",
    )
    signed_cols = [col for col in signed_candidates if col in display.columns]
    if signed_cols:
        styler = styler.map(_signed_fill, subset=signed_cols)

    rank_cols = [col for col in ("Weekly Rank Change", "Rank Δ") if col in display.columns]
    if rank_cols:
        styler = styler.map(_rank_fill, subset=rank_cols)

    breadth_cols = [
        col
        for col in ("Above 50D", "Above 200D", ">50D", ">200D")
        if col in display.columns
    ]
    if breadth_cols:
        styler = styler.map(_breadth_fill, subset=breadth_cols)
    if "State" in display.columns:
        styler = styler.map(_state_fill, subset=["State"])

    percent_candidates = (
        "1W Rel",
        "1M Rel",
        "3M Rel",
        "Weekly Rel Change",
        "1W Δ Rel",
        "1M Abs",
        "vs Parent 1M",
        "vs Parent",
        "Dist. 50D",
        "vs 50D",
        "52W Drawdown",
        "52W DD",
    )
    percent_cols = [col for col in percent_candidates if col in display.columns]
    formats = {col: "{:+.1%}" for col in percent_cols}

    for col in ("Above 50D", "Above 200D", ">50D", ">200D"):
        if col in display.columns:
            formats[col] = "{:.0f}%"
    for col in ("Breadth 1M Chg", "Breadth Δ"):
        if col in display.columns:
            formats[col] = "{:+.0f}%"
    for col in rank_cols:
        formats[col] = "{:+.0f}"

    styler = styler.format(formats, na_rep="N/A")
    return styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#111111"),
                    ("font-weight", "600"),
                    ("border", "1px solid #E5E7EB"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #ECEFF1"),
                    ("white-space", "nowrap"),
                ],
            },
        ]
    )


__all__ = [
    "STATE_EDGE",
    "STATE_PALETTE",
    "clip_to_axis",
    "display_name",
    "robust_axis_range",
    "select_auto_labels",
    "style_rotation_table",
]
