"""Original Yahoo Treasury-yield normalization and regime logic."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL, PASTEL_RATES_SCALE
TITLE = "Yield Curve Rates Regime Monitor"


YAHOO_YIELD_TICKERS: Dict[str, Dict[str, object]] = {
    "^IRX": {"label": "3M", "field": "Y3M", "years": 0.25},
    "^FVX": {"label": "5Y", "field": "Y5", "years": 5.0},
    "^TNX": {"label": "10Y", "field": "Y10", "years": 10.0},
    "^TYX": {"label": "30Y", "field": "Y30", "years": 30.0},
}


YIELD_LABELS = {
    str(v["field"]): str(v["label"]) for v in YAHOO_YIELD_TICKERS.values()
}


YIELD_TICKER_TO_FIELD = {
    ticker: str(meta["field"]) for ticker, meta in YAHOO_YIELD_TICKERS.items()
}


FIELD_TO_TICKER = {
    field: ticker for ticker, field in YIELD_TICKER_TO_FIELD.items()
}


PERIODS: Dict[str, Dict[str, object]] = {
    "Today": {"kind": "row", "rows": 1, "threshold": 4},
    "1W": {"kind": "calendar", "days": 7, "threshold": 8},
    "1M": {"kind": "calendar", "months": 1, "threshold": 15},
    "3M": {"kind": "calendar", "months": 3, "threshold": 30},
    "YTD": {"kind": "ytd", "threshold": 35},
}


CURVE_OPTIONS = ["3m10y", "5s10s", "10s30s", "5s30s"]


COLORS = {
    "ink": "#111111",
    "muted": "#666666",
    "border": "#d0d0d0",
    "soft": "#f5f5f3",
    "blue": PASTEL["blue"],
    "purple": PASTEL["lavender"],
    "green": PASTEL["sage"],
    "red": PASTEL["rose"],
    "amber": PASTEL["amber"],
    "slate": PASTEL["slate_blue"],
    "grey": "#A8ADB5",
}


def safe_float(x: object) -> float:
    try:
        value = float(x)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return safe_float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    clean = df.dropna(how="all")
    if clean.empty:
        return None
    return pd.Timestamp(clean.index[-1])


def value_on_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index <= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[-1])


def first_value_on_or_after(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index >= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[0])


def anchor_value(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan

    last_idx = pd.Timestamp(clean.index[-1])
    spec = PERIODS[period]

    if spec["kind"] == "row":
        rows = int(spec.get("rows", 1))
        if len(clean) <= rows:
            return np.nan
        return safe_float(clean.iloc[-rows - 1])

    if spec["kind"] == "calendar":
        if "months" in spec:
            target = last_idx - pd.DateOffset(months=int(spec["months"]))
        else:
            target = last_idx - pd.DateOffset(days=int(spec.get("days", 0)))
        return value_on_or_before(clean, target)

    if spec["kind"] == "ytd":
        jan_first = pd.Timestamp(date(last_idx.year, 1, 1))
        return first_value_on_or_after(clean, jan_first)

    return np.nan


def change_bp(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan
    last_value = safe_float(clean.iloc[-1])
    anchor = anchor_value(clean, period)
    if not np.isfinite(last_value) or not np.isfinite(anchor):
        return np.nan
    return float((last_value - anchor) * 100.0)


def fmt_pct(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.2f}%"


def fmt_bp(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:+.0f} bp"


def normalize_yahoo_yield_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    median = out.dropna().tail(260).median()
    if np.isfinite(median) and median > 20:
        out = out / 10.0
    return out


def split_yahoo_yields(close: pd.DataFrame) -> pd.DataFrame:
    yield_cols = [
        ticker for ticker in YAHOO_YIELD_TICKERS if ticker in close.columns
    ]
    yields = (
        close[yield_cols].copy()
        if yield_cols
        else pd.DataFrame(index=close.index)
    )
    for ticker in yield_cols:
        yields[ticker] = normalize_yahoo_yield_series(yields[ticker])
    yields = yields.rename(columns=YIELD_TICKER_TO_FIELD)
    return yields.ffill().dropna(how="all")


def add_derived_yahoo_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"Y10", "Y3M"}.issubset(out.columns):
        out["3m10y"] = out["Y10"] - out["Y3M"]
    if {"Y10", "Y5"}.issubset(out.columns):
        out["5s10s"] = out["Y10"] - out["Y5"]
    if {"Y30", "Y10"}.issubset(out.columns):
        out["10s30s"] = out["Y30"] - out["Y10"]
    if {"Y30", "Y5"}.issubset(out.columns):
        out["5s30s"] = out["Y30"] - out["Y5"]

    return out


def available_curve_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in CURVE_OPTIONS
        if c in df.columns and df[c].dropna().any()
    ]


def label_for_series(col: str) -> str:
    labels = {
        "Y3M": "3M",
        "Y5": "5Y",
        "Y10": "10Y",
        "Y30": "30Y",
        "3m10y": "3M/10Y",
        "5s10s": "5s10s",
        "10s30s": "10s30s",
        "5s30s": "5s30s",
    }
    return labels.get(col, col)


def classify_regime(
    df: pd.DataFrame, period: str, curve_col: str
) -> Tuple[str, str, str]:
    ten = change_bp(df["Y10"], period) if "Y10" in df else np.nan
    curve = change_bp(df[curve_col], period) if curve_col in df else np.nan
    threshold = float(PERIODS[period]["threshold"])

    if not np.isfinite(ten):
        return (
            "Insufficient Data",
            "Need a valid Yahoo 10Y series.",
            COLORS["amber"],
        )

    if not np.isfinite(curve):
        if ten > threshold:
            return (
                "Rates Rising",
                f"10Y yield up {fmt_bp(ten)} over {period}; selected curve unavailable.",
                COLORS["red"],
            )
        if ten < -threshold:
            return (
                "Rates Falling",
                f"10Y yield down {fmt_bp(ten)} over {period}; selected curve unavailable.",
                COLORS["green"],
            )
        return (
            "Range / Mixed",
            f"10Y move is inside the {threshold:.0f} bp signal band.",
            COLORS["amber"],
        )

    if abs(ten) < threshold and abs(curve) < threshold:
        return (
            "Range / Mixed",
            f"10Y and {label_for_series(curve_col)} are inside the {threshold:.0f} bp signal band.",
            COLORS["amber"],
        )
    if ten > threshold and curve > threshold:
        return (
            "Bear Steepener",
            f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.",
            COLORS["red"],
        )
    if ten > threshold and curve < -threshold:
        return (
            "Bear Flattener",
            f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.",
            COLORS["red"],
        )
    if ten < -threshold and curve > threshold:
        return (
            "Bull Steepener",
            f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.",
            COLORS["green"],
        )
    if ten < -threshold and curve < -threshold:
        return (
            "Bull Flattener",
            f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.",
            COLORS["green"],
        )
    if ten > threshold:
        return (
            "Bearish Rates Impulse",
            f"10Y up {fmt_bp(ten)}; curve signal is mixed.",
            COLORS["red"],
        )
    if ten < -threshold:
        return (
            "Bullish Rates Impulse",
            f"10Y down {fmt_bp(ten)}; curve signal is mixed.",
            COLORS["green"],
        )
    return (
        "Curve Signal",
        f"10Y quiet, but {label_for_series(curve_col)} moved {fmt_bp(curve)} over {period}.",
        COLORS["amber"],
    )


def regime_read(regime: str) -> str:
    reads = {
        "Bear Steepener": (
            "Long-end yields are rising and the curve is steepening. The market is "
            "adding duration pressure while raising the probability that nominal growth, "
            "term premium, fiscal supply, or some combination of the three is dominating."
        ),
        "Bear Flattener": (
            "Yields are rising while the curve compresses. That is a tighter-policy or "
            "front-end pressure regime rather than a clean reflation signal."
        ),
        "Bull Steepener": (
            "Yields are falling while the curve steepens. The market is moving toward "
            "easier policy, weaker growth, or a stronger duration bid."
        ),
        "Bull Flattener": (
            "Yields are falling while the curve flattens. The long end is outperforming "
            "the front of the available curve, consistent with a stronger duration bid."
        ),
        "Bearish Rates Impulse": (
            "The outright level move matters more than curve shape. Duration pressure is "
            "rising, but the selected curve is not confirming a clean steepening or flattening regime."
        ),
        "Bullish Rates Impulse": (
            "The outright level move is supportive for duration, while the selected curve "
            "has not moved enough to define a clean steepener or flattener."
        ),
        "Curve Signal": (
            "Curve shape is moving more than the outright 10Y level. The information is in "
            "policy-path and term-premium repricing rather than a broad duration shock."
        ),
        "Range / Mixed": (
            "Neither the 10Y level nor the selected curve has cleared the regime threshold. "
            "Treat the rates tape as range-bound until one side breaks."
        ),
    }
    return reads.get(
        regime,
        "Signal quality is low. Check data freshness and missing Yahoo yield symbols.",
    )


def period_matrix(df: pd.DataFrame, rows: List[str]) -> pd.DataFrame:
    out: List[Dict[str, object]] = []

    for col in rows:
        if col not in df.columns or df[col].dropna().empty:
            continue

        row = {
            "Series": label_for_series(col),
            "Latest": latest(df[col]),
        }
        for period in PERIODS:
            row[period] = change_bp(df[col], period)
        out.append(row)

    return pd.DataFrame(out)


def curve_comparison_values(
    curve_data: pd.DataFrame, compare_window: str
) -> Tuple[pd.Series, List[float]]:
    latest_curve = curve_data.iloc[-1]
    last_idx = pd.Timestamp(curve_data.index[-1])

    if compare_window == "1W":
        compare_target = last_idx - pd.DateOffset(days=7)
    elif compare_window == "1M":
        compare_target = last_idx - pd.DateOffset(months=1)
    elif compare_window == "3M":
        compare_target = last_idx - pd.DateOffset(months=3)
    else:
        compare_target = pd.Timestamp(date(last_idx.year, 1, 1))

    comparison_curve: List[float] = []
    for tenor in curve_data.columns:
        if compare_window == "YTD":
            comparison_curve.append(
                first_value_on_or_after(curve_data[tenor], compare_target)
            )
        else:
            comparison_curve.append(
                value_on_or_before(curve_data[tenor], compare_target)
            )

    return latest_curve, comparison_curve


