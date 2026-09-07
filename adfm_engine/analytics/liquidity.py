"""Original liquidity formulas and scoring; no dynamic loading or UI."""
from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from adfm_engine.analytics.liquidity_definitions import *

def latest(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def obs_change(series: pd.Series, periods: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1] - clean.iloc[-1 - periods]) if len(clean) > periods else np.nan


def fmt_score(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:+.2f}"


def fmt_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.0f}%"


def fmt_raw(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "N/A"
    if fmt == "mm_tn":
        return f"${value / 1_000_000:.2f}tn"
    if fmt == "bn_tn":
        return f"${value / 1_000:.2f}tn"
    if fmt == "pct_bp":
        return f"{value * 100:+.1f} bp"
    if fmt == "pct":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def score_bucket(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.90:
        return "Strong easing"
    if value >= 0.35:
        return "Easing"
    if value > -0.35:
        return "Mixed"
    if value > -0.90:
        return "Tightening"
    return "Strong tightening"


def classify_regime(level: float, impulse: float, breadth: float) -> Tuple[str, str]:
    if pd.isna(level) or pd.isna(impulse):
        return "Unavailable", "Insufficient primary-source coverage."
    broad_up = pd.notna(breadth) and breadth >= 60
    broad_down = pd.notna(breadth) and breadth <= 40
    if level >= 0.35 and impulse >= 0.35 and broad_up:
        return "Liquidity Expansion", "Conditions are easy and improving with broad confirmation."
    if level >= 0.35 and impulse <= -0.35:
        return "Easy, Deteriorating", "Liquidity remains supportive, but the marginal impulse is rolling over."
    if level <= -0.35 and impulse >= 0.35:
        return "Tight, Improving", "Conditions remain restrictive, but the marginal impulse has turned positive."
    if level <= -0.35 and impulse <= -0.35 and broad_down:
        return "Liquidity Contraction", "Conditions are restrictive and becoming tighter across the major sleeves."
    if impulse >= 0.35:
        return "Improving", "The marginal impulse is positive, but the level is not yet easy."
    if impulse <= -0.35:
        return "Deteriorating", "The marginal impulse is negative, though the level is not yet deeply tight."
    return "Neutral / Mixed", "Level and impulse are near trailing norms or offsetting one another."


def zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    mean = clean.rolling(window, min_periods=min_periods).mean()
    std = clean.rolling(window, min_periods=min_periods).std()
    return ((clean - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def change(series: pd.Series, periods: int, kind: str) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.pct_change(periods, fill_method=None) * 100 if kind == "pct" else clean.diff(periods)


def filter_lookback(obj: pd.Series | pd.DataFrame, lookback: str) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if out.empty or lookback == "max":
        return out
    offsets = {
        "6m": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
    }
    return out.loc[out.index >= out.index.max() - offsets[lookback]]


def rebase(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    valid = clean.dropna()
    return clean / valid.iloc[0] * 100 if not valid.empty and valid.iloc[0] != 0 else pd.Series(index=clean.index, dtype=float)


def color_score(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(x):
        return ""
    if x >= 0.90:
        return "background-color:#d9ead3;color:#274e13;"
    if x >= 0.35:
        return "background-color:#e2f0d9;color:#385723;"
    if x > -0.35:
        return "background-color:#f2f2f2;color:#404040;"
    if x > -0.90:
        return "background-color:#fce4d6;color:#843c0c;"
    return "background-color:#f4cccc;color:#990000;"


def market_tickers() -> List[str]:
    tickers = {"SPY", "QQQ"}
    for spec in MARKET_SPECS:
        if "ticker" in spec:
            tickers.add(str(spec["ticker"]))
        else:
            tickers.update((str(spec["numerator"]), str(spec["denominator"])))
    return sorted(tickers)


def build_primary(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    series_map: Dict[str, pd.Series] = {}
    specs: List[Dict[str, object]] = []
    for raw in PRIMARY_SPECS:
        spec = dict(raw)
        if spec.get("formula") == "spread":
            left, right = tuple(spec["inputs"])
            if left not in panel or right not in panel:
                continue
            series = panel[left] - panel[right]
        else:
            series_id = str(spec["series"])
            if series_id not in panel:
                continue
            series = panel[series_id]
        if series.dropna().shape[0] >= 180:
            series_map[str(spec["name"])] = series
            specs.append(spec)
    return (pd.DataFrame(series_map).sort_index().dropna(how="all"), specs) if series_map else (pd.DataFrame(), [])


def build_market_components(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    series_map: Dict[str, pd.Series] = {}
    specs: List[Dict[str, object]] = []
    for raw in MARKET_SPECS:
        spec = dict(raw)
        if "ticker" in spec:
            ticker = str(spec["ticker"])
            if ticker not in prices:
                continue
            series = prices[ticker]
            spec["display_ticker"] = ticker
        else:
            numerator, denominator = str(spec["numerator"]), str(spec["denominator"])
            if numerator not in prices or denominator not in prices:
                continue
            series = prices[numerator] / prices[denominator].replace(0, np.nan)
            spec["display_ticker"] = f"{numerator}/{denominator}"
        if series.dropna().shape[0] >= 180:
            series_map[str(spec["name"])] = series
            specs.append(spec)
    return (pd.DataFrame(series_map).sort_index().dropna(how="all"), specs) if series_map else (pd.DataFrame(), [])


def component_scores(components: pd.DataFrame, specs: Sequence[Mapping[str, object]], window: int, min_periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    levels = pd.DataFrame(index=components.index)
    impulses = pd.DataFrame(index=components.index)
    for spec in specs:
        name = str(spec["name"])
        raw = pd.to_numeric(components[name], errors="coerce")
        orientation = float(spec.get("orientation", 1.0))
        kind = str(spec.get("change_kind", "diff"))
        z21 = zscore(change(raw, 21, kind) * orientation, window, min_periods)
        z63 = zscore(change(raw, 63, kind) * orientation, window, min_periods)
        z126 = zscore(change(raw, 126, kind) * orientation, window, min_periods)
        impulses[name] = (0.50 * z21 + 0.35 * z63 + 0.15 * z126).clip(-3, 3)
        if bool(spec.get("include_level", True)):
            levels[name] = zscore(raw * orientation, window, min_periods).clip(-3, 3)
    return levels, impulses


def sleeve_composite(
    scores: pd.DataFrame,
    specs: Sequence[Mapping[str, object]],
    min_component_coverage: float,
    min_group_coverage: float,
    min_groups: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for raw in specs:
        spec = dict(raw)
        name = str(spec["name"])
        if name in scores and scores[name].notna().any():
            groups.setdefault(str(spec["category"]), []).append(spec)

    sleeves = pd.DataFrame(index=scores.index)
    for group, members in groups.items():
        names = [str(member["name"]) for member in members]
        weights = pd.Series({str(member["name"]): float(member.get("weight", 1.0)) for member in members})
        total = float(weights.sum())

        def score_row(
            row: pd.Series,
            weights: pd.Series = weights,
            total: float = total,
        ) -> float:
            valid = row.dropna()
            if valid.empty:
                return np.nan
            active = weights.loc[valid.index]
            if float(active.sum()) / total < min_component_coverage:
                return np.nan
            return float((valid * active).sum() / active.sum())

        sleeves[group] = scores[names].apply(score_row, axis=1)

    if sleeves.empty:
        empty = pd.Series(index=scores.index, dtype=float)
        return sleeves, empty, empty, empty

    group_weights = pd.Series({group: SLEEVE_WEIGHTS[group] for group in sleeves.columns})
    total_weight = float(group_weights.sum())
    composite: List[float] = []
    breadth: List[float] = []
    coverage: List[float] = []

    for _, row in sleeves.iterrows():
        valid = row.dropna()
        if valid.empty:
            composite.append(np.nan)
            breadth.append(np.nan)
            coverage.append(np.nan)
            continue
        active = group_weights.loc[valid.index]
        active_weight = float(active.sum())
        cov = active_weight / total_weight
        coverage.append(cov * 100)
        if len(valid) < min_groups or cov < min_group_coverage:
            composite.append(np.nan)
            breadth.append(np.nan)
            continue
        composite.append(float((valid * active).sum() / active_weight))
        positive_weight = float(active.loc[valid.index[valid > 0]].sum())
        breadth.append(positive_weight / active_weight * 100)

    return (
        sleeves,
        pd.Series(composite, index=sleeves.index, dtype=float),
        pd.Series(breadth, index=sleeves.index, dtype=float),
        pd.Series(coverage, index=sleeves.index, dtype=float),
    )


def scorecard(components: pd.DataFrame, levels: pd.DataFrame, impulses: pd.DataFrame, specs: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        name = str(spec["name"])
        if name not in components:
            continue
        rows.append(
            {
                "Sleeve": str(spec["category"]),
                "Component": name,
                "Latest": fmt_raw(latest(components[name]), str(spec.get("format", "index"))),
                "Level Score": latest(levels[name]) if name in levels else np.nan,
                "Impulse Score": latest(impulses[name]) if name in impulses else np.nan,
                "Signal": score_bucket(latest(impulses[name])) if name in impulses else "Unavailable",
                "Within-Sleeve Weight": float(spec.get("weight", 1.0)),
                "Source": str(spec.get("source", "Yahoo Finance")),
                "Description": str(spec.get("description", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["Sleeve", "Impulse Score"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame()


def _read_bucket(value: float, positive: str, negative: str) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.35:
        return positive
    if value <= -0.35:
        return negative
    return "Mixed"


def _color_score(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(x):
        return ""
    if x >= 0.90:
        return "background-color:#d9ead3;color:#274e13;"
    if x >= 0.35:
        return "background-color:#e2f0d9;color:#385723;"
    if x > -0.35:
        return "background-color:#f2f2f2;color:#404040;"
    if x > -0.90:
        return "background-color:#fce4d6;color:#843c0c;"
    return "background-color:#f4cccc;color:#990000;"


def compute_liquidity(fred, prices, z_window=756, min_periods=252, smoothing=3, lookback="5y"):
    primary, primary_specs = build_primary(fred)
    market, market_specs = (
        build_market_components(prices)
        if not prices.empty
        else (pd.DataFrame(), [])
    )


    primary_levels, primary_impulses = component_scores(
        primary,
        primary_specs,
        int(z_window),
        int(min_periods),
    )


    market_levels, market_impulses = (
        component_scores(
            market,
            market_specs,
            int(z_window),
            int(min_periods),
        )
        if not market.empty
        else (pd.DataFrame(), pd.DataFrame())
    )


    all_impulses = pd.concat(
        [primary_impulses, market_impulses],
        axis=1,
    ).sort_index()


    all_specs = primary_specs + market_specs


    sleeve_impulses, liquidity_impulse, easing_breadth, impulse_coverage = (
        sleeve_composite(
            all_impulses,
            all_specs,
            0.65,
            0.70,
            3,
        )
    )


    sleeve_levels, liquidity_level, _, level_coverage = sleeve_composite(
        primary_levels,
        primary_specs,
        0.65,
        0.70,
        2,
    )


    if int(smoothing) > 1:
        liquidity_impulse = liquidity_impulse.rolling(
            int(smoothing),
            min_periods=1,
        ).mean()
        liquidity_level = liquidity_level.rolling(
            int(smoothing),
            min_periods=1,
        ).mean()


    market_confirmation = (
        sleeve_impulses["Market Confirmation"]
        if "Market Confirmation" in sleeve_impulses
        else pd.Series(index=liquidity_impulse.index, dtype=float)
    )


    display_level = filter_lookback(liquidity_level, lookback)


    display_impulse = filter_lookback(liquidity_impulse, lookback)


    display_sleeve_impulses = filter_lookback(sleeve_impulses, lookback)


    current_level = latest(display_level)


    current_impulse = latest(display_impulse)


    level_read = _read_bucket(current_level, "Easy", "Tight")


    impulse_read = _read_bucket(current_impulse, "Improving", "Deteriorating")


    return locals()
