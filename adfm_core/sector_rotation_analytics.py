"""Pure sector-rotation calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from adfm_core.sector_rotation_catalog import (
    BASKET_MIN_COVERAGE,
    BREADTH_MIN_COVERAGE,
    MOVEMENT_LOOKBACK,
    NEUTRAL_MAP_THRESHOLD,
    STATE_CONFIRM_DAYS,
)


@dataclass(frozen=True)
class RotationWindow:
    short_window: int
    long_window: int
    short_label: str
    long_label: str


def compute_equal_weight_basket(prices: pd.DataFrame, members: Sequence[str], min_coverage: float = BASKET_MIN_COVERAGE) -> pd.Series:
    members = [m for m in members if m in prices.columns]
    if not members:
        return pd.Series(index=prices.index, dtype=float)
    member_prices = prices[members].apply(pd.to_numeric, errors="coerce")
    returns = member_prices.pct_change(fill_method=None)
    coverage = returns.notna().sum(axis=1) / max(len(members), 1)
    basket_return = returns.mean(axis=1, skipna=True).where(coverage >= min_coverage)
    level = (1.0 + basket_return.fillna(0.0)).cumprod() * 100.0
    level = level.where(basket_return.notna())
    if len(level):
        initial_coverage = member_prices.iloc[0].notna().sum() / max(len(members), 1)
        level.iloc[0] = 100.0 if initial_coverage >= min_coverage else np.nan
    return level


def build_asset_levels(raw_prices: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    levels = pd.DataFrame(index=raw_prices.index)
    for _, row in catalog.iterrows():
        key = row["Key"]
        if row["Kind"] == "Stock Basket":
            levels[key] = compute_equal_weight_basket(raw_prices, row["Members"] or [])
        else:
            ticker = row["Ticker"]
            levels[key] = pd.to_numeric(raw_prices[ticker], errors="coerce") if ticker in raw_prices else np.nan
    return levels


def trailing_return(series: pd.Series, periods: int, end_offset: int = 0) -> float:
    s = pd.to_numeric(series, errors="coerce")
    end_pos = len(s) - 1 - end_offset
    start_pos = end_pos - periods
    if start_pos < 0 or end_pos < 0:
        return np.nan
    start, end = s.iloc[start_pos], s.iloc[end_pos]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return np.nan
    return float(end / start - 1.0)


def relative_series(asset: pd.Series, benchmark: pd.Series) -> pd.Series:
    aligned = pd.concat([asset, benchmark], axis=1)
    aligned.columns = ["asset", "benchmark"]
    return aligned["asset"].div(aligned["benchmark"]).replace([np.inf, -np.inf], np.nan)


def relative_return(asset: pd.Series, benchmark: pd.Series, periods: int, end_offset: int = 0) -> float:
    return trailing_return(relative_series(asset, benchmark), periods, end_offset=end_offset)


def distance_from_ma(series: pd.Series, window: int = 50) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if len(s) < window or pd.isna(s.iloc[-1]):
        return np.nan
    ma = s.rolling(window, min_periods=window).mean().iloc[-1]
    return np.nan if pd.isna(ma) or ma == 0 else float(s.iloc[-1] / ma - 1.0)


def drawdown_from_high(series: pd.Series, window: int = 252) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.empty or pd.isna(s.iloc[-1]):
        return np.nan
    high = s.iloc[-window:].max(skipna=True)
    return np.nan if pd.isna(high) or high == 0 else float(s.iloc[-1] / high - 1.0)


def classify_quadrant(long_value: float, short_value: float, neutral_threshold: float = NEUTRAL_MAP_THRESHOLD) -> str:
    if pd.isna(long_value) or pd.isna(short_value):
        return "Neutral"
    if abs(float(long_value)) <= neutral_threshold or abs(float(short_value)) <= neutral_threshold:
        return "Neutral"
    if long_value > 0 and short_value > 0:
        return "Leading"
    if long_value < 0 and short_value > 0:
        return "Improving"
    if long_value < 0 and short_value < 0:
        return "Lagging"
    if long_value > 0 and short_value < 0:
        return "Weakening"
    return "Neutral"


def map_coordinate_history(relative: pd.Series, window: RotationWindow) -> pd.DataFrame:
    rel = pd.to_numeric(relative, errors="coerce")
    return pd.DataFrame({
        "x": rel.pct_change(window.long_window, fill_method=None),
        "y": rel.pct_change(window.short_window, fill_method=None),
    }).replace([np.inf, -np.inf], np.nan)


def confirmed_state_series(raw_states: pd.Series, confirm_days: int = STATE_CONFIRM_DAYS) -> pd.Series:
    raw = raw_states.astype("object")
    if raw.empty:
        return raw.copy()
    output: List[str] = []
    current = str(raw.iloc[0])
    candidate = None
    count = 0
    for value in raw:
        state = str(value)
        if state == current:
            candidate, count = None, 0
        else:
            if state == candidate:
                count += 1
            else:
                candidate, count = state, 1
            if count >= confirm_days:
                current, candidate, count = state, None, 0
        output.append(current)
    return pd.Series(output, index=raw.index, dtype="object")


def latest_state_metrics(confirmed: pd.Series) -> tuple[str, int]:
    if confirmed.empty:
        return "Neutral", 0
    state = str(confirmed.iloc[-1])
    days = 0
    for value in reversed(confirmed.tolist()):
        if str(value) != state:
            break
        days += 1
    return state, max(days, 1)


def movement_metrics(coords: pd.DataFrame, lookback: int = MOVEMENT_LOOKBACK) -> tuple[float, float, float, float]:
    clean = coords.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    if len(clean) <= lookback:
        return np.nan, np.nan, np.nan, np.nan
    latest, prior = clean.iloc[-1], clean.iloc[-(lookback + 1)]
    dx, dy = float(latest["x"] - prior["x"]), float(latest["y"] - prior["y"])
    return dx, dy, float(math.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)))


def _benchmark_series(raw_prices: pd.DataFrame, ticker: str) -> pd.Series:
    if not ticker or ticker not in raw_prices.columns:
        return pd.Series(index=raw_prices.index, dtype=float)
    return pd.to_numeric(raw_prices[ticker], errors="coerce")


def compute_snapshot(raw_prices: pd.DataFrame, asset_levels: pd.DataFrame, catalog: pd.DataFrame, window: RotationWindow) -> pd.DataFrame:
    records: List[dict] = []
    for _, item in catalog.iterrows():
        key = item["Key"]
        if key not in asset_levels.columns:
            continue
        asset = pd.to_numeric(asset_levels[key], errors="coerce")
        broad_name = item["Broad Benchmark"]
        broad = _benchmark_series(raw_prices, broad_name)
        rel = relative_series(asset, broad)
        coords = map_coordinate_history(rel, window)
        raw_states = pd.Series([classify_quadrant(x, y) for x, y in zip(coords["x"], coords["y"])], index=coords.index)
        confirmed = confirmed_state_series(raw_states, STATE_CONFIRM_DAYS)
        state, days = latest_state_metrics(confirmed)
        dx, dy, speed, angle = movement_metrics(coords, MOVEMENT_LOOKBACK)
        parent_name = item["Parent Benchmark"]
        parent_rel = relative_return(asset, _benchmark_series(raw_prices, parent_name), 21) if parent_name else np.nan
        map_x = float(coords["x"].iloc[-1]) if len(coords) and pd.notna(coords["x"].iloc[-1]) else np.nan
        map_y = float(coords["y"].iloc[-1]) if len(coords) and pd.notna(coords["y"].iloc[-1]) else np.nan
        weekly_now, weekly_prior = trailing_return(rel, 5), trailing_return(rel, 5, end_offset=5)
        records.append({
            "Key": key, "Ticker": item["Ticker"], "Name": item["Name"], "Industry": item["Name"],
            "Sector Group": item["Sector Group"], "Tier": item["Tier"], "Universe": item["Universe"],
            "Kind": item["Kind"], "Broad Benchmark": broad_name, "Parent Benchmark": parent_name,
            "State": state, "Days in State": days,
            "1W Rel": relative_return(asset, broad, 5), "1M Rel": relative_return(asset, broad, 21),
            "3M Rel": relative_return(asset, broad, 63),
            "Weekly Rel Δ": weekly_now - weekly_prior if pd.notna(weekly_now) and pd.notna(weekly_prior) else np.nan,
            "1M Abs": trailing_return(asset, 21), "Parent 1M Rel": parent_rel,
            "vs 50D": distance_from_ma(asset, 50), "DD from 52W High": drawdown_from_high(asset, 252),
            "Map X": map_x, "Map Y": map_y, "Movement ΔX": dx, "Movement ΔY": dy,
            "Movement Speed": speed, "Movement Angle": angle,
        })
    snap = pd.DataFrame(records)
    if snap.empty:
        return snap
    snap["Rank"] = snap["1M Rel"].rank(method="min", ascending=False, na_option="bottom").astype(int)
    prior_scores = {}
    for _, item in catalog.iterrows():
        key = item["Key"]
        if key in asset_levels:
            prior_scores[key] = relative_return(asset_levels[key], _benchmark_series(raw_prices, item["Broad Benchmark"]), 21, end_offset=5)
    prior_rank = pd.Series(prior_scores, dtype=float).rank(method="min", ascending=False, na_option="bottom")
    snap["Prior Rank"] = snap["Key"].map(prior_rank)
    snap["Rank Δ"] = snap["Prior Rank"] - snap["Rank"]
    return snap.sort_values(["Rank", "Name"], na_position="last").reset_index(drop=True)


def _breadth_at(prices: pd.DataFrame, members: Sequence[str], end_pos: int, min_coverage: float) -> dict:
    members = [m for m in members if m in prices.columns]
    if not members or end_pos < 0:
        return {"above50": np.nan, "above200": np.nan, "coverage": 0.0}
    frame = prices[members].apply(pd.to_numeric, errors="coerce")
    latest = frame.iloc[end_pos]
    ma50 = frame.rolling(50, min_periods=50).mean().iloc[end_pos]
    ma200 = frame.rolling(200, min_periods=200).mean().iloc[end_pos]
    eligible50, eligible200 = latest.notna() & ma50.notna(), latest.notna() & ma200.notna()
    coverage50, coverage200 = float(eligible50.sum() / len(members)), float(eligible200.sum() / len(members))
    above50 = float((latest[eligible50] > ma50[eligible50]).mean()) if coverage50 >= min_coverage else np.nan
    above200 = float((latest[eligible200] > ma200[eligible200]).mean()) if coverage200 >= min_coverage else np.nan
    return {"above50": above50, "above200": above200, "coverage": min(coverage50, coverage200)}


def compute_breadth(prices: pd.DataFrame, members: Sequence[str], min_coverage: float = BREADTH_MIN_COVERAGE) -> dict:
    if prices.empty:
        return {"% > 50D": np.nan, "% > 200D": np.nan, "Breadth 1M Δ": np.nan, "Coverage": 0.0}
    current = _breadth_at(prices, members, len(prices) - 1, min_coverage)
    prior_pos = len(prices) - 22
    prior = _breadth_at(prices, members, prior_pos, min_coverage) if prior_pos >= 0 else {"above50": np.nan}
    change = current["above50"] - prior["above50"] if pd.notna(current["above50"]) and pd.notna(prior.get("above50")) else np.nan
    return {"% > 50D": current["above50"], "% > 200D": current["above200"], "Breadth 1M Δ": change, "Coverage": current["coverage"]}


def attach_breadth(snapshot: pd.DataFrame, raw_prices: pd.DataFrame, breadth_members: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    out = snapshot.copy()
    rows = []
    for key in out["Key"]:
        members = breadth_members.get(key, [])
        rows.append(compute_breadth(raw_prices, members) if members else {"% > 50D": np.nan, "% > 200D": np.nan, "Breadth 1M Δ": np.nan, "Coverage": np.nan})
    breadth = pd.DataFrame(rows, index=out.index)
    out["% > 50D"], out["% > 200D"] = breadth["% > 50D"], breadth["% > 200D"]
    out["Breadth 1M Δ"], out["Breadth Coverage"] = breadth["Breadth 1M Δ"], breadth["Coverage"]
    return out


def build_breadth_member_map(catalog: pd.DataFrame, sector_holdings: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for _, row in catalog.iterrows():
        if row["Kind"] == "Stock Basket":
            mapping[row["Key"]] = list(row["Members"] or [])
        elif row["Universe"] == "Sectors" and row["Ticker"] in sector_holdings:
            mapping[row["Key"]] = list(sector_holdings[row["Ticker"]])
    return mapping


def trail_for_selected(raw_prices: pd.DataFrame, asset_levels: pd.DataFrame, catalog: pd.DataFrame, key: str, window: RotationWindow, trail_weeks: int) -> pd.DataFrame:
    if trail_weeks <= 0 or key not in asset_levels.columns:
        return pd.DataFrame(columns=["x", "y"])
    match = catalog[catalog["Key"] == key]
    if match.empty:
        return pd.DataFrame(columns=["x", "y"])
    broad = _benchmark_series(raw_prices, match.iloc[0]["Broad Benchmark"])
    coords = map_coordinate_history(relative_series(asset_levels[key], broad), window).dropna()
    return coords if coords.empty else coords.resample("W-FRI").last().dropna().tail(trail_weeks)


def adaptive_axis_range(values: pd.Series, trail_values: pd.Series | None = None) -> tuple[float, float]:
    pieces = [pd.to_numeric(values, errors="coerce")]
    if trail_values is not None:
        pieces.append(pd.to_numeric(trail_values, errors="coerce"))
    clean = pd.concat(pieces, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -0.05, 0.05
    low, high = min(float(clean.min()), 0.0), max(float(clean.max()), 0.0)
    span = high - low
    pad = max(0.005, span * 0.12, max(abs(low), abs(high)) * 0.04)
    return low - pad, high + pad
