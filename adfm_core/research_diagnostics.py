"""Auditable PM signal attribution, history, analog, and performance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data_registry import COCKPIT_PROXIES, ProxyDefinition
from .market_data import safe_divide
from .pm_cockpit import SCORE_WEIGHTS, build_signal_snapshot

FORWARD_HORIZONS = {"1W": 5, "1M": 21, "3M": 63}
ANALOG_ASSETS = ("SPY", "TLT", "UUP", "DBC", "HYG")


@dataclass(frozen=True)
class AnalogResult:
    """Nearest historical regimes and their subsequent asset returns."""

    current_date: str | None
    features: tuple[str, ...]
    matches: pd.DataFrame
    distribution: pd.DataFrame


def _proxy_series(prices: pd.DataFrame, definition: ProxyDefinition) -> pd.Series:
    if definition.numerator not in prices:
        return pd.Series(dtype=float)
    numerator = pd.to_numeric(prices[definition.numerator], errors="coerce")
    if definition.denominator:
        if definition.denominator not in prices:
            return pd.Series(dtype=float)
        denominator = pd.to_numeric(prices[definition.denominator], errors="coerce")
        proxy = safe_divide(numerator, denominator)
    else:
        proxy = numerator
    if definition.direction < 0:
        proxy = safe_divide(pd.Series(1.0, index=proxy.index), proxy)
    return proxy.replace([np.inf, -np.inf], np.nan)


def build_snapshot_history(
    prices: pd.DataFrame,
    definitions: Sequence[ProxyDefinition] = COCKPIT_PROXIES,
    *,
    frequency: str = "W-FRI",
    minimum_observations: int = 190,
) -> pd.DataFrame:
    """Reconstruct causal point-in-time snapshots on a controlled schedule."""

    if prices.empty:
        return pd.DataFrame()
    dates = (
        pd.Series(prices.index, index=prices.index)
        .resample(frequency)
        .last()
        .dropna()
        .tolist()
    )
    rows: list[pd.DataFrame] = []
    for date in dates:
        window = prices.loc[:date]
        if len(window) < minimum_observations:
            continue
        snapshot = build_signal_snapshot(window, definitions)
        snapshot.insert(0, "Snapshot Date", pd.Timestamp(date).normalize())
        rows.append(snapshot)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def signal_attribution(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Explain each composite change using weighted component contributions."""

    if current.empty:
        return pd.DataFrame()
    component_weights = {**SCORE_WEIGHTS, "Trend": 0.20}
    prior = previous.set_index("Key") if not previous.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in current.iterrows():
        key = str(row["Key"])
        old = (
            prior.loc[key]
            if not prior.empty and key in prior.index
            else pd.Series(dtype=object)
        )
        current_values = {
            component: pd.to_numeric(row.get(component), errors="coerce")
            for component in component_weights
        }
        active = {
            component: weight
            for component, weight in component_weights.items()
            if np.isfinite(current_values[component])
        }
        total_weight = sum(active.values())
        current_composite = pd.to_numeric(row.get("Composite"), errors="coerce")
        previous_composite = pd.to_numeric(old.get("Composite"), errors="coerce")
        for component, weight in active.items():
            normalized_weight = weight / total_weight if total_weight else np.nan
            current_value = current_values[component]
            prior_value = pd.to_numeric(old.get(component), errors="coerce")
            contribution = current_value * normalized_weight
            prior_contribution = (
                prior_value * normalized_weight if np.isfinite(prior_value) else np.nan
            )
            rows.append(
                {
                    "Signal": row["Signal"],
                    "Key": key,
                    "Group": row["Group"],
                    "Component": component,
                    "Current Input": current_value,
                    "Prior Input": prior_value,
                    "Normalized Weight": normalized_weight,
                    "Current Contribution": contribution,
                    "Prior Contribution": prior_contribution,
                    "Contribution Change": (
                        contribution - prior_contribution
                        if np.isfinite(prior_contribution)
                        else np.nan
                    ),
                    "Current Composite": current_composite,
                    "Prior Composite": previous_composite,
                    "Composite Change": (
                        current_composite - previous_composite
                        if np.isfinite(current_composite)
                        and np.isfinite(previous_composite)
                        else np.nan
                    ),
                    "Data Through": row.get("Data Through"),
                }
            )
    return pd.DataFrame(rows)


def signal_performance(
    history: pd.DataFrame,
    prices: pd.DataFrame,
    definitions: Sequence[ProxyDefinition] = COCKPIT_PROXIES,
) -> pd.DataFrame:
    """Measure subsequent proxy returns, hit rate, turnover, and drawdown."""

    if history.empty:
        return pd.DataFrame()
    definitions_by_key = {item.key: item for item in definitions}
    rows: list[dict[str, object]] = []
    for key, signal_history in history.groupby("Key"):
        definition = definitions_by_key.get(str(key))
        if definition is None:
            continue
        proxy = _proxy_series(prices, definition).dropna()
        if proxy.empty:
            continue
        ordered = signal_history.sort_values("Snapshot Date").copy()
        ordered["Composite"] = pd.to_numeric(ordered["Composite"], errors="coerce")
        dates = pd.to_datetime(ordered["Snapshot Date"])
        positions = proxy.index.searchsorted(dates)
        for horizon, periods in FORWARD_HORIZONS.items():
            outcomes: list[float] = []
            scores: list[float] = []
            last_entry_position = -periods
            for score, position in zip(ordered["Composite"], positions, strict=False):
                if (
                    not np.isfinite(score)
                    or position >= len(proxy)
                    or position + periods >= len(proxy)
                    or position < last_entry_position + periods
                ):
                    continue
                forward = float(
                    proxy.iloc[position + periods] / proxy.iloc[position] - 1.0
                )
                outcomes.append(forward)
                scores.append(float(score))
                last_entry_position = int(position)
            if not outcomes:
                continue
            outcome_series = pd.Series(outcomes, dtype=float)
            score_series = pd.Series(scores, dtype=float)
            signed = np.sign(score_series) * outcome_series
            strategy_curve = (1.0 + signed.fillna(0.0)).cumprod()
            drawdown = strategy_curve / strategy_curve.cummax() - 1.0
            active = score_series.abs() >= 0.12
            rows.append(
                {
                    "Signal": definition.label,
                    "Key": key,
                    "Group": definition.group,
                    "Horizon": horizon,
                    "Observations": len(outcome_series),
                    "Hit Rate": float((signed[active] > 0).mean())
                    if active.any()
                    else np.nan,
                    "Average Forward Return": float(outcome_series.mean()),
                    "Average Signed Return": float(signed.mean()),
                    "Worst Strategy Drawdown": float(drawdown.min()),
                    "Turnover": float(score_series.diff().abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def evidence_weights(
    diagnostics: pd.DataFrame,
    *,
    horizon: str = "1M",
    floor: float = 0.50,
    ceiling: float = 1.25,
) -> pd.DataFrame:
    """Translate out-of-sample evidence into bounded proposed signal weights."""

    frame = diagnostics.loc[diagnostics["Horizon"] == horizon].copy()
    if frame.empty:
        return frame
    hit = pd.to_numeric(frame["Hit Rate"], errors="coerce")
    signed = pd.to_numeric(frame["Average Signed Return"], errors="coerce")
    evidence = (hit - 0.50) * 2.0 + np.sign(signed) * np.minimum(
        signed.abs() * 20.0, 0.25
    )
    frame["Proposed Weight"] = (1.0 + evidence).clip(floor, ceiling)
    frame["Evidence Status"] = np.select(
        [frame["Proposed Weight"] < 0.85, frame["Proposed Weight"] > 1.10],
        ["De-emphasize", "Strengthen"],
        default="Neutral",
    )
    return frame


def regime_feature_panel(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Build a daily rates/inflation/liquidity/market regime feature panel."""

    if prices.empty:
        return pd.DataFrame()
    index = prices.index
    features = pd.DataFrame(index=index)
    macro_aligned = macro.reindex(index).ffill(limit=10)
    if {"dgs10", "dgs2"}.issubset(macro_aligned):
        features["Rates Curve"] = macro_aligned["dgs10"] - macro_aligned["dgs2"]
    if "t10yie" in macro_aligned:
        features["Inflation"] = macro_aligned["t10yie"]
    if {"walcl", "tga", "rrp"}.issubset(macro_aligned):
        features["Liquidity"] = (
            macro_aligned["walcl"] - macro_aligned["tga"] - macro_aligned["rrp"]
        )
    if "UUP" in prices:
        features["Dollar"] = np.log(prices["UUP"]).diff(63)
    if {"HYG", "LQD"}.issubset(prices):
        features["Credit"] = np.log(safe_divide(prices["HYG"], prices["LQD"])).diff(63)
    if "^VIX" in prices:
        features["Volatility"] = np.log(prices["^VIX"])
    if {"RSP", "SPY"}.issubset(prices):
        features["Breadth"] = np.log(safe_divide(prices["RSP"], prices["SPY"])).diff(63)
    return features.replace([np.inf, -np.inf], np.nan)


def historical_regime_analogs(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    matches: int = 12,
    exclusion_sessions: int = 63,
) -> AnalogResult:
    """Find nearest standardized regimes and show forward-return distributions."""

    complete = features.dropna()
    if complete.empty:
        return AnalogResult(
            None, tuple(features.columns), pd.DataFrame(), pd.DataFrame()
        )
    current_date = complete.index[-1]
    history = (
        complete.iloc[:-exclusion_sessions]
        if len(complete) > exclusion_sessions
        else complete.iloc[:0]
    )
    if history.empty:
        return AnalogResult(
            current_date.date().isoformat(),
            tuple(complete.columns),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    combined = pd.concat([history, complete.tail(1)])
    means = combined.iloc[:-1].mean()
    standard_deviation = combined.iloc[:-1].std(ddof=0).replace(0, np.nan)
    zscores = (combined - means) / standard_deviation
    current = zscores.iloc[-1]
    distances = np.sqrt(((zscores.iloc[:-1] - current) ** 2).mean(axis=1))
    nearest = distances.nsmallest(min(matches, len(distances)))
    match_rows: list[dict[str, object]] = []
    for date, distance in nearest.items():
        row: dict[str, object] = {
            "Analog Date": pd.Timestamp(date).date().isoformat(),
            "Distance": float(distance),
        }
        position = prices.index.searchsorted(pd.Timestamp(date))
        for asset in ANALOG_ASSETS:
            if asset not in prices or position >= len(prices):
                continue
            series = pd.to_numeric(prices[asset], errors="coerce")
            for label, periods in FORWARD_HORIZONS.items():
                if position + periods < len(series):
                    row[f"{asset} {label}"] = float(
                        series.iloc[position + periods] / series.iloc[position] - 1.0
                    )
        match_rows.append(row)
    match_frame = pd.DataFrame(match_rows)
    return_columns = [
        column
        for column in match_frame.columns
        if any(column.startswith(f"{asset} ") for asset in ANALOG_ASSETS)
    ]
    distributions = []
    for column in return_columns:
        values = pd.to_numeric(match_frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        asset, horizon = column.split(" ", 1)
        distributions.append(
            {
                "Asset": asset,
                "Horizon": horizon,
                "Observations": len(values),
                "Median": float(values.median()),
                "Mean": float(values.mean()),
                "10th Percentile": float(values.quantile(0.10)),
                "90th Percentile": float(values.quantile(0.90)),
                "Positive Rate": float((values > 0).mean()),
            }
        )
    return AnalogResult(
        current_date.date().isoformat(),
        tuple(complete.columns),
        match_frame,
        pd.DataFrame(distributions),
    )


def dashboard_route(signal_key: str, *, lookback: str = "3y") -> Mapping[str, str]:
    """Resolve a signal to an existing analytical page and drilldown context."""

    routes = {
        "credit_risk": ("pages/13_Credit_Conditions_Monitor.py", "HYG/LQD"),
        "duration_risk": ("pages/12_Yield_Curve_Rates_Regime_Monitor.py", "TLT/SHY"),
        "dollar_liquidity": ("pages/14_Liquidity_Tracker.py", "UUP"),
        "volatility": ("pages/15_Market_Stress_Composite.py", "^VIX"),
        "breadth": ("pages/2_Sector_Breadth_and_Rotation.py", "RSP/SPY"),
    }
    page, instrument = routes.get(
        signal_key,
        ("pages/11_Global_Macro_Regime_Dashboard.py", signal_key),
    )
    return {"page": page, "instrument": instrument, "lookback": lookback}
