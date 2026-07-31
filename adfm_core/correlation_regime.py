"""Cross-asset correlation, beta, and diversification calculations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationSnapshot:
    """Current cross-asset correlation state."""

    average_correlation: float
    prior_average_correlation: float
    market_mode_share: float
    effective_bets: float
    observations: int
    assets: int


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns without filling missing prices."""

    numeric = prices.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return np.log(numeric).diff().replace([np.inf, -np.inf], np.nan)


def window_correlation(
    returns: pd.DataFrame,
    window: int,
    *,
    offset: int = 0,
    min_coverage: float = 0.80,
) -> pd.DataFrame:
    """Return a complete correlation matrix for one trailing sample window."""

    if returns.empty or window < 2:
        return pd.DataFrame()
    end = len(returns) - max(0, int(offset))
    start = max(0, end - int(window))
    sample = returns.iloc[start:end]
    if len(sample) < 2:
        return pd.DataFrame()
    minimum = max(2, int(len(sample) * min_coverage))
    eligible = [column for column in sample if sample[column].notna().sum() >= minimum]
    if len(eligible) < 2:
        return pd.DataFrame()
    return sample[eligible].corr(min_periods=minimum)


def average_off_diagonal(matrix: pd.DataFrame) -> float:
    """Average unique pairwise correlation."""

    if matrix.empty or len(matrix) < 2:
        return np.nan
    values = matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices(len(matrix), k=1)]
    finite = upper[np.isfinite(upper)]
    return float(finite.mean()) if len(finite) else np.nan


def eigen_diagnostics(matrix: pd.DataFrame) -> tuple[float, float]:
    """Return first-component variance share and entropy effective rank."""

    if matrix.empty or len(matrix) < 2:
        return np.nan, np.nan
    clean = matrix.dropna(axis=0, how="any").dropna(axis=1, how="any")
    common = clean.index.intersection(clean.columns)
    clean = clean.loc[common, common]
    if len(clean) < 2:
        return np.nan, np.nan
    eigenvalues = np.linalg.eigvalsh(clean.to_numpy(dtype=float))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        return np.nan, np.nan
    weights = eigenvalues / total
    positive = weights[weights > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    return float(eigenvalues.max() / total), effective_rank


def correlation_snapshot(
    returns: pd.DataFrame,
    window: int,
) -> CorrelationSnapshot:
    """Build the current and prior-window diversification snapshot."""

    current = window_correlation(returns, window)
    prior = window_correlation(returns, window, offset=window)
    market_mode, effective_bets = eigen_diagnostics(current)
    sample = returns.tail(window)
    return CorrelationSnapshot(
        average_correlation=average_off_diagonal(current),
        prior_average_correlation=average_off_diagonal(prior),
        market_mode_share=market_mode,
        effective_bets=effective_bets,
        observations=int(sample.notna().any(axis=1).sum()),
        assets=len(current),
    )


def pair_table(
    current: pd.DataFrame,
    prior: pd.DataFrame,
    display_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Flatten current and prior matrices into auditable pair rows."""

    if current.empty:
        return pd.DataFrame(
            columns=("Asset 1", "Asset 2", "Correlation", "Prior", "Change")
        )
    names = display_names or {}
    rows: list[dict[str, object]] = []
    for left, right in combinations(current.columns, 2):
        value = pd.to_numeric(current.loc[left, right], errors="coerce")
        old = (
            pd.to_numeric(prior.loc[left, right], errors="coerce")
            if left in prior.index and right in prior.columns
            else np.nan
        )
        rows.append(
            {
                "Ticker 1": left,
                "Ticker 2": right,
                "Asset 1": names.get(left, left),
                "Asset 2": names.get(right, right),
                "Correlation": float(value) if np.isfinite(value) else np.nan,
                "Prior": float(old) if np.isfinite(old) else np.nan,
                "Change": (
                    float(value - old)
                    if np.isfinite(value) and np.isfinite(old)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False)


def rolling_average_correlation(
    returns: pd.DataFrame,
    window: int,
    *,
    min_assets: int = 4,
) -> pd.Series:
    """Calculate the rolling average of unique pairwise correlations."""

    if returns.empty:
        return pd.Series(dtype=float, name="Average Correlation")
    pairwise = {
        f"{left}|{right}": returns[left]
        .rolling(window, min_periods=window)
        .corr(returns[right])
        for left, right in combinations(returns.columns, 2)
    }
    if not pairwise:
        return pd.Series(index=returns.index, dtype=float, name="Average Correlation")
    panel = pd.DataFrame(pairwise)
    minimum_pairs = max(1, min_assets * (min_assets - 1) // 2)
    result = panel.mean(axis=1).where(panel.notna().sum(axis=1) >= minimum_pairs)
    return result.rename("Average Correlation")


def rolling_pair_metrics(
    returns: pd.DataFrame,
    asset: str,
    benchmark: str,
    window: int,
) -> pd.DataFrame:
    """Rolling correlation, beta, and relative cumulative return for one pair."""

    if asset not in returns or benchmark not in returns:
        return pd.DataFrame()
    pair = returns[[asset, benchmark]].dropna()
    if pair.empty:
        return pd.DataFrame()
    covariance = pair[asset].rolling(window, min_periods=window).cov(pair[benchmark])
    variance = pair[benchmark].rolling(window, min_periods=window).var()
    beta = covariance.div(variance.replace(0, np.nan))
    correlation = pair[asset].rolling(window, min_periods=window).corr(pair[benchmark])
    relative = (pair[asset] - pair[benchmark]).cumsum()
    return pd.DataFrame(
        {
            "Correlation": correlation,
            "Beta": beta,
            "Relative Log Return": relative,
        }
    ).replace([np.inf, -np.inf], np.nan)


def conditional_pair_statistics(
    returns: pd.DataFrame,
    benchmark: str,
    *,
    vix_levels: pd.Series | None = None,
    drawdown: pd.Series | None = None,
) -> pd.DataFrame:
    """Correlations and betas conditional on benchmark and volatility regimes."""

    if returns.empty or benchmark not in returns:
        return pd.DataFrame()
    index = returns.index
    masks: dict[str, pd.Series] = {
        "All Sessions": pd.Series(True, index=index),
        f"{benchmark} Up Sessions": returns[benchmark] > 0,
        f"{benchmark} Down Sessions": returns[benchmark] < 0,
    }
    if vix_levels is not None:
        aligned_vix = pd.to_numeric(vix_levels, errors="coerce").reindex(index)
        threshold = aligned_vix.quantile(0.75)
        masks["High VIX Quartile"] = aligned_vix >= threshold
    if drawdown is not None:
        aligned_drawdown = pd.to_numeric(drawdown, errors="coerce").reindex(index)
        masks["SPY Drawdown > 10%"] = aligned_drawdown <= -0.10

    rows: list[dict[str, object]] = []
    for regime, mask in masks.items():
        sample = returns.loc[mask.fillna(False)]
        for asset in returns.columns:
            if asset == benchmark:
                continue
            paired = sample[[asset, benchmark]].dropna()
            correlation = paired[asset].corr(paired[benchmark])
            benchmark_variance = paired[benchmark].var()
            beta = (
                paired[asset].cov(paired[benchmark]) / benchmark_variance
                if np.isfinite(benchmark_variance) and benchmark_variance > 0
                else np.nan
            )
            rows.append(
                {
                    "Regime": regime,
                    "Ticker": asset,
                    "Observations": len(paired),
                    "Correlation to Benchmark": correlation,
                    "Beta to Benchmark": beta,
                    "Annualized Volatility": paired[asset].std(ddof=1) * np.sqrt(252),
                }
            )
    return pd.DataFrame(rows)


def current_beta_table(
    returns: pd.DataFrame,
    benchmark: str,
    windows: Sequence[int] = (21, 63, 126, 252),
) -> pd.DataFrame:
    """Current correlation and beta across standard lookback windows."""

    if benchmark not in returns:
        return pd.DataFrame()
    rows = []
    for asset in returns.columns:
        if asset == benchmark:
            continue
        row: dict[str, object] = {"Ticker": asset}
        for window in windows:
            sample = returns[[asset, benchmark]].dropna().tail(window)
            variance = sample[benchmark].var()
            row[f"Corr {window}D"] = sample[asset].corr(sample[benchmark])
            row[f"Beta {window}D"] = (
                sample[asset].cov(sample[benchmark]) / variance
                if np.isfinite(variance) and variance > 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
