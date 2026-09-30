"""Coverage-aware basket mathematics, independent of Streamlit and providers."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_DAILY_MEMBER_COVERAGE = 0.60
BENCH = "SPY"


def trailing_valid_segment(series):
    """Never compress a data gap into an indicator's observation window."""
    clean = series.replace([np.inf, -np.inf], np.nan)
    missing = np.flatnonzero(clean.isna().to_numpy())
    return clean.iloc[missing[-1] + 1 :] if len(missing) else clean


def reliable_price_sessions(
    levels: pd.DataFrame, calendar: pd.DatetimeIndex, constituents: List[str]
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Discard dates where the provider lost most prices across the universe.

    A basket return after an omitted date spans the two observed closes;
    individual missing quotes still obey the basket coverage requirement.
    Compare each date with nearby dates so IPOs do not invalidate older history.
    """
    calendar = pd.DatetimeIndex(calendar)
    if calendar.empty:
        return (calendar, calendar)
    symbols = list(
        dict.fromkeys((t for t in constituents if t in levels.columns and t != BENCH))
    )
    observed = levels.reindex(calendar)
    counts = (
        observed[symbols].notna().sum(axis=1)
        if symbols
        else pd.Series(0, index=calendar)
    )
    nearby = counts.rolling(11, center=True, min_periods=3).median()
    unreliable = (nearby >= 2) & (counts < nearby * 0.7)
    if BENCH in observed:
        unreliable |= observed[BENCH].isna()
    return (calendar[~unreliable], calendar[unreliable])


def ew_rets_from_levels(
    levels: pd.DataFrame,
    baskets: Dict[str, List[str]],
    min_daily_coverage: float = MIN_DAILY_MEMBER_COVERAGE,
) -> pd.DataFrame:
    """Coverage-aware equal-weight returns on observed price sessions.

    Constituent returns require prices on both adjacent observed
    sessions. Missing returns are never compressed across a gap. A basket
    remains observable when the configured coverage floor is met, so one
    missing constituent or a recent IPO does not blank an otherwise valid
    multi-year basket.
    """
    if levels.empty:
        return pd.DataFrame()
    rets = pd.DataFrame(
        {
            column: levels[column].pct_change(fill_method=None)
            for column in levels.columns
        },
        index=levels.index,
    )
    out: Dict[str, pd.Series] = {}
    for basket_name, basket_tickers in baskets.items():
        cols = list(
            dict.fromkeys(
                (
                    str(t).upper()
                    for t in basket_tickers
                    if str(t).upper() in rets.columns
                )
            )
        )
        if not cols:
            continue
        member_rets = rets[cols]
        required_count = (
            1
            if len(cols) == 1
            else max(2, int(math.ceil(len(cols) * min_daily_coverage)))
        )
        valid_count = member_rets.notna().sum(axis=1)
        basket_ret = member_rets.mean(axis=1, skipna=True)
        basket_ret[valid_count < required_count] = np.nan
        price_count = levels[cols].notna().sum(axis=1)
        inception_dates = price_count[price_count >= required_count].index
        if len(inception_dates):
            inception_date = inception_dates[0]
            if pd.isna(basket_ret.loc[inception_date]):
                basket_ret.loc[inception_date] = 0.0
        out[basket_name] = basket_ret
    return pd.DataFrame(out, index=levels.index)


def basket_vs_dma_pct(series: pd.Series, window: int) -> float:
    """

    Current basket level versus the basket's own moving average.



    The basket series is already equal-weighted by ew_rets_from_levels().

    This returns the percent distance between today's basket level and

    today's rolling DMA:



        current basket level / current basket DMA - 1



    A positive value means the basket is trading above that DMA.

    A negative value means the basket is trading below that DMA.

    """
    clean = trailing_valid_segment(series)
    if clean.shape[0] < window:
        return np.nan
    dma = clean.rolling(window=window, min_periods=window).mean()
    latest_px = clean.iloc[-1]
    latest_dma = dma.dropna().iloc[-1] if dma.dropna().shape[0] else np.nan
    if pd.isna(latest_px) or pd.isna(latest_dma) or latest_dma == 0:
        return np.nan
    return float((latest_px / latest_dma - 1.0) * 100.0)


def basket_observation(returns: pd.Series) -> dict[str, object]:
    """Report the actual basket observation; never fill a missing endpoint.

    Age is calendar days relative to the requested return calendar endpoint.
    Prior-close change requires a valid current return. The page's return
    calendar may omit a provider-wide outage; it is labeled accordingly.
    """
    clean = returns.replace([np.inf, -np.inf], np.nan)
    observed = clean.last_valid_index()
    age = (
        (pd.Timestamp(clean.index[-1]) - pd.Timestamp(observed)).days
        if observed is not None
        else None
    )
    change = (
        float(clean.iloc[-1] * 100)
        if len(clean) > 1
        and observed != clean.first_valid_index()
        and pd.notna(clean.iloc[-1])
        else np.nan
    )
    return {
        "observed_as_of": observed,
        "observation_age_days": age,
        "prior_close_change_pct": change,
    }
