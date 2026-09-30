"""Official daily 10Y country curves, kept separate from OECD monthly averages.

The existing currency adapter cache is read only. Every refresh attempts direct
sources; cached values keep their dates and never substitute a different basis.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cte.adapters import sovereign_yields as adapters
from cte.adapters.base import http_get, tidy_yields

# Country, currency, exact source identifier, curve basis, official reference.
DAILY_SOVEREIGNS = (
    ('United States', 'USD', 'us_treasury', 'Treasury par constant-maturity curve', 'https://home.treasury.gov/resource-center-data-chart-center/interest-rates'),
    ('Canada', 'CAD', 'boc_valet', 'Government benchmark bond yield', 'https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/'),
    ('Germany', 'EUR', 'bundesbank', 'Federal securities residual-maturity yield', 'https://www.bundesbank.de/en/statistics/money-and-capital-markets/interest-rates-and-yields'),
    ('United Kingdom', 'GBP', 'boe_glc', 'Government nominal spot curve', 'https://www.bankofengland.co.uk/statistics/yield-curves'),
    ('Switzerland', 'CHF', 'snb_airchart_spot', 'Confederation spot curve', 'https://data.snb.ch/en/topics/ziredev/cube/rendeidglfzch'),
    ('Australia', 'AUD', 'rba_f2', 'Government benchmark bond yield', 'https://www.rba.gov.au/statistics/tables/'),
    ('Japan', 'JPY', 'jp_mof', 'JGB reference constant-maturity curve', 'https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/'),
    ('New Zealand', 'NZD', 'rbnz_b2', 'Government secondary-market closing yield', 'https://www.rbnz.govt.nz/statistics/series/exchange-and-interest-rates/wholesale-interest-rates'),
)
CACHE_PATH = Path(__file__).resolve().parents[1] / 'data/cache/yields.parquet'


def _japan():
    # The adapter merges history before the fresh tail and keeps first duplicates.
    # Apply its parser to the tail again so official revisions/tail win.
    history = adapters.fetch_jp()
    try:
        tail = tidy_yields(adapters._parse_jgb_csv(http_get('https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv').content), 'jp_mof')
        return pd.concat([history, tail], ignore_index=True)
    except Exception:
        return history


def default_fetchers(cached: pd.DataFrame) -> dict:
    # One current annual Treasury file refreshes the tail; valid persisted
    # history supplies the backfill. Cold starts request ten annual files.
    has_us = not _validated(cached, DAILY_SOVEREIGNS[0], pd.Timestamp.today()).empty
    return {
        'United States': lambda: adapters.fetch_us(years_back=0 if has_us else 10),
        'Canada': adapters.fetch_boc, 'Germany': adapters.fetch_bundesbank,
        'United Kingdom': adapters.fetch_boe, 'Switzerland': adapters.fetch_snb,
        'Australia': adapters.fetch_rba, 'Japan': _japan, 'New Zealand': adapters.fetch_rbnz,
    }


def _validated(frame: pd.DataFrame, definition: tuple, today: pd.Timestamp) -> pd.Series:
    required = {'date', 'ccy', 'tenor', 'value', 'source', 'fetched_at'}
    if not required <= set(frame.columns):
        return pd.Series(dtype=float)
    _, currency, source, _, _ = definition
    rows = frame.loc[(frame.ccy == currency) & (frame.source == source) & (frame.tenor == '10Y')].copy()
    dates = pd.to_datetime(rows['date'], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    values = pd.to_numeric(rows['value'], errors='coerce')
    fetched = pd.to_datetime(rows['fetched_at'], errors='coerce', utc=True).dt.tz_convert(None)
    valid = dates.notna() & fetched.notna() & np.isfinite(values) & dates.le(today) & values.between(-10, 40)
    data = pd.Series(values.loc[valid].to_numpy(dtype=float), index=pd.DatetimeIndex(dates.loc[valid]))
    data = data.loc[~data.index.duplicated(keep='last')].sort_index()
    # Reject monthly-shaped/malformed sparse snapshots claiming daily tenor
    # coverage. In particular the legacy RBNZ cache has only scattered dates;
    # its nonofficial fill-in source is deliberately not accepted either.
    if len(data) >= 4 and np.median((data.index[1:] - data.index[:-1]).total_seconds() / 86400) > 10:
        return pd.Series(dtype=float)
    return data


def load_daily_sovereigns(*, cache_path: Path = CACHE_PATH, fetchers: dict | None = None,
                          today: pd.Timestamp | None = None, timeout: float = 20.) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh each official source independently, returning panel and status.

    A failed/invalid source returns validated cached history and a visible error.
    The shared currency cache is never overwritten by this page.
    """
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).tz_localize(None).normalize()
    cache_error = ''
    try:
        cached = pd.read_parquet(cache_path) if Path(cache_path).exists() else pd.DataFrame()
    except Exception as exc:
        cached, cache_error = pd.DataFrame(), f'Cache unavailable: {type(exc).__name__}'
    registry = default_fetchers(cached) if fetchers is None else fetchers
    refreshed, errors = {}, {}
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        jobs = {pool.submit(fetch): country for country, fetch in registry.items()}
        for job in as_completed(jobs, timeout=timeout):
            country = jobs[job]
            try:
                refreshed[country] = job.result()
            except Exception as exc:
                errors[country] = f'{type(exc).__name__}: {exc}'
    except TimeoutError:
        for job, country in jobs.items():
            if country not in refreshed and country not in errors:
                errors[country] = 'Official source refresh deadline exceeded; cached observations retained'
                job.cancel()
    finally:
        # Requests already running are bounded by adapter socket timeouts;
        # their late results cannot alter the returned validated panel.
        pool.shutdown(wait=False, cancel_futures=True)
    panel, status = {}, []
    for definition in DAILY_SOVEREIGNS:
        country, _, source, basis, url = definition
        prior = _validated(cached, definition, today)
        fresh = _validated(refreshed.get(country, pd.DataFrame()), definition, today)
        error = errors.get(country, '')
        if country in registry and fresh.empty and not error:
            error = 'No validated official daily 10Y observations returned'
        combined = pd.concat([prior, fresh])
        combined = combined.loc[~combined.index.duplicated(keep='last')].sort_index()
        panel[country] = combined
        status.append({'country': country, 'source': source, 'basis': basis, 'frequency': 'daily',
                       'observed': combined.index.max() if not combined.empty else pd.NaT,
                       'delivery': 'direct + validated history' if not fresh.empty else 'validated cache' if not prior.empty else 'unavailable',
                       'error': error or cache_error, 'url': url})
    return pd.DataFrame(panel).sort_index(), pd.DataFrame(status)
