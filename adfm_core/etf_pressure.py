"""Comparable ETF trading pressure and published category net issuance.

Pressure is a bounded close-location statistic, not a capital-flow estimate.
All calculations use actual observations on the supplied completed-session calendar.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .market_data import canonicalize_date_index

METRICS = ('Pressure (%)', 'Pressure change (pp)', 'Return (%)',
           'Up/down volume (%)', 'Activity (x)', 'Avg daily traded ($M)')
ICI_URL = 'https://www.ici.org/research/stats/etf_flows'
ICI_CATEGORIES = {'Domestic': 'US equity', 'World': 'International equity',
                  'Taxable': 'Taxable bonds', 'Municipal': 'Municipal bonds',
                  'Hybrid': 'Hybrid', 'Commodity': 'Commodities', 'Total': 'Total'}


def pressure_reading(frame: pd.DataFrame, sessions: pd.DatetimeIndex, window: int) -> dict:
    """Read one ETF on the same window/end date as the rest of the universe."""
    result = {key: np.nan for key in METRICS}
    result.update({'Status': 'Incomplete window', 'As of': None})
    sessions = pd.DatetimeIndex(sessions).sort_values().unique()
    window = max(int(window), 1)
    if len(sessions) < window or frame.empty:
        return result
    raw = canonicalize_date_index(frame).reindex(sessions)
    for column in ('High', 'Low', 'Close', 'Volume', 'Adj Close'):
        raw[column] = pd.to_numeric(raw.get(column, pd.Series(index=sessions, dtype=float)), errors='coerce')
    if raw['Close'].notna().any():
        result['As of'] = raw['Close'].last_valid_index().date().isoformat()
    current = raw.iloc[-window:]

    def valid(values):
        return (np.isfinite(values[['High', 'Low', 'Close', 'Volume']]).all(axis=1)
                & (values['Low'] > 0) & (values['High'] >= values['Low'])
                & values['Close'].between(values['Low'], values['High'])
                & (values['Volume'] >= 0)).all()

    def pressure(values):
        if values.empty or not valid(values):
            return np.nan
        weights = values['Close'] * values['Volume']
        if weights.sum() <= 0:
            return np.nan
        spread = values['High'] - values['Low']
        location = ((2 * values['Close'] - values['High'] - values['Low'])
                    .div(spread.replace(0, np.nan)).where(spread.ne(0), 0.))
        return float((location * weights).sum() / weights.sum() * 100.)

    result['Pressure (%)'] = pressure(current)
    if not np.isfinite(result['Pressure (%)']):
        return result
    result['Status'] = 'Available'
    weights = current['Close'] * current['Volume']
    result['Avg daily traded ($M)'] = float(weights.mean() / 1e6)
    previous = raw.iloc[-2 * window:-window]
    if len(previous) == window:
        result['Pressure change (pp)'] = result['Pressure (%)'] - pressure(previous)
    adjusted = raw['Adj Close'].where(raw['Adj Close'] > 0)
    returns = adjusted.pct_change(fill_method=None).iloc[-window:]
    if returns.notna().all() and len(adjusted) > window:
        result['Return (%)'] = float((adjusted.iloc[-1] / adjusted.iloc[-window - 1] - 1) * 100.)
        result['Up/down volume (%)'] = float((np.sign(returns) * weights).sum() / weights.sum() * 100.)
    # Use a prior baseline, never the window being measured. Median resists isolated volume spikes.
    baseline = raw['Volume'].iloc[max(0, len(raw) - window - 63):len(raw) - window]
    baseline = baseline.where(baseline > 0).dropna()
    if len(baseline) >= 20:
        result['Activity (x)'] = float(current['Volume'].mean() / baseline.median())
    return result


def parse_ici_issuance(html: str) -> pd.DataFrame:
    """Parse ICI's reported weekly estimates into non-overlapping categories ($bn)."""
    soup = BeautifulSoup(html, 'html.parser')
    if 'Millions of dollars' not in soup.get_text(' ', strip=True):
        raise ValueError('ICI units not identified')
    for table in soup.find_all('table'):
        text = table.get_text(' ', strip=True)
        if not all(label in text for label in ICI_CATEGORIES):
            continue
        rows = [[cell.get_text(' ', strip=True) for cell in row.find_all(['th', 'td'])]
                for row in table.find_all('tr')]
        date_row = next((row for row in rows if sum(bool(re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', cell)) for cell in row) >= 2), None)
        if date_row is None:
            continue
        dates = pd.to_datetime(date_row[1:], format='%m/%d/%Y', errors='coerce')
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError('Invalid ICI report dates')
        values = {}
        for row in rows:
            if row and row[0] in ICI_CATEGORIES:
                numbers = pd.to_numeric(pd.Series(row[1:]).str.replace(',', '', regex=False), errors='coerce')
                if len(numbers) != len(dates) or numbers.isna().any():
                    raise ValueError('Incomplete ICI observations')
                values[ICI_CATEGORIES[row[0]]] = numbers.to_numpy(dtype=float) / 1000.
        if len(values) != len(ICI_CATEGORIES):
            raise ValueError('Incomplete ICI categories')
        result = pd.DataFrame.from_dict(values, orient='index', columns=dates)
        result = result.reindex(list(ICI_CATEGORIES.values())).sort_index(axis=1, ascending=False)
        # Parent equity/bond subtotals are deliberately excluded to avoid double counting.
        if (result.iloc[:-1].sum() - result.loc['Total']).abs().gt(.01).any():
            raise ValueError('ICI categories do not reconcile')
        return result
    raise ValueError('ICI ETF issuance table unavailable')
