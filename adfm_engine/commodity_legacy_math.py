"""Commodity event study calculations extracted from adfm_core/commodity_event_study_page.py.
Regenerate with python scripts/extract_commodity_engine.py after changing source math."""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

COMMODITY_GROUPS: Dict[str, List[Tuple[str, str]]] = {'Energy': [('WTI Crude Oil', 'CL=F'), ('Brent Crude Oil', 'BZ=F'), ('Natural Gas', 'NG=F'), ('Heating Oil', 'HO=F'), ('RBOB Gasoline', 'RB=F'), ('Mont Belvieu Propane', 'B0=F')], 'Metals': [('Gold', 'GC=F'), ('Micro Gold', 'MGC=F'), ('Silver', 'SI=F'), ('Micro Silver', 'SIL=F'), ('Copper', 'HG=F'), ('Platinum', 'PL=F'), ('Palladium', 'PA=F')], 'Grains + Oilseeds': [('Corn', 'ZC=F'), ('Chicago Wheat', 'ZW=F'), ('KC HRW Wheat', 'KE=F'), ('Soybeans', 'ZS=F'), ('Soybean Meal', 'ZM=F'), ('Soybean Oil', 'ZL=F'), ('Oats', 'ZO=F'), ('Rough Rice', 'ZR=F')], 'Livestock': [('Live Cattle', 'LE=F'), ('Feeder Cattle', 'GF=F'), ('Lean Hogs', 'HE=F')], 'Softs': [('Cocoa', 'CC=F'), ('Coffee', 'KC=F'), ('Sugar #11', 'SB=F'), ('Cotton', 'CT=F'), ('Orange Juice', 'OJ=F'), ('Random Length Lumber', 'LBS=F')]}

CONTRACT_LABEL_TO_SYMBOL: Dict[str, str] = {}

CONTRACT_SYMBOL_TO_NAME: Dict[str, str] = {}

for _group, _contracts in COMMODITY_GROUPS.items():
    for _name, _symbol in _contracts:
        _label = f'{_group} · {_name} ({_symbol})'
        CONTRACT_LABEL_TO_SYMBOL[_label] = _symbol
        CONTRACT_SYMBOL_TO_NAME[_symbol] = _name

CONTRACT_OPTIONS = list(CONTRACT_LABEL_TO_SYMBOL) + ['Custom Yahoo futures symbol…']

RETURN_WINDOWS = {'1M': 21, '2M': 42, '3M': 63, '6M': 126, '12M': 252}

FORWARD_HORIZONS = {'1D': 1, '1W': 5, '2W': 10, '3W': 15, '1M': 21, '2M': 42, '3M': 63, '6M': 126, '9M': 189, '12M': 252}

SPACING_OPTIONS = {'1M': 21, '2M': 42, '3M': 63, '6M': 126, '12M': 252}

LOOKBACK_OPTIONS = {'Max': None, '10Y': 10, '25Y': 25, '50Y': 50}

def _flatten_yfinance_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if symbol in values:
            try:
                return frame.xs(symbol, axis=1, level=level, drop_level=True)
            except Exception:
                pass
    out = frame.copy()
    out.columns = out.columns.get_level_values(0)
    return out

def load_contract_history(symbol: str) -> pd.DataFrame:
    symbol = str(symbol).strip().upper()
    last_error = None
    for attempt in range(3):
        try:
            frame = yf.download(symbol, period='max', interval='1d', auto_adjust=False, progress=False, threads=False)
            if frame is None or frame.empty:
                raise ValueError('Yahoo Finance returned no rows.')
            frame = _flatten_yfinance_columns(frame, symbol)
            if 'Close' not in frame.columns:
                raise ValueError('Yahoo Finance returned no Close column.')
            close = pd.to_numeric(frame['Close'], errors='coerce')
            volume = pd.to_numeric(frame['Volume'], errors='coerce') if 'Volume' in frame.columns else pd.Series(index=frame.index, dtype=float)
            out = pd.DataFrame({'Close': close, 'Volume': volume})
            out.index = pd.to_datetime(out.index)
            if getattr(out.index, 'tz', None) is not None:
                out.index = out.index.tz_localize(None)
            out = out[~out.index.duplicated(keep='last')].sort_index()
            out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['Close'])
            if len(out) < 260:
                raise ValueError('Insufficient daily history for an event study.')
            return out
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f'Could not load {symbol}: {last_error}')

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return rsi.clip(lower=0.0, upper=100.0)

def _window_label(window_days: int) -> str:
    for label, days in RETURN_WINDOWS.items():
        if days == window_days:
            return label
    return f'{window_days}D'

def build_signal(close: pd.Series, signal_type: str, *, direction: str='Rally', window_days: int=63, threshold: float=25.0, rsi_period: int=14) -> Tuple[pd.Series, pd.Series, str, str]:
    if signal_type == 'Return threshold':
        metric = close.pct_change(window_days) * 100.0
        if direction == 'Rally':
            condition = metric >= threshold
            label = f'{_window_label(window_days)} return ≥ +{threshold:.2f}%'
        else:
            condition = metric <= -threshold
            label = f'{_window_label(window_days)} return ≤ -{threshold:.2f}%'
        return (metric, condition.fillna(False), label, 'percent')
    if signal_type == '52-week breakout':
        if direction == 'High':
            prior_extreme = close.shift(1).rolling(252, min_periods=200).max()
            metric = close / prior_extreme * 100.0 - 100.0
            condition = close >= prior_extreme
            label = 'New 52-week high'
        else:
            prior_extreme = close.shift(1).rolling(252, min_periods=200).min()
            metric = close / prior_extreme * 100.0 - 100.0
            condition = close <= prior_extreme
            label = 'New 52-week low'
        return (metric, condition.fillna(False), label, 'percent')
    if signal_type == 'RSI extreme':
        metric = _rsi(close, rsi_period)
        if direction == 'Overbought':
            condition = metric >= threshold
            label = f'RSI({rsi_period}) ≥ {threshold:.2f}'
        else:
            condition = metric <= threshold
            label = f'RSI({rsi_period}) ≤ {threshold:.2f}'
        return (metric, condition.fillna(False), label, 'number')
    moving_average = close.rolling(200, min_periods=180).mean()
    metric = (close / moving_average - 1.0) * 100.0
    if direction == 'Above':
        condition = metric >= threshold
        label = f'Price ≥ {threshold:.2f}% above 200D MA'
    else:
        condition = metric <= -threshold
        label = f'Price {threshold:.2f}% below 200D MA'
    return (metric, condition.fillna(False), label, 'percent')

def detect_events(condition: pd.Series, spacing_days: int, *, continuous: bool=False) -> pd.DatetimeIndex:
    condition = condition.fillna(False).astype(bool)
    candidates = condition if continuous else condition & ~condition.shift(1, fill_value=False)
    candidate_positions = np.flatnonzero(candidates.to_numpy())
    kept_positions: List[int] = []
    last_position = -10 ** 9
    for position in candidate_positions:
        if position - last_position >= spacing_days:
            kept_positions.append(int(position))
            last_position = int(position)
    return pd.DatetimeIndex(condition.index[kept_positions])

def build_event_observations(close: pd.Series, events: Iterable[pd.Timestamp], signal_metric: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    close = close.dropna().astype(float)
    index_positions = {timestamp: i for i, timestamp in enumerate(close.index)}
    rows: List[dict] = []
    store: Dict[str, Dict[str, List[float]]] = {label: {'return': [], 'signal_dd': [], 'path_dd': []} for label in FORWARD_HORIZONS}
    for event_date in events:
        if event_date not in index_positions:
            continue
        start_pos = index_positions[event_date]
        start_price = float(close.iloc[start_pos])
        row = {'Date': event_date, 'Price': start_price, 'Signal': float(signal_metric.reindex([event_date]).iloc[0])}
        for label, horizon in FORWARD_HORIZONS.items():
            end_pos = start_pos + horizon
            if end_pos >= len(close):
                row[label] = np.nan
                continue
            path = close.iloc[start_pos:end_pos + 1].astype(float)
            end_return = float(path.iloc[-1] / start_price - 1.0)
            from_signal = path / start_price - 1.0
            path_drawdown = path / path.cummax() - 1.0
            row[label] = end_return
            store[label]['return'].append(end_return)
            store[label]['signal_dd'].append(float(from_signal.min()))
            store[label]['path_dd'].append(float(path_drawdown.min()))
        rows.append(row)
    history = pd.DataFrame(rows)
    arrays = {label: {metric: np.asarray(values, dtype=float) for metric, values in metrics.items()} for label, metrics in store.items()}
    return (history, arrays)

def summarize_forward_performance(arrays: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = ['Average', 'Median', 'Best', 'Worst', '% Positive', 'Avg DD From Signal', 'Worst DD From Signal', 'Avg Peak-to-Trough DD', 'Sample']
    summary = pd.DataFrame(index=rows, columns=list(FORWARD_HORIZONS), dtype=float)
    for horizon in FORWARD_HORIZONS:
        returns = arrays[horizon]['return']
        signal_dd = arrays[horizon]['signal_dd']
        path_dd = arrays[horizon]['path_dd']
        if returns.size == 0:
            continue
        summary.loc['Average', horizon] = np.mean(returns)
        summary.loc['Median', horizon] = np.median(returns)
        summary.loc['Best', horizon] = np.max(returns)
        summary.loc['Worst', horizon] = np.min(returns)
        summary.loc['% Positive', horizon] = np.mean(returns > 0.0)
        summary.loc['Avg DD From Signal', horizon] = np.mean(signal_dd)
        summary.loc['Worst DD From Signal', horizon] = np.min(signal_dd)
        summary.loc['Avg Peak-to-Trough DD', horizon] = np.mean(path_dd)
        summary.loc['Sample', horizon] = float(returns.size)
    return summary

def _format_signal_value(value: float, value_kind: str) -> str:
    if pd.isna(value):
        return 'n/a'
    return f'{value:+.2f}%' if value_kind == 'percent' else f'{value:.2f}'

def make_price_chart(close: pd.Series, event_dates: pd.DatetimeIndex, signal_metric: pd.Series, signal_label: str, signal_value_kind: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=close.index, y=close.values, mode='lines', line={'color': '#2f7fd1', 'width': 1.65}, hovertemplate='%{x|%b %d, %Y}<br>Price: %{y:,.2f}<extra></extra>', name='Price'))
    if len(event_dates):
        event_prices = close.reindex(event_dates)
        hover_text = []
        for event_date in event_dates:
            value = signal_metric.reindex([event_date]).iloc[0]
            hover_text.append(f'{event_date:%b %d, %Y}<br>Price: {event_prices.loc[event_date]:,.2f}<br>Signal: {_format_signal_value(value, signal_value_kind)}')
        figure.add_trace(go.Scatter(x=event_dates, y=event_prices.values, mode='markers', marker={'color': '#e52822', 'size': 7, 'line': {'color': '#ffffff', 'width': 0.65}}, text=hover_text, hovertemplate='%{text}<extra></extra>', name=signal_label))
    figure.update_layout(height=500, margin={'l': 8, 'r': 8, 't': 12, 'b': 8}, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', showlegend=False, hovermode='closest', font={'family': 'Arial, Helvetica, sans-serif', 'color': '#1b1b1b', 'size': 12})
    figure.update_xaxes(showgrid=False, showline=True, linecolor='#8d8d8d', linewidth=1, ticks='outside', tickcolor='#8d8d8d', tickformat='%Y', fixedrange=False)
    figure.update_yaxes(showgrid=True, gridcolor='#e3e8ec', gridwidth=1, zeroline=False, showline=False, tickformat=',.2f', title=None, fixedrange=False)
    return figure

def _history_display(history: pd.DataFrame, signal_kind: str) -> pd.DataFrame:
    if history.empty:
        return history
    display = history.copy().sort_values('Date', ascending=False)
    display['Date'] = pd.to_datetime(display['Date']).dt.strftime('%Y-%m-%d')
    display['Price'] = display['Price'].map(lambda value: f'{value:,.2f}')
    display['Signal'] = display['Signal'].map(lambda value: _format_signal_value(value, signal_kind))
    for column in FORWARD_HORIZONS:
        display[column] = display[column].map(lambda value: '—' if pd.isna(value) else f'{value * 100.0:+.2f}%')
    return display
