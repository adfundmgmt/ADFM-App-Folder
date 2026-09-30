"""OHLCV normalization and technical calculations for the Chart Terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
CAP_MAX_ROWS = 250_000


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    required = set(REQUIRED_PRICE_COLUMNS)
    level_0 = list(df.columns.get_level_values(0))
    level_1 = list(df.columns.get_level_values(1))
    if required.issubset(set(level_0)):
        out = df.copy()
        out.columns = df.columns.get_level_values(0)
        return out
    if required.issubset(set(level_1)):
        out = df.copy()
        out.columns = df.columns.get_level_values(1)
        return out
    out = df.copy()
    out.columns = [
        "_".join((str(x) for x in col if str(x) != "")) for col in df.columns
    ]
    return out


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = flatten_yfinance_columns(df).copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    rename_map = {}
    for col in out.columns:
        clean = str(col).strip()
        if clean in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            rename_map[col] = clean
    out = out.rename(columns=rename_map)
    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in out.columns]
    if missing:
        return pd.DataFrame()
    for col in REQUIRED_PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    else:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")
    out = out.dropna(subset=REQUIRED_PRICE_COLUMNS)
    if len(out) > CAP_MAX_ROWS:
        out = out.tail(CAP_MAX_ROWS)
    return out


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return rsi


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    return (macd, signal_line, hist)


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return (mid, upper, lower)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for window in [8, 20, 50, 65, 100, 130, 195, 200, 260]:
        out[f"SMA{window}"] = (
            out["Close"].rolling(window=window, min_periods=window).mean()
        )
    out["RSI14"] = compute_rsi(out["Close"], length=14)
    macd, signal, hist = compute_macd(out["Close"])
    out["MACD"] = macd
    out["MACD_SIGNAL"] = signal
    out["MACD_HIST"] = hist
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(out["Close"])
    out["BB_MID"] = bb_mid
    out["BB_UPPER"] = bb_upper
    out["BB_LOWER"] = bb_lower
    out["ATR14"] = compute_atr(out)
    out["ATR14_PCT"] = out["ATR14"] / out["Close"].replace(0, np.nan)
    out["ROLLING_VOL_20"] = out["Close"].pct_change().rolling(
        20, min_periods=20
    ).std() * np.sqrt(252)
    out["DRAWDOWN_252"] = (
        out["Close"] / out["Close"].rolling(252, min_periods=30).max() - 1.0
    )
    return out
