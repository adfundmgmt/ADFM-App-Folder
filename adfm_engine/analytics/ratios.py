"""Preserved from the original Cross-Asset Ratio Chartbook; no UI runtime."""
from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd
from adfm_engine.analytics.ratio_universe import DEFAULT_STALE_DAYS

def first_valid_on_or_after(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[ts:]

    if not sub.empty:
        return sub.index[0], float(sub.iloc[0])

    return s.index[-1], float(s.iloc[-1])


def last_valid_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[:ts]

    if not sub.empty:
        return sub.index[-1], float(sub.iloc[-1])

    return s.index[0], float(s.iloc[0])


def rebase_series(series: pd.Series, base_date: pd.Timestamp, base: float = 100.0) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return pd.Series(dtype=float)

    base_ts, base_val = last_valid_on_or_before(s, base_date)

    if pd.isna(base_ts) or not np.isfinite(base_val) or base_val == 0:
        return pd.Series(dtype=float)

    return s / base_val * base


def compute_price_ratio(
    s1: pd.Series,
    s2: pd.Series,
    base_date: pd.Timestamp,
    base: float = 100.0,
) -> pd.Series:
    a, b = s1.align(s2, join="inner")
    raw_ratio = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
    return rebase_series(raw_ratio, base_date=base_date, base=base)


def rsi_wilder(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return pd.Series(dtype=float)

    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def period_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    prev = s.iloc[-periods - 1]
    latest = s.iloc[-1]

    if not np.isfinite(prev) or prev == 0:
        return np.nan

    return latest / prev - 1.0


def ytd_change(series: pd.Series) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    latest_date = s.index[-1]
    year_start = pd.Timestamp(datetime(latest_date.year, 1, 1))

    _, base_val = first_valid_on_or_after(s, year_start)
    latest = s.iloc[-1]

    if not np.isfinite(base_val) or base_val == 0:
        return np.nan

    return latest / base_val - 1.0


def days_since_window_extreme(series: pd.Series, window: int, kind: str) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    view = s.tail(window)

    if view.empty:
        return np.nan

    extreme_date = view.idxmax() if kind == "high" else view.idxmin()

    return float(len(s.loc[extreme_date:]) - 1)


def fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{value:+.1%}"


def fmt_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{value:,.1f}"


def fmt_days(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"

    return f"{int(value)}d"


def ratio_signal_line(
    ratio: pd.Series,
    rsi_len: int,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> str:
    s = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return "No usable data."

    latest = float(s.iloc[-1])
    latest_date = pd.Timestamp(s.index[-1]).date()
    today = pd.Timestamp(datetime.today()).date()
    stale = (today - latest_date).days > stale_days

    rsi = rsi_wilder(s, window=rsi_len).dropna()
    rsi_latest = float(rsi.iloc[-1]) if not rsi.empty else np.nan

    ma50 = s.rolling(50, min_periods=20).mean().dropna()
    ma100 = s.rolling(100, min_periods=40).mean().dropna()
    ma200 = s.rolling(200, min_periods=80).mean().dropna()

    vs_ma50 = latest / float(ma50.iloc[-1]) - 1.0 if not ma50.empty and ma50.iloc[-1] else np.nan
    vs_ma100 = latest / float(ma100.iloc[-1]) - 1.0 if not ma100.empty and ma100.iloc[-1] else np.nan
    vs_ma200 = latest / float(ma200.iloc[-1]) - 1.0 if not ma200.empty and ma200.iloc[-1] else np.nan

    parts = [
        f"Last {fmt_num(latest)}",
        f"1M {fmt_pct(period_change(s, 21))}",
        f"3M {fmt_pct(period_change(s, 63))}",
        f"6M {fmt_pct(period_change(s, 126))}",
        f"YTD {fmt_pct(ytd_change(s))}",
        f"RSI {fmt_num(rsi_latest)}",
        f"vs 50D {fmt_pct(vs_ma50)}",
        f"vs 100D {fmt_pct(vs_ma100)}",
        f"vs 200D {fmt_pct(vs_ma200)}",
        f"3M high {fmt_days(days_since_window_extreme(s, 63, 'high'))} ago",
        f"3M low {fmt_days(days_since_window_extreme(s, 63, 'low'))} ago",
        f"Last data {latest_date}",
    ]

    if stale:
        parts.append("data may be stale")

    return " | ".join(parts)


