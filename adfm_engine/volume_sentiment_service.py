"""Native data and serialization boundary for the volume sentiment tool."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from adfm_engine.data.market import fetch_daily_ohlcv
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable
from adfm_engine import volume_sentiment_legacy_math as math


def _shares_outstanding(symbol: str) -> float | None:
    try:
        value = yf.Ticker(symbol).fast_info.get("shares")
        return float(value) if value and np.isfinite(value) else None
    except Exception:
        return None


def load_volume_sentiment(*, symbol="QQQ", lookback_months=18,
                          volume_mode="Dollar volume", percentile_window=126,
                          smooth_window=20, high_cutoff=90, low_cutoff=10,
                          show_price_mas=True, event_filter="All extremes",
                          max_event_rows=12):
    frames, _ = fetch_daily_ohlcv((symbol,), period="10y")
    raw = frames.get(symbol)
    if raw is None or raw.empty:
        raise DataUnavailable(f"No daily price and volume history was returned for {symbol}.")
    prepared = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    prepared["Raw_Close"] = raw["Close"]
    prepared["Close"] = raw.get("Adj Close", raw["Close"])
    prepared = math.validate_ohlcv(prepared, "Yahoo Finance")
    if prepared["Volume"].sum() <= 0:
        raise DataUnavailable("This symbol has no usable exchange volume.")
    shares = _shares_outstanding(symbol) if volume_mode == "Turnover %" else None
    full, label, fallback = math.compute_volume_framework(
        prepared, volume_mode, shares, percentile_window,
        smooth_window, high_cutoff, low_cutoff,
    )
    visible_start = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date()) - pd.DateOffset(months=lookback_months)
    visible = full.loc[full.index >= visible_start].copy()
    if visible.empty:
        raise DataUnavailable("No observations remain in the selected visible history.")
    latest = visible.iloc[-1]
    outcomes = math.build_setup_outcomes(full)
    events = math.build_recent_events(visible, event_filter, max_event_rows)
    event_columns = {
        "Close": "Close", "Ret_1D": "1D %", "Ret_5D": "5D %",
        "Ret_20D": "20D %", "Volume_Display": "Volume",
        "Volume_Ratio": "Vs Baseline", "Volume_Pctl": "Percentile",
        "Close_Location": "Close Location", "Fwd_5D": "Forward 5D %",
        "Fwd_20D": "Forward 20D %", "Max_DD_20D": "Max Drawdown 20D %",
    }
    event_rows = events[["Setup", *event_columns]].rename(columns=event_columns).copy() if not events.empty else pd.DataFrame()
    if not event_rows.empty:
        event_rows.insert(0, "Date", events.index.strftime("%Y-%m-%d"))
        event_rows["Close Location"] *= 100
    chart = math.build_chart(visible, symbol, label, show_price_mas, [])
    return {
        "symbol": symbol, "as_of": str(visible.index[-1].date()),
        "source": "Yahoo Finance adjusted close and reported daily volume",
        "volume_label": label, "turnover_fallback": fallback,
        "setup": str(latest["Setup"]), "state": str(latest["State"]),
        "latest": {
            key: float(latest[key]) if np.isfinite(latest[key]) else None
            for key in ("Close", "Volume_Pctl", "Volume_Ratio", "Ret_1D",
                        "Close_Location", "Volume_Display", "Volume_Baseline")
        },
        "events": records(event_rows), "outcomes": records(outcomes),
        "figure": figure_json(chart),
    }
