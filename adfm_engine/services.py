"""Reusable page services: jobs, HTTP, and reports call the same functions."""
from datetime import datetime, timezone

import pandas as pd

from adfm_engine.analytics.rate_of_change import add_trading_session_axis, compute_features
from adfm_engine.charts.rate_of_change import ROC_PERIODS, TIMEFRAME_MAP, build_figure
from adfm_engine.data.integrity import DataIntegrityPolicy, build_data_quality_report
from adfm_engine.data.market import fetch_daily_ohlcv
from adfm_engine.serialization import figure_json, records


class DataUnavailable(RuntimeError):
    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


def rate_of_change(frame: pd.DataFrame, *, symbol="^SPX", window="3Y", roc="63D", view="Candlestick", inflections=True) -> dict:
    """Exact page-12 calculation and chart pipeline on an injected raw frame."""
    if frame.empty:
        raise DataUnavailable(f"No valid OHLCV data returned for {symbol}.")
    warnings = []
    if len(frame) < 60:
        warnings.append("The selected window has fewer than 60 observations, so derivative readings may be unstable.")
    quality = build_data_quality_report({symbol: frame}, symbol, policy=DataIntegrityPolicy(min_valid_sessions=1, max_stale_sessions=0))
    feat = compute_features(frame, ROC_PERIODS[roc]).dropna(subset=["ROC", "Second_Derivative"], how="any")
    feat = add_trading_session_axis(feat)
    if feat.empty:
        warnings.append("Data became empty after indicator calculations. Try a longer analysis window.")
    through = quality.data_through.date().isoformat() if quality.data_through is not None else None
    return {
        "schema_version": 1,
        "parameters": {"symbol": symbol, "window": window, "roc": roc, "view": view, "inflections": inflections},
        "source": "Yahoo Finance",
        "data_through": through,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "quality": "complete daily sessions" if quality.benchmark_ready else quality.reason_for(symbol),
        "quality_diagnostics": records(quality.diagnostics),
        "observation_count": len(frame),
        "warnings": warnings,
        "figure": figure_json(build_figure(feat, symbol, roc, view, inflections)) if not feat.empty else None,
        "rows": records(feat.reset_index(names="Date")),
    }


def load_rate_of_change(*, symbol="^SPX", window="3Y", roc="63D", view="Candlestick", inflections=True) -> dict:
    frames, dropped = fetch_daily_ohlcv((symbol,), TIMEFRAME_MAP[window])
    if symbol not in frames:
        raise DataUnavailable(f"Failed to fetch {symbol}: No valid OHLCV data returned.")
    return rate_of_change(frames[symbol], symbol=symbol, window=window, roc=roc, view=view, inflections=inflections)


def overview() -> dict:
    """Home reuses the verified ROC engine, with per-symbol dates and failures."""
    symbols = ("^SPX", "QQQ", "TLT", "GLD", "USDJPY=X", "CL=F")
    frames, dropped = fetch_daily_ohlcv(symbols, "1y")
    rows = []
    for symbol in symbols:
        row = dict(symbol=symbol, data_through=None, close=None, roc=None, acceleration=None)
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            row["error"] = "Market history unavailable"
        else:
            features = compute_features(frame, 63)
            last = features.iloc[-1]
            row.update(data_through=features.index[-1].date().isoformat(), close=last["Close"], roc=last["ROC"], acceleration=last["Second_Derivative"])
        rows.append(row)
    return {"source": "Yahoo Finance", "rows": records(pd.DataFrame(rows))}
