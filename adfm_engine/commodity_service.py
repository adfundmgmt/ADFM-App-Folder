"""Native orchestration for the Streamlit commodity event study formulas."""
import pandas as pd

from adfm_engine import commodity_legacy_math as math
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable


def load_commodity_event_study(*, symbol="CL=F", signal_type="Return threshold",
                               direction="Rally", return_window="3M", threshold=25.0,
                               rsi_period=14, spacing="3M", lookback="Max",
                               session_hour=None):
    try:
        data = math.load_contract_history(symbol)
    except Exception as exc:
        raise DataUnavailable(f"Price history is unavailable for {symbol}. Please retry later.") from exc
    close = data["Close"].dropna().astype(float)
    window = (math.RETURN_WINDOWS[return_window] if signal_type == "Return threshold"
              else 252 if signal_type == "52-week breakout"
              else rsi_period if signal_type == "RSI extreme" else 200)
    metric, condition, label, kind = math.build_signal(
        close, signal_type, direction=direction, window_days=window,
        threshold=threshold, rsi_period=rsi_period)
    all_events = math.detect_events(condition, math.SPACING_OPTIONS[spacing],
                                    continuous=signal_type == "52-week breakout")
    years = math.LOOKBACK_OPTIONS[lookback]
    chart_close = close if years is None else close.loc[
        close.index >= close.index.max() - pd.DateOffset(years=years)]
    study_events = all_events[all_events >= chart_close.index.min()]
    history, arrays = math.build_event_observations(close, study_events, metric)
    summary = math.summarize_forward_performance(arrays)
    latest = close.index.max()
    signal_value = metric.loc[latest]
    history["Date"] = pd.to_datetime(history["Date"]).dt.strftime("%Y-%m-%d") if not history.empty else pd.Series(dtype=str)
    return {
        "symbol": symbol, "name": math.CONTRACT_SYMBOL_TO_NAME.get(symbol, symbol),
        "signal": label, "active": bool(condition.loc[latest]),
        "signal_value": None if pd.isna(signal_value) else float(signal_value),
        "signal_kind": kind, "count": len(study_events),
        "as_of": str(latest.date()),
        "latest_event": str(study_events.max().date()) if len(study_events) else None,
        "summary": records(summary.reset_index(names="Metric")),
        "history": records(history.sort_values("Date", ascending=False)),
        "figure": figure_json(math.make_price_chart(chart_close, study_events, metric, label, kind)),
        "source": "Yahoo Finance continuous futures",
    }
