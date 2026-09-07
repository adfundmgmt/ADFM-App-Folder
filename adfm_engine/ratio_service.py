"""Reusable ratio chartbook orchestration; provider inputs can be replayed."""
from datetime import date, timedelta

import pandas as pd

from adfm_engine.analytics.ratio_universe import RATIO_FAMILIES, MA_DEFAULTS, parse_custom_ratio_text, unique_keep_order, make_display_title
from adfm_engine.analytics.ratios import compute_price_ratio, ratio_signal_line
from adfm_engine.charts.ratios import make_fig
from adfm_engine.data.ratios import fetch_closes
from adfm_engine.serialization import figure_json
from adfm_engine.services import DataUnavailable

SPANS = {"3 Months": 90, "6 Months": 180, "9 Months": 270, "YTD": None, "1 Year": 365, "3 Years": 1095, "5 Years": 1825, "10 Years": 3650, "20 Years": 7300}


def date_bounds(history, today):
    days = max(1, (today - date(today.year, 1, 1)).days) if history == "YTD" else SPANS[history]
    start = pd.Timestamp(date(today.year, 1, 1) if history == "YTD" else today - timedelta(days=days))
    return today - timedelta(days=max(900, days + 450)), today + timedelta(days=1), start


def ratios(closes_static, closes_custom=None, *, families=None, history="3 Years", rsi_window=14, show_rsi=False, show_signal_strip=True, moving_averages=None, custom="", market_date=None):
    families = list(RATIO_FAMILIES) if families is None else families
    moving_averages = [k for k, v in MA_DEFAULTS.items() if v] if moving_averages is None else moving_averages
    if history not in SPANS or set(families) - set(RATIO_FAMILIES) or not 5 <= rsi_window <= 30 or set(moving_averages) - set(MA_DEFAULTS):
        raise ValueError("Unsupported ratio chartbook controls.")
    result = dict(schema_version=1, charts=[], warnings=[], unavailable=[], unavailable_custom=[])
    if not families:
        result["warnings"].append("Select at least one chart family.")
        return result
    if closes_static.empty:
        raise DataUnavailable("Failed to download price data.")
    _, _, start = date_bounds(history, market_date or date.today())
    settings = {k: k in moving_averages for k in MA_DEFAULTS}

    def append(spec, closes, family, compact):
        a, b = spec.ticker_1, spec.ticker_2
        unavailable = result["unavailable" if compact else "unavailable_custom"]
        if a not in closes or b not in closes:
            unavailable.append(f"{a}/{b}")
            return
        ratio = compute_price_ratio(closes[a], closes[b], base_date=start, base=100.0)
        if ratio.empty:
            unavailable.append(f"{a}/{b}")
            return
        title = make_display_title(spec)
        result["charts"].append(dict(key=f"{family}:{a}/{b}", family=family, title=title, note=spec.note, compact=compact,
            figure=figure_json(make_fig(ratio, title, start, settings, show_rsi, rsi_window, compact)),
            signal=ratio_signal_line(ratio, rsi_len=rsi_window) if show_signal_strip else None))

    for family in families:
        for spec in RATIO_FAMILIES[family]:
            append(spec, closes_static, family, True)
    custom_specs = parse_custom_ratio_text(custom)
    if custom_specs:
        if closes_custom is None or closes_custom.empty:
            result["warnings"].append("No custom ratio data available.")
        else:
            for spec in custom_specs:
                append(spec, closes_custom, "Custom Ratios", False)
    result["unavailable"] = sorted(set(result["unavailable"]))
    result["unavailable_custom"] = sorted(set(result["unavailable_custom"]))
    return result


def load_ratios(**parameters):
    today = date.today()
    start, end, _ = date_bounds(parameters.get("history", "3 Years"), today)
    families = parameters.get("families")
    families = list(RATIO_FAMILIES) if families is None else families
    if not families:
        return ratios(pd.DataFrame(), market_date=today, **parameters)
    specs = [spec for family in families for spec in RATIO_FAMILIES[family]]
    def fetch(specs):
        tickers = unique_keep_order(t for spec in specs for t in (spec.ticker_1, spec.ticker_2))
        return fetch_closes(tuple(tickers), start, end) if tickers else pd.DataFrame()
    static = fetch(specs)
    custom = fetch(parse_custom_ratio_text(parameters.get("custom", ""))) if not static.empty else None
    return ratios(static, custom, market_date=today, **parameters)
