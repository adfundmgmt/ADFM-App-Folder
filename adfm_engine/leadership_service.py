"""Equity leadership orchestration, retaining the full fixed ranking universe."""
from datetime import date, timedelta

import pandas as pd

from adfm_engine.analytics.leadership import build_leadership_frame
from adfm_engine.analytics.leadership_universe import ALL_SPECS, FAMILY_BY_KEY, LEADERSHIP_FAMILIES, SPEC_BY_KEY, raw_ratio
from adfm_engine.charts.leadership import make_detail_figure, make_rotation_map
from adfm_engine.data.leadership import fetch_closes, unique_keep_order
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable

DETAIL_SPANS = {"6 Months": 180, "1 Year": 365, "3 Years": 365 * 3, "5 Years": 365 * 5}
STATES = ["Leading", "Improving", "Weakening", "Lagging"]


def leadership(closes, *, families=None, states=None, history="3 Years", market_date=None):
    market_date = market_date or date.today()
    families = list(LEADERSHIP_FAMILIES) if families is None else families
    states = STATES if states is None else states
    if history not in DETAIL_SPANS or set(families) - set(LEADERSHIP_FAMILIES) or set(states) - set(STATES):
        raise ValueError("Unsupported leadership controls.")
    result = dict(schema_version=1, families=list(LEADERSHIP_FAMILIES), states=STATES, history_options=list(DETAIL_SPANS), warnings=[], unavailable=[], rotation=None, charts=[], rows=[])
    if not families:
        result["warnings"].append("Select at least one leadership family.")
        return result
    if closes.empty:
        raise DataUnavailable("Failed to download leadership data.")
    ratios = {}
    for spec in ALL_SPECS:
        if spec.ticker_1 not in closes or spec.ticker_2 not in closes:
            result["unavailable"].append(spec.key)
            continue
        series = raw_ratio(closes[spec.ticker_1], closes[spec.ticker_2])
        if len(series) < 22:
            result["unavailable"].append(spec.key)
            continue
        ratios[spec.key] = series
    metadata = pd.DataFrame({"Family": [FAMILY_BY_KEY[spec.key] for spec in ALL_SPECS], "Relationship": [spec.label for spec in ALL_SPECS], "Pair": [spec.key for spec in ALL_SPECS], "Note": [spec.note for spec in ALL_SPECS]}, index=[spec.key for spec in ALL_SPECS])
    all_rows = build_leadership_frame(ratios, metadata)
    if all_rows.empty:
        raise DataUnavailable("The available histories were insufficient to calculate leadership scores.")
    # Filters apply AFTER the fixed-universe cross-sectional score calculation.
    visible = all_rows.loc[all_rows["Family"].isin(families) & all_rows["State"].isin(states)].copy()
    result["rows"] = records(visible)
    if visible.empty:
        result["warnings"].append("No relationships match the selected family and state filters.")
        return result
    result["rotation"] = figure_json(make_rotation_map(visible))
    start = pd.Timestamp(market_date - timedelta(days=DETAIL_SPANS[history]))
    for family in families:
        for spec in LEADERSHIP_FAMILIES[family]:
            if spec.key in visible.index:
                row = records(visible.loc[[spec.key]])[0]
                result["charts"].append(dict(key=spec.key, family=family, title=spec.label, note=spec.note, metrics=row, figure=figure_json(make_detail_figure(ratios[spec.key], start))))
    return result


def load_leadership(**parameters):
    today = date.today()
    tickers = unique_keep_order(ticker for spec in ALL_SPECS for ticker in (spec.ticker_1, spec.ticker_2))
    closes = fetch_closes(tuple(tickers), today - timedelta(days=365 * 5 + 120), today + timedelta(days=1))
    return leadership(closes, market_date=today, **parameters)
