"""Reusable Credit Conditions Monitor orchestration."""
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from adfm_engine.analytics.credit import compute_credit, latest_timestamp
from adfm_engine.analytics.credit_definitions import FOCUS_WINDOWS, GLOBAL_WINDOWS, HISTORY_DAYS, MARKET_TICKERS, CREDIT_FRED_DEFINITIONS
from adfm_engine.charts.credit import spread_chart, funding_chart, appetite_chart, sovereign_bar_chart
from adfm_engine.data.credit import fetch_market_prices, load_global_sovereign_moves
from adfm_engine.data.primary import fetch_fred_series
from adfm_engine.serialization import records, figure_json
from adfm_engine.services import DataUnavailable


def credit(fred, market, fred_status=None, sovereign_moves=None, sovereign_source="Unavailable", sovereign_note="", *, focus_window="1M", global_window="1Y", history="3 Years"):
    if focus_window not in FOCUS_WINDOWS or global_window not in GLOBAL_WINDOWS or history not in HISTORY_DAYS:
        raise ValueError("Unsupported credit monitor controls.")
    model = compute_credit(fred, market, focus_window)
    if model["hy_oas"].empty and market.empty:
        raise DataUnavailable("Neither primary credit spreads nor market confirmation data loaded.")
    start = pd.Timestamp(date.today() - timedelta(days=HISTORY_DAYS[history]))
    spread = spread_chart(fred, model["hy_oas"], model["bbb_oas"], model["ig_oas"], start)
    funding = funding_chart(fred, model["dgs10"], model["dgs30"], start)
    appetite = appetite_chart(model["proxy"], start)
    sovereign_moves = pd.DataFrame() if sovereign_moves is None else sovereign_moves
    sovereign = []
    summary = []
    if not sovereign_moves.empty:
        max_abs = float(sovereign_moves["Move bp"].abs().max())
        x_limit = max(75.0, np.ceil((max_abs * 1.42 + 20.0) / 25.0) * 25.0)
        for group in ("Developed", "Emerging"):
            sovereign.append(figure_json(sovereign_bar_chart(sovereign_moves, group, x_limit)))
            selected = sovereign_moves.loc[sovereign_moves["Group"] == group, "Move bp"]
            summary.append(dict(group=group, count=len(selected), median=selected.median()))
    audit = pd.DataFrame({"Ticker": market.columns, "Latest Observation": [latest_timestamp(market[col]).date().isoformat() if latest_timestamp(market[col]) is not None else "N/A" for col in market]})
    return dict(schema_version=1, cards=model["cards"], narrative=model["active_read"],
        spread=figure_json(spread) if spread is not None else None, funding=figure_json(funding) if funding is not None else None,
        appetite=figure_json(appetite) if appetite is not None else None, sovereign=sovereign,
        sovereign_summary=records(pd.DataFrame(summary)), sovereign_source=sovereign_source, sovereign_note=sovereign_note,
        rows=records(pd.DataFrame(model["rows"])), fred_status=records(fred_status) if fred_status is not None else [], market_status=records(audit))


def load_credit(**parameters):
    today = date.today()
    start = (today - timedelta(days=365 * 10 + 45)).isoformat()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fred_task = pool.submit(fetch_fred_series, CREDIT_FRED_DEFINITIONS, start=start, end=today.isoformat())
        market_task = pool.submit(fetch_market_prices, MARKET_TICKERS, start, (today + timedelta(days=1)).isoformat())
        sovereign_task = pool.submit(load_global_sovereign_moves, parameters.get("global_window", "1Y"))
        fred, status = fred_task.result()
        market = market_task.result()
        moves, source, note = sovereign_task.result()
    return credit(fred, market, status, moves, source, note, **parameters)
