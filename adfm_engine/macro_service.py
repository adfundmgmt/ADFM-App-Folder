"""Global Macro Regime page outputs, shared by API and future reports."""
from datetime import timedelta

import pandas as pd

from adfm_engine.analytics import macro as engine
from adfm_engine.data.macro import fetch_market_prices, fetch_macro_data
from adfm_engine.serialization import records
from adfm_engine.services import DataUnavailable


def performance_cells(frame):
    """Transport the original per-column Styler shading as plain CSS values."""
    colors = [{} for _ in range(len(frame))]
    for column in ("1W", "1M", "3M", "YTD"):
        values = pd.to_numeric(frame[column], errors="coerce")
        bound = max(float(values.abs().quantile(.90)) if values.notna().any() else 1., .01)
        for i, value in enumerate(values):
            if engine.is_valid(value) and value != 0:
                alpha = .05 + .12 * min(abs(float(value)) / bound, 1.)
                rgb = engine.POSITIVE_RGB if value > 0 else engine.NEGATIVE_RGB
                colors[i][column] = f"rgba({rgb},{alpha:.3f})"
    return colors


def macro_regime(prices, macro, macro_status=None, failed=None):
    if prices.empty:
        raise DataUnavailable("Market data did not load.")
    macro_status = pd.DataFrame() if macro_status is None else macro_status
    dates = [engine.latest_date(prices[col]) for col in prices]
    dates = [d for d in dates if d is not None]
    asof = max(dates) if dates else pd.Timestamp.today().normalize()
    current = engine.build_snapshot(prices, macro, asof)
    one_month = engine.build_snapshot(prices, macro, asof - timedelta(days=30))
    three_month = engine.build_snapshot(prices, macro, asof - timedelta(days=90))
    names = [("Growth", "growth"), ("Inflation", "inflation"), ("Rates", "rates"), ("Liquidity", "liquidity"), ("Risk confirmation", "risk")]
    drivers = pd.DataFrame([[label, current[key], engine.evidence(current[key + "_indicators"])] for label, key in names], columns=["Regime component", "Current read", "Evidence"])
    performance = engine.cross_asset_table(prices, asof)
    return dict(schema_version=1, data_through=asof.date().isoformat(), failed=failed or [],
        market_loaded=sum(not engine.clean_series(prices[col]).empty for col in prices), market_total=len(engine.TICKERS),
        macro_loaded=int((macro_status["status"] == "OK").sum()) if "status" in macro_status else 0, macro_total=len(macro_status),
        current={k: current[k] for k in ("regime", "growth", "inflation", "rates", "liquidity", "risk")}, narrative=engine.narrative(current),
        drivers=records(drivers), tensions=engine.build_tensions(current, prices, macro),
        states=records(engine.state_table(current, one_month, three_month)), rates=records(engine.rates_fci_table(current, macro)),
        performance=records(performance), performance_colors=performance_cells(performance) if not performance.empty else [], macro_status=records(macro_status))


def load_macro_regime():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        market = pool.submit(fetch_market_prices, tuple(engine.TICKERS))
        macro = pool.submit(fetch_macro_data)
        prices, failed = market.result()
        panel, status = macro.result()
    return macro_regime(prices, panel, status, failed)
