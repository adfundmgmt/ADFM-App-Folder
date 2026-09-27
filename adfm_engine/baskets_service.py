"""Public equity baskets: original formulas with a native orchestration boundary."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from adfm_engine import baskets_legacy_math as b
from adfm_engine.services import DataUnavailable


def load_baskets(*, preset="YTD", categories=None, market_cap_filter=False,
                 stale_days=30, session_date=None):
    now = datetime.now(ZoneInfo("America/New_York"))
    today = now.date()
    selected = categories if categories is not None else list(b.CATEGORIES)
    if not selected:
        return {"rows": [], "as_of": None, "warnings": ["Select a category."], "categories": list(b.CATEGORIES)}
    definitions = {key: b.CATEGORIES[key] for key in selected}
    display_start = b.compute_display_start(preset, today)
    fetch_start = b.compute_fetch_start(display_start)
    include_today = now.weekday() >= 5 or now.time() >= b.COMPLETED_SESSION_TIME
    fetch_end = today + timedelta(days=1) if include_today else today
    raw = b.flatten_baskets(definitions)
    symbols = b.unique_tickers_from_baskets(raw)
    need = sorted(set(symbols + b.required_fx_tickers(symbols) + [b.BENCH]))
    levels, meta = b.fetch_daily_levels(need, pd.Timestamp(fetch_start), pd.Timestamp(fetch_end))
    if levels.empty:
        raise DataUnavailable("No price history was returned for the selected baskets.")
    usd, fx_issues = b.convert_foreign_levels_to_usd(levels)
    if b.BENCH not in usd or usd[b.BENCH].dropna().empty:
        raise DataUnavailable("SPY price history is unavailable.")
    reference = pd.Timestamp(usd[b.BENCH].dropna().index.max())
    start = pd.Timestamp(b.compute_display_start(preset, reference.date()))
    metadata = b.fetch_market_metadata(symbols) if market_cap_filter else {}
    if metadata:
        metadata = b.convert_market_metadata_to_usd(metadata, levels)
    live, basket_meta, dropped = b.build_live_baskets(
        usd, definitions, metadata, b.MIN_MARKET_CAP if market_cap_filter else None,
        stale_days, reference)
    mapped = b.flatten_baskets(live)
    if not mapped:
        raise DataUnavailable("No baskets passed the data quality filters.")
    available = [symbol for symbol in b.unique_tickers_from_baskets(mapped, extra=[b.BENCH]) if symbol in usd]
    aligned = b.align_levels_to_calendar(usd[available], usd[b.BENCH].dropna().index)
    returns = b.ew_rets_from_levels(aligned, mapped)
    if returns.empty:
        raise DataUnavailable("Insufficient price history for basket returns.")
    benchmark = aligned[b.BENCH].dropna().pct_change(fill_method=None).dropna()
    panel = b.build_panel_df(returns, start, preset, basket_meta, benchmark)
    rows = []
    for key, values in panel.iterrows():
        row = {column: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
               for column, value in values.items()}
        row["Category"] = basket_meta[key]["Category"]
        row["Members"] = basket_meta[key]["Members"]
        rows.append(row)
    warnings = []
    if meta.get("source") != "yahoo":
        warnings.append("Some prices came from the last good local cache; check the observation date.")
    if fx_issues:
        warnings.append(f"FX conversion unavailable for {len(fx_issues)} symbols.")
    if dropped:
        warnings.append(f"{len(dropped)} baskets failed the minimum live-member coverage check.")
    return {"rows": rows, "as_of": str(reference.date()), "warnings": warnings,
            "categories": list(b.CATEGORIES), "requested_tickers": len(need),
            "returned_tickers": meta.get("returned_tickers"), "source": meta.get("source")}
