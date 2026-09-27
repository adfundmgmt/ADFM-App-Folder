"""Native data boundary for the original sector breadth and rotation calculations."""
import pandas as pd
from functools import lru_cache
from time import monotonic

from adfm_engine import sector_rotation_legacy_math as rotation
from adfm_engine.serialization import figure_json, records
from adfm_engine.services import DataUnavailable


@lru_cache(maxsize=32)
def _cached_prices(tickers: tuple[str, ...], bucket: int) -> pd.DataFrame:
    """Reuse a market snapshot while visitors change presentation controls."""
    return rotation.fetch_prices(list(tickers))


def load_sector_rotation(*, universe="Core subsectors", benchmark="SPY",
                         mode="Benchmark-relative rotation", window="Fast (1M vs 3M)",
                         trail="4 weeks", groups=None, selected_ticker="SMH",
                         label_mode="Top ranked only", session_hour=None):
    base = rotation.build_universe(universe)
    selected = rotation.filter_universe_by_groups(base, groups if groups is not None else
                                                  base["Sector Group"].drop_duplicates().tolist())
    if selected.empty:
        return {"rows": [], "coverage": 0, "requested": 0, "charts": [], "as_of": None}
    cfg = rotation.get_rotation_config(window)
    min_rows = cfg.long_window + 30
    tickers = list(dict.fromkeys(selected["Ticker"].tolist() + [benchmark]))
    raw = _cached_prices(tuple(tickers), int(monotonic() // 900))
    ok, message = rotation.validate_benchmark(raw, benchmark, min_rows)
    if not ok:
        raise DataUnavailable(message)
    eligible, diagnostics = rotation.filter_universe_by_data(selected, raw, benchmark, min_rows)
    if eligible.empty:
        raise DataUnavailable("No sector ETFs have sufficient current history for this window.")
    item_tickers = eligible["Ticker"].tolist()
    prices = rotation.prepare_analysis_prices(raw, list(dict.fromkeys(item_tickers + [benchmark])),
                                               benchmark)
    ok, message = rotation.validate_benchmark(prices, benchmark, min_rows)
    if not ok:
        raise DataUnavailable(message)
    basis = rotation.ROTATION_MODES[mode]
    snap = rotation.build_snapshot(prices, eligible, benchmark, cfg, basis)
    rank_delta = rotation.compute_rank_delta(prices, eligible, benchmark, cfg, basis, compare_days=5)
    trails = rotation.build_rotation_trails(prices, eligible, benchmark, cfg,
                                           rotation.TRAIL_OPTIONS[trail], basis)
    table = rotation.build_display_table(snap, rank_delta, cfg).sort_values("Rank")
    selected_ticker = selected_ticker if selected_ticker in item_tickers else item_tickers[0]
    lookup = eligible.set_index("Ticker")
    rs = rotation.compute_relative_strength(prices, item_tickers, benchmark)
    label = f"{lookup.loc[selected_ticker, 'Name']} ({selected_ticker})"
    available = snap.sort_values("Rank")["Ticker"].tolist()
    latest = pd.to_datetime(raw[benchmark].dropna().index.max()).date()
    return {
        "rows": records(table), "coverage": len(item_tickers), "requested": len(selected),
        "as_of": str(latest), "benchmark": benchmark, "selected_ticker": selected_ticker,
        "tickers": available,
        "excluded": records(diagnostics.loc[diagnostics["Status"] != "OK"]),
        "rotation_chart": figure_json(rotation.make_rotation_scatter(
            snap, cfg, trails, basis, benchmark, label_mode, 25)),
        "rs_chart": figure_json(rotation.make_rs_chart(rs[selected_ticker].dropna(), label, benchmark)),
    }
