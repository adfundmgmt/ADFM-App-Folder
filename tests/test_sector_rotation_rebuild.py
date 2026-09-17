from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from adfm_core.sector_rotation import (
    RotationWindow,
    adaptive_axis_range,
    build_asset_levels,
    build_catalog,
    classify_quadrant,
    compute_breadth,
    compute_equal_weight_basket,
    compute_snapshot,
    confirmed_state_series,
    drop_incomplete_us_session,
    movement_metrics,
    parse_state_street_holdings_frame,
)


def test_recovered_catalog_scope():
    catalog = build_catalog()
    assert len(catalog) == 178
    assert (catalog["Kind"] == "Stock Basket").sum() == 36
    assert (catalog["Universe"] == "Countries").sum() == 40
    assert set(catalog.query("Universe == 'Countries'")["Broad Benchmark"]) == {"ACWI"}


def test_equal_weight_basket_is_daily_rebalanced_and_keeps_gaps_visible():
    idx = pd.bdate_range("2026-01-02", periods=4)
    prices = pd.DataFrame({
        "A": [100.0, 110.0, 110.0, 121.0],
        "B": [100.0, 100.0, 110.0, 110.0],
    }, index=idx)
    out = compute_equal_weight_basket(prices, ["A", "B"], min_coverage=1.0)
    expected = (1 + pd.Series([0.0, 0.05, 0.05, 0.05], index=idx)).cumprod() * 100
    pd.testing.assert_series_equal(out, expected, check_names=False)

    gappy = pd.DataFrame({
        "A": [100.0, 101.0, np.nan, 103.0],
        "B": [100.0, 101.0, np.nan, 103.0],
        "C": [100.0, 101.0, 102.0, 103.0],
    }, index=idx)
    assert pd.isna(compute_equal_weight_basket(gappy, ["A", "B", "C"], 0.60).iloc[2])


def test_neutral_band_and_persistent_state_change():
    assert classify_quadrant(0.005, 0.02) == "Neutral"
    assert classify_quadrant(0.02, 0.03) == "Leading"
    raw = pd.Series(["Leading"] * 4 + ["Weakening", "Weakening", "Leading", "Weakening", "Weakening", "Weakening"])
    confirmed = confirmed_state_series(raw, confirm_days=3)
    assert confirmed.iloc[-2] == "Leading"
    assert confirmed.iloc[-1] == "Weakening"


def test_movement_uses_five_session_coordinate_change():
    coords = pd.DataFrame({
        "x": [0.10, 0.11, 0.12, 0.13, 0.14, 0.16],
        "y": [0.20, 0.20, 0.21, 0.21, 0.22, 0.23],
    })
    dx, dy, speed, angle = movement_metrics(coords, lookback=5)
    assert np.isclose(dx, 0.06)
    assert np.isclose(dy, 0.03)
    assert np.isclose(speed, np.hypot(0.06, 0.03))
    assert np.isclose(angle, np.degrees(np.arctan2(0.03, 0.06)))


def test_adaptive_axes_remove_fifteen_point_floor_and_include_trail():
    low, high = adaptive_axis_range(pd.Series([-0.02, 0.03]), pd.Series([0.09]))
    assert low > -0.10
    assert high > 0.09
    assert high < 0.15


def test_constituent_breadth_reports_50d_and_200d_participation():
    idx = pd.bdate_range("2025-01-02", periods=260)
    base = np.arange(260, dtype=float)
    prices = pd.DataFrame({
        "UP1": 100 + base,
        "UP2": 120 + base * 0.5,
        "DOWN": 400 - base,
    }, index=idx)
    result = compute_breadth(prices, ["UP1", "UP2", "DOWN"], min_coverage=1.0)
    assert np.isclose(result["% > 50D"], 2 / 3)
    assert np.isclose(result["% > 200D"], 2 / 3)


def test_state_street_parser_preserves_firstcash_and_drops_cash_rows():
    frame = pd.DataFrame([
        ["Fund", "XLF", None],
        [None, None, None],
        ["Name", "Ticker", "Weight"],
        ["FirstCash Holdings", "FCFS", 0.1],
        ["Apple", "AAPL", 0.2],
        ["US DOLLAR", "USD", 0.01],
        ["Cash", "CASH", 0.01],
    ])
    tickers = parse_state_street_holdings_frame(frame)
    assert "FCFS" in tickers and "AAPL" in tickers
    assert "USD" not in tickers and "CASH" not in tickers


def test_current_us_session_is_excluded_before_vendor_grace_time():
    idx = pd.DatetimeIndex(["2026-09-16", "2026-09-17"])
    prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=idx)
    now = datetime(2026, 9, 17, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    out = drop_incomplete_us_session(prices, now=now)
    assert list(out.index) == [pd.Timestamp("2026-09-16")]


def test_snapshot_ranks_transparently_by_one_month_relative_performance():
    catalog = build_catalog().query("Universe == 'Sectors'").head(2).copy()
    idx = pd.bdate_range("2025-01-02", periods=320)
    raw = pd.DataFrame(index=idx)
    raw["SPY"] = 100 * (1.0005 ** np.arange(len(idx)))
    for i, ticker in enumerate(catalog["Ticker"]):
        raw[ticker] = 100 * ((1.001 if i == 0 else 1.0002) ** np.arange(len(idx)))
    levels = build_asset_levels(raw, catalog)
    snap = compute_snapshot(raw, levels, catalog, RotationWindow(21, 63, "1M", "3M"))
    leader = snap.sort_values("Rank").iloc[0]
    assert leader["Key"] == catalog.iloc[0]["Key"]
    assert "Parent 1M Rel" in snap and "Weekly Rel Δ" in snap
