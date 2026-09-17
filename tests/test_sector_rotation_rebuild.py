import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from adfm_core.sector_rotation import (
    RotationWindow,
    adaptive_axis_range,
    attach_breadth,
    build_asset_levels,
    build_breadth_member_map,
    build_catalog,
    classify_quadrant,
    compute_breadth,
    compute_equal_weight_basket,
    compute_snapshot,
    confirmed_state_series,
    drop_incomplete_us_session,
    latest_state_metrics,
    movement_metrics,
    parse_state_street_holdings_frame,
    relative_series,
    required_tickers,
    select_catalog,
    trail_for_selected,
)


class SectorRotationRebuildTests(unittest.TestCase):
    def test_recovered_catalog_scope(self):
        catalog = build_catalog()
        self.assertEqual(len(catalog), 178)
        self.assertEqual(int((catalog["Kind"] == "Stock Basket").sum()), 36)
        self.assertEqual(int((catalog["Universe"] == "Countries").sum()), 40)
        self.assertEqual(set(catalog.query("Universe == 'Countries'")["Broad Benchmark"]), {"ACWI"})
        self.assertEqual(len(select_catalog(catalog, "Industries")), 91)
        self.assertFalse(select_catalog(catalog, "Countries", ["Countries & Regions"]).empty)

    def test_equal_weight_basket_is_daily_rebalanced_and_keeps_gaps_visible(self):
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
        self.assertTrue(pd.isna(compute_equal_weight_basket(gappy, ["A", "B", "C"], 0.60).iloc[2]))
        self.assertTrue(compute_equal_weight_basket(prices, ["MISSING"]).isna().all())

    def test_neutral_band_and_persistent_state_change(self):
        self.assertEqual(classify_quadrant(0.005, 0.02), "Neutral")
        self.assertEqual(classify_quadrant(0.02, 0.03), "Leading")
        self.assertEqual(classify_quadrant(-0.02, 0.03), "Improving")
        self.assertEqual(classify_quadrant(-0.02, -0.03), "Lagging")
        self.assertEqual(classify_quadrant(0.02, -0.03), "Weakening")
        raw = pd.Series([
            "Leading", "Leading", "Leading", "Leading", "Weakening", "Weakening",
            "Leading", "Weakening", "Weakening", "Weakening",
        ])
        confirmed = confirmed_state_series(raw, confirm_days=3)
        self.assertEqual(confirmed.iloc[-2], "Leading")
        self.assertEqual(confirmed.iloc[-1], "Weakening")
        state, days = latest_state_metrics(confirmed)
        self.assertEqual(state, "Weakening")
        self.assertEqual(days, 1)

    def test_movement_uses_five_session_coordinate_change(self):
        coords = pd.DataFrame({
            "x": [0.10, 0.11, 0.12, 0.13, 0.14, 0.16],
            "y": [0.20, 0.20, 0.21, 0.21, 0.22, 0.23],
        })
        dx, dy, speed, angle = movement_metrics(coords, lookback=5)
        self.assertTrue(np.isclose(dx, 0.06))
        self.assertTrue(np.isclose(dy, 0.03))
        self.assertTrue(np.isclose(speed, np.hypot(0.06, 0.03)))
        self.assertTrue(np.isclose(angle, np.degrees(np.arctan2(0.03, 0.06))))
        self.assertTrue(all(pd.isna(x) for x in movement_metrics(coords.head(2), lookback=5)))

    def test_adaptive_axes_remove_fifteen_point_floor_and_include_trail(self):
        low, high = adaptive_axis_range(pd.Series([-0.02, 0.03]), pd.Series([0.09]))
        self.assertGreater(low, -0.10)
        self.assertGreater(high, 0.09)
        self.assertLess(high, 0.15)
        self.assertEqual(adaptive_axis_range(pd.Series(dtype=float)), (-0.05, 0.05))

    def test_constituent_breadth_reports_50d_and_200d_participation(self):
        idx = pd.bdate_range("2025-01-02", periods=260)
        base = np.arange(260, dtype=float)
        prices = pd.DataFrame({
            "UP1": 100 + base,
            "UP2": 120 + base * 0.5,
            "DOWN": 400 - base,
        }, index=idx)
        result = compute_breadth(prices, ["UP1", "UP2", "DOWN"], min_coverage=1.0)
        self.assertTrue(np.isclose(result["% > 50D"], 2 / 3))
        self.assertTrue(np.isclose(result["% > 200D"], 2 / 3))
        self.assertGreater(result["Coverage"], 0.99)
        empty = compute_breadth(pd.DataFrame(), ["UP1"])
        self.assertTrue(pd.isna(empty["% > 50D"]))

    def test_breadth_member_map_and_attachment(self):
        catalog = build_catalog()
        subset = pd.concat([
            catalog.query("Universe == 'Sectors'").head(1),
            catalog.query("Kind == 'Stock Basket'").head(1),
        ]).reset_index(drop=True)
        sector_key = subset.iloc[0]["Key"]
        basket_key = subset.iloc[1]["Key"]
        members = build_breadth_member_map(subset, {subset.iloc[0]["Ticker"]: ["A", "B"]})
        self.assertEqual(members[sector_key], ["A", "B"])
        self.assertEqual(members[basket_key], subset.iloc[1]["Members"])

        idx = pd.bdate_range("2025-01-02", periods=260)
        prices = pd.DataFrame({"A": np.arange(260) + 100.0, "B": np.arange(260) + 120.0}, index=idx)
        snapshot = pd.DataFrame({"Key": [sector_key]})
        attached = attach_breadth(snapshot, prices, members)
        self.assertEqual(float(attached.iloc[0]["% > 50D"]), 1.0)

    def test_state_street_parser_preserves_firstcash_and_drops_cash_rows(self):
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
        self.assertIn("FCFS", tickers)
        self.assertIn("AAPL", tickers)
        self.assertNotIn("USD", tickers)
        self.assertNotIn("CASH", tickers)
        self.assertEqual(parse_state_street_holdings_frame(pd.DataFrame([["No", "Header"]])), [])

    def test_current_us_session_is_excluded_before_vendor_grace_time(self):
        idx = pd.DatetimeIndex(["2026-09-16", "2026-09-17"])
        prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=idx)
        before = datetime(2026, 9, 17, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        after = datetime(2026, 9, 17, 16, 30, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(list(drop_incomplete_us_session(prices, now=before).index), [pd.Timestamp("2026-09-16")])
        self.assertEqual(len(drop_incomplete_us_session(prices, now=after)), 2)
        self.assertTrue(drop_incomplete_us_session(pd.DataFrame()).empty)

    def test_required_tickers_includes_members_and_benchmarks(self):
        catalog = build_catalog().query("Kind == 'Stock Basket'").head(1).copy()
        tickers = required_tickers(catalog, {"extra": ["EXTRA"]})
        self.assertIn("SPY", tickers)
        self.assertIn("EXTRA", tickers)
        for member in catalog.iloc[0]["Members"]:
            self.assertIn(member, tickers)

    def test_relative_series_and_snapshot_rank_transparently(self):
        catalog = build_catalog().query("Universe == 'Sectors'").head(2).copy()
        idx = pd.bdate_range("2025-01-02", periods=320)
        raw = pd.DataFrame(index=idx)
        raw["SPY"] = 100 * (1.0005 ** np.arange(len(idx)))
        for i, ticker in enumerate(catalog["Ticker"]):
            raw[ticker] = 100 * ((1.001 if i == 0 else 1.0002) ** np.arange(len(idx)))
        levels = build_asset_levels(raw, catalog)
        rs = relative_series(levels[catalog.iloc[0]["Key"]], raw["SPY"])
        self.assertGreater(float(rs.iloc[-1]), float(rs.iloc[0]))
        snap = compute_snapshot(raw, levels, catalog, RotationWindow(21, 63, "1M", "3M"))
        leader = snap.sort_values("Rank").iloc[0]
        self.assertEqual(leader["Key"], catalog.iloc[0]["Key"])
        self.assertIn("Parent 1M Rel", snap)
        self.assertIn("Weekly Rel Δ", snap)
        self.assertIn("Days in State", snap)

    def test_selected_trail_uses_same_rotation_coordinates(self):
        catalog = build_catalog().query("Universe == 'Sectors'").head(1).copy()
        idx = pd.bdate_range("2025-01-02", periods=320)
        ticker = catalog.iloc[0]["Ticker"]
        raw = pd.DataFrame({
            "SPY": 100 * (1.0005 ** np.arange(len(idx))),
            ticker: 100 * (1.0008 ** np.arange(len(idx))),
        }, index=idx)
        levels = build_asset_levels(raw, catalog)
        trail = trail_for_selected(raw, levels, catalog, ticker, RotationWindow(21, 63, "1M", "3M"), 8)
        self.assertLessEqual(len(trail), 8)
        self.assertEqual(list(trail.columns), ["x", "y"])
        self.assertFalse(trail.empty)


if __name__ == "__main__":
    unittest.main()
