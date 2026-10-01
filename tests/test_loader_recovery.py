import ast
import hashlib  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import tempfile
import time  # noqa: F401
import unittest
from datetime import date, datetime, timedelta  # noqa: F401
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple  # noqa: F401
from unittest.mock import Mock, patch

import numpy as np  # noqa: F401
import pandas as pd

source = Path(__file__).resolve().parents[1] / "pages/1_ADFM_Public_Equities_Baskets.py"
if not source.exists():
    source = Path(__file__).with_name("loader_fixed.py")
scope = dict(
    globals(),
    CACHE_VERSION=3,
    CACHE_MAX_AGE_DAYS=7,
    BASKET_FETCH_BUDGET_SECONDS=40.0,
    BASKET_BATCH_BUDGET_SECONDS=9.0,
    BASKET_DOWNLOAD_CHUNK_SIZE=300,
    BASKET_DOWNLOAD_THREADS=32,
    BENCH="SPY",
    MIN_DAILY_MEMBER_COVERAGE=0.6,
)
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.ClassDef) and node.name == "PriceFeedUnavailable":
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
            scope,
        )
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
            scope,
        )


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        scope["CACHE_DIR"] = Path(self.temp.name)
        self.start = pd.Timestamp("2026-09-01")
        self.end = pd.Timestamp("2026-09-10")
        self.cache = pd.DataFrame(
            {"SPY": [100.0, 101.0, 102.0], "AAA": [50.0, 51.0, 52.0]},
            index=pd.to_datetime(["2026-09-01", "2026-09-04", "2026-09-08"]),
        )

    def tearDown(self):
        self.temp.cleanup()

    def save(self, frame=None):
        scope["save_last_good_levels"](
            self.cache if frame is None else frame,
            {"source": "yahoo"},
            "legacy_version2_key",
        )

    def test_legacy_snapshot_recognized(self):
        self.save()
        data, meta = scope["compatible_snapshot"](
            ["SPY", "AAA"], self.start, self.end, "new_version3_key"
        )
        pd.testing.assert_frame_equal(data, self.cache, check_like=True)

    def test_outage_falls_back_whole(self):
        self.save()
        with patch.dict(scope, {"_download_close": Mock(return_value=pd.DataFrame())}):
            data, meta = scope["fetch_daily_levels"](
                ["SPY", "AAA"], self.start, self.end
            )
        pd.testing.assert_frame_equal(data, self.cache, check_like=True)
        self.assertEqual(meta["source"], "last_good_cache")

    def test_partial_benchmark_does_not_defeat_fallback(self):
        self.save()
        fresh = pd.DataFrame({"SPY": [999.0]}, index=[pd.Timestamp("2026-09-09")])
        with patch.dict(scope, {"_download_close": Mock(return_value=fresh)}):
            data, meta = scope["fetch_daily_levels"](
                ["SPY", "AAA"], self.start, self.end
            )
        pd.testing.assert_frame_equal(data, self.cache, check_like=True)
        self.assertEqual(meta["source"], "last_good_cache")

    def test_current_snapshot_avoids_requests(self):
        self.cache.index = pd.to_datetime(["2026-09-01", "2026-09-04", "2026-09-09"])
        self.save()
        download = Mock(side_effect=AssertionError("Unexpected network call"))
        with patch.dict(scope, {"_download_close": download}):
            data, meta = scope["fetch_daily_levels"](
                ["SPY", "AAA"], self.start, self.end
            )
        self.assertEqual(meta["source"], "saved_snapshot")
        download.assert_not_called()

    def test_current_partial_snapshot_does_not_prevent_constituent_recovery(self):
        self.cache.index = pd.to_datetime(["2026-09-01", "2026-09-04", "2026-09-09"])
        self.cache["BBB"] = [20.0, 21.0, 22.0]
        self.save()
        recovered = self.cache.assign(CCC=[30.0, 31.0, 32.0])
        with patch.dict(scope, {"_download_close": Mock(return_value=recovered)}):
            data, meta = scope["fetch_daily_levels"](
                ["SPY", "AAA", "BBB", "CCC"], self.start, self.end
            )
        self.assertIn("CCC", data)
        self.assertEqual(data["CCC"].tolist(), [30.0, 31.0, 32.0])
        self.assertEqual(meta["missing_tickers"], [])

    def test_broad_partial_live_universe_fills_missing_symbols_from_cache(self):
        current_dates = pd.to_datetime(["2026-09-01", "2026-09-04", "2026-09-08"])
        cached = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0],
                "AAA": [50.0, 51.0, 52.0],
                "BBB": [60.0, 61.0, 62.0],
                "CCC": [70.0, 71.0, 72.0],
            },
            index=current_dates,
        )
        self.save(cached)
        fresh = cached.drop(columns=["CCC"]).copy()
        fresh[["SPY", "AAA", "BBB"]] += 1.0
        with patch.dict(scope, {"_download_close": Mock(return_value=fresh)}):
            data, meta = scope["fetch_daily_levels"](
                ["SPY", "AAA", "BBB", "CCC"], self.start, self.end
            )
        self.assertEqual(meta["source"], "yahoo+saved_snapshot")
        self.assertEqual(meta["cache_fallback_symbols"], ["CCC"])
        self.assertEqual(data["CCC"].tolist(), [70.0, 71.0, 72.0])
        self.assertEqual(data["AAA"].tolist(), [51.0, 52.0, 53.0])

    def test_empty_feed_raises_and_stops_request_storm(self):
        download = Mock(return_value=pd.DataFrame())
        with patch.dict(scope, {"_download_close": download}):
            with self.assertRaises(scope["PriceFeedUnavailable"]):
                scope["fetch_daily_levels"](
                    ["SPY"] + [f"T{i}" for i in range(200)], self.start, self.end
                )
        self.assertLessEqual(download.call_count, 3)

    def test_old_or_inadequate_snapshot_rejected(self):
        self.cache.index = pd.to_datetime(["2026-08-01", "2026-08-02", "2026-08-03"])
        self.save()
        data, _ = scope["compatible_snapshot"](
            ["SPY", "AAA"], self.start, self.end, "new"
        )
        self.assertTrue(data.empty)


if __name__ == "__main__":
    unittest.main()
