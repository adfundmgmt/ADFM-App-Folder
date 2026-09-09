"""Exercise the real basket loader without rendering or calling providers."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd


class BasketDataLoadingTests(unittest.TestCase):
    def setUp(self):
        page = Path(__file__).resolve().parents[1] / "pages" / "1_ADFM_Public_Equities_Baskets.py"
        tree = ast.parse(page.read_text(encoding="utf-8"))
        names = {"_chunk", "_clean_index", "_to_float_frame", "_download_close_once", "_download_close", "fetch_daily_levels"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in functions:
            node.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
        self.download = Mock()
        self.cached = Mock(return_value=(pd.DataFrame(), {}))
        self.usable = Mock(return_value=False)
        self.namespace = {"pd": pd, "yf": SimpleNamespace(download=self.download), "time": SimpleNamespace(sleep=Mock()),
                          "BENCH": "SPY", "_cache_key": Mock(return_value="fixture"),
                          "load_last_good_levels": self.cached, "_cache_is_usable": self.usable,
                          "save_last_good_levels": Mock(return_value=None)}
        exec(compile(ast.fix_missing_locations(module), str(page), "exec"), self.namespace)
        self.dates = pd.bdate_range("2026-01-02", periods=3)

    def raw(self, values):
        result = pd.DataFrame(values, index=self.dates)
        result.columns = pd.MultiIndex.from_tuples([("Close", ticker) for ticker in result.columns])
        return result

    def fetch(self):
        return self.namespace["fetch_daily_levels"](["SPY", "AAA"], self.dates[0], self.dates[-1] + pd.Timedelta(days=1))

    def test_all_missing_provider_column_gets_retried(self):
        self.download.side_effect = [self.raw({"SPY": [100, 101, 102], "AAA": [None, None, None]}),
                                     self.raw({"AAA": [50, 51, 52]})]
        result, metadata = self.fetch()
        self.assertEqual(result["AAA"].tolist(), [50, 51, 52])
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(self.download.call_args.kwargs["tickers"], ["AAA"])
        self.assertEqual(metadata["missing_tickers"], [])

    def test_filling_a_cached_observation_is_disclosed(self):
        self.download.return_value = self.raw({"SPY": [100, 101, 102], "AAA": [50, None, 52]})
        self.cached.return_value = (pd.DataFrame({"SPY": [99, 100, 101], "AAA": [49, 51, 51]}, index=self.dates), {"source": "older"})
        self.usable.return_value = True
        result, metadata = self.fetch()
        self.assertEqual(result["AAA"].tolist(), [50, 51, 52])
        self.assertEqual(metadata["source"], "yahoo+cache")
        self.assertEqual(metadata["cache_meta"], {"source": "older"})

    def test_unused_cache_is_not_reported_as_used(self):
        self.download.return_value = self.raw({"SPY": [100, 101, 102], "AAA": [50, 51, 52]})
        self.cached.return_value = (pd.DataFrame({"SPY": [99, 100, 101], "AAA": [49, 50, 51]}, index=self.dates), {})
        self.usable.return_value = True
        _, metadata = self.fetch()
        self.assertEqual(metadata["source"], "yahoo")
