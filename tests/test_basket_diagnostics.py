"""Regression tests for the public-equities basket decision fields."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class BasketDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        page = (
            Path(__file__).resolve().parents[1]
            / "pages"
            / "1_ADFM_Public_Equities_Baskets.py"
        )
        tree = ast.parse(page.read_text(encoding="utf-8"))
        names = {
            "pct_since",
            "basket_vs_dma_pct",
            "ema_regime",
            "build_basket_diagnostics",
            "build_constituent_table",
            "build_panel_df",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *functions,
            ],
            type_ignores=[],
        )
        cls.namespace = {
            "pd": pd,
            "np": np,
            "BASKET_KEY_SEPARATOR": " :: ",
        }
        exec(compile(ast.fix_missing_locations(module), str(page), "exec"), cls.namespace)

    def test_breadth_dispersion_and_member_count(self):
        dates = pd.bdate_range("2026-01-02", periods=60)
        levels = pd.DataFrame(
            {
                "AAA": 100.0 + np.arange(60),
                "BBB": 200.0 - np.arange(60),
            },
            index=dates,
        )
        result = self.namespace["build_basket_diagnostics"](
            levels,
            {"Test :: Pair": ["AAA", "BBB"]},
        )["Test :: Pair"]

        self.assertEqual(result["Breadth >50DMA %"], 50.0)
        self.assertGreater(result["1M Dispersion %"], 0.0)
        self.assertEqual(result["Members"], "2")

    def test_relative_trend_uses_basket_to_spy_ratio(self):
        dates = pd.bdate_range("2026-01-02", periods=40)
        basket_returns = pd.DataFrame(
            {"Test :: Slower": np.repeat(0.005, len(dates))},
            index=dates,
        )
        benchmark_returns = pd.Series(np.repeat(0.01, len(dates)), index=dates)
        panel = self.namespace["build_panel_df"](
            basket_returns_full=basket_returns,
            display_start=dates[0],
            dynamic_label="YTD",
            basket_metadata={
                "Test :: Slower": {
                    "Category": "Test",
                    "Basket": "Slower",
                    "Live Members": 2,
                }
            },
            benchmark_series_full=benchmark_returns,
            basket_diagnostics={
                "Test :: Slower": {
                    "Breadth >50DMA %": 50.0,
                    "1M Dispersion %": 2.0,
                    "Members": "2",
                }
            },
        )

        self.assertEqual(panel.loc["Test :: Slower", "Relative Trend"], "Down")
        self.assertLess(panel.loc["Test :: Slower", "vs SPY YTD"], 0.0)


if __name__ == "__main__":
    unittest.main()
