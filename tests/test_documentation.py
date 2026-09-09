"""Regression checks for the public application catalog and documentation."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from adfm_core.catalog import (
    SIDEBAR_GUIDES,
    TOOL_CATALOG,
    sidebar_guide_for_page,
    tool_descriptions,
    tool_for_page,
    tool_groups,
    tool_order,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_catalog_contains_25_unique_existing_pages(self) -> None:
        self.assertEqual(len(TOOL_CATALOG), 25)
        self.assertEqual([tool.number for tool in TOOL_CATALOG], list(range(1, 26)))
        self.assertEqual(len({tool.title for tool in TOOL_CATALOG}), 25)
        self.assertEqual(TOOL_CATALOG[0].title, "Equity Baskets")
        self.assertEqual(TOOL_CATALOG[8].page_filename, "9_ADFM_Underwriter.py")
        for tool in TOOL_CATALOG:
            self.assertTrue((REPOSITORY_ROOT / "pages" / tool.page_filename).is_file())
            self.assertTrue(tool.page_filename.startswith(f"{tool.number}_"))

    def test_catalog_follows_the_research_workflow(self) -> None:
        self.assertEqual(
            [tool.title for tool in TOOL_CATALOG],
            [
                "Equity Baskets",
                "Global Macro",
                "Liquidity",
                "Rates & Yield Curve",
                "Credit Conditions",
                "FX Regime",
                "Sector Rotation",
                "Equity Leadership",
                "Equity Underwriter",
                "Chart Terminal",
                "Cross-Asset Ratios",
                "Momentum & Rate of Change",
                "Relative Volatility",
                "ETF Flow Pressure",
                "Volume Sentiment",
                "Options Positioning",
                "13F Holdings",
                "CFTC Positioning",
                "Market Stress",
                "Catalyst Calendar",
                "Hedge Timing",
                "Position Sizing",
                "Market Memory",
                "Seasonality",
                "Commodity Event Study",
            ],
        )

    def test_every_tool_has_one_concise_sidebar_guide(self) -> None:
        self.assertEqual(set(SIDEBAR_GUIDES), {tool.page_filename for tool in TOOL_CATALOG})
        for tool in TOOL_CATALOG:
            guide = sidebar_guide_for_page(tool.page_filename)
            self.assertIsNotNone(guide)
            self.assertEqual(len(guide.read_order), 3)
            self.assertTrue(all(step.endswith(".") for step in guide.read_order))

    def test_visible_titles_are_independent_from_legacy_filenames(self) -> None:
        legacy_labels = [
            re.sub(r"^\d+_", "", Path(tool.page_filename).stem).replace("_", " ")
            for tool in TOOL_CATALOG
        ]
        self.assertEqual(len(set(legacy_labels)), 25)
        self.assertTrue(
            any(label != tool.title for label, tool in zip(legacy_labels, TOOL_CATALOG))
        )

    def test_home_navigation_maps_to_catalog(self) -> None:
        self.assertEqual(tool_order(), [tool.title for tool in TOOL_CATALOG])
        self.assertEqual(tool_groups()["All tools"], tool_order())
        self.assertEqual(tool_descriptions(), {tool.title: tool.description for tool in TOOL_CATALOG})
        for tool in TOOL_CATALOG:
            self.assertEqual(tool_for_page(f"pages/{tool.page_filename}"), tool)
        self.assertIsNone(tool_for_page("unknown.py"))

    def test_readme_catalog_matches_the_shared_tool_catalog(self) -> None:
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        for tool in TOOL_CATALOG:
            self.assertIn(f"| {tool.number} | {tool.title} |", readme)


if __name__ == "__main__":
    unittest.main()
