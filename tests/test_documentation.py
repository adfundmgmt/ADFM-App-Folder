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
        self.assertEqual(TOOL_CATALOG[0].title, "ADFM Public Equities Baskets")
        self.assertEqual(TOOL_CATALOG[8].page_filename, "9_ADFM_Underwriter.py")
        for tool in TOOL_CATALOG:
            self.assertTrue((REPOSITORY_ROOT / "pages" / tool.page_filename).is_file())
            self.assertTrue(tool.page_filename.startswith(f"{tool.number}_"))

    def test_catalog_follows_the_research_workflow(self) -> None:
        self.assertEqual(
            [tool.title for tool in TOOL_CATALOG],
            [
                "ADFM Public Equities Baskets",
                "Global Macro Regime",
                "Liquidity Conditions Monitor",
                "Yield Curve Rates Regime Monitor",
                "Credit Conditions Monitor",
                "Currency Tension Engine",
                "Sector Breadth and Rotation",
                "Equity Leadership & Rotation",
                "ADFM Underwriter",
                "ADFM Chart Terminal",
                "Cross-Asset Ratio Chartbook",
                "Rate of Change Regime Explorer",
                "Relative Volatility Lab",
                "ETF Flow Pressure Proxy",
                "Volume Based Sentiment Indicator",
                "Options Positioning Compass",
                "SEC 13F Exposure Browser",
                "CFTC Positioning Monitor",
                "Market Stress Composite",
                "Catalyst Calendar",
                "Hedge Timer",
                "Position Sizing Lab",
                "Market Memory Explorer",
                "Monthly Seasonality Explorer",
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

    def test_sidebar_labels_match_catalog_titles(self) -> None:
        for tool in TOOL_CATALOG:
            stem = Path(tool.page_filename).stem
            sidebar_label = re.sub(r"^\d+_", "", stem).replace("_", " ")
            self.assertEqual(sidebar_label, tool.title)

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
