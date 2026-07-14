"""Regression checks for the public application catalog and documentation."""

from __future__ import annotations

import unittest
from pathlib import Path

from adfm_core.catalog import (
    TOOL_CATALOG,
    tool_descriptions,
    tool_for_page,
    tool_groups,
    tool_order,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_catalog_contains_19_unique_existing_pages(self) -> None:
        self.assertEqual(len(TOOL_CATALOG), 19)
        self.assertEqual([tool.number for tool in TOOL_CATALOG], list(range(1, 20)))
        self.assertEqual(len({tool.title for tool in TOOL_CATALOG}), 19)
        for tool in TOOL_CATALOG:
            self.assertTrue((REPOSITORY_ROOT / "pages" / tool.page_filename).is_file())

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
