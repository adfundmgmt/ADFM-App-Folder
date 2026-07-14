"""Regression checks for the public application catalog."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def home_tool_order() -> list[str]:
    """Read TOOL_ORDER without importing Streamlit's application module."""
    tree = ast.parse((REPOSITORY_ROOT / "Home.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TOOL_ORDER" for target in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("Home.py does not define TOOL_ORDER.")


class DocumentationTests(unittest.TestCase):
    def test_readme_catalog_matches_home_tool_order(self) -> None:
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        tools = home_tool_order()

        self.assertEqual(len(tools), 19)
        for number, tool in enumerate(tools, start=1):
            self.assertIn(f"| {number} | {tool} |", readme)


if __name__ == "__main__":
    unittest.main()
