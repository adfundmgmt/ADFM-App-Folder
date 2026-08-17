from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from adfm_core.catalog import tool_definitions


class HomeResearchDirectoryTests(unittest.TestCase):
    def test_home_renders_ordered_native_page_links(self):
        app = AppTest.from_file("Home.py", default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 0)

        expected_tools = tool_definitions()
        page_links = list(app.get("page_link"))
        self.assertEqual(len(page_links), len(expected_tools))
        self.assertEqual(
            [link.label for link in page_links],
            [f"**{tool.title}**" for tool in expected_tools],
        )
        self.assertEqual(
            [link.page for link in page_links],
            [
                Path(tool.page_filename).stem.partition("_")[2]
                for tool in expected_tools
            ],
        )
        self.assertTrue(all(not link.external for link in page_links))


if __name__ == "__main__":
    unittest.main()
