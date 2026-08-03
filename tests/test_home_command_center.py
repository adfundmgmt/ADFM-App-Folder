from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class HomeResearchDirectoryTests(unittest.TestCase):
    def test_home_renders_research_directory_and_clickable_tool_links(self):
        app = AppTest.from_file("Home.py", default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("page_link")), 20)


if __name__ == "__main__":
    unittest.main()
