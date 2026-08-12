from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class HomeResearchDirectoryTests(unittest.TestCase):
    def test_home_renders_research_directory_and_clickable_tool_links(self):
        app = AppTest.from_file("Home.py", default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 0)
        markup = "\n".join(element.value for element in app.markdown)
        self.assertEqual(markup.count("class='directory-tool-link'"), 22)
        self.assertIn("href='/ADFM_Public_Equities_Baskets'", markup)
        self.assertIn("href='/ADFM_Underwriter'", markup)
        self.assertIn("href='/Global_Macro_Regime_Dashboard'", markup)
        self.assertLess(
            markup.index("ADFM Public Equities Baskets"),
            markup.index("Global Macro Regime Dashboard"),
        )
        self.assertLess(
            markup.index("Global Macro Regime Dashboard"),
            markup.index("Sector Breadth and Rotation"),
        )
        self.assertLess(
            markup.index("Factor Momentum Leadership"),
            markup.index("ADFM Underwriter"),
        )


if __name__ == "__main__":
    unittest.main()
