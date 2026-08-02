from __future__ import annotations

import unittest
from unittest.mock import patch

from adfm_core.ui import (
    PageHeader,
    inject_institutional_theme,
    render_footer,
    render_page_header,
)


class InstitutionalThemeTests(unittest.TestCase):
    @patch("adfm_core.ui.st.markdown")
    def test_shared_theme_is_black_white_and_square(self, markdown):
        inject_institutional_theme(max_width_px=1440)

        body = markdown.call_args.args[0]
        self.assertIn("--adfm-black: #000000", body)
        self.assertIn("--adfm-white: #ffffff", body)
        self.assertIn("max-width: 1440px", body)
        self.assertIn("border-radius: 0", body)
        self.assertNotIn("prefers-color-scheme: dark", body)

    @patch("adfm_core.ui.st.markdown")
    def test_page_header_uses_the_institutional_masthead(self, markdown):
        render_page_header(
            PageHeader(
                title="Liquidity Tracker",
                description="Tracks major liquidity drivers.",
                eyebrow="ADFM Analytics",
                as_of="2026-07-31",
            )
        )

        body = markdown.call_args.args[0]
        self.assertTrue(body.startswith("<header class='adfm-page-header'>"))
        self.assertTrue(body.endswith("</header>"))
        self.assertIn("Liquidity Tracker", body)
        self.assertIn("2026-07-31", body)

    @patch("adfm_core.ui.st.markdown")
    def test_footer_uses_the_shared_rule_and_firm_lockup(self, markdown):
        render_footer(data_note="Primary inputs: official data.")

        body = markdown.call_args.args[0]
        self.assertTrue(body.startswith("<footer class='adfm-footer'>"))
        self.assertIn("Primary inputs: official data.", body)
        self.assertIn("AD Fund Management LP", body)


if __name__ == "__main__":
    unittest.main()
