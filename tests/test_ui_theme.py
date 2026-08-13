from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adfm_core.ui import (
    PageHeader,
    inject_institutional_theme,
    inject_institutional_tool_finish,
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

        contract = markdown.call_args_list[0].args[0]
        self.assertIn("@media (max-width: 760px)", contract)
        self.assertIn(".modebar-container", contract)
        self.assertIn("overflow-x: clip", contract)
        self.assertIn("padding-top: calc(4rem + env(safe-area-inset-top, 0px))", contract)
        self.assertIn("padding: calc(4.25rem + env(safe-area-inset-top, 0px)) .9rem", contract)
        self.assertIn("white-space: normal !important", contract)
        self.assertIn("overflow: visible !important", contract)

    @patch("adfm_core.ui.tool_for_page")
    @patch("adfm_core.ui.st.markdown")
    def test_page_header_uses_catalog_title_and_group(self, markdown, tool_for_page):
        tool_for_page.return_value = SimpleNamespace(
            title="ADFM Public Equities Baskets",
            group="Equity Discovery",
        )

        render_page_header(
            PageHeader(
                title="Legacy page title",
                description="Catalog-controlled identity.",
                eyebrow="ADFM Equity Leadership",
            )
        )

        body = markdown.call_args.args[0]
        self.assertIn("ADFM Public Equities Baskets", body)
        self.assertIn("ADFM Equity Discovery", body)
        self.assertNotIn("Legacy page title", body)
        self.assertNotIn("ADFM Equity Leadership", body)

    def test_every_analytics_page_uses_the_shared_header(self):
        root = Path(__file__).resolve().parents[1]
        pages = sorted((root / "pages").glob("*.py"))

        self.assertEqual(22, len(pages))
        for page in pages:
            source = page.read_text(encoding="utf-8")
            with self.subTest(page=page.name):
                self.assertIn("render_page_header(", source)
                self.assertNotIn("st.title(", source)

    def test_mobile_first_render_contracts_for_legacy_tools(self):
        root = Path(__file__).resolve().parents[1]
        chart_terminal = (root / "pages" / "10_ADFM_Chart_Terminal.py").read_text(
            encoding="utf-8"
        )
        ratio_charts = (root / "pages" / "11_Ratio_Charts.py").read_text(
            encoding="utf-8"
        )
        sector_breadth = (
            root / "pages" / "7_Sector_Breadth_and_Rotation.py"
        ).read_text(encoding="utf-8")

        self.assertIn("grid-auto-flow: column", chart_terminal)
        self.assertIn('"displayModeBar": False', chart_terminal)
        self.assertIn('"scrollZoom": False', chart_terminal)

        self.assertIn("market_date = date.today()", ratio_charts)
        self.assertIn("history_days = max(900, display_days + 450)", ratio_charts)
        self.assertIn("ratio-chart-heading", ratio_charts)
        self.assertNotIn("yf_end = now + timedelta", ratio_charts)

        self.assertIn(
            "ticker for ticker in missing_tickers if ticker in BENCHMARKS",
            sector_breadth,
        )
        self.assertIn('config={"displayModeBar": False, "responsive": True}', sector_breadth)
        self.assertLess(
            sector_breadth.index("config={\"displayModeBar\": False"),
            sector_breadth.index("Dropped {len(excluded_tickers)} ticker(s)"),
        )

    @patch("adfm_core.ui.st.markdown")
    def test_legacy_tool_finish_removes_colored_rounded_chrome(self, markdown):
        inject_institutional_tool_finish()

        body = markdown.call_args.args[0]
        self.assertIn(".hero-title", body)
        self.assertIn(".metric-card", body)
        self.assertIn("border-radius: 0", body)
        self.assertIn("background: #ffffff", body)
        self.assertIn("color: #000000", body)

    @patch("adfm_core.ui.st.markdown")
    def test_footer_uses_the_shared_rule_and_firm_lockup(self, markdown):
        render_footer(data_note="Primary inputs: official data.")

        body = markdown.call_args.args[0]
        self.assertTrue(body.startswith("<footer class='adfm-footer'>"))
        self.assertIn("Primary inputs: official data.", body)
        self.assertIn("AD Fund Management LP", body)


if __name__ == "__main__":
    unittest.main()
