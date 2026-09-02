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
    render_sidebar_about,
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

    def test_home_and_shared_headers_reserve_toolbar_safe_area(self):
        root = Path(__file__).resolve().parents[1]
        home = (root / "Home.py").read_text(encoding="utf-8")

        self.assertIn(
            "padding: calc(3.25rem + env(safe-area-inset-top, 0px)) 2rem",
            home,
        )
        self.assertIn(
            "padding: calc(3.2rem + env(safe-area-inset-top, 0px)) 1rem",
            home,
        )
        self.assertIn("white-space: normal !important", home)
        self.assertIn("overflow: visible !important", home)

    def test_every_analytics_page_uses_the_shared_header(self):
        root = Path(__file__).resolve().parents[1]
        pages = sorted((root / "pages").glob("*.py"))
        shared_renderers = {
            "17_SEC_13F_Exposure_Browser.py": root / "adfm_core" / "sec_13f_browser.py",
            "20_Catalyst_Calendar.py": root / "adfm_core" / "catalyst_calendar_exact_page.py",
            "25_Commodity_Event_Study.py": root / "adfm_core" / "commodity_top_exhaustion_page.py",
        }

        self.assertEqual(25, len(pages))
        for page in pages:
            source = page.read_text(encoding="utf-8")
            if page.name in shared_renderers:
                source += shared_renderers[page.name].read_text(encoding="utf-8")
            with self.subTest(page=page.name):
                self.assertIn("render_page_header(", source)
                self.assertNotIn("st.title(", source)

    @patch("adfm_core.ui.st.divider")
    @patch("adfm_core.ui.st.caption")
    @patch("adfm_core.ui.st.markdown")
    @patch("adfm_core.ui.st.header")
    def test_sidebar_about_uses_the_shared_reading_flow(
        self, header, markdown, caption, divider
    ):
        render_sidebar_about("17_SEC_13F_Exposure_Browser.py")

        header.assert_called_once_with("About This Tool")
        body = markdown.call_args.args[0]
        self.assertIn("**Purpose**", body)
        self.assertIn("**Read it in this order**", body)
        self.assertIn("1. Choose a security search", body)
        self.assertEqual(caption.call_count, 2)
        self.assertIn("Keep in mind", caption.call_args_list[0].args[0])
        self.assertIn("Primary inputs", caption.call_args_list[1].args[0])
        divider.assert_called_once_with()

    def test_mobile_first_render_contracts_for_legacy_tools(self):
        root = Path(__file__).resolve().parents[1]
        chart_terminal = (root / "pages" / "10_ADFM_Chart_Terminal.py").read_text(
            encoding="utf-8"
        )
        ratio_charts = (
            root / "pages" / "11_Cross-Asset_Ratio_Chartbook.py"
        ).read_text(
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

    def test_leadership_and_ratio_pages_default_to_expanded_chartbooks(self):
        root = Path(__file__).resolve().parents[1]
        leadership = (
            root / "pages" / "8_Equity_Leadership_&_Rotation.py"
        ).read_text(encoding="utf-8")
        ratio_chartbook = (
            root / "pages" / "11_Cross-Asset_Ratio_Chartbook.py"
        ).read_text(encoding="utf-8")

        self.assertIn('st.subheader("Rotation Map")', leadership)
        self.assertIn('st.subheader("Leadership Charts")', leadership)
        self.assertLess(
            leadership.index('st.subheader("Rotation Map")'),
            leadership.index('st.subheader("Leadership Charts")'),
        )
        self.assertIn('st.columns(2, gap="large")', leadership)
        self.assertNotIn("Current read", leadership)
        self.assertNotIn("Leadership by Family", leadership)
        self.assertNotIn("Multi-Horizon Leadership Matrix", leadership)
        self.assertNotIn("Leadership Ranking", leadership)
        self.assertNotIn("Selected Detail", leadership)

        self.assertIn("for family, specs in selected_groups.items():", ratio_chartbook)
        self.assertIn("render_spec(spec, compact=True)", ratio_chartbook)
        self.assertNotIn("Focused relationship", ratio_chartbook)
        self.assertNotIn("Full chartbook", ratio_chartbook)
        self.assertNotIn("view_mode", ratio_chartbook)

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
