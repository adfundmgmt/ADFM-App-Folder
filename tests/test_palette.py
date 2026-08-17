from __future__ import annotations

import re
import unittest
from pathlib import Path

from adfm_core.catalog import TOOL_CATALOG
from adfm_core.palette import EXCEL, EXCEL_20, PASTEL, PASTEL_20, pastel

ROOT = Path(__file__).resolve().parents[1]


class ExcelPaletteTests(unittest.TestCase):
    def test_palette_has_exactly_twenty_distinct_hex_colors(self) -> None:
        self.assertEqual(len(PASTEL), 20)
        self.assertEqual(len(PASTEL_20), 20)
        self.assertEqual(len(set(PASTEL_20)), 20)
        for color in PASTEL_20:
            self.assertRegex(color, re.compile(r"^#[0-9A-F]{6}$"))

        self.assertIs(PASTEL, EXCEL)
        self.assertIs(PASTEL_20, EXCEL_20)

    def test_primary_colors_match_the_excel_finance_convention(self) -> None:
        self.assertEqual(EXCEL["blue"], "#4472C4")
        self.assertEqual(EXCEL["coral"], "#ED7D31")
        self.assertEqual(EXCEL["sage"], "#70AD47")
        self.assertEqual(EXCEL["amber"], "#FFC000")
        self.assertEqual(EXCEL["rose"], "#C0504D")

    def test_palette_cycles_stably_after_twenty_series(self) -> None:
        self.assertEqual(pastel(0), PASTEL_20[0])
        self.assertEqual(pastel(19), PASTEL_20[19])
        self.assertEqual(pastel(20), PASTEL_20[0])

    def test_every_analytical_page_uses_the_shared_palette(self) -> None:
        indirect_palette_pages = {
            "7_Sector_Breadth_and_Rotation.py": ROOT / "adfm_sector_rotation_config.py",
            "6_Currency_Tension_Dashboard.py": ROOT / "cte" / "dashboard" / "plots.py",
        }
        positioning_convention_pages = {
            "24_CFTC_Positioning_Monitor.py",
        }
        for tool in TOOL_CATALOG:
            page = ROOT / "pages" / tool.page_filename
            source = page.read_text(encoding="utf-8")
            if tool.page_filename in positioning_convention_pages:
                self.assertIn("POSITION_COLOR", source)
                self.assertIn("PRICE_COLOR", source)
                continue
            if tool.page_filename in indirect_palette_pages:
                source += indirect_palette_pages[tool.page_filename].read_text(
                    encoding="utf-8"
                )
            self.assertIn(
                "adfm_core.palette",
                source,
                msg=f"{tool.page_filename} is outside the shared output palette.",
            )


if __name__ == "__main__":
    unittest.main()
