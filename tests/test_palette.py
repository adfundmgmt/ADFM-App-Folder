from __future__ import annotations

import re
import unittest
from pathlib import Path

from adfm_core.palette import PASTEL, PASTEL_20, pastel
from adfm_core.catalog import TOOL_CATALOG


ROOT = Path(__file__).resolve().parents[1]


class PastelPaletteTests(unittest.TestCase):
    def test_palette_has_exactly_twenty_distinct_hex_colors(self) -> None:
        self.assertEqual(len(PASTEL), 20)
        self.assertEqual(len(PASTEL_20), 20)
        self.assertEqual(len(set(PASTEL_20)), 20)
        for color in PASTEL_20:
            self.assertRegex(color, re.compile(r"^#[0-9A-F]{6}$"))

    def test_palette_cycles_stably_after_twenty_series(self) -> None:
        self.assertEqual(pastel(0), PASTEL_20[0])
        self.assertEqual(pastel(19), PASTEL_20[19])
        self.assertEqual(pastel(20), PASTEL_20[0])

    def test_every_analytical_page_uses_the_shared_palette(self) -> None:
        indirect_palette_pages = {
            "2_Sector_Breadth_and_Rotation.py": ROOT / "adfm_sector_rotation_config.py",
            "18_Currency_Tension_Dashboard.py": ROOT / "cte" / "dashboard" / "plots.py",
        }
        for tool in TOOL_CATALOG:
            page = ROOT / "pages" / tool.page_filename
            source = page.read_text(encoding="utf-8")
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
