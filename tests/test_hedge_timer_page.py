"""Integration guards for the Hedge Timer Streamlit page."""

from __future__ import annotations

import unittest
from pathlib import Path

from adfm_core.catalog import sidebar_guide_for_page, tool_for_page

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "21_Hedge_Timer.py"


class HedgeTimerPageTests(unittest.TestCase):
    def test_page_uses_recall_model_and_keeps_spx_ndx_separate(self) -> None:
        source = PAGE.read_text(encoding="utf-8")

        self.assertIn("from adfm_core.hedge_timer_model import", source)
        self.assertIn("calibrate_watch_threshold", source)
        self.assertIn("episode_audit", source)
        self.assertIn("warning_summary", source)
        self.assertNotIn("pick_target_today", source)
        self.assertNotIn('"TLT"', source)

    def test_chart_has_one_score_line_and_confirmation_dots_on_price(self) -> None:
        source = PAGE.read_text(encoding="utf-8")

        self.assertIn('label="Hedge Score"', source)
        self.assertIn('label="Confirmed"', source)
        self.assertIn("ax_price.scatter(", source)
        self.assertNotIn("ax_score.scatter(", source)
        self.assertNotIn('label="Hedge Watch onset"', source)
        self.assertNotIn('label="Short allowed onset"', source)
        self.assertNotIn('label="Confirmation"', source)

    def test_about_metadata_matches_the_recall_model(self) -> None:
        tool = tool_for_page("21_Hedge_Timer.py")
        guide = sidebar_guide_for_page("21_Hedge_Timer.py")

        self.assertIsNotNone(tool)
        self.assertIsNotNone(guide)
        assert tool is not None
        assert guide is not None
        self.assertIn("high-recall", tool.description.lower())
        self.assertIn("10%+", tool.description)
        self.assertIn("IWM", tool.primary_inputs)
        self.assertIn("sector ETFs", tool.primary_inputs)
        self.assertIn("Hedge Watch", " ".join(guide.read_order))
        self.assertIn("local", " ".join(guide.read_order).lower())


if __name__ == "__main__":
    unittest.main()
