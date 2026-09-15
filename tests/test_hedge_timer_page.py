"""Integration guards for the Hedge Timer Streamlit page."""

from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
