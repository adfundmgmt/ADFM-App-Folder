from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adfm_core.chart_patterns import SUPPORTED_PATTERN_NAMES, detect_chart_patterns

ROOT = Path(__file__).resolve().parents[1]


def price_frame(anchors: list[tuple[int, float]]) -> pd.DataFrame:
    positions = np.arange(anchors[-1][0] + 1)
    close = np.interp(
        positions,
        [point[0] for point in anchors],
        [point[1] for point in anchors],
    )
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.35,
            "Low": close - 0.35,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000),
        },
        index=pd.date_range("2025-01-02", periods=len(close), freq="B"),
    )


def detected_names(anchors: list[tuple[int, float]]) -> set[str]:
    return {item.name for item in detect_chart_patterns(price_frame(anchors), max_patterns=5)}


class ChartPatternTests(unittest.TestCase):
    def test_registry_covers_every_requested_pattern(self) -> None:
        self.assertEqual(set(SUPPORTED_PATTERN_NAMES), {
        "Double Bottom",
        "Double Top",
        "Inverse Head and Shoulders",
        "Head and Shoulders Top",
        "Triple Bottom",
        "Triple Top",
        "Rounding Bottom",
        "Rounding Top",
        "Cup and Handle",
        "Inverse Cup and Handle",
        "Diamond Bottom",
        "Diamond Top",
        "V-Bottom",
        "V-Top",
        "Ascending Triangle",
        "Descending Triangle",
        "Symmetrical Triangle",
        "Bull Flag",
        "Bear Flag",
        "Bull Pennant",
        "Bear Pennant",
        "Rectangle",
        "Rising Wedge",
        "Falling Wedge",
        "Megaphone / Broadening Formation",
        })


    def test_detects_confirmed_double_bottom(self) -> None:
        data = price_frame([(0, 120), (20, 108), (35, 95), (52, 108), (70, 95.5), (92, 112)])
        patterns = detect_chart_patterns(data, max_patterns=5)
        pattern = next(item for item in patterns if item.name == "Double Bottom")
        self.assertEqual(pattern.status, "Confirmed")
        self.assertEqual(pattern.bias, "bullish")
        self.assertIsNotNone(pattern.breakout_level)
        self.assertIsNotNone(pattern.target_level)


    def test_detects_head_and_shoulders_top(self) -> None:
        names = detected_names([(0, 90), (15, 105), (25, 96), (38, 114), (51, 96.5), (64, 104.5), (80, 91)])
        self.assertIn("Head and Shoulders Top", names)


    def test_detects_ascending_triangle(self) -> None:
        names = detected_names([(0, 90), (12, 100), (22, 92), (34, 100), (44, 94), (56, 100), (68, 96), (82, 100.5)])
        self.assertIn("Ascending Triangle", names)


    def test_detects_bull_flag(self) -> None:
        names = detected_names([(0, 90), (20, 92), (36, 115), (42, 113), (48, 114), (54, 111.5), (60, 112.5), (68, 109.5), (74, 112)])
        self.assertIn("Bull Flag", names)

    def test_detects_additional_reversal_structures(self) -> None:
        cases = {
            "Inverse Head and Shoulders": [(0, 120), (15, 104), (25, 113), (38, 94), (51, 112.5), (64, 103.5), (80, 120)],
            "Triple Bottom": [(0, 120), (15, 100), (27, 112), (39, 100.5), (51, 112.2), (63, 99.8), (80, 116)],
            "Triple Top": [(0, 90), (15, 110), (27, 99), (39, 110.3), (51, 98.8), (63, 109.8), (80, 94)],
        }
        for expected, anchors in cases.items():
            with self.subTest(pattern=expected):
                self.assertIn(expected, detected_names(anchors))

    def test_detects_triangle_wedge_rectangle_and_broadening_structures(self) -> None:
        cases = {
            "Descending Triangle": [(0, 110), (12, 100), (22, 108), (34, 100), (44, 106), (56, 100), (68, 104), (82, 99)],
            "Symmetrical Triangle": [(0, 90), (12, 110), (22, 92), (34, 107), (44, 95), (56, 104), (68, 98), (82, 101)],
            "Rectangle": [(0, 100), (12, 110), (24, 100), (36, 110), (48, 100), (60, 110), (72, 100), (84, 105)],
            "Rising Wedge": [(0, 90), (12, 100), (22, 94), (34, 104), (44, 100), (56, 107), (68, 105), (82, 108)],
            "Falling Wedge": [(0, 120), (12, 110), (22, 116), (34, 106), (44, 111), (56, 102), (68, 106), (82, 104)],
            "Megaphone / Broadening Formation": [(0, 100), (12, 108), (22, 96), (34, 112), (44, 92), (56, 117), (68, 87), (82, 101)],
        }
        for expected, anchors in cases.items():
            with self.subTest(pattern=expected):
                self.assertIn(expected, detected_names(anchors))

    def test_detects_bear_flag_and_both_pennants(self) -> None:
        cases = {
            "Bear Flag": [(0, 120), (20, 118), (36, 95), (42, 97), (48, 96), (54, 99), (60, 98), (68, 101), (74, 98)],
            "Bull Pennant": [(0, 90), (20, 92), (35, 116), (40, 113), (45, 116), (50, 113.5), (55, 115.5), (60, 114), (65, 115), (72, 116)],
            "Bear Pennant": [(0, 120), (20, 118), (35, 94), (40, 97), (45, 94), (50, 96.5), (55, 94.5), (60, 96), (65, 95), (72, 94)],
        }
        for expected, anchors in cases.items():
            with self.subTest(pattern=expected):
                self.assertIn(expected, detected_names(anchors))

    def test_detects_rounded_and_cup_structures(self) -> None:
        x = np.linspace(-1, 1, 100)
        rounding_bottom = 100 + 20 * x**2
        rounding_top = 120 - 20 * x**2

        for expected, close in (("Rounding Bottom", rounding_bottom), ("Rounding Top", rounding_top)):
            data = pd.DataFrame({"Open": close, "High": close + 0.3, "Low": close - 0.3, "Close": close})
            with self.subTest(pattern=expected):
                self.assertIn(expected, {item.name for item in detect_chart_patterns(data, max_patterns=5)})

        cup = 100 + 20 * np.linspace(-1, 1, 80) ** 2
        handle = np.r_[np.linspace(120, 114, 10), np.linspace(114, 118, 10)]
        for expected, close in (("Cup and Handle", np.r_[cup, handle]), ("Inverse Cup and Handle", 220 - np.r_[cup, handle])):
            data = pd.DataFrame({"Open": close, "High": close + 0.3, "Low": close - 0.3, "Close": close})
            with self.subTest(pattern=expected):
                self.assertIn(expected, {item.name for item in detect_chart_patterns(data, max_patterns=5)})


    def test_empty_or_short_data_returns_no_pattern(self) -> None:
        self.assertEqual(detect_chart_patterns(pd.DataFrame()), [])
        self.assertEqual(detect_chart_patterns(price_frame([(0, 100), (10, 105)])), [])


    def test_chart_terminal_exposes_toggle_overlay_and_summary(self) -> None:
        source = (ROOT / "pages" / "5_ADFM_Chart_Terminal.py").read_text(encoding="utf-8")
        self.assertIn('"Automatic chart patterns"', source)
        self.assertIn("add_chart_pattern_overlay", source)
        self.assertIn("pattern_summary_html", source)


if __name__ == "__main__":
    unittest.main()
