import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adfm_core.sector_rotation_ui import (
    STATE_PALETTE,
    display_name,
    full_extent_axis_range,
    select_auto_labels,
    style_rotation_table,
)


class SectorRotationUiTests(unittest.TestCase):
    def test_state_palette_uses_distinct_pastel_colors(self):
        self.assertEqual(set(STATE_PALETTE), {"Leading", "Improving", "Weakening", "Lagging", "Neutral"})
        self.assertEqual(STATE_PALETTE["Leading"], "#B9DFC4")
        self.assertEqual(STATE_PALETTE["Improving"], "#BFDDEC")
        self.assertEqual(STATE_PALETTE["Weakening"], "#F4DEAE")
        self.assertEqual(STATE_PALETTE["Lagging"], "#E9B9BD")
        self.assertEqual(STATE_PALETTE["Neutral"], "#DDE2E7")

    def test_display_name_humanizes_stock_baskets(self):
        self.assertEqual(display_name("BASKET_REFINERS", "Refiners"), "Refiners")
        self.assertEqual(display_name("XLK", "Technology"), "XLK")

    def test_full_extent_axis_range_keeps_single_outlier_visible(self):
        values = pd.Series([-0.12, -0.08, -0.04, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.80])
        low, high = full_extent_axis_range(values)
        self.assertLess(low, -0.12)
        self.assertGreater(high, 0.80)
        self.assertGreater(high - low, 0.92)

    def test_full_extent_axis_range_contracts_for_tight_cross_section(self):
        values = pd.Series([-0.004, -0.003, -0.002, 0.0, 0.001, 0.002, 0.004])
        low, high = full_extent_axis_range(values)
        self.assertLess(low, -0.004)
        self.assertGreater(high, 0.004)
        self.assertLess(high - low, 0.02)

    def test_full_extent_axis_range_expands_for_wide_cross_section(self):
        values = pd.Series([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
        low, high = full_extent_axis_range(values)
        self.assertLess(low, -0.30)
        self.assertGreater(high, 0.30)
        self.assertGreater(high - low, 0.60)

    def test_auto_labels_limit_dense_universe_and_keep_selected(self):
        frame = pd.DataFrame({
            "Id": [f"ID{i}" for i in range(30)],
            "Map X": np.linspace(-0.2, 0.2, 30),
            "Map Y": np.linspace(-0.1, 0.1, 30),
            "5D Speed": np.linspace(0.0, 0.08, 30),
        })
        labels = select_auto_labels(frame, selected_ids=["ID10"], max_labels=8)
        self.assertIn("ID10", labels)
        self.assertLessEqual(len(labels), 8)

    def test_rotation_table_uses_heatmap_and_state_fill(self):
        frame = pd.DataFrame({
            "ETF": ["XLK", "BASKET_REFINERS"],
            "Industry": ["Technology", "Refiners"],
            "State": ["Leading", "Lagging"],
            "1W Rel": [0.025, -0.018],
            "1M Rel": [0.06, -0.04],
            "3M Rel": [0.11, -0.07],
            "Weekly Rank Change": [3.0, -2.0],
            "1M Abs": [0.04, -0.03],
            "Dist. 50D": [0.08, -0.06],
            "Above 50D": [80.0, 25.0],
        })
        html = style_rotation_table(frame).to_html()
        self.assertIn("#B9DFC4", html)
        self.assertIn("#E9B9BD", html)
        self.assertIn("background-color", html)
        self.assertNotIn("BASKET_REFINERS", html)
        self.assertIn("Refiners", html)

    def test_compact_table_aliases_remain_formatted_and_heatmapped(self):
        frame = pd.DataFrame({
            "Exposure": ["Refiners"],
            "State": ["Leading"],
            "1W Δ Rel": [0.025],
            "Rank Δ": [3.0],
            "vs Parent": [0.04],
            "vs 50D": [0.05],
            ">50D": [80.0],
            "Breadth Δ": [12.0],
        })
        html = style_rotation_table(frame).to_html()
        self.assertIn("+2.5%", html)
        self.assertIn("+3", html)
        self.assertIn("80%", html)
        self.assertIn("#CFE8D8", html)
        self.assertIn("#A8D4B6", html)

    def test_page_orders_map_then_relative_strength_then_table(self):
        source = Path("pages/7_Sector_Breadth_and_Rotation.py").read_text(encoding="utf-8")
        map_pos = source.index('"Rotation map"')
        rs_pos = source.index('"Relative strength"')
        table_pos = source.index('"Rotation table"')
        self.assertLess(map_pos, rs_pos)
        self.assertLess(rs_pos, table_pos)

    def test_rotation_map_uses_short_uniform_tails(self):
        source = Path("pages/7_Sector_Breadth_and_Rotation.py").read_text(encoding="utf-8")
        self.assertIn(
            'trail_sessions = st.selectbox("Tail length", [3, 5, 8, 12], index=1',
            source,
        )
        self.assertIn('line=dict(color=_rgba(edge, 0.22), width=0.9)', source)
        self.assertNotIn('0.82 if selected else 0.28', source)
        self.assertNotIn('2.8 if selected else 1.15', source)

    def test_rotation_map_does_not_clip_outlier_coordinates(self):
        source = Path("pages/7_Sector_Breadth_and_Rotation.py").read_text(encoding="utf-8")
        self.assertIn('x_range = full_extent_axis_range(snapshot["Map X"])', source)
        self.assertIn('y_range = full_extent_axis_range(snapshot["Map Y"])', source)
        self.assertNotIn("clip_to_axis", source)
        self.assertNotIn("Diamond markers are clipped", source)


if __name__ == "__main__":
    unittest.main()