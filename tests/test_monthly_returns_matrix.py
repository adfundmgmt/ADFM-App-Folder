import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adfm_core.monthly_returns_matrix import (
    _build_matrix_html,
    _compound_pct,
    _display_years,
    _prepare_month_table,
)


class MonthlyReturnsMatrixTests(unittest.TestCase):
    def test_compound_pct_uses_geometric_linking(self) -> None:
        self.assertAlmostEqual(_compound_pct([10.0, -5.0]), 4.5)
        self.assertTrue(np.isnan(_compound_pct([np.nan, None])))

    def test_prepare_month_table_recovers_period_fields(self) -> None:
        raw = pd.DataFrame(
            {"total_ret": [1.5, -2.0]},
            index=pd.PeriodIndex(["2025-12", "2026-01"], freq="M"),
        )
        prepared = _prepare_month_table(raw)
        self.assertEqual(prepared["year"].tolist(), [2025, 2026])
        self.assertEqual(prepared["month"].tolist(), [12, 1])

    def test_display_years_keeps_current_year_first(self) -> None:
        raw = pd.DataFrame(
            {
                "year": [2022, 2023, 2024, 2025],
                "month": [12, 12, 12, 12],
                "total_ret": [1.0, 2.0, 3.0, 4.0],
            },
            index=pd.period_range("2022-12", periods=4, freq="12M"),
        )
        years = _display_years(raw, years_to_show=5, as_of=pd.Timestamp("2026-07-16"))
        self.assertEqual(years, [2026, 2025, 2024, 2023, 2022])

    def test_matrix_marks_live_period_and_compounds_ytd(self) -> None:
        months = pd.DataFrame(
            {
                "year": [2026, 2026, 2026],
                "month": [1, 2, 7],
                "total_ret": [10.0, -5.0, 2.0],
            },
            index=pd.PeriodIndex(["2026-01", "2026-02", "2026-07"], freq="M"),
        )
        stats = pd.DataFrame(
            {
                "mean_total": [1.0] * 12,
                "hit_rate": [60.0] * 12,
            },
            index=pd.Index(range(1, 13), name="month"),
        )
        html, values = _build_matrix_html(
            months=months,
            stats=stats,
            years=[2026],
            current_period=pd.Period("2026-07", freq="M"),
        )
        self.assertIn("monthly-now-badge", html)
        self.assertIn("is-now", html)
        self.assertIn("+6.6%", html)
        self.assertEqual(values.tolist(), [10.0, -5.0, 2.0])

    def test_retired_3d_terrain_stays_out_of_seasonality_page(self) -> None:
        page = Path("pages/8_Monthly_Seasonality_Explorer.py").read_text(encoding="utf-8")
        self.assertNotIn("3D Seasonal Waterfall Terrain", page)
        self.assertNotIn("plot_3d_seasonality_terrain", page)
        self.assertNotIn("plotly.graph_objects", page)
        self.assertIn("render_monthly_returns_lens", page)

    def test_currency_snapshot_force_stages_ignored_cache(self) -> None:
        workflow = Path(".github/workflows/sync_currency_tension_snapshot.yml").read_text(encoding="utf-8")
        self.assertIn("git add -f -- data/cache", workflow)


if __name__ == "__main__":
    unittest.main()

