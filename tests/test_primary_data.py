from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from adfm_core.data_registry import SeriesDefinition
from adfm_core.primary_data import fetch_fred_series


class PrimaryDataTests(unittest.TestCase):
    @patch("adfm_core.primary_data.web.DataReader")
    def test_fetch_preserves_missing_values_and_reports_source_status(self, reader):
        index = pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29"])
        reader.return_value = pd.DataFrame({"DGS10": [4.50, None, 4.55]}, index=index)
        definition = SeriesDefinition(
            key="dgs10",
            label="US 10-Year Treasury",
            symbol="DGS10",
            provider="Federal Reserve FRED",
            group="Rates",
            description="Primary yield.",
        )

        panel, diagnostics = fetch_fred_series(
            (definition,), start="2026-07-01", end="2026-07-30"
        )

        self.assertEqual(panel["dgs10"].notna().sum(), 2)
        self.assertTrue(pd.isna(panel.loc[pd.Timestamp("2026-07-28"), "dgs10"]))
        self.assertEqual(diagnostics.loc[0, "status"], "OK")
        self.assertEqual(diagnostics.loc[0, "data_through"], "2026-07-29")

    @patch("adfm_core.primary_data.web.DataReader", side_effect=RuntimeError("offline"))
    def test_one_provider_failure_returns_diagnostics(self, _reader):
        definition = SeriesDefinition(
            key="dgs10",
            label="US 10-Year Treasury",
            symbol="DGS10",
            provider="Federal Reserve FRED",
            group="Rates",
            description="Primary yield.",
        )
        panel, diagnostics = fetch_fred_series((definition,))
        self.assertTrue(panel.empty)
        self.assertEqual(diagnostics.loc[0, "status"], "FAILED")
        self.assertIn("RuntimeError", diagnostics.loc[0, "error"])


if __name__ == "__main__":
    unittest.main()
