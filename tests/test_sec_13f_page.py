"""Streamlit render tests for the SEC 13F Exposure Browser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.sec_13f import QuarterDataset
from tests.test_sec_13f import prepared_fixture

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "23_SEC_13F_Exposure_Browser.py"


class Sec13FPageTests(unittest.TestCase):
    def test_page_renders_real_chart_and_detail_table_from_fixture(self) -> None:
        release = QuarterDataset(
            slug="fixture",
            label="Fixture SEC release",
            url="https://www.sec.gov/fixture.zip",
            size_label="1 KB",
        )
        ticker_directory = pd.DataFrame(
            [{"TICKER": "INTC", "COMPANY_NAME": "Intel Corp", "CIK": 50863}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepared_fixture(Path(temporary))
            with (
                patch(
                    "adfm_core.sec_13f.discover_quarter_datasets",
                    return_value=[release],
                ),
                patch(
                    "adfm_core.sec_13f.prepare_dataset",
                    return_value=prepared,
                ),
                patch(
                    "adfm_core.sec_13f.load_company_tickers",
                    return_value=ticker_directory,
                ),
            ):
                app = AppTest.from_file(str(PAGE)).run(timeout=30)
                self.assertEqual(app.number_input[0].value, 1.0)
                app.number_input[0].set_value(0.0)
                app = app.button[0].click().run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Overview", "Fund holdings", "Methodology"],
        )
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertGreaterEqual(len(app.dataframe), 1)
        self.assertTrue(
            any(item.label == "Filter managers" for item in app.text_input)
        )
        self.assertTrue(
            any(item.label == "Customize columns" for item in app.multiselect)
        )
        self.assertTrue(
            any("Highest allocation" in block.value for block in app.markdown)
        )

    def test_manager_mode_opens_portfolio_by_cik(self) -> None:
        release = QuarterDataset(
            slug="fixture",
            label="Fixture SEC release",
            url="https://www.sec.gov/fixture.zip",
            size_label="1 KB",
        )
        ticker_directory = pd.DataFrame(
            [{"TICKER": "INTC", "COMPANY_NAME": "Intel Corp", "CIK": 50863}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepared_fixture(Path(temporary))
            with (
                patch(
                    "adfm_core.sec_13f.discover_quarter_datasets",
                    return_value=[release],
                ),
                patch(
                    "adfm_core.sec_13f.prepare_dataset",
                    return_value=prepared,
                ),
                patch(
                    "adfm_core.sec_13f.load_company_tickers",
                    return_value=ticker_directory,
                ),
            ):
                app = AppTest.from_file(str(PAGE)).run(timeout=30)
                app = app.radio[0].set_value("Manager").run(timeout=30)
                manager_input = next(
                    item for item in app.text_input if item.label == "Manager name or CIK"
                )
                self.assertEqual(
                    manager_input.value,
                    "Duquesne Family Office LLC (CIK: 0001536411)",
                )
                manager_input.set_value("0000000001")
                app = app.button[0].click().run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any("ALPHA CAPITAL" in block.value for block in app.markdown)
        )
        self.assertTrue(
            any(item.label == "Filter portfolio" for item in app.text_input)
        )
        self.assertGreaterEqual(len(app.dataframe), 1)


if __name__ == "__main__":
    unittest.main()
